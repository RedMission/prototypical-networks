import os
import json
import time
from functools import partial
from tqdm import tqdm

import numpy as np

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler 
import torchvision
import torchnet as tnt
# import torchtnt as tnt
from protonets.data.palmprint import palmdataloader
from protonets.engine import Engine

import protonets.utils.data as data_utils # 加载数据的方法
import protonets.utils.model as model_utils # 模型用到的类
import protonets.utils.log as log_utils

def main(opt):
    if not os.path.isdir(opt['log.exp_dir']):
        os.makedirs(opt['log.exp_dir'])

    # save opts
    with open(os.path.join(opt['log.exp_dir'], 'opt.json'), 'w') as f:
        json.dump(opt, f)
        f.write('\n')

    trace_file = os.path.join(opt['log.exp_dir'], 'trace.txt')

    # Postprocess arguments
    opt['model.x_dim'] = list(map(int, opt['model.x_dim'].split(',')))
    opt['log.fields'] = opt['log.fields'].split(',')

    torch.manual_seed(1234)
    if opt['data.cuda']:
        torch.cuda.manual_seed(1234)

    # 加载字符数据
    # if opt['data.trainval']:
    #     data = data_utils.load(opt, ['trainval'])
    #     train_loader = data['trainval']
    #     val_loader = None
    # else:
    #     data = data_utils.load(opt, ['train', 'val'])
    #     train_loader = data['train']
    #     val_loader = data['val']
    #

    # 创建掌纹数据加载器
    raw_train_data = np.load(opt['traindata_path'],
                       allow_pickle=True).copy()
    raw_val_data = np.load(opt['valdata_path'],
                             allow_pickle=True).copy()

    train_loader = palmdataloader(raw_train_data, opt['data.way'], opt['data.shot'],
                                  opt['data.query'], opt['data.train_episodes'], True)
    val_loader = palmdataloader(raw_val_data, opt['data.test_way'], opt['data.test_shot'],
                                  opt['data.test_query'], opt['data.train_episodes'], True)
    # 分析一下train_loader
    '''
    三项：way项的class（标签）；[way,data.shot,c,h,w] [way,data.query,c,h,w]
    '''
    # for data in enumerate(val_loader):
    #     a,b = data
    #     cla = b['class']
    #     xs = b['xs']
    #     xq = b['xq']
    #     print(xs[0])
    #     print(xq[0])

    # 初始化网络对象
    model = model_utils.load(opt)

    if opt['data.cuda']:
        model.cuda()

    engine = Engine()

    meters = { 'train': { field: tnt.meter.AverageValueMeter() for field in opt['log.fields'] } }

    if val_loader is not None:
        meters['val'] = { field: tnt.meter.AverageValueMeter() for field in opt['log.fields'] }

    def on_start(state):
        if os.path.isfile(trace_file):
            os.remove(trace_file)
        state['scheduler'] = lr_scheduler.StepLR(state['optimizer'], opt['train.decay_every'], gamma=0.5)
    engine.hooks['on_start'] = on_start

    def on_start_epoch(state):
        for split, split_meters in meters.items():
            for field, meter in split_meters.items():
                meter.reset()
        state['scheduler'].step()
    engine.hooks['on_start_epoch'] = on_start_epoch

    def on_update(state):
        for field, meter in meters['train'].items():
            meter.add(state['output'][field])
    engine.hooks['on_update'] = on_update

    def on_end_epoch(hook_state, state):
        if val_loader is not None:
            if 'best_loss' not in hook_state:
                hook_state['best_loss'] = np.inf
            if 'wait' not in hook_state:
                hook_state['wait'] = 0
        if val_loader is not None:
            model_utils.evaluate(state['model'],
                                 val_loader,
                                 meters['val'],
                                 is_cuda = opt['data.cuda'],
                                 desc="Epoch {:d} valid".format(state['epoch'], ),
                                 )

        meter_vals = log_utils.extract_meter_values(meters)
        print("Epoch {:02d}: {:s}".format(state['epoch'], log_utils.render_meter_values(meter_vals)))
        meter_vals['epoch'] = state['epoch']
        with open(trace_file, 'a') as f:
            json.dump(meter_vals, f)
            f.write('\n')

        if val_loader is not None:
            if meter_vals['val']['loss'] < hook_state['best_loss']:
                hook_state['best_loss'] = meter_vals['val']['loss']
                print("==> best model (loss = {:0.6f}), saving model...".format(hook_state['best_loss']))

                state['model'].cpu()
                torch.save(state['model'], os.path.join(opt['log.exp_dir'],
                                                         time.strftime("%Y%m%d-%H%M",time.localtime())+'best_model.pt'))
                if opt['data.cuda']:
                    state['model'].cuda()

                hook_state['wait'] = 0
            else:
                hook_state['wait'] += 1

                if hook_state['wait'] > opt['train.patience']:
                    print("==> patience {:d} exceeded".format(opt['train.patience']))
                    state['stop'] = True
        else:
            state['model'].cpu()
            torch.save(state['model'], os.path.join(opt['log.exp_dir'],
                                                    time.strftime("%Y%m%d-%H%M",time.localtime())+'best_model.pt'))
            if opt['data.cuda']:
                state['model'].cuda()

    engine.hooks['on_end_epoch'] = partial(on_end_epoch, { })
    engine.train(
        model = model, # 传入model
        loader = train_loader, # 放入数据
        optim_method = getattr(optim, opt['train.optim_method']),
        optim_config = { 'lr': opt['train.learning_rate'],
                         'weight_decay': opt['train.weight_decay'] },
        max_epoch = opt['train.epochs'],
        device = opt['data.cuda']

    )
