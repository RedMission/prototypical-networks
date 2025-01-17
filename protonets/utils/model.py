from tqdm import tqdm

from protonets.utils import filter_opt
from protonets.models import get_model

def load(opt): # 传入name参数 加载模型
    model_opt = filter_opt(opt, 'model')
    model_name = model_opt['model_name']

    del model_opt['model_name']

    return get_model(model_name, model_opt)

def evaluate(model, data_loader, meters, is_cuda,desc=None, ):
    model.eval()

    for field,meter in meters.items():
        meter.reset()

    if desc is not None:
        data_loader = tqdm(data_loader, desc=desc)

    for sample in data_loader:

        _, output = model.loss(sample, is_cuda)
        for field, meter in meters.items():
            meter.add(output[field])

    return meters
