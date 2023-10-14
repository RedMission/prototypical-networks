import protonets.data

def load(opt, splits):
    if opt['data.dataset'] == 'omniglot':
        ds = protonets.data.omniglot.load(opt, splits) # 特定数据集加载、划分方法['train', 'val']
    else:
        raise ValueError("Unknown dataset: {:s}".format(opt['data.dataset']))

    return ds
