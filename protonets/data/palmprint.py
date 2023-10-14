import random

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import warnings


class Dataset(Dataset):

    def __init__(self, cal, xs, xq, transform=None):
        self.classes = cal
        self.xs = xs # 传入list[way个ndarray]
        self.xq = xq
        self.transform = transform

    def __len__(self):
        return len(self.classes)

    def __getitem__(self, idx):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            dic = {'class':self.classes[idx],
                   # 'xs':self.transform(self.xs[idx]),
                   # 'xq':self.transform(self.xq[idx])
                   'xs': self.xs[idx],
                   'xq': self.xq[idx]
                   }
            return dic

def render_img(arr):
    '''
    处理array的功能函数
    '''
    arr = (arr * 0.5) + 0.5
    arr = np.uint8(arr * 255)
    # 转换格式为PIL.Image.Image mode='RGB'
    img = Image.fromarray(np.squeeze(arr), mode='L').convert('RGB')
    return img
def render_transform(x):
    return render_img(x)
def palmdataloader(raw_data, n_way, k_shot, k_query, shuffle, batch_size):
    mid_pixel_value = 1.0 / 2
    transform = transforms.Compose([render_transform,
                                    transforms.Resize((84, 84)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                    ])
    raw_shape = raw_data.shape
    xs = []
    xq = []
    # 随机选n_way
    cal = random.sample(range(raw_shape[0]), n_way) #生成的随机nway个类别
    # 在一类中随机选s+q张，前s张作为xs，剩余作为xq
    for i in cal:
        xs_ = []
        xq_ = []
        xsxq_ind = random.sample(range(raw_shape[1]), k_shot+k_query)
        for xs_i in xsxq_ind[:k_shot]:
            xs_.append(raw_data[i][xs_i]) # 添加的是ndarray
        for xq_i in xsxq_ind[k_shot:]:
            xq_.append(raw_data[i][xq_i])
        xs.append(np.array(xs_)) # 类别总和为list
        xq.append(np.array(xq_))
    # 实例化对象
    train_dataset = Dataset(cal, xs, xq, transform)
    print("dataset长度：",train_dataset.__len__())
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

if __name__ == '__main__':
    raw_data = np.load("F:\jupyter_notebook\DAGAN\datasets\IITDdata_left_PSA+SC+MC+W_10.npy",
                       allow_pickle=True).copy()  # numpy.ndarray

    n_way = 20
    k_shot = 5
    k_query = 2
    batch_size = 100

    # 创建训练数据加载器
    dataloader = palmdataloader(raw_data, n_way, k_shot, k_query, True, batch_size)

    for i,item  in enumerate(dataloader):
        print(i)
        print(item)
