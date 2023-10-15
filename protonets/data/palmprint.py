import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import warnings

from tqdm import tqdm


class Dataset(Dataset):

    def __init__(self, cal_train_episodes, xs_train_episodes, xq_train_episodes, transform=None):
        self.cal_train_episodes = cal_train_episodes
        self.xs_train_episodes = xs_train_episodes # 传入list[way个ndarray]
        self.xq_train_episodes = xq_train_episodes
        self.transform = transform

    def __len__(self):
        return len(self.cal_train_episodes)

    def __getitem__(self, idx):
        '''
        接受一个索引，返回一个样本或者标签
        :param idx:
        :return:
        '''
        # 转换图片格式为张量的一系列操作 array->tensor->list->array->tensor
        xs_list = self.xs_train_episodes[idx]
        xq_list = self.xq_train_episodes[idx]
        xs = [[self.transform(j) for j in i ]for i in xs_list ]
        xs_tensor = torch.tensor([[tensor.numpy() for tensor in sublist] for sublist in xs])
        xq = [[self.transform(j) for j in i ]for i in xq_list ]
        xq_tensor = torch.tensor([[tensor.numpy() for tensor in sublist] for sublist in xq])

        # xq = [self.transform(i) for i in xq_list]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            dic = {'class':self.cal_train_episodes[idx],
                   'xs': xs_tensor,
                   'xq': xq_tensor
                   }
            return dic

def render_img(arr):
    '''
    处理array的功能函数
    '''
    arr = (arr * 0.5) + 0.5
    arr = np.uint8(arr * 255)
    # 转换格式为PIL.Image.Image mode='RGB'
    # img = Image.fromarray(np.squeeze(arr), mode='L').convert('RGB') # 彩色图像
    img = Image.fromarray(np.squeeze(arr), mode='L') # 单通道图像
    return img
def render_transform(x):
    return render_img(x)
def palmdataloader(raw_data, n_way, k_shot, k_query,train_episodes, shuffle):
    mid_pixel_value = 1.0 / 2
    transform = transforms.Compose([render_transform,
                                    transforms.Resize((84, 84)),
                                    transforms.ToTensor(),
                                    # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # 彩色图像
                                    transforms.Normalize((0.5,), (0.5,))
                                    ])
    raw_shape = raw_data.shape
    cal_train_episodes = []
    xs_train_episodes = []
    xq_train_episodes = []
    # 总共需要选train_episodes次
    for _ in range(train_episodes):
        # 随机选n_way
        cal = random.sample(range(raw_shape[0]), n_way) #生成的随机nway个类别
        cal_train_episodes.append(cal) # 记录class
        # 在一类中随机选s+q张，前s张作为xs，剩余作为xq
        xs = []
        xq = []
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
        xs_train_episodes.append(np.array(xs))
        xq_train_episodes.append(np.array(xq))
    # 实例化对象
    train_dataset = Dataset(cal_train_episodes, xs_train_episodes, xq_train_episodes, transform)
    return DataLoader(train_dataset, batch_size= None, shuffle=shuffle, num_workers=0)

if __name__ == '__main__':
    raw_data = np.load("F:\jupyter_notebook\DAGAN\datasets\IITDdata_left_PSA+SC+MC+W_10.npy",
                       allow_pickle=True).copy()  # numpy.ndarray

    n_way = 20
    k_shot = 5
    k_query = 2
    batch_size = 100
    train_episodes = 100
    # 创建训练数据加载器
    dataloader = palmdataloader(raw_data, n_way, k_shot, k_query, train_episodes,True)

    # for i,item  in enumerate(dataloader):
    #     print(i)
    #     print(item)

    for sample in tqdm(dataloader):
        print("++")
        print(sample)
