import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import scipy.stats as st


def gkern(kernlen=16, nsig=3):
    interval = (2*nsig+1.)/kernlen
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    # np.diff是对矩阵中求后一个减一个元素的差值，https://blog.csdn.net/hanshuobest/article/details/78558826 可以参考
    kern1d = np.diff(st.norm.cdf(x))    # scipy.stats.norm.cdf是正态分布累计概率密度函数
    # np.outer是求两个矩阵的外积，注意函数会自动将输入的矩阵拉成一维，如果np.out(a,b), a为m维，b为n维，则结果是m*n维矩阵
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel


def min_max_norm(in_):
    # out = (in_ - min_in) / (max_in - min_in)，这里的min和max取值为第二三维的极值
    max_ = in_.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    min_ = in_.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    in_ = in_ - min_
    return in_.div(max_-min_+1e-8)


class HA(nn.Module):
    # holistic attention module，这里其实就是用一个高斯矩阵 gaussian 对之前的显著性图 saliency_a (attention) 进行卷积操作
    # 然后再进行标准化， 到注意力图 soft_attention 结果在[0,1]之间
    # 然后将这一结果与之前的 attention 求较大值，可以得到范围更广的显著性映射图，这里之所以要这么做，是因为简单的使用attention范围太小
    # 作者试图通过这一方法扩大注意力范围，防止漏检测
    def __init__(self):
        super(HA, self).__init__()
        gaussian_kernel = np.float32(gkern(31, 4))
        gaussian_kernel = gaussian_kernel[np.newaxis, np.newaxis, ...]
        self.gaussian_kernel = Parameter(torch.from_numpy(gaussian_kernel), requires_grad=True)

    def forward(self, attention, x):
        soft_attention = F.conv2d(attention, self.gaussian_kernel, padding=15)
        soft_attention = min_max_norm(soft_attention)
        x = torch.mul(x, soft_attention.max(attention))
        return x
