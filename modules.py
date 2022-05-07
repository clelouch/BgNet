import torch
from torch import nn
import torch.nn.functional as F


def upsample(tensor, size):
    return F.interpolate(tensor, size, mode='bilinear', align_corners=True)


def norm_layer(channel, norm_name='bn'):
    if norm_name == 'bn':
        return nn.BatchNorm2d(channel)
    elif norm_name == 'gn':
        return nn.GroupNorm(min(32, channel // 4), channel)


class ChannelCompress(nn.Module):
    def __init__(self, in_c, out_c):
        super(ChannelCompress, self).__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, bias=False),
            norm_layer(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.reduce(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=1):
        super(ChannelAttention, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class RFB(nn.Module):
    """ receptive field block """

    def __init__(self, in_channel, out_channel=256):
        super(RFB, self).__init__()
        self.relu = nn.ReLU()
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=3, dilation=3)  # 当kernel=3，如果dilation=padding则shape不变
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.Conv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            nn.Conv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = nn.Conv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = nn.Conv2d(in_channel, out_channel, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = torch.cat((x0, x1, x2, x3), 1)
        x_cat = self.conv_cat(x_cat)

        out = self.relu(x_cat + self.conv_res(x))
        return out


# locating module
class LM(nn.Module):
    def __init__(self, channel):
        super(LM, self).__init__()
        # non-local
        temp_c = channel // 4
        self.query_conv = nn.Conv2d(in_channels=channel, out_channels=temp_c, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=channel, out_channels=temp_c, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

        # local
        self.local1 = nn.Sequential(
            nn.Conv2d(channel, channel, 1, bias=False),
            norm_layer(channel),
            nn.ReLU(inplace=True)
        )
        self.local2 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            norm_layer(channel),
            nn.ReLU(inplace=True)
        )
        self.local3 = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 2, dilation=2, bias=False),
            norm_layer(channel),
            nn.ReLU(inplace=True)
        )
        self.conv_cat = nn.Sequential(
            nn.Conv2d(channel * 3, channel, 3, 1, 1, bias=False),
            norm_layer(channel),
            nn.ReLU(inplace=True)
        )

        # residual connection
        self.conv_res = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1, bias=False),
            norm_layer(channel),
        )

    def forward(self, x):
        # non-local
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out1 = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out1 = out1.view(m_batchsize, C, height, width)

        # local
        branch1 = self.local1(x)
        branch2 = self.local2(x)
        branch3 = self.local3(x)
        out2 = self.conv_cat(torch.cat([branch1, branch2, branch3], dim=1))

        out = F.relu(x + self.conv_res(out1 + out2))
        return out

# Boundary-guided fusion module
class BFM(nn.Module):
    def __init__(self, in_c, out_c, groups=8):
        super(BFM, self).__init__()
        self.rfb = RFB(in_c, out_c)
        self.groups = groups
        sc_channel = (out_c // groups + 1) * groups  # split then concate channel

        self.foreground_conv = nn.Conv2d(sc_channel, sc_channel, 3, 1, 1, bias=False)
        self.foreground_bn = norm_layer(sc_channel)
        self.foreground_relu = nn.ReLU()
        self.background_conv = nn.Conv2d(sc_channel, sc_channel, 3, 1, 1, bias=False)
        self.background_bn = norm_layer(sc_channel)
        self.background_relu = nn.ReLU()

        self.edge_conv = nn.Sequential(
            nn.Conv2d(sc_channel, out_c, 3, 1, 1, bias=False),
            norm_layer(out_c),
            nn.ReLU(inplace=True)
        )
        self.mask_conv = nn.Sequential(
            nn.Conv2d(2 * sc_channel, out_c, 3, 1, 1, bias=False),
            norm_layer(out_c),
            nn.ReLU(inplace=True)
        )
        self.mask_pred_conv = nn.Conv2d(out_c, 1, 3, 1, 1)
        self.edge_pred_conv = nn.Conv2d(out_c, 1, 3, 1, 1)

    def split_and_concate(self, x1, x2):
        N, C, H, W = x1.shape
        x2 = x2.repeat(1, self.groups, 1, 1)
        x1 = x1.reshape(N, self.groups, C // self.groups, H, W)
        x2 = x2.unsqueeze(2)
        x = torch.cat([x1, x2], dim=2)
        x = x.reshape(N, -1, H, W)
        return x

    def forward(self, low, high, mask_pred, edge_pred, sig=True):
        low = self.rfb(low)
        if high is not None:
            low += upsample(high, low.shape[2:])
        mask_pred = upsample(mask_pred, low.shape[2:])
        edge_pred = upsample(edge_pred, low.shape[2:])
        if sig:
            mask_pred = torch.sigmoid(mask_pred)
            edge_pred = torch.sigmoid(edge_pred)
        foreground = low * mask_pred
        background = low * (1 - mask_pred)

        foreground = self.foreground_conv(self.split_and_concate(foreground, edge_pred))
        background = self.background_conv(self.split_and_concate(background, edge_pred))

        edge_feature = (foreground - foreground.min()) / (foreground.max() - foreground.min()) * (
                background - background.min()) / (background.max() - background.min())

        foreground = self.foreground_relu(self.foreground_bn(foreground))
        background = self.background_relu(self.background_bn(background))
        mask_feature = torch.cat((foreground, background), dim=1)

        edge_feature = self.edge_conv(edge_feature)
        mask_feature = self.mask_conv(mask_feature)

        mask = self.mask_pred_conv(mask_feature)
        edge = self.edge_pred_conv(edge_feature)
        return mask_feature, mask, edge
