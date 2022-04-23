import torch
from modules import upsample, ChannelCompress, BFM, LM
import torch.nn.functional as F
from HolisticAttention import HA
from torch import nn
from torchvision import models
from ResNet import B2_ResNet
from Res2Net import B2_Res2Net50


def upsample_sigmoid(pred_list, target_size):
    for i in range(len(pred_list)):
        pred_list[i] = torch.sigmoid(upsample(pred_list[i], target_size))
    return pred_list


class PositioningDecoder(nn.Module):
    def __init__(self, config):
        super(PositioningDecoder, self).__init__()
        self.compress3 = ChannelCompress(2048, 256)
        self.compress2 = ChannelCompress(1024, 256)
        self.compress1 = ChannelCompress(512, 256)

        self.locate1 = LM(256)
        self.locate2 = LM(256)
        self.locate3 = LM(256)

        self.predict = nn.Conv2d(256, 1, 3, 1, 1)

    def forward(self, x2, x3, x4):
        x2 = self.compress1(x2)
        x3 = self.compress2(x3)
        x4 = self.compress3(x4)

        x4 = self.locate1(x4)
        x3 = x3 + upsample(x4, x3.shape[2:])
        x3 = self.locate2(x3)
        x2 = x2 + upsample(x3, x2.shape[2:])
        x2 = self.locate3(x2)

        attention_map = torch.sigmoid(self.predict(x2))
        edge = torch.abs(F.avg_pool2d(attention_map, kernel_size=3, stride=1, padding=1) - attention_map)
        return attention_map, edge


class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config
        if config.backbone == 'resnet':
            self.resnet = B2_ResNet()
        elif config.backbone == 'res2net':
            self.resnet = B2_Res2Net50()

        self.decoder1 = PositioningDecoder(config)

        self.ha = HA()

        self.refine1 = BFM(2048, 64, config.groups)
        self.refine2 = BFM(1024, 64, config.groups)
        self.refine3 = BFM(512, 64, config.groups)

        self._initialize_weight()

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)

        x2_1 = x2
        x3_1 = self.resnet.layer3_1(x2_1)
        x4_1 = self.resnet.layer4_1(x3_1)

        attention_map, edge = self.decoder1(x2_1, x3_1, x4_1)
        x2_2 = self.ha(attention_map, x2)

        x3_2 = self.resnet.layer3_2(x2_2)
        x4_2 = self.resnet.layer4_2(x3_2)

        x4_refined, pred4, edge4 = self.refine1(x4_2, None, attention_map, edge, sig=False)
        x3_refined, pred3, edge3 = self.refine2(x3_2, x4_refined, pred4, edge4)
        x2_refined, pred2, edge2 = self.refine3(x2_2, x3_refined, pred3, edge3)
        mask_list = upsample_sigmoid([pred2, pred3, pred4], self.config.trainsize)
        edge_list = upsample_sigmoid([edge2, edge3, edge4], self.config.trainsize)

        return upsample(attention_map, self.config.trainsize), upsample(edge, self.config.trainsize), mask_list, edge_list

    def _initialize_weight(self):
        if self.config.backbone == 'resnet':
            res50 = models.resnet50(pretrained=True)
            pretrained_dict = res50.state_dict()
        else:
            pretrained_dict = torch.load('./models/res2net50_v1b_26w_4s-3cf99910.pth')
        all_params = {}
        for k, v in self.resnet.state_dict().items():
            if k in pretrained_dict.keys():
                v = pretrained_dict[k]
                all_params[k] = v
            elif '_1' in k:
                name = k.split('_1')[0] + k.split('_1')[1]
                v = pretrained_dict[name]
                all_params[k] = v
            elif '_2' in k:
                name = k.split('_2')[0] + k.split('_2')[1]
                v = pretrained_dict[name]
                all_params[k] = v
        assert len(all_params.keys()) == len(self.resnet.state_dict().keys())
        self.resnet.load_state_dict(all_params)


if __name__ == '__main__':
    from options import opt

    model = Net(opt).cuda()
    img = torch.randn(2, 3, 352, 352).cuda()
    model.load_state_dict(torch.load('models/resnet/BgNet_epoch_best.pth'))
    with torch.no_grad():
        out = model(img)