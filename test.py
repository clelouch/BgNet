import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
import cv2
from model import Net
from data import test_dataset
from options import opt

dataset_path = opt.test_path

# set device for test
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

# load the model
model = Net(opt)
model.load_state_dict(torch.load(opt.load))
model.cuda()
model.eval()

# test
with torch.no_grad():
    test_datasets = ['CAMO', 'CHAMELEON', 'COD10K', 'NC4K']
    for dataset in test_datasets:
        save_path = os.path.join(opt.test_save_path, opt.test_name, dataset)
        edge_save_path = os.path.join(opt.test_edge_path, opt.test_name, dataset)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(edge_save_path):
            os.makedirs(edge_save_path)
        image_root = os.path.join(dataset_path, dataset, 'Image')
        gt_root = os.path.join(dataset_path, dataset, 'GT')
        test_loader = test_dataset(image_root, gt_root, opt.trainsize)
        for i in range(test_loader.size):
            image, gt, name, image_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()

            # test first branch
            # res, edge, _, _ = model(image)
            # res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            # edge_res = F.upsample(edge, size=gt.shape, mode='bilinear', align_corners=False)

            # test second branch
            _, _, res, edge = model(image)
            res = F.upsample(res[0], size=gt.shape, mode='bilinear', align_corners=False)
            edge_res = F.upsample(edge[0], size=gt.shape, mode='bilinear', align_corners=False)

            res = res.data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            print('save img to: ', os.path.join(save_path, name))
            cv2.imwrite(os.path.join(save_path, name), res * 255)

            edge_res = edge_res.data.cpu().numpy().squeeze()
            edge_res = (edge_res - edge_res.min()) / (edge_res.max() - edge_res.min() + 1e-8)
            cv2.imwrite(os.path.join(edge_save_path, name), edge_res * 255)
        print('Test Done!')
