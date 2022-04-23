import argparse
parser = argparse.ArgumentParser()
# training settings
parser.add_argument('--epoch', type=int, default=150, help='epoch number')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=40, help='every n epochs decay learning rate')
parser.add_argument('--load', type=str, default=None, help='train from checkpoints')

# device settings
parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')

# dataset settings
parser.add_argument('--img_root', type=str, default='/home/workstation/diskA/datasets/COD_datasets/TrainingSets/Image/', help='the training images root')
parser.add_argument('--edge_root', type=str, default='/home/workstation/diskA/datasets/COD_datasets/TrainingSets/edge/', help='the training edge root')
parser.add_argument('--gt_root', type=str, default='/home/workstation/diskA/datasets/COD_datasets/TrainingSets/GT/', help='the training gt images root')
parser.add_argument('--test_img_root', type=str, default='/home/workstation/diskA/datasets/COD_datasets/TestingSets/CAMO/Image/', help='the validation images root')
parser.add_argument('--test_gt_root', type=str, default='/home/workstation/diskA/datasets/COD_datasets/TestingSets/CAMO/GT/', help='the validation gt images root')
# parser.add_argument('--test_path', type=str, default=r'E:\COD_datasets\TestingSets')
parser.add_argument('--test_path', type=str, default='/home/workstation/diskA/datasets/COD_datasets/TestingSets')
parser.add_argument('--test_save_path', type=str, default='./results')
parser.add_argument('--test_edge_path', type=str, default='./edge_results')
parser.add_argument('--test_name', type=str, default='run-1')

# save settings
parser.add_argument('--save_path', type=str, default='/home/workstation/diskA/BGNet_work_dir/', help='the path to save models and logs')

# architecture settings
parser.add_argument('--backbone', type=str, default='resnet', choices=['resnet', 'res2net'])
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--ratio', type=float, default=1)
parser.add_argument('--n', type=float, default=4)
parser.add_argument('--groups', type=int, default=1)

# loss settings
parser.add_argument('--mask_loss', type=str, default='f3', choices=['bi', 'bas', 'f3'])
parser.add_argument('--edge_loss', type=str, default='f3', choices=['bi', 'bas', 'f3'])

opt = parser.parse_args()
