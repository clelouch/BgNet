import os
import numpy as np
import torch
import time
from datetime import datetime, timedelta
from torchvision.utils import make_grid
from model import Net
from data import get_loader, test_dataset
from utils import clip_gradient, adjust_lr, setup_seed
from loss import build_loss
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import logging

from options import opt

# set the device for training
torch.autograd.set_detect_anomaly(True)
setup_seed()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
print('USE GPU', opt.gpu_id)

# build the model
model = Net(opt)
if (opt.load is not None):
    model.load_state_dict(torch.load(opt.load))
    print('load model from ', opt.load)

model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

# set the path
image_root = opt.img_root
gt_root = opt.gt_root
edge_root = opt.edge_root
test_image_root = opt.test_img_root
test_gt_root = opt.test_gt_root
save_path = opt.save_path

if os.path.exists(os.path.join(save_path, 'models')):
    raise Exception("directory exists! Please change save path")
if not os.path.exists(os.path.join(save_path, 'models')):
    os.makedirs(os.path.join(save_path, 'models'))
with open('%s/args.txt' % (opt.save_path), 'w') as f:
    for arg in vars(opt):
        print('%s: %s' % (arg, getattr(opt, arg)), file=f)

# load data
print('load data...')
train_loader = get_loader(image_root, gt_root, edge_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = test_dataset(test_image_root, test_gt_root, opt.trainsize)
total_step = len(train_loader)

logging.basicConfig(filename=os.path.join(save_path, 'log.log'),
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("CFNet-Train")
logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_path,
        opt.decay_epoch))

# set loss function
criterior_mask, criterior_edge = build_loss(opt)

step = 0
writer = SummaryWriter(save_path + '/summary')
best_mae = 1
best_epoch = 0


# train function
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        start_time = time.time()
        for i, (images, gts, gt_edges) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()
            gt_edges = gt_edges.cuda()

            pred0, edge0, [pred2, pred3, pred4], [edge2, edge3, edge4] = model(images)
            mask_loss0 = criterior_mask(pred0, gts)
            mask_loss2 = criterior_mask(pred2, gts)
            mask_loss3 = criterior_mask(pred3, gts)
            mask_loss4 = criterior_mask(pred4, gts)

            edge_loss0 = criterior_edge(edge0, gt_edges)
            edge_loss2 = criterior_edge(edge2, gt_edges)
            edge_loss3 = criterior_edge(edge3, gt_edges)
            edge_loss4 = criterior_edge(edge4, gt_edges)

            mask_loss = (mask_loss0 + mask_loss2) + mask_loss3 / 2 + mask_loss4 / 4
            edge_loss = (edge_loss0 + edge_loss2) + edge_loss3 / 2 + edge_loss4 / 4

            total_loss = mask_loss + opt.ratio * edge_loss

            total_loss.backward()

            # clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += total_loss.data
            if i % 100 == 0:
                end_time = time.time()
                duration_time = end_time - start_time
                time_second_avg = duration_time / (opt.batchsize * 100)
                eta_sec = time_second_avg * (
                        (opt.epoch - epoch) * len(train_loader) * opt.batchsize + (
                        len(train_loader) - i - 1) * opt.batchsize
                )
                eta_str = str(timedelta(seconds=int(eta_sec)))
                print(
                    '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], eta: {},'
                    ' MaskLoss0: {:0.4f}, MaskLoss2: {:0.4f}, MaskLoss3: {:0.4f}, MaskLoss4: {:0.4f}'.
                        format(datetime.now(), epoch, opt.epoch, i, total_step, eta_str,
                               mask_loss0.data, mask_loss2.data, mask_loss3.data, mask_loss4.data))
                logging.info(
                    '#TRAIN Mask#: {} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], eta: {}, MaskLoss0: {:0.4f}, '
                    'MaskLoss2: {:0.4f}, MaskLoss3: {:0.4f}, MaskLoss4: {:0.4f}'.
                        format(datetime.now(), epoch, opt.epoch, i, total_step, eta_str,
                               mask_loss0.data, mask_loss2.data, mask_loss3.data, mask_loss4.data))
                logging.info(
                    '#TRAIN Edge#: {} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], eta: {}, EdgeLoss0: {:0.4f}, '
                    'EdgeLoss2: {:0.4f}, EdgeLoss3: {:0.4f}, EdgeLoss4: {:0.4f}'.
                        format(datetime.now(), epoch, opt.epoch, i, total_step, eta_str, edge_loss0.data,
                               edge_loss2.data, edge_loss3.data, edge_loss4.data))
                writer.add_scalar('Loss', total_loss.data, global_step=step)

                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('Ground_truth', grid_image, step)

                res = pred0[0][0].clone()
                res = res.data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('Init_Pred', torch.tensor(res), step, dataformats='HW')

                res = edge0[0][0].clone()
                res = res.data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('Init_Edge', torch.tensor(res), step, dataformats='HW')

                res = pred2[0][0].clone()
                res = res.data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('Final_Pred', torch.tensor(res), step, dataformats='HW')

                res = edge2[0][0].clone()
                res = res.data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('Final_Edge', torch.tensor(res), step, dataformats='HW')
                start_time = time.time()

        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if (epoch) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, 'models', 'BgNet_epoch_{}.pth'.format(epoch)))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), os.path.join(save_path, 'models', 'BgNet_epoch_{}.pth'.format(epoch + 1)))
        print('save checkpoints successfully!')
        raise


# test function
def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, name, img_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            _, _, res, _ = model(image)
            res = F.upsample(res[0], size=gt.shape, mode='bilinear', align_corners=False)
            res = res.data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(save_path, 'models', 'BgNet_epoch_best.pth'))
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    print("Start train...")
    cur_lr = opt.lr
    for epoch in range(1, opt.epoch + 1):
        if epoch % opt.decay_epoch == 0:
            cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path)
        test(test_loader, model, epoch, save_path)
