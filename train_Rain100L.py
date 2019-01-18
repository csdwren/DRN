import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from DerainDataset import prepare_data_Rain100L, Dataset
from utils import *
from torch.optim.lr_scheduler import MultiStepLR
from SSIM import *
from network import DRN, print_network

parser = argparse.ArgumentParser(description="DRN_train_Rain100L")
parser.add_argument("--preprocess", type=bool, default=True, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=16, help="Training batch size")
parser.add_argument("--intra_iter", type=int, default=7, help="Number of intra iteration")
parser.add_argument("--inter_iter", type=int, default=7, help="Number of inter iteration")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=[30,50,80], help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--save_folder", type=str, default="logs/Rain100L", help='path of log files')
parser.add_argument("--save_freq",type=int,default=1,help='save intermediate model')
parser.add_argument("--data_path",type=str, default="./train/RainTrainL", help='path to training data')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
opt = parser.parse_args()

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

opt.save_folder = opt.save_folder + "_inter%d"%opt.inter_iter + "_intra%d"%opt.intra_iter

def main():
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True, data_path=opt.data_path)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model

    model = DRN(channel=3, inter_iter=opt.inter_iter, intra_iter=opt.intra_iter, use_GPU=opt.use_GPU)
    print_network(model)

    criterion = SSIM()

    # Move to GPU
    if opt.use_GPU:
        model = model.cuda()
        criterion.cuda()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.5)  # learning rates
    # training
    writer = SummaryWriter(opt.save_folder)
    step = 0

    initial_epoch = findLastCheckpoint(save_dir=opt.save_folder)  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        model.load_state_dict(torch.load(os.path.join(opt.save_folder, 'net_epoch%d.pth' % initial_epoch)))

    for epoch in range(initial_epoch, opt.epochs):

        scheduler.step(epoch)
        # set learning rate
        for param_group in optimizer.param_groups:
            #param_group["lr"] = current_lr
            print('learning rate %f' % param_group["lr"])
        # train
        for i, (input, target) in enumerate(loader_train, 0):
            # training step
            loss_list = []
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            input_train, target_train = Variable(input.cuda()), Variable(target.cuda())

            out_train, outs = model(input_train)

            pixel_loss = criterion(target_train, out_train)

            for lossi in range(opt.inter_iter):
                loss1 = criterion(target_train, outs[lossi])
                loss_list.append(loss1)

            loss = -pixel_loss
            index = 0.1
            for lossi in range(opt.inter_iter):
                loss += -index * loss_list[lossi]
                index = index + 0.1
            loss.backward()
            optimizer.step()
            # results
            model.eval()
            out_train, _ = model(input_train)
            out_train = torch.clamp(out_train, 0., 1.)
            psnr_train = batch_PSNR(out_train, target_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f, loss1: %.4f, loss2: %.4f, loss3: %.4f, loss4: %.4f, PSNR_train: %.4f" %
                  (epoch + 1, i + 1, len(loader_train), loss.item(), loss_list[0].item(), loss_list[1].item(), loss_list[2].item(),
                   loss_list[3].item(), psnr_train))
            # print("[epoch %d][%d/%d] loss: %.4f, PSNR_train: %.4f" %
            #       (epoch + 1, i + 1, len(loader_train), loss.item(), psnr_train))
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1
        ## the end of each epoch

        model.eval()
        # log the images
        out_train,_ = model(input_train)
        out_train = torch.clamp(out_train, 0., 1.)
        Img = utils.make_grid(target_train.data, nrow=8, normalize=True, scale_each=True)
        Imgn = utils.make_grid(input_train.data, nrow=8, normalize=True, scale_each=True)
        Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        writer.add_image('clean image', Img, epoch)
        writer.add_image('noisy image', Imgn, epoch)
        writer.add_image('reconstructed image', Irecon, epoch)
        # save model
        torch.save(model.state_dict(), os.path.join(opt.save_folder, 'net_latest.pth'))

        if epoch % opt.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(opt.save_folder, 'net_epoch%d.pth' % (epoch+1)))




if __name__ == "__main__":
    if opt.preprocess:
        prepare_data_Rain100L(data_path=opt.data_path, patch_size=100, stride=80, aug_times=1)

    main()
