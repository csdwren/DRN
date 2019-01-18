import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import *

from network import *
import time



parser = argparse.ArgumentParser(description="DRN_test_Rain100L")
parser.add_argument("--logdir", type=str, default="logs/Rain100L", help='path of log files')
parser.add_argument("--data_path",type=str, default="./test/Rain100L", help='path to test data')
parser.add_argument("--save_path", type=str, default="results_Rain100L", help='path to save results')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="1", help='GPU id')
parser.add_argument("--intra_iter", type=int, default=7, help="Number of intra iteration")
parser.add_argument("--inter_iter", type=int, default=7, help="Number of inter iteration")
opt = parser.parse_args()

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

opt.logdir = opt.logdir + "_inter%d"%opt.inter_iter + "_intra%d"%opt.intra_iter
opt.save_path = os.path.join(opt.data_path, opt.save_path)
opt.save_path = opt.save_path + "_inter%d"%opt.inter_iter + "_intra%d"%opt.intra_iter

if not os.path.exists(opt.save_path):
    os.mkdir(opt.save_path)

def normalize(data):
    return data/255.

def main():
    # Build model
    print('Loading model ...\n')

    model = DRN(channel=3, inter_iter=opt.inter_iter, intra_iter=opt.intra_iter, use_GPU=opt.use_GPU)
    print_network(model)

    if opt.use_GPU:
        model = model.cuda()

    if os.path.exists(os.path.join(opt.logdir, 'net_latest.pth')):
        model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net_latest.pth')))
    else:
        model.load_state_dict(torch.load(os.path.join(opt.logdir, 'pretrained', 'net_latest.pth')))

    model.eval()
    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join(opt.data_path,'rainy/*.png'))

    files_source.sort()
    # process data
    time_test = 0
    for f in files_source:
        img_name = os.path.basename(f)


        # image
        Img = cv2.imread(f)

        b, g, r = cv2.split(Img)
        Img = cv2.merge([r, g, b])
        #Img = cv2.resize(Img, (int(1024), int(1024)), interpolation=cv2.INTER_CUBIC)
        Img = normalize(np.float32(Img))
        Img = np.expand_dims(Img.transpose(2, 0, 1), 0)
        ISource = torch.Tensor(Img)

        if opt.use_GPU:
            ISource = Variable(ISource.cuda())
        else:
            ISource = Variable(ISource)

        with torch.no_grad(): # this can save much memory
            if opt.use_GPU:
                torch.cuda.synchronize()
            start_time = time.time()
            out, _ = model(ISource)
            out = torch.clamp(out, 0., 1.)
            if opt.use_GPU:
                torch.cuda.synchronize()
            end_time = time.time()
            dur_time = end_time - start_time
            print(img_name)
            print(dur_time)
            time_test += dur_time

        if opt.use_GPU:
            save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())
        else:
            save_out = np.uint8(255 * out.data.numpy().squeeze())

        save_out = save_out.transpose(1,2,0)
        b, g, r = cv2.split(save_out)
        save_out = cv2.merge([r, g, b])

        save_path = opt.save_path
        cv2.imwrite(os.path.join(save_path,img_name), save_out)

    print(time_test/100)

if __name__ == "__main__":
    main()
