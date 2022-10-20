from asyncio import base_tasks
import logging
from logging.handlers import TimedRotatingFileHandler
import os
import torch.optim as optim

from torchvision import transforms
from PIL import Image
from transfer import ImageTransfer

from transform_net import TransformNet
from style_network import *
from loss_network import *
from dataset import get_loader, get_image_loader
from opticalflow import opticalflow
import cv2
import argparse

trHandler = TimedRotatingFileHandler("train_log.log", when="w1", interval=4, backupCount=12)
formatter = logging.Formatter('%(asctime)s.%(msecs)03d:%(filename)-12s[%(lineno)4d] %(levelname)-6s %(message)s',
                                  '%Y-%m-%d %H:%M:%S')
level = logging.DEBUG
trHandler.setFormatter(formatter)
trHandler.setLevel(level)
logger = logging.getLogger()
logger.addHandler(trHandler)

class ImageTransfer:
    def __init__(self, epoch, data_path, style_path, vgg_path, batch_size, lr, spatial_a, spatial_b, spatial_r, temporal_lambda, gpu=False, img_shape=(640, 360)):
        self.epoch = epoch
        self.data_path = data_path
        self.style_path = style_path
        self.lr = lr
        self.batch_size = batch_size
        self.gpu = gpu

        self.s_a = spatial_a
        self.s_b = spatial_b
        self.s_r = spatial_r
        self.t_l = temporal_lambda

        self.style_net = StyleNet()
        self.loss_net = LossNet(vgg_path)
        self.style_layer = ['conv1_2', 'conv2_2', 'conv3_4', 'conv4_4']

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406),
                                                                  (0.229, 0.224, 0.225))])
        self.img_shape = img_shape

        self.device = torch.device("cuda" if self.gpu else "cpu")

    def load_style(self):
        img = Image.open(self.style_path)
        img = img.resize(self.img_shape)
        img = np.asarray(img, np.float32) / 255.0
        img = self.transform(img)
        img = img.unsqueeze(0)
        img = img.to(self.device)
        # img = Variable(img, requires_grad=True)
        return img

    def calculate_losse():
        pass

    def train(self):
        style_img = self.load_style()

        if self.gpu:
            self.style_net.cuda()
            self.loss_net.cuda()
        

        adam = optim.Adam(self.style_net.parameters(), lr=self.lr)
        loader = get_loader(self.batch_size, self.data_path, self.img_shape, self.transform)
        print('Data Load Success!!')
        
        print('Training Start!!')
        for count in range(self.epoch):
            for step, frames in enumerate(loader):
                logger.info('step {}'.format(str(step)))
                for i in range(1, len(frames)):

                    x_t = frames[i]
                    x_t1 = frames[i-1]
                    
                    if self.gpu:
                        x_t = torch.stack(x_t).to(self.device)
                        x_t1 = torch.stack(x_t1).to(self.device)

                    # h_xt = self.style_net(x_t)
                    # h_xt1 = self.style_net(x_t1)


                    # s_xt = self.loss_net(x_t, self.style_layer)
                    # s_xt1 = self.loss_net(x_t1, self.style_layer)
                    # s_hxt = self.loss_net(h_xt, self.style_layer)
                    # s_hxt1 = self.loss_net(h_xt1, self.style_layer)
                    # s = self.loss_net(style_img, self.style_layer)

                    breakpoint()

def main():
    parser = argparse.ArgumentParser(description='Style Transfer in PyTorch')
    # subparsers = parser.add_subparsers(title="subcommands", dest="subcommand")
    parser.add_argument("--epochs", type=int, required=True, default=1, help="No of epochs")
    parser.add_argument("--dataset", type=str, required=True, help="Path to source dataset")
    parser.add_argument("--style-image", type=str, required=True, help="Path to style image")
    parser.add_argument("--vgg-path", type=str, required=True, default='models/vgg19-dcbb9e9d.pth',  help="Path to VGG model")
    parser.add_argument('--batch-size', type=int, default=4, help = "Batch size for data loader")
    parser.add_argument('--lr', type=float, required=True, default=0.01, help="Learning rate")
    parser.add_argument('--gpu', type=bool, default=True, help = "Use GPU")
    args = parser.parse_args()
    alpha = 1
    beta = 10
    regularizer = 1e-3
    temporal_lambda= 1e4
    gpu=True
    img_shape=(640, 360)

    transfer = ImageTransfer(
        epoch = args.epochs,
        data_path = args.dataset,
        style_path = args.style_image,
        vgg_path = args.vgg_path,
        batch_size = args.batch_size,
        lr = args.lr,
        spatial_a = alpha,
        spatial_b = beta,
        spatial_r = regularizer,
        temporal_lambda = temporal_lambda,
        gpu = gpu,
        img_shape = img_shape)
    
    transfer.train()

if __name__ == '__main__':
    main()