# -*- coding: UTF-8 -*-
from __future__ import print_function
from six.moves import range

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import os
import time
from PIL import Image, ImageFont, ImageDraw
from copy import deepcopy

from miscc.config import cfg
from miscc.utils import mkdir_p


from torch.utils.tensorboard import SummaryWriter





from tensorboardX import summary  #
# from tensorboard import summary
#from tensorboard import FileWriter


#from torch.utils.tensorboard import FileWriter
from tensorboardX import FileWriter   #
#from torch.utils.tensorboard import summary



from model import G_NET, D_NET64, D_NET128, D_NET256, D_NET512, D_NET1024, INCEPTION_V3



# ################## Shared functions ###################
def compute_mean_covariance(img):   # 计算均值,协方差
    batch_size = img.size(0)
    channel_num = img.size(1)
    height = img.size(2)
    width = img.size(3)
    num_pixels = height * width

    # batch_size * channel_num * 1 * 1
    mu = img.mean(2, keepdim=True).mean(3, keepdim=True)

    # batch_size * channel_num * num_pixels
    img_hat = img - mu.expand_as(img)
    img_hat = img_hat.view(batch_size, channel_num, num_pixels)
    # batch_size * num_pixels * channel_num
    img_hat_transpose = img_hat.transpose(1, 2)  # 维度1，2交换
    # batch_size * channel_num * channel_num
    covariance = torch.bmm(img_hat, img_hat_transpose)  # 矩阵相乘
    covariance = covariance / num_pixels

    return mu, covariance


def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)    #(-0.5)*平均值
    return KLD


def weights_init(m):    # 参数初始化函数，conv,bn,linear层
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_G_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


def compute_inception_score(predictions, num_splits=1):
    print('predictions.shape:', predictions.shape)
    scores = []
    for i in range(num_splits):
        istart = i * predictions.shape[0] // num_splits
        iend = (i + 1) * predictions.shape[0] // num_splits
        part = predictions[istart:iend, :]
        kl = part * \
            (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)


def negative_log_posterior_probability(predictions, num_splits=1):  # 负对数后验概率
    # print('predictions', predictions.shape)
    scores = []
    for i in range(num_splits):
        istart = i * predictions.shape[0] // num_splits
        iend = (i + 1) * predictions.shape[0] // num_splits
        part = predictions[istart:iend, :]
        result = -1. * np.log(np.max(part, 1))
        result = np.mean(result)
        scores.append(result)
    return np.mean(scores), np.std(scores)


def load_network(gpus):
    netG = G_NET()
    netG.apply(weights_init)    # 对网络模型G进行参数初始化
    netG = torch.nn.DataParallel(netG, device_ids=gpus)
    print(netG)

    netsD = []
    if cfg.TREE.BRANCH_NUM > 0: # netsD包含三个判别器
        netsD.append(D_NET64()) # 在netsD()列表后面添加D_NET64()
    if cfg.TREE.BRANCH_NUM > 1:
        netsD.append(D_NET128())
    if cfg.TREE.BRANCH_NUM > 2:
        netsD.append(D_NET256())
    if cfg.TREE.BRANCH_NUM > 3:
        netsD.append(D_NET512())
    if cfg.TREE.BRANCH_NUM > 4:
        netsD.append(D_NET1024())
    # if cfg.TREE.BRANCH_NUM > 0: # netsD包含三个判别器
    #     netsD.append(D_NET32()) # 在netsD()列表后面添加D_NET64()
    # if cfg.TREE.BRANCH_NUM > 1:
    #     netsD.append(D_NET64_gai())
    # if cfg.TREE.BRANCH_NUM > 2:
    #     netsD.append(D_NET128())
    # if cfg.TREE.BRANCH_NUM > 3:
    #     netsD.append(D_NET256())
    # if cfg.TREE.BRANCH_NUM > 4:
    #     netsD.append(D_NET512())
    # TODO: if cfg.TREE.BRANCH_NUM > 5:

    for i in range(len(netsD)):
        netsD[i].apply(weights_init)
        netsD[i] = torch.nn.DataParallel(netsD[i], device_ids=gpus)
        print(netsD[i])
    print('# of netsD', len(netsD))

    count = 0
    if cfg.TRAIN.NET_G != '':
        state_dict = torch.load(cfg.TRAIN.NET_G)
        netG.load_state_dict(state_dict)
        print('Load ', cfg.TRAIN.NET_G)

        istart = cfg.TRAIN.NET_G.rfind('_') + 1
        iend = cfg.TRAIN.NET_G.rfind('.')
        count = cfg.TRAIN.NET_G[istart:iend]
        count = int(count) + 1

    if cfg.TRAIN.NET_D != '':
        for i in range(len(netsD)):
            print('Load %s_%d.pth' % (cfg.TRAIN.NET_D, i))
            state_dict = torch.load('%s%d.pth' % (cfg.TRAIN.NET_D, i))
            netsD[i].load_state_dict(state_dict)

    inception_model = INCEPTION_V3()

    if cfg.CUDA:
        netG.cuda()
        for i in range(len(netsD)):
            netsD[i].cuda()
        inception_model = inception_model.cuda()
    inception_model.eval()

    return netG, netsD, len(netsD), inception_model, count


def define_optimizers(netG, netsD):
    optimizersD = []
    num_Ds = len(netsD)
    for i in range(num_Ds):
        opt = optim.Adam(netsD[i].parameters(),
                         lr=cfg.TRAIN.DISCRIMINATOR_LR,
                         betas=(0.5, 0.999))
        optimizersD.append(opt)

    # G_opt_paras = []
    # for p in netG.parameters():
    #     if p.requires_grad:
    #         G_opt_paras.append(p)
    optimizerG = optim.Adam(netG.parameters(),
                            lr=cfg.TRAIN.GENERATOR_LR,
                            betas=(0.5, 0.999))
    return optimizerG, optimizersD


def save_model(netG, avg_param_G, netsD, epoch, model_dir):
    load_params(netG, avg_param_G)
    torch.save(
        netG.state_dict(),
        '%s/netG_%d.pth' % (model_dir, epoch))
    for i in range(len(netsD)): # 长度为3,
        netD = netsD[i]
        torch.save(
            netD.state_dict(),
            '%s/netD%d.pth' % (model_dir, i))
    print('Save G/Ds models.')



def save_img_results(imgs_tcpu, fake_imgs, num_imgs,
                     count, image_dir, summary_writer):
    num = cfg.TRAIN.VIS_COUNT

    # The range of real_img (i.e., self.imgs_tcpu[i][0:num])
    # is changed to [0, 1] by function vutils.save_image
    print("====================%d",num)
    real_img = imgs_tcpu[-1][0:num]
    vutils.save_image(
        real_img, '%s/real_samples.png' % (image_dir),
        normalize=True)
    real_img_set = vutils.make_grid(real_img).numpy()   # (C,H,W)  图像的网格化输出
    real_img_set = np.transpose(real_img_set, (1, 2, 0))    # 反转数组，按索引（1，2，0）排列(H,W,C)
    real_img_set = real_img_set * 255
    real_img_set = real_img_set.astype(np.uint8)    # 数据类型转换为8位无符号整型uint8
    sup_real_img = summary.image('real_img', real_img_set)  # tf.summary.image(name, tensor, max_outputs=3, collections=None),输出带图像的probuf
    print(sup_real_img)
    print(count)
    summary_writer.add_summary(sup_real_img, count)
             #向FileWriter的缓存对象中添加event data

    for i in range(num_imgs):   # i为 0,1,2
        fake_img = fake_imgs[i][0:num]
        # The range of fake_img.data (i.e., self.fake_imgs[i][0:num])
        # is still [-1. 1]...
        vutils.save_image(
            fake_img.data, '%s/count_%09d_fake_samples%d.png' %
            (image_dir, count, i), normalize=True)

        fake_img_set = vutils.make_grid(fake_img.data).cpu().numpy()

        fake_img_set = np.transpose(fake_img_set, (1, 2, 0))
        fake_img_set = (fake_img_set + 1) * 255 / 2
        fake_img_set = fake_img_set.astype(np.uint8)

        sup_fake_img = summary.image('fake_img%d' % i, fake_img_set)    #汇总数据的图像的的形式如下： ’ tag /image/0’, ’ tag /image/1’…，如：input/image/0等
        summary_writer.add_summary(sup_fake_img, count)
        summary_writer.flush()
        #######     #每多少秒向disk中写数据，并清空对象缓存


# ################## For uncondional tasks ######################### #
class GANTrainer(object):
    def __init__(self, output_dir, data_loader, imsize):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.log_dir = os.path.join(output_dir, 'Log')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.log_dir)
            self.summary_writer = FileWriter(self.log_dir)

        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True

        self.batch_size = cfg.TRAIN.BATCH_SIZE * self.num_gpus
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)

    def prepare_data(self, data):
        imgs = data

        vimgs = []
        for i in range(self.num_Ds):
            if cfg.CUDA:
                vimgs.append(Variable(imgs[i]).cuda())
            else:
                vimgs.append(Variable(imgs[i]))

        return imgs, vimgs

    def train_Dnet(self, idx, count):   # 判别器网络
        flag = count % 100
        batch_size = self.real_imgs[0].size(0)
        criterion = self.criterion

        netD, optD = self.netsD[idx], self.optimizersD[idx]
        real_imgs = self.real_imgs[idx]
        fake_imgs = self.fake_imgs[idx]
        real_labels = self.real_labels[:batch_size]
        fake_labels = self.fake_labels[:batch_size]
        #
        netD.zero_grad()    # 梯度清零(pytorch中，若不清零，会梯度累加)
        #
        real_logits = netD(real_imgs)
        fake_logits = netD(fake_imgs.detach())
        # real_logits[0]：包含c向量的;real_logits[1]：不包含c向量的
        errD_real = criterion(real_logits[0], real_labels)  # 求解loss
        errD_fake = criterion(fake_logits[0], fake_labels)
        #
        errD = errD_real + errD_fake    #
        errD.backward() # 反向传播计算梯度，标量不需要指定任意参数
        # update parameters
        optD.step()  # 更新权重参数
        #
        if flag == 0:
            summary_D = summary.scalar('D_loss%d' % idx, errD.item())
            self.summary_writer.add_summary(summary_D, count)
        return errD

    def train_Gnet(self, count):    # 生成器网络
        self.netG.zero_grad()
        errG_total = 0
        flag = count % 100
        batch_size = self.real_imgs[0].size(0)
        criterion = self.criterion
        real_labels = self.real_labels[:batch_size]

        for i in range(self.num_Ds):
            netD = self.netsD[i]
            outputs = netD(self.fake_imgs[i])
            errG = criterion(outputs[0], real_labels)
            # errG = self.stage_coeff[i] * errG
            errG_total = errG_total + errG  # 3个生成器损失累加
            if flag == 0:
                summary_G = summary.scalar('G_loss%d' % i, errG.item())
                self.summary_writer.add_summary(summary_G, count)

        # Compute color preserve losses         covariance协方差
        if cfg.TRAIN.COEFF.COLOR_LOSS > 0:
            if self.num_Ds > 1:
                mu1, covariance1 = compute_mean_covariance(self.fake_imgs[-1])  # 【-1】：表示最后一个元素
                mu2, covariance2 = \
                    compute_mean_covariance(self.fake_imgs[-2].detach())
                like_mu2 = cfg.TRAIN.COEFF.COLOR_LOSS * nn.MSELoss()(mu1, mu2)
                like_cov2 = cfg.TRAIN.COEFF.COLOR_LOSS * 5 * \
                    nn.MSELoss()(covariance1, covariance2)
                errG_total = errG_total + like_mu2 + like_cov2
            if self.num_Ds > 2:
                mu1, covariance1 = compute_mean_covariance(self.fake_imgs[-2])
                mu2, covariance2 = \
                    compute_mean_covariance(self.fake_imgs[-3].detach())
                like_mu1 = cfg.TRAIN.COEFF.COLOR_LOSS * nn.MSELoss()(mu1, mu2)
                like_cov1 = cfg.TRAIN.COEFF.COLOR_LOSS * 5 * \
                    nn.MSELoss()(covariance1, covariance2)
                errG_total = errG_total + like_mu1 + like_cov1
                ###################################
            if self.num_Ds > 3:
                mu1, covariance1 = compute_mean_covariance(self.fake_imgs[-3])
                mu2, covariance2 = \
                    compute_mean_covariance(self.fake_imgs[-4].detach())
                like_mu1 = cfg.TRAIN.COEFF.COLOR_LOSS * nn.MSELoss()(mu1, mu2)
                like_cov1 = cfg.TRAIN.COEFF.COLOR_LOSS * 5 * \
                    nn.MSELoss()(covariance1, covariance2)
                errG_total = errG_total + like_mu1 + like_cov1


            if flag == 0:
                sum_mu = summary.scalar('G_like_mu2', like_mu2.item())
                self.summary_writer.add_summary(sum_mu, count)
                sum_cov = summary.scalar('G_like_cov2', like_cov2.item())   # 256和128图片的颜色损失
                self.summary_writer.add_summary(sum_cov, count)
                if self.num_Ds > 2:
                    sum_mu = summary.scalar('G_like_mu1', like_mu1.item())
                    self.summary_writer.add_summary(sum_mu, count)
                    sum_cov = summary.scalar('G_like_cov1', like_cov1.item())   # 128和64的图片颜色损失
                    self.summary_writer.add_summary(sum_cov, count)

        errG_total.backward()
        self.optimizerG.step()
        return errG_total

    def train(self):
        self.netG, self.netsD, self.num_Ds,\
            self.inception_model, start_count = load_network(self.gpus)
        avg_param_G = copy_G_params(self.netG)

        self.optimizerG, self.optimizersD = \
            define_optimizers(self.netG, self.netsD)

        self.criterion = nn.BCELoss()   # 即torch.nn.BCELoss() 二进制交叉熵损失函数
        # self.criterion = nn.MSELoss()

        self.real_labels = \
            Variable(torch.FloatTensor(self.batch_size).fill_(1))
        self.fake_labels = \
            Variable(torch.FloatTensor(self.batch_size).fill_(0))
        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(self.batch_size, nz))
        fixed_noise = \
            Variable(torch.FloatTensor(self.batch_size, nz).normal_(0, 1))

        if cfg.CUDA:
            self.criterion.cuda()
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
            self.real_labels = self.real_labels.cuda()
            self.fake_labels = self.fake_labels.cuda()

        predictions = []
        count = start_count
        start_epoch = start_count // (self.num_batches)
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()

            for step, data in enumerate(self.data_loader, 0):
                #######################################################
                # (0) Prepare training data
                ######################################################
                self.imgs_tcpu, self.real_imgs = self.prepare_data(data)

                #######################################################
                # (1) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                self.fake_imgs, _, _ = self.netG(noise)

                #######################################################
                # (2) Update D network
                ######################################################
                errD_total = 0
                for i in range(self.num_Ds):
                    errD = self.train_Dnet(i, count)
                    errD_total += errD

                #######################################################
                # (3) Update G network: maximize log(D(G(z)))
                ######################################################
                errG_total = self.train_Gnet(count)
                # errG_total = errG_total * 0.5

                for p, avg_p in zip(self.netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                # for inception score
                pred = self.inception_model(self.fake_imgs[-1].detach())
                predictions.append(pred.data.cpu().numpy())

                if count % 3 == 0:     #   if count % 100 == 0:
                    summary_D = summary.scalar('D_loss', errD_total.data[0])
                    summary_G = summary.scalar('G_loss', errG_total.data[0])
                    self.summary_writer.add_summary(summary_D, count)
                    self.summary_writer.add_summary(summary_G, count)
                if step == 0:
                    print('''[%d/%d][%d/%d] Loss_D: %.2f Loss_G: %.2f'''
                           % (epoch, self.max_epoch, step, self.num_batches,
                              errD_total.item(), errG_total.item))
                count = count + 1

                if count % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:
                #if count % 1 == 0:
                    save_model(self.netG, avg_param_G, self.netsD, count, self.model_dir)
                    save_model(self.netG, avg_param_G, self.netsD, count, self.model_dir)
                    # Save images
                    backup_para = copy_G_params(self.netG)
                    load_params(self.netG, avg_param_G)
                    #
                    self.fake_imgs, _, _ = self.netG(fixed_noise)
                    save_img_results(self.imgs_tcpu, self.fake_imgs, self.num_Ds,
                                    count, self.image_dir, self.summary_writer)
                    #
                    load_params(self.netG, backup_para)

                    # Compute inception score
                    if len(predictions) > 500:
                        predictions = np.concatenate(predictions, 0)    # 完成多个数组的拼接
                        mean, std = compute_inception_score(predictions, 10)
                        # print('mean:', mean, 'std', std)
                        m_incep = summary.scalar('Inception_mean', mean)
                        self.summary_writer.add_summary(m_incep, count)
                        #
                        mean_nlpp, std_nlpp = \
                            negative_log_posterior_probability(predictions, 10)
                        m_nlpp = summary.scalar('NLPP_mean', mean_nlpp)
                        self.summary_writer.add_summary(m_nlpp, count)
                        #
                        predictions = []

            end_t = time.time()
            print('Total Time: %.2fsec' % (end_t - start_t))

        save_model(self.netG, avg_param_G, self.netsD, count, self.model_dir)
        save_model(self.netG, avg_param_G, self.netsD, count, self.model_dir)

        self.summary_writer.close()

    def save_superimages(self, images, folder, startID, imsize):
        fullpath = '%s/%d_%d.png' % (folder, startID, imsize)
        vutils.save_image(images.data, fullpath, normalize=True)

    def save_singleimages(self, images, folder, startID, imsize):
        for i in range(images.size(0)):
            fullpath = '%s/%d_%d.png' % (folder, startID + i, imsize)
            # range from [-1, 1] to [0, 1]
            img = (images[i] + 1.0) / 2
            img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            # range from [0, 1] to [0, 255]
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath)

    def evaluate(self, split_dir):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            # Build and load the generator
            netG = G_NET()
            netG.apply(weights_init)
            netG = torch.nn.DataParallel(netG, device_ids=self.gpus)
            print(netG)
            # state_dict = torch.load(cfg.TRAIN.NET_G)
            state_dict = \
                torch.load(cfg.TRAIN.NET_G,
                           map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load ', cfg.TRAIN.NET_G)

            # the path to save generated images
            s_tmp = cfg.TRAIN.NET_G
            istart = s_tmp.rfind('_') + 1
            iend = s_tmp.rfind('.')
            iteration = int(s_tmp[istart:iend])
            s_tmp = s_tmp[:s_tmp.rfind('/')]
            save_dir = '%s/iteration%d/%s' % (s_tmp, iteration, split_dir)
            if cfg.TEST.B_EXAMPLE:
                folder = '%s/super' % (save_dir)
            else:
                folder = '%s/single' % (save_dir)
            print('Make a new folder: ', folder)
            mkdir_p(folder)

            nz = cfg.GAN.Z_DIM
            noise = Variable(torch.FloatTensor(self.batch_size, nz))
            if cfg.CUDA:
                netG.cuda()
                noise = noise.cuda()

            # switch to evaluate mode
            netG.eval()
            num_batches = int(cfg.TEST.SAMPLE_NUM / self.batch_size)
            cnt = 0
            for step in xrange(num_batches):
                noise.data.normal_(0, 1)
                fake_imgs, _, _ = netG(noise)
                if cfg.TEST.B_EXAMPLE:
                    self.save_superimages(fake_imgs[-1], folder, cnt, 256)
                else:
                    self.save_singleimages(fake_imgs[-1], folder, cnt, 256)
                    # self.save_singleimages(fake_imgs[-2], folder, 128)
                    # self.save_singleimages(fake_imgs[-3], folder, 64)
                cnt += self.batch_size


# ################# Text to image task############################ #
class condGANTrainer(object):
    def __init__(self, output_dir, data_loader, imsize):
        if cfg.TRAIN.FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.log_dir = os.path.join(output_dir, 'Log')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.log_dir)
            self.summary_writer = FileWriter(self.log_dir)

        s_gpus = cfg.GPU_ID.split(',')
        self.gpus = [int(ix) for ix in s_gpus]
        self.num_gpus = len(self.gpus)
        torch.cuda.set_device(self.gpus[0])
        cudnn.benchmark = True

        self.batch_size = cfg.TRAIN.BATCH_SIZE * self.num_gpus
        self.max_epoch = cfg.TRAIN.MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN.SNAPSHOT_INTERVAL

        self.data_loader = data_loader
        self.num_batches = len(self.data_loader)

    def prepare_data(self, data):
        imgs, w_imgs, t_embedding, _ = data

        real_vimgs, wrong_vimgs = [], []
        if cfg.CUDA:
            vembedding = Variable(t_embedding).cuda()
        else:
            vembedding = Variable(t_embedding)
        for i in range(self.num_Ds):    # 本来是3，现在变成了4个D
            if cfg.CUDA:
                real_vimgs.append(Variable(imgs[i]).cuda())
                wrong_vimgs.append(Variable(w_imgs[i]).cuda())
            else:
                real_vimgs.append(Variable(imgs[i]))
                wrong_vimgs.append(Variable(w_imgs[i]))
        return imgs, real_vimgs, wrong_vimgs, vembedding

    def train_Dnet(self, idx, count):
        flag = count % 100
        batch_size = self.real_imgs[0].size(0)
        criterion, mu = self.criterion, self.mu

        netD, optD = self.netsD[idx], self.optimizersD[idx]
        real_imgs = self.real_imgs[idx]
        wrong_imgs = self.wrong_imgs[idx]
        fake_imgs = self.fake_imgs[idx]
        #
        netD.zero_grad()
        # Forward
        real_labels = self.real_labels[:batch_size]
        fake_labels = self.fake_labels[:batch_size]
        # for real
        real_logits = netD(real_imgs, mu.detach())  # logits 为softmax层输入的量，用于转换为概率的值
        wrong_logits = netD(wrong_imgs, mu.detach())
        fake_logits = netD(fake_imgs.detach(), mu.detach())
        #
        errD_real = criterion(real_logits[0], real_labels)  # 真实图片，对应标签
        errD_wrong = criterion(wrong_logits[0], fake_labels)    # 真实图片，不对应标签
        errD_fake = criterion(fake_logits[0], fake_labels)  # 生成图片，对应标签
        if len(real_logits) > 1 and cfg.TRAIN.COEFF.UNCOND_LOSS > 0:
            errD_real_uncond = cfg.TRAIN.COEFF.UNCOND_LOSS * \
                criterion(real_logits[1], real_labels)
            errD_wrong_uncond = cfg.TRAIN.COEFF.UNCOND_LOSS * \
                criterion(wrong_logits[1], real_labels)
            errD_fake_uncond = cfg.TRAIN.COEFF.UNCOND_LOSS * \
                criterion(fake_logits[1], fake_labels)
            # 每个判别器包含一个条件损失(判别匹配)，一个无条件损失(判别真伪)
            errD_real = errD_real + errD_real_uncond
            errD_wrong = errD_wrong + errD_wrong_uncond
            errD_fake = errD_fake + errD_fake_uncond
            # 三部分损失之和
            errD = errD_real + errD_wrong + errD_fake
            # errD = (errD_real + errD_wrong + errD_fake) * 0.5   #用LSGAN的公式
        else:
            errD = errD_real + 0.5 * (errD_wrong + errD_fake)
        # backward
        errD.backward() # 求梯度
        # update parameters
        optD.step()

        # log
        if flag == 0:
            #summary_D = summary.scalar('D_loss%d' % idx, errD.data[0])
            summary_D = summary.scalar('D_loss%d' % idx, errD.item())

            self.summary_writer.add_summary(summary_D, count)
        return errD

    def train_Gnet(self, count):
        self.netG.zero_grad()
        errG_total = 0
        flag = count % 100
        batch_size = self.real_imgs[0].size(0)
        criterion, mu, logvar = self.criterion, self.mu, self.logvar
        real_labels = self.real_labels[:batch_size]
        for i in range(self.num_Ds):
            outputs = self.netsD[i](self.fake_imgs[i], mu)
            errG = criterion(outputs[0], real_labels)
            if len(outputs) > 1 and cfg.TRAIN.COEFF.UNCOND_LOSS > 0:
                errG_patch = cfg.TRAIN.COEFF.UNCOND_LOSS *\
                    criterion(outputs[1], real_labels)
                errG = errG + errG_patch
            errG_total = errG_total + errG
            if flag == 0:
                summary_D = summary.scalar('G_loss%d' % i, errG.item())   #######
                self.summary_writer.add_summary(summary_D, count)

        # Compute color consistency losses
        if cfg.TRAIN.COEFF.COLOR_LOSS > 0:
            if self.num_Ds > 1:
                mu1, covariance1 = compute_mean_covariance(self.fake_imgs[-1])
                mu2, covariance2 = \
                    compute_mean_covariance(self.fake_imgs[-2].detach())
                like_mu2 = cfg.TRAIN.COEFF.COLOR_LOSS * nn.MSELoss()(mu1, mu2)
                like_cov2 = cfg.TRAIN.COEFF.COLOR_LOSS * 5 * \
                    nn.MSELoss()(covariance1, covariance2)
                errG_total = errG_total + like_mu2 + like_cov2
                if flag == 0:
                    sum_mu = summary.scalar('G_like_mu2', like_mu2.item())
                    self.summary_writer.add_summary(sum_mu, count)
                    sum_cov = summary.scalar('G_like_cov2', like_cov2.item())  #tf.summary.scalar(name, tensor, collections=None)
                    self.summary_writer.add_summary(sum_cov, count)
            if self.num_Ds > 2:
                mu1, covariance1 = compute_mean_covariance(self.fake_imgs[-2])
                mu2, covariance2 = \
                    compute_mean_covariance(self.fake_imgs[-3].detach())
                like_mu1 = cfg.TRAIN.COEFF.COLOR_LOSS * nn.MSELoss()(mu1, mu2)
                like_cov1 = cfg.TRAIN.COEFF.COLOR_LOSS * 5 * \
                    nn.MSELoss()(covariance1, covariance2)
                errG_total = errG_total + like_mu1 + like_cov1
                if flag == 0:
                    sum_mu = summary.scalar('G_like_mu1', like_mu1.item())
                    self.summary_writer.add_summary(sum_mu, count)
                    sum_cov = summary.scalar('G_like_cov1', like_cov1.item())
                    self.summary_writer.add_summary(sum_cov, count)

        kl_loss = KL_loss(mu, logvar) * cfg.TRAIN.COEFF.KL
        errG_total = errG_total + kl_loss
        errG_total.backward()
        self.optimizerG.step()
        return kl_loss, errG_total

    def train(self):
        self.netG, self.netsD, self.num_Ds,\
            self.inception_model, start_count = load_network(self.gpus)
        avg_param_G = copy_G_params(self.netG)

        self.optimizerG, self.optimizersD = \
            define_optimizers(self.netG, self.netsD)

        self.criterion = nn.BCELoss()
        # self.criterion = nn.MSELoss()   #用LSGAN的公式

        self.real_labels = \
            Variable(torch.FloatTensor(self.batch_size).fill_(1))
        self.fake_labels = \
            Variable(torch.FloatTensor(self.batch_size).fill_(0))

        self.gradient_one = torch.FloatTensor([1.0])
        self.gradient_half = torch.FloatTensor([0.5])

        nz = cfg.GAN.Z_DIM
        noise = Variable(torch.FloatTensor(self.batch_size, nz))
        fixed_noise = \
            Variable(torch.FloatTensor(self.batch_size, nz).normal_(0, 1))

        if cfg.CUDA:
            self.criterion.cuda()
            self.real_labels = self.real_labels.cuda()
            self.fake_labels = self.fake_labels.cuda()
            self.gradient_one = self.gradient_one.cuda()
            self.gradient_half = self.gradient_half.cuda()
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        predictions = []
        count = start_count
        start_epoch = start_count // (self.num_batches)
        for epoch in range(start_epoch, self.max_epoch):
            start_t = time.time()

            for step, data in enumerate(self.data_loader, 0):   #enumerate()函数将一遍历对象变成索引序列，同时列出数据和数据下标
                #######################################################
                # (0) Prepare training data
                ######################################################
                self.imgs_tcpu, self.real_imgs, self.wrong_imgs, \
                    self.txt_embedding = self.prepare_data(data)

                #######################################################
                # (1) Generate fake images
                ######################################################
                noise.data.normal_(0, 1)
                self.fake_imgs, self.mu, self.logvar = \
                    self.netG(noise, self.txt_embedding)

                #######################################################
                # (2) Update D network
                ######################################################
                errD_total = 0
                for i in range(self.num_Ds):
                    errD = self.train_Dnet(i, count)
                    errD_total += errD  # 总损失为三个判别器的累加

                #######################################################
                # (3) Update G network: maximize log(D(G(z)))
                ######################################################
                kl_loss, errG_total = self.train_Gnet(count)

                # errD_total = errD_total *0.5
                for p, avg_p in zip(self.netG.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)

                # for inception score
                pred = self.inception_model(self.fake_imgs[-1].detach())
                predictions.append(pred.data.cpu().numpy())

                if count % 100 == 0:
                    summary_D = summary.scalar('D_loss', errD_total.item())
                    summary_G = summary.scalar('G_loss', errG_total.item())
                    summary_KL = summary.scalar('KL_loss', kl_loss.item())
                    self.summary_writer.add_summary(summary_D, count)
                    self.summary_writer.add_summary(summary_G, count)
                    self.summary_writer.add_summary(summary_KL, count)

                count = count + 1

                if count % cfg.TRAIN.SNAPSHOT_INTERVAL == 0:
                    save_model(self.netG, avg_param_G, self.netsD, count, self.model_dir)
                    # Save images
                    backup_para = copy_G_params(self.netG)
                    load_params(self.netG, avg_param_G)
                    #
                    self.fake_imgs, _, _ = \
                        self.netG(fixed_noise, self.txt_embedding)
                    save_img_results(self.imgs_tcpu, self.fake_imgs, self.num_Ds,
                                     count, self.image_dir, self.summary_writer)
                    #
                    load_params(self.netG, backup_para)

                    # Compute inception score
                    if len(predictions) > 500:
                        predictions = np.concatenate(predictions, 0)
                        mean, std = compute_inception_score(predictions, 10)
                        # print('mean:', mean, 'std', std)
                        m_incep = summary.scalar('Inception_mean', mean)
                        self.summary_writer.add_summary(m_incep, count)
                        #
                        mean_nlpp, std_nlpp = \
                            negative_log_posterior_probability(predictions, 10)
                        m_nlpp = summary.scalar('NLPP_mean', mean_nlpp)
                        self.summary_writer.add_summary(m_nlpp, count)
                        #
                        predictions = []

            end_t = time.time()
            print('''[%d/%d][%d]
                         Loss_D: %.2f Loss_G: %.2f Loss_KL: %.2f Time: %.2fs
                      '''  # D(real): %.4f D(wrong):%.4f  D(fake) %.4f
                  % (epoch, self.max_epoch, self.num_batches,
                     errD_total.item(), errG_total.item(),
                     kl_loss.item(), end_t - start_t))

            TensorWriter = SummaryWriter('../data/birds')  #######
            for epoch in range(100):
                TensorWriter.add_scalar('Loss_D', errD_total.item(), epoch)    #########

        save_model(self.netG, avg_param_G, self.netsD, count, self.model_dir)
        self.summary_writer.close()

    def save_superimages(self, images_list, filenames,
                         save_dir, split_dir, imsize):
        batch_size = images_list[0].size(0)
        num_sentences = len(images_list)
        for i in range(batch_size):
            s_tmp = '%s/super/%s/%s' %\
                (save_dir, split_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)
            #
            savename = '%s_%d.png' % (s_tmp, imsize)
            super_img = []
            for j in range(num_sentences):
                img = images_list[j][i]
                # print(img.size())
                img = img.view(1, 3, imsize, imsize)
                # print(img.size())
                super_img.append(img)
                # break
            super_img = torch.cat(super_img, 0)
            vutils.save_image(super_img, savename, nrow=10, normalize=True)

    def save_singleimages(self, images, filenames,
                          save_dir, split_dir, sentenceID, imsize):
        for i in range(images.size(0)):
            s_tmp = '%s/single_samples/%s/%s' %\
                (save_dir, split_dir, filenames[i])
            folder = s_tmp[:s_tmp.rfind('/')]
            if not os.path.isdir(folder):
                print('Make a new folder: ', folder)
                mkdir_p(folder)

            fullpath = '%s_%d_sentence%d.png' % (s_tmp, imsize, sentenceID)
            # range from [-1, 1] to [0, 255]
            img = images[i].add(1).div(2).mul(255).clamp(0, 255).byte()
            ndarr = img.permute(1, 2, 0).data.cpu().numpy()
            im = Image.fromarray(ndarr)
            im.save(fullpath)

    def evaluate(self, split_dir):
        if cfg.TRAIN.NET_G == '':
            print('Error: the path for morels is not found!')
        else:
            # Build and load the generator
            if split_dir == 'test':
                split_dir = 'valid'
            netG = G_NET()
            netG.apply(weights_init)
            netG = torch.nn.DataParallel(netG, device_ids=self.gpus)
            print(netG)
            # state_dict = torch.load(cfg.TRAIN.NET_G)
            state_dict = \
                torch.load(cfg.TRAIN.NET_G,
                           map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)    # 加载预训练模型的参数
            print('Load ', cfg.TRAIN.NET_G)

            # the path to save generated images
            s_tmp = cfg.TRAIN.NET_G
            istart = s_tmp.rfind('_') + 1
            iend = s_tmp.rfind('.')
            iteration = int(s_tmp[istart:iend])
            s_tmp = s_tmp[:s_tmp.rfind('/')]
            save_dir = '%s/iteration%d' % (s_tmp, iteration)

            nz = cfg.GAN.Z_DIM
            noise = Variable(torch.FloatTensor(self.batch_size, nz))
            if cfg.CUDA:
                netG.cuda()
                noise = noise.cuda()

            # switch to evaluate mode
            netG.eval()
            for step, data in enumerate(self.data_loader, 0):
                imgs, t_embeddings, filenames = data
                if cfg.CUDA:
                    t_embeddings = Variable(t_embeddings).cuda()
                else:
                    t_embeddings = Variable(t_embeddings)
                # print(t_embeddings[:, 0, :], t_embeddings.size(1))

                embedding_dim = t_embeddings.size(1)
                batch_size = imgs[0].size(0)
                noise.data.resize_(batch_size, nz)
                noise.data.normal_(0, 1)

                fake_img_list = []
                for i in range(embedding_dim):
                    fake_imgs, _, _ = netG(noise, t_embeddings[:, i, :])
                    if cfg.TEST.B_EXAMPLE:
                        # fake_img_list.append(fake_imgs[0].data.cpu())
                        # fake_img_list.append(fake_imgs[1].data.cpu())
                        fake_img_list.append(fake_imgs[2].data.cpu())
                    else:
                        self.save_singleimages(fake_imgs[-1], filenames,
                                               save_dir, split_dir, i, 256)
                        # self.save_singleimages(fake_imgs[-2], filenames,
                        #                        save_dir, split_dir, i, 128)
                        # self.save_singleimages(fake_imgs[-3], filenames,
                        #                        save_dir, split_dir, i, 64)
                    # break
                if cfg.TEST.B_EXAMPLE:
                    # self.save_superimages(fake_img_list, filenames,
                    #                       save_dir, split_dir, 64)
                    # self.save_superimages(fake_img_list, filenames,
                    #                       save_dir, split_dir, 128)
                    self.save_superimages(fake_img_list, filenames,
                                          save_dir, split_dir, 256)