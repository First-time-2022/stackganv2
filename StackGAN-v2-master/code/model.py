# -*- coding: UTF-8 -*-

import torch
import torch.nn as nn
import torch.nn.parallel
from miscc.config import cfg
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import models
import torch.utils.model_zoo as model_zoo

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from GlobalAttention import GlobalAttentionGeneral as ATT_NET

# ############################## For Compute inception score ##############################
# Besides the inception score computed by pretrained model, especially for fine-grained datasets (such as birds, bedroom),
#  it is also good to compute inception score using fine-tuned model and manually examine the image quality.
class INCEPTION_V3(nn.Module):
    def __init__(self):
        super(INCEPTION_V3, self).__init__()
        self.model = models.inception_v3()
        url = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'
        # print(next(model.parameters()).data)
        state_dict = \
            model_zoo.load_url(url, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state_dict)
        for param in self.model.parameters():
            param.requires_grad = False
        print('Load pretrained model from ', url)
        # print(next(self.model.parameters()).data)
        # print(self.model)

    def forward(self, input):
        # [-1.0, 1.0] --> [0, 1.0]
        x = input * 0.5 + 0.5
        # mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]
        # --> mean = 0, std = 1
        x[:, 0] = (x[:, 0] - 0.485) / 0.229
        x[:, 1] = (x[:, 1] - 0.456) / 0.224
        x[:, 2] = (x[:, 2] - 0.406) / 0.225
        #
        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.Upsample(size=(299, 299), mode='bilinear')(x)
        # 299 x 299 x 3
        x = self.model(x)
        x = nn.Softmax()(x)
        return x


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc / 2)
        return x[:, :nc] * F.sigmoid(x[:, nc:])


def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=False)

def conv5x5(in_planes, out_planes):
    "5x5 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=5, stride=1,
                     padding=2, bias=False)

def conv7x7(in_planes, out_planes):
    "7x7 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=1,
                     padding=3, bias=False)
# ############## G networks ################################################
# Upsale the spatial size by a factor of 2  将空间大小增加2倍
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        # nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block


def upBlock1(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=4, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block


# Keep the spatial size
def Block3x3_relu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        # nn.GroupNorm(8, out_planes * 2),
        GLU()

    )
    return block


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm2d(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num)
        )

    def forward(self, x):
        residual = x
        # print('residual.shape:', residual.shape)
        out = self.block(x)
        out += residual
        # print('out_resblock.shape:', out.shape)
        return out


class ResBlock1(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock1, self).__init__()
        self.seq = conv1x1(channel_num, channel_num)
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num),
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm2d(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num)
        )

    def forward(self, x):
        x = self.seq(x)
        # print('x.shape:', x.shape)
        residual = x
        # print('residual1.shape:', residual.shape)
        out = self.block(x)
        out += residual
        # print('out1_resblock.shape:', out.shape)
        return out


######################################################################################################## RRDB模块
class ResidualDenseBlock(nn.Module):
    def __init__(self, nf, gc=32, res_scale=0.2):
        super(ResidualDenseBlock, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(nf + 0 * gc, gc, 3, padding=1, bias=True), nn.LeakyReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(nf + 1 * gc, gc, 3, padding=1, bias=True), nn.LeakyReLU())
        self.layer3 = nn.Sequential(nn.Conv2d(nf + 2 * gc, gc, 3, padding=1, bias=True), nn.LeakyReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(nf + 3 * gc, gc, 3, padding=1, bias=True), nn.LeakyReLU())
        self.layer5 = nn.Sequential(nn.Conv2d(nf + 4 * gc, nf, 3, padding=1, bias=True), nn.LeakyReLU())

        self.res_scale = res_scale

    def forward(self, x):
        layer1 = self.layer1(x)
        layer2 = self.layer2(torch.cat((x, layer1), 1))
        layer3 = self.layer3(torch.cat((x, layer1, layer2), 1))
        layer4 = self.layer4(torch.cat((x, layer1, layer2, layer3), 1))
        layer5 = self.layer5(torch.cat((x, layer1, layer2, layer3, layer4), 1))
        return layer5.mul(self.res_scale) + x


class ResidualInResidualDenseBlock(nn.Module):   # RDDB(ESRGAN,增强型超分辨率中提出)
    def __init__(self, nf, gc=32, res_scale=0.2):
        super(ResidualInResidualDenseBlock, self).__init__()
        self.layer1 = ResidualDenseBlock(nf, gc)
        self.layer2 = ResidualDenseBlock(nf, gc)
        self.layer3 = ResidualDenseBlock(nf, gc, )
        self.res_scale = res_scale

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        return out.mul(self.res_scale) + x


def upsample_block(nf, scale_factor=2):
    block = []
    for _ in range(scale_factor // 2):
        block += [
            nn.Conv2d(nf, nf * (2 ** 2), 1),
            nn.PixelShuffle(2),    # 像素混洗
            nn.ReLU()
        ]

    return nn.Sequential(*block)


######################################################################################################## RRDB模块

class CA_NET(nn.Module):  # （CA）条件增强
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self):
        super(CA_NET, self).__init__()
        self.t_dim = cfg.TEXT.DIMENSION
        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.fc = nn.Linear(self.t_dim, self.ef_dim * 4, bias=True)
        self.relu = GLU()

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.ef_dim]
        logvar = x[:, self.ef_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if cfg.CUDA:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        mu, logvar = self.encode(text_embedding)
        c_code = self.reparametrize(mu, logvar)
        return c_code, mu, logvar


class Self_Attn_PAM(nn.Module):
    """ Self attention Layer"""
    # in_dim=128
    def __init__(self, in_dim, activation):
        super(Self_Attn_PAM, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """

        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, height * width).permute(0, 2, 1)  # B X N X C
        proj_key = self.key_conv(x).view(m_batchsize, -1, height * width)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, height * width)  # B X C X N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        #
        out = self.gamma * out + x
        return out


class Self_Attn_CAM(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim, activation):
        super(Self_Attn_CAM, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """

        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, C, -1)  # B C N
        proj_key = self.key_conv(x).view(m_batchsize, C, -1).permute(0, 2, 1)   # B N C
        energy = torch.bmm(proj_query, proj_key)    # B C N X B N C -> B C C
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = self.value_conv(x).view(m_batchsize, C, -1)    # B C N
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class Self_Res(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Res, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.maxpool = nn.MaxPool2d(2, stride=2)    # H,W 缩小2倍
        self.conv1X1 = conv1x1(in_dim, in_dim)
        self.conv1X1_down = conv1x1(in_dim * 2, in_dim)
        self.conv3X3 = conv3x3(in_dim, in_dim)
        self.conv5X5 = conv5x5(in_dim, in_dim)
        self.conv7X7 = conv7x7(in_dim, in_dim)
        self.upsample = upBlock(in_dim, in_dim)
        self.block = Block3x3_relu(in_dim * 4, in_dim)
        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, height, width = x.size()
        max_pool_x = self.maxpool(x)    # 最大池化  B C H//2 W//2
        max_pool_x1 = self.conv1X1(max_pool_x)  # 最大池化后1x1卷积 B C H//2 W//2
        # proj_query = self.query_conv(x).view(m_batchsize, -1, height * width).permute(0, 2, 1)  # B X CX(N)
        # proj_key = self.key_conv(x).view(m_batchsize, -1, height * width)  # B X C x (*W*H)
        max_pool_x1_out = self.upsample(max_pool_x1)    # B C H W
        # print('max_pool_x1_out.shape:', max_pool_x1_out.shape)
        # attention = self.softmax(energy)  # BX (N) X (N)
        conv3x3_x = self.conv3X3(x)
        conv5x5_x = self.conv5X5(x)
        conv7x7_x = self.conv7X7(x)
        conv_cat = torch.cat((conv3x3_x, conv5x5_x, conv7x7_x, max_pool_x1_out), 1) # B 4*C H W (8,256,64,64)
        # print('conv_cat.shape:', conv_cat.shape)
        # out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = self.block(conv_cat)   # B C H W
        # out = self.gamma * out + x
        return out




class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        if cfg.GAN.B_CONDITION:
            self.in_dim = cfg.GAN.Z_DIM + cfg.GAN.EMBEDDING_DIM
        else:
            self.in_dim = cfg.GAN.Z_DIM
        self.define_module()


    def define_module(self):
        in_dim = self.in_dim
        ngf = self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, ngf * 4 * 4 * 2, bias=False), # (B,in_dim)->(B,ngf*4*4*2)
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU())

        # ngf//4 :除法，返回结果的整数部分
        self.upsample1 = upBlock(ngf, ngf // 2)  # 通道数64->64
        self.upsample2 = upBlock(ngf // 2, ngf // 4)  # 32->32
        self.upsample3 = upBlock(ngf // 4, ngf // 8)  # 16->16
        self.upsample4 = upBlock(ngf // 8, ngf // 16)  # 8->8

        # self.attn = Self_Attn(ngf // 4, 'relu')

    def forward(self, z_code, c_code=None): # c_code:(batch_size,128) z_code:(batch_size,100)  in_code(batch_size,228)
        if cfg.GAN.B_CONDITION and c_code is not None:
            in_code = torch.cat((c_code, z_code), 1)
        else:
            in_code = z_code
        # state size 16ngf x 4 x 4
        out_code = self.fc(in_code) # out_code:(batch_size,16384)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        # state size 8ngf x 8 x 8
        out_code = self.upsample1(out_code)
        # state size 4ngf x 16 x 16
        out_code = self.upsample2(out_code)
        # out_code = self.attn(out_code)
        # state size 2ngf x 32 x 32
        out_code = self.upsample3(out_code)
        # # state size ngf x 64 x 64
        out_code = self.upsample4(out_code)

        return out_code


class NEXT_STAGE_G(nn.Module):
    def __init__(self, ngf, num_residual=cfg.GAN.R_NUM):
        super(NEXT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        if cfg.GAN.B_CONDITION:
            self.ef_dim = cfg.GAN.EMBEDDING_DIM
        else:
            self.ef_dim = cfg.GAN.Z_DIM
        self.num_residual = num_residual
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(self.num_residual):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def _make_layer1(self, block, channel_num):
        layers1 = []
        # for i in range(self.num_residual):
        #     layers.append(block(channel_num))
        layers1.append(block(channel_num))
        return nn.Sequential(*layers1)

    def define_module(self):
        ngf = self.gf_dim
        efg = self.ef_dim

        self.jointConv = Block3x3_relu(ngf + efg, ngf)
        # self.attn = Self_Attn(ngf, 'relu')
        self.residual = self._make_layer(ResBlock, ngf)
        self.upsample = upBlock(ngf, ngf // 2)

        # self.attn = Self_Attn(ngf // 2, 'relu')

    def forward(self, h_code, c_code):
        s_size = h_code.size(2)
        c_code = c_code.view(-1, self.ef_dim, 1, 1) # c_code(batch_size, 128, 1, 1)
        c_code = c_code.repeat(1, 1, s_size, s_size)    # c_code(batch_size, 128, 64, 64)
        # state size (ngf+egf) x in_size x in_size
        h_c_code = torch.cat((c_code, h_code), 1)
        # state size ngf x in_size x in_size
        out_code = self.jointConv(h_c_code)
        # out_code = self.attn(out_code)
        out_code = self.residual(out_code)
        # out_c_code = torch.cat((c_code, out_code), 1)   # 在第二个残差块加入c_code
        # out_code = self.jointConv(out_c_code)
        # out_code = self.residual1(out_code)
        # state size ngf/2 x 2in_size x 2in_size
        out_code = self.upsample(out_code)

        return out_code


class NEXT_STAGE_G1(nn.Module):
    def __init__(self, ngf, num_residual=cfg.GAN.R_NUM):
        super(NEXT_STAGE_G1, self).__init__()
        self.gf_dim = ngf
        if cfg.GAN.B_CONDITION:
            self.ef_dim = cfg.GAN.EMBEDDING_DIM
        else:
            self.ef_dim = cfg.GAN.Z_DIM
        self.num_residual = num_residual
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(self.num_residual):
            layers.append(block(channel_num))
        # layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def _make_layer1(self, block, channel_num):
        layers1 = []
        # for i in range(self.num_residual):
        #     layers.append(block(channel_num))
        layers1.append(block(channel_num))
        return nn.Sequential(*layers1)

    def define_module(self):
        ngf = self.gf_dim
        efg = self.ef_dim

        self.jointConv = Block3x3_relu(ngf + efg, ngf)
        # self.attn_PAM= Self_Attn_PAM(ngf, 'relu')   # 空间注意力
        # self.attn_CAM = Self_Attn_CAM(ngf, 'relu')  # 通道注意力
        self.residual = self._make_layer(ResBlock, ngf)
        self.upsample = upBlock(ngf, ngf // 2)
        # self.upsample1 = upBlock(ngf * 2, ngf // 2)  #
        # self.attn_PAM= Self_Attn_PAM(ngf, 'relu')   # 空间注意力
        # self.attn_CAM = Self_Attn_CAM(ngf, 'relu')  # 通道注意力

    def forward(self, h_code, c_code):  # h_code(8,32,128,128)

        """
            h_code1(query):  batch x idf x ih x iw (queryL=ihxiw)
            word_embs(context): batch x cdf x sourceL (sourceL=seq_len)
            c_code1: batch x idf x queryL
            att1: batch x sourceL x queryL
        """
        # print('h_code64.shape:', h_code64.shape)
        # h_code64 = self.upsample1(h_code64)
        # print('h_code64_upsample.shape:', h_code64.shape)

        s_size = h_code.size(2)
        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, s_size, s_size)
        # state size (ngf+egf) x in_size x in_size
        # print('h_code.shape:', h_code.shape)
        h_c_code = torch.cat((c_code, h_code), 1)
        # # state size ngf x in_size x in_size
        out_code = self.jointConv(h_c_code)
        # out_code = self.res(out_code)
        # print('out_code.shape:', out_code.shape)

        # out_code1 = self.attn_PAM(out_code)
        # out_code2 = self.attn_CAM(out_code)
        # out_code = torch.cat((out_code1, out_code2), 1) # B 2*C H W
        out_code = self.residual(out_code)
        # out_c_code = torch.cat((c_code, out_code), 1)
        # out_code = self.residual1(out_code)
        # state size ngf/2 x 2in_size x 2in_size
        # out_code1 = self.attn_PAM(out_code)
        # out_code2 = self.attn_CAM(out_code)
        # out_code = torch.cat((out_code1, out_code2), 1) # B 2*C H W


        out_code = self.upsample(out_code)
        # out_code = self.upsample1(out_code)
        # out_code = self.attn(out_code)

        return out_code


class GET_IMAGE_G(nn.Module):  # 对应每个生成器后的Conv3*3
    def __init__(self, ngf):
        super(GET_IMAGE_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, 3),  # 通道变成3
            nn.Tanh()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


class G_NET(nn.Module):
    def __init__(self):
        super(G_NET, self).__init__()
        self.gf_dim = cfg.GAN.GF_DIM
        self.define_module()

    def define_module(self):
        if cfg.GAN.B_CONDITION:
            self.ca_net = CA_NET()

        if cfg.TREE.BRANCH_NUM > 0:
            self.h_net1 = INIT_STAGE_G(self.gf_dim * 16)
            self.img_net1 = GET_IMAGE_G(self.gf_dim)
        if cfg.TREE.BRANCH_NUM > 1:
            self.h_net2 = NEXT_STAGE_G1(self.gf_dim)
            self.img_net2 = GET_IMAGE_G(self.gf_dim // 2)
        if cfg.TREE.BRANCH_NUM > 2:
            # self.h_net3 = NEXT_STAGE_G(self.gf_dim // 2)
            self.h_net3 = NEXT_STAGE_G1(self.gf_dim // 2)  # 在生成256*256的生成器上改动残差模块
            self.img_net3 = GET_IMAGE_G(self.gf_dim // 4)

    def forward(self, z_code, text_embedding=None):
        c_code, mu, logvar = self.ca_net(text_embedding)
        fake_imgs = []
        if cfg.TREE.BRANCH_NUM > 0:
            h_code1 = self.h_net1(z_code, c_code)
            fake_img1 = self.img_net1(h_code1)
            fake_imgs.append(fake_img1)
        if cfg.TREE.BRANCH_NUM > 1:
            h_code2 = self.h_net2(h_code1, c_code)
            fake_img2 = self.img_net2(h_code2)
            fake_imgs.append(fake_img2)
        if cfg.TREE.BRANCH_NUM > 2:
            h_code3 = self.h_net3(h_code2, c_code)
            fake_img3 = self.img_net3(h_code3)
            fake_imgs.append(fake_img3)
        if cfg.TREE.BRANCH_NUM > 3:
            h_code4 = self.h_net4(h_code3, c_code)
            fake_img4 = self.img_net4(h_code4)
            fake_imgs.append(fake_img4)

        return fake_imgs, mu, logvar


# ############## D networks ################################################
def Block3x3_leakRelu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


# Downsale the spatial size by a factor of 2    图片尺寸缩小一半
def downBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


# Downsale the spatial size by a factor of 16   # 空间尺寸缩小16倍
def encode_image_by_16times(ndf):
    encode_img = nn.Sequential(
        # --> state size. ndf x in_size/2 x in_size/2   64*128*128
        nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 2ndf x x in_size/4 x in_size/4 128*64*64
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 4ndf x in_size/8 x in_size/8   256*32*32
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        # Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 8ndf x in_size/16 x in_size/16 512*16*16
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return encode_img


# For 64 x 64 images
class D_NET64(nn.Module):
    def __init__(self):
        super(D_NET64, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM  # config.py中，64
        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

        if cfg.GAN.B_CONDITION:
            self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
            self.uncond_logits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),  # 512*16*16->1*4*4
                nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)

        if cfg.GAN.B_CONDITION and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((c_code, x_code), 1)  # 按维度1拼接，横着拼
            # state size ngf x in_size x in_size
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = x_code

        output = self.logits(h_c_code)
        if cfg.GAN.B_CONDITION:
            out_uncond = self.uncond_logits(x_code)
            return [output.view(-1), out_uncond.view(-1)]
        else:
            return [output.view(-1)]  # .view(-1)将所有维度数据转化为一维的，按顺序排列


# For 128 x 128 images
class D_NET128(nn.Module):
    def __init__(self):
        super(D_NET128, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM  # 128
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s32_1 = Block3x3_leakRelu(ndf * 16, ndf * 8)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

        if cfg.GAN.B_CONDITION:
            self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
            self.uncond_logits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s32_1(x_code)

        if cfg.GAN.B_CONDITION and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((c_code, x_code), 1)
            # state size ngf x in_size x in_size
            h_c_code = self.jointConv(h_c_code)  # Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
        else:
            h_c_code = x_code

        output = self.logits(h_c_code)
        if cfg.GAN.B_CONDITION:
            out_uncond = self.uncond_logits(x_code)
            return [output.view(-1), out_uncond.view(-1)]  # 返回两个元素的列表,一个含c向量，一个不含
        else:
            return [output.view(-1)]


# For 256 x 256 images
class D_NET256(nn.Module):
    def __init__(self):
        super(D_NET256, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s64 = downBlock(ndf * 16, ndf * 32)
        self.img_code_s64_1 = Block3x3_leakRelu(ndf * 32, ndf * 16)
        self.img_code_s64_2 = Block3x3_leakRelu(ndf * 16, ndf * 8)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

        if cfg.GAN.B_CONDITION:
            self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
            self.uncond_logits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s64(x_code)
        x_code = self.img_code_s64_1(x_code)
        x_code = self.img_code_s64_2(x_code)

        if cfg.GAN.B_CONDITION and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((c_code, x_code), 1)
            # state size ngf x in_size x in_size
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = x_code

        output = self.logits(h_c_code)
        if cfg.GAN.B_CONDITION:
            out_uncond = self.uncond_logits(x_code)
            return [output.view(-1), out_uncond.view(-1)]
        else:
            return [output.view(-1)]


# For 512 x 512 images: Recommended structure, not test yet
class D_NET512(nn.Module):
    def __init__(self):
        super(D_NET512, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s64 = downBlock(ndf * 16, ndf * 32)
        self.img_code_s128 = downBlock(ndf * 32, ndf * 64)
        self.img_code_s128_1 = Block3x3_leakRelu(ndf * 64, ndf * 32)
        self.img_code_s128_2 = Block3x3_leakRelu(ndf * 32, ndf * 16)
        self.img_code_s128_3 = Block3x3_leakRelu(ndf * 16, ndf * 8)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

        if cfg.GAN.B_CONDITION:
            self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
            self.uncond_logits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s64(x_code)
        x_code = self.img_code_s128(x_code)
        x_code = self.img_code_s128_1(x_code)
        x_code = self.img_code_s128_2(x_code)
        x_code = self.img_code_s128_3(x_code)

        if cfg.GAN.B_CONDITION and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((c_code, x_code), 1)
            # state size ngf x in_size x in_size
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = x_code

        output = self.logits(h_c_code)
        if cfg.GAN.B_CONDITION:
            out_uncond = self.uncond_logits(x_code)
            return [output.view(-1), out_uncond.view(-1)]
        else:
            return [output.view(-1)]


# For 1024 x 1024 images: Recommended structure, not test yet
class D_NET1024(nn.Module):
    def __init__(self):
        super(D_NET1024, self).__init__()
        self.df_dim = cfg.GAN.DF_DIM
        self.ef_dim = cfg.GAN.EMBEDDING_DIM
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s64 = downBlock(ndf * 16, ndf * 32)
        self.img_code_s128 = downBlock(ndf * 32, ndf * 64)
        self.img_code_s256 = downBlock(ndf * 64, ndf * 128)
        self.img_code_s256_1 = Block3x3_leakRelu(ndf * 128, ndf * 64)
        self.img_code_s256_2 = Block3x3_leakRelu(ndf * 64, ndf * 32)
        self.img_code_s256_3 = Block3x3_leakRelu(ndf * 32, ndf * 16)
        self.img_code_s256_4 = Block3x3_leakRelu(ndf * 16, ndf * 8)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

        if cfg.GAN.B_CONDITION:
            self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
            self.uncond_logits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s64(x_code)
        x_code = self.img_code_s128(x_code)
        x_code = self.img_code_s256(x_code)
        x_code = self.img_code_s256_1(x_code)
        x_code = self.img_code_s256_2(x_code)
        x_code = self.img_code_s256_3(x_code)
        x_code = self.img_code_s256_4(x_code)

        if cfg.GAN.B_CONDITION and c_code is not None:
            c_code = c_code.view(-1, self.ef_dim, 1, 1)
            c_code = c_code.repeat(1, 1, 4, 4)
            # state size (ngf+egf) x 4 x 4
            h_c_code = torch.cat((c_code, x_code), 1)
            # state size ngf x in_size x in_size
            h_c_code = self.jointConv(h_c_code)
        else:
            h_c_code = x_code

        output = self.logits(h_c_code)
        if cfg.GAN.B_CONDITION:
            out_uncond = self.uncond_logits(x_code)
            return [output.view(-1), out_uncond.view(-1)]
        else:
            return [output.view(-1)]
