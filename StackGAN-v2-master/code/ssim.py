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

from torchvision.models.vgg import vgg19


from tensorboardX import summary  #
# from tensorboard import summary
#from tensorboard import FileWriter


#from torch.utils.tensorboard import FileWriter
from tensorboardX import FileWriter   #
#from torch.utils.tensorboard import summary

# from skimage.metrics import structural_similarity as ssim
from skimage.measure import compare_ssim as ssim


from model import G_NET, D_NET64, D_NET128, D_NET256, D_NET512, D_NET1024, INCEPTION_V3

path = os.getcwd()
print(path)
img1 = np.array(Image.open('Black_Billed_Cuckoo_0001_26242.jpg'))
img2 = np.array(Image.open('Black_Billed_Cuckoo_0001_26242_128_sentence1.png'))
def compute_ssim(fake_img, real_img):


    # fake_img = fake_img.numpy()
    # real_img = real_img.numpy()
    SSIM = ssim(fake_img, real_img, multichannel=True)
    return SSIM


if __name__ == "__main__":

    ssim_scores = compute_ssim(img1, img2)
    print('ssim_scores:', ssim_scores)

