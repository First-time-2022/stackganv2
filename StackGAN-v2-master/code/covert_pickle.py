# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import PIL
import os
import os.path
import pickle
import random
import numpy as np
import pandas as pd
from miscc.config import cfg

import torch.utils.data as data
from PIL import Image
import os
import os.path
import six
import string
import sys
import torch
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']

data_dir = '../data/Flickr8k'
path = os.path.join(data_dir + '/' + 'train')  # 需自定义路径
txt_name = path + '/' + 'images' + '.txt'
df_filenames = pd.read_csv(txt_name, delim_whitespace=True, header=None)
print(df_filenames)
df2_filenames = df_filenames[0].tolist()
file1 = open(r'filenames1.pickle', 'wb')  # 一定要用二进制的形式打开
pickle.dump(df2_filenames, file1)  # 将列表存入打开的文件中
file1.close()