import argparse
import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import time
import math
from gc_net import *
import cv2
from PIL import Image
from torch.autograd import Variable

torch.cuda.manual_seed(1)
tsfm=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])

h=256
w=512
maxdisp=64
batch =2
model=GcNet(h,w,maxdisp)
model=model.cuda()
model=torch.nn.DataParallel(model)

checkpoint=torch.load('.checkpoint/checkpoint.ckpt.t7')
model.load_state_dict(checkpoint['net'])
start_epoch=checkpoint['epoch']
accu=checkpoint['accur']

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

model.eval()

imL=Variable(torch.FloatTensor(1).cuda())
imR=Variable(torch.FloatTensor(1).cuda())
dispL=Variable(torch.FloatTensor(1).cuda())

randomH=np.random.randint(0,160)
randomW=np.random.randint(0,400)

#todo
imageL = cv2.imread(self.paths_left[idx]).reshape(540,960,3)#.transpose((2, 0, 1))
imageR = cv2.imread(self.paths_right[idx]).reshape(540,960,3)#.transpose((2, 0, 1))
sample = {'imL': imageL, 'imR': imageR}

"""
transform
"""
sample['imL']=tsfm(sample['imL'])
sample['imR']=tsfm(sample['imR'])


imageL_rand = sample['imL'][:,:,randomH:(randomH+h),randomW:(randomW+w)]
imageR_rand = sample['imR'][:, :, randomH:(randomH + h), randomW:(randomW + w)]

with torch.no_grad():
    imL.resize_(imageL_rand.size()).copy_(imageL_rand)
    imR.resize_(imageR_rand.size()).copy_(imageR_rand)

#todo
loss_mul_list_test=[]
for d in range(maxdisp):
    loss_mul_temp=Variable(torch.Tensor(np.ones([1,1,h,w])*d)).cuda()
    loss_mul_list_test.append(loss_mul_temp)
losss_mul_test=torch.cat(loss_mul_list_test,1)

with torch.no_grad():
    result=model(imL,imR)

disp=torch.sum(result.mul(losss_mul_test),1)
im=disp.data.cpu().numpy().astype('uint8')
im=np.transpose(im,(1,2,0))

cv2.imshow('disparity',im)
cv2.waitkey(100)

