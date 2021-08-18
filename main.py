'''
Author: your name
Date: 2021-08-13 18:19:58
LastEditTime: 2021-08-16 18:42:33
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /GC-Net/GCNET/main.py
'''
import os
import torch
import torchvision
import torchvision.transforms as transforms
import cv2
from torch.utils.data import Dataset, DataLoader

from torch.autograd import Variable
import numpy as np
from read_data import sceneDisp
import torch.optim as optim

from gc_net import *
from python_pfm import *

def normalizeRGB(img):
    return img

os.environ['CUDA_VISIBLE_DEVICES']='0,1'
tsfm=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))])

h=256
w=512
maxdisp=64
batch =2
net=GcNet(h,w,maxdisp)
net=net.cuda()
net=torch.nn.DataParallel(net)

def train(epoch_total,loadstate):
    print('I am running')
    loss_mul_list=[]
    for d in range(maxdisp):
        loss_mul_temp=Variable(torch.Tensor(np.ones([batch,1,h,w])*d)).cuda()
        loss_mul_list.append(loss_mul_temp)
    loss_mul=torch.cat(loss_mul_list,1)

    optimizer=optim.RMSprop(net.parameters(),lr=0.001,alpha=0.9)
    dataset=sceneDisp('','train',tsfm)
    loss_fn=nn.L1Loss()
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=batch,shuffle=True,num_workers=1)

    imL=Variable(torch.FloatTensor(1).cuda())
    imR=Variable(torch.FloatTensor(1).cuda())
    dispL=Variable(torch.FloatTensor(1).cuda())

    loss_list=[]
    print(len(dataloader))

    start_epoch=0
    accu=0
    if loadstate==True:
        checkpoint=torch.load('./checkpoint.ckpt.t7')
        net.load_state_dict(checkpoint['net'])
        start_epoch=checkpoint['epoch']
        accu=checkpoint['accur']

    print('start epoch:%d accuracy:%f'%(start_epoch,accu))
    
    for epoch in range(start_epoch,epoch_total):
        net.train()
        data_iter=iter(dataloader)

        print('\n Epoch:%d'% epoch)
        train_loss=0
        acc_total=0
        for step in range(len(dataloader)-1):
            print('----epoch:%d-------step:%d------'%(epoch,step))
            data=next(data_iter)

            randomH=np.random.randint(0,160)
            randomW=np.random.randint(0,400)
            imageL = data['imL'][:,:,randomH:(randomH+h),randomW:(randomW+w)]
            imageR = data['imR'][:, :, randomH:(randomH + h), randomW:(randomW + w)]
            disL = data['dispL'][:, :, randomH:(randomH + h), randomW:(randomW + w)]
            with torch.no_grad():
                imL.resize_(imageL.size()).copy_(imageL)
            
                imR.resize_(imageR.size()).copy_(imageR)
            
                dispL.resize_(disL.size()).copy_(disL)

            net.zero_grad()
            optimizer.zero_grad()

            x=net(imL,imR)
            result=torch.sum(x.mul(loss_mul),1)
            tt=loss_fn(result,dispL)
            train_loss+=tt.data

            tt.backward()
            optimizer.step()

            print('=======loss value for every step=======:%f' % (tt.data))
            print('=======average loss value for every step=======:%f' %(train_loss/(step+1)))
            result=result.view(batch,1,h,w)
            diff=torch.abs(result.data.cpu()-dispL.data.cpu())
            print(diff.shape)
            accuracy=torch.sum(diff<3)/float(h*w*batch)
            acc_total+=accuracy
            print('====accuracy for the result less than 3 pixels===:%f' %accuracy)
            print('====average accuracy for the result less than 3 pixels===:%f' % (acc_total/(step+1)))

            if step%100==0:
                loss_list.append(train_loss/(step+1))
            if (step>1 and step%200==0) or step==len(dataloader)-2:
                print('=======>saving model......')
                state={'net':net.state_dict(),'step':step,
                       'loss_list':loss_list,'epoch':epoch,'accur':acc_total}
                torch.save(state,'checkpoint/ckpt.t7')

                im = result[0, :, :, :].data.cpu().numpy().astype('uint8')
                im = np.transpose(im, (1, 2, 0))
                cv2.imwrite('train_result.png', im, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
                gt = np.transpose(dispL[0, :, :, :].data.cpu().numpy(), (1, 2, 0))
                cv2.imwrite('train_gt.png', gt, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    fp=open('./checkpoint/loss.txt','w')
    for i in range(len(loss_list)):
        fp.write(str(loss_list[i][0]))
        fp.write('\n')
    fp.close()


def test(loadstate):

    if loadstate==True:
        checkpoint = torch.load('./checkpoint/ckpt.t7')
        net.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        accu=checkpoint['accu']
    net.eval()
    imL = Variable(torch.FloatTensor(1).cuda())
    imR = Variable(torch.FloatTensor(1).cuda())
    dispL = Variable(torch.FloatTensor(1).cuda())

    dataset = sceneDisp('', 'test',tsfm)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    data_iter = iter(dataloader)
    data = next(data_iter)

    randomH = np.random.randint(0, 160)
    randomW = np.random.randint(0, 400)
    print('test')
    imageL = data['imL'][:, :, randomH:(randomH + h), randomW:(randomW + w)]
    imageR = data['imR'][:, :, randomH:(randomH + h), randomW:(randomW + w)]
    disL = data['dispL'][:, :, randomH:(randomH + h), randomW:(randomW + w)]
    imL.data.resize_(imageL.size()).copy_(imageL)
    imR.data.resize_(imageR.size()).copy_(imageR)
    dispL.data.resize_(disL.size()).copy_(disL)
    loss_mul_list_test = []
    for d in range(maxdisp):
        loss_mul_temp = Variable(torch.Tensor(np.ones([1, 1, h, w]) * d)).cuda()
        loss_mul_list_test.append(loss_mul_temp)
    loss_mul_test = torch.cat(loss_mul_list_test, 1)

    with torch.no_grad():
        result=net(imL,imR)

    disp=torch.sum(result.mul(loss_mul_test),1)
    diff = torch.abs(disp.data.cpu() -dispL.data.cpu())  # end-point-error

    accuracy = torch.sum(diff < 3) / float(h * w)
    print('test accuracy less than 3 pixels:%f' %accuracy)

    # save
    im=disp.data.cpu().numpy().astype('uint8')
    im=np.transpose(im,(1,2,0))
    cv2.imwrite('./checkpoint/test_result.png',im,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    gt=np.transpose(dispL[0,:,:,:].data.cpu().numpy(),(1,2,0))
    cv2.imwrite('./checkpoint/test_gt.png',gt,[int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    return disp

def main():
    epoch_total=20
    load_state=False
    print('second')
    train(epoch_total,load_state)
    test(load_state)

if __name__=='__main__':
    print('first')
    main()
