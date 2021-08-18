# -*- coding: UTF-8 -*- 
'''
Author: your name
Date: 2021-08-16 10:23:05
LastEditTime: 2021-08-16 14:00:40
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /GC-Net/GCNET/read_sceneflow.py
'''
import os
import pickle
# dir = "disparity"
dir_driving='/workspace/data_aanet/SceneFlow/Driving/disparity'
dir_flying='/workspace/data_aanet/SceneFlow/FlyingThings3D/disparity/TRAIN'
dir_monkk='/workspace/data_aanet/SceneFlow/Monkaa/disparity'

paths=[]
paths_test=[]
for root, dirs, files in os.walk(dir_driving):
    for file in files:
        paths.append(os.path.join(root,file))
for root, dirs, files in os.walk(dir_flying):
    for file in files:
        paths.append(os.path.join(root,file))
for root, dirs, files in os.walk(dir_monkk):
    for file in files:
        paths.append(os.path.join(root,file))
paths_left=[]
paths_right=[]
for i in range(len(paths)):
	if paths[i].find('left')>-1:
		paths_left.append(paths[i])
	elif paths[i].find('right')>-1:
		paths_right.append(paths[i])
foutl=open('dataset/disp_left_train.pkl','wb')
foutr=open('dataset/disp_right_train.pkl','wb')
# paths_left=paths_left.encode("utf-8")
# paths_right=paths_right.encode("utf-8")
pickle.dump(paths_left,foutl)
pickle.dump(paths_right,foutr)
foutl.close()
foutr.close()
print(len(paths))
