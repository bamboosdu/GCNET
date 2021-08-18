'''
Author: your name
Date: 2021-08-16 10:23:05
LastEditTime: 2021-08-16 12:03:06
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /GC-Net/GCNET/read_frame_cleanpass.py
'''
import os
import pickle
# dir = "disparity"
dir_driving='/workspace/data_aanet/SceneFlow/Driving/frames_cleanpass'
dir_flying='/workspace/data_aanet/SceneFlow/FlyingThings3D/frames_cleanpass/TRAIN'
dir_monkk='/workspace/data_aanet/SceneFlow/Monkaa/frames_cleanpass'
paths=[]
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
foutl=open('dataset/paths_left_train.pkl','wb')
foutr=open('dataset/paths_right_train.pkl','wb')
pickle.dump(paths_left,foutl)
pickle.dump(paths_right,foutr)
foutl.close()
foutr.close()
print(len(paths))
