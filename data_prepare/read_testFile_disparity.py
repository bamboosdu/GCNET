'''
Author: your name
Date: 2021-08-16 10:23:05
LastEditTime: 2021-08-16 16:08:45
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /GC-Net/read_testFile.py
'''
import os
import pickle

dir_flying='/workspace/data_aanet/SceneFlow/FlyingThings3D/disparity/TEST'
paths=[]

for root, dirs, files in os.walk(dir_flying):
    for file in files:
        paths.append(os.path.join(root,file))

paths_left=[]
paths_right=[]
for i in range(len(paths)):
	if paths[i].find('left')>-1:
		paths_left.append(paths[i])
	elif paths[i].find('right')>-1:
		paths_right.append(paths[i])
foutl=open('dataset/disp_left_test.pkl','wb')
foutr=open('dataset/disp_right_test.pkl','wb')
pickle.dump(paths_left,foutl)
pickle.dump(paths_right,foutr)
foutl.close()
foutr.close()
print(len(paths))
