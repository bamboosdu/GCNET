import cv2
import os.path
import glob
import numpy as np
from PIL import Image
 
def convertPNG(pngfile,outdir):
    # READ THE DEPTH
    im_depth = cv2.imread(pngfile)
    #apply colormap on deoth image(image must be converted to 8-bit per pixel first)
    im_color=cv2.applyColorMap(cv2.convertScaleAbs(im_depth,alpha=3),cv2.COLORMAP_JET)
    #convert to mat png
    im=Image.fromarray(im_color)
    #save image
    im.save(os.path.join(outdir,os.path.basename(pngfile)))
 
for pngfile in glob.glob("./*.png"):#C:/Users/BAMBOO/Desktop/source pics/rgbd_6/depth/*.png
    print(pngfile)
    convertPNG(pngfile,"./out_png")#C:/Users/BAMBOO/Desktop/source pics/rgbd_6/color
