import numpy as np
import scipy.io as sio
from skimage.io import imread, imsave
import cv2
import os
import matplotlib.pyplot as plt
from api import PRN
import utils.depth_image as DepthImage



prn = PRN(is_dlib = True, is_opencv = False) 


image = imread(r'TestImages\1_1.avi_25_real.jpg')
image_shape = [image.shape[0], image.shape[1]]
pos, cropped_image = prn.process(image, None, None, image_shape)


kpt = prn.get_landmarks(pos)
        # 3D vertices
vertices = prn.get_vertices(pos)
depth_scene_map = DepthImage.generate_depth_image(vertices, kpt, image.shape, isMedFilter=True)


# cv2.imwrite(r'TestImages\1_1.avi_25_real_cropped.jpg',cv2.convertScaleAbs(cropped_image, alpha=(255.0))[:,:,::-1]   )
plt.imshow(depth_scene_map,cmap = 'gray')
plt.show()