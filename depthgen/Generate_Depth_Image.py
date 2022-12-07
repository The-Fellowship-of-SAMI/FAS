import numpy as np
import scipy.io as sio
from skimage.io import imread, imsave
import cv2
import os

from api import PRN
import utils.depth_image as DepthImage

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

prn = PRN(is_dlib = True, is_opencv = False) 

# path_image = './TestImages/0.jpg'

out_dir ='./zalo_data/depth/'

in_dir = './zalo_data/color'

# for imfile in os.listdir(in_dir):
#     image = imread(in_dir+f'/{imfile}')
#     image_shape = [image.shape[0], image.shape[1]]
#     try:
#         pos, cropped_image = prn.process(image, None, None, image_shape)
#         cv2.imwrite(in_dir+f'/{imfile}',cv2.convertScaleAbs(cropped_image, alpha=(255.0))[:,:,::-1] )
#     except:
#         os.remove(in_dir+f'/{imfile}')
    

for imfile in os.listdir(in_dir):
    image = imread(in_dir+f'/{imfile}')
    # image = imread(path_image)
    # print(in_dir+f'/{imfile}')
    if os.path.exists(out_dir+f'/{imfile}'):
        continue
    image_shape = [image.shape[0], image.shape[1]]

    if 'fake' in imfile:
        try:
            _, cropped_image = prn.process(image, None, None, image_shape)
            cv2.imwrite(out_dir+f'/{imfile}',np.zeros([256,256,3]))
        except:
            os.remove(in_dir+f'/{imfile}')
        
    else:
        try:
            pos, cropped_image = prn.process(image, None, None, image_shape)

            kpt = prn.get_landmarks(pos)

        # 3D vertices
            vertices = prn.get_vertices(pos)

            depth_scene_map = DepthImage.generate_depth_image(vertices, kpt, image.shape, isMedFilter=True)

            cv2.imwrite(out_dir+f'/{imfile}',depth_scene_map)


        except:
            os.remove(in_dir+f'/{imfile}')
    


