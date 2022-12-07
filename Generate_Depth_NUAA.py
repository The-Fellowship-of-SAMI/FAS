import numpy as np
import scipy.io as sio
from skimage.io import imread, imsave
import cv2
import os

from depthgen.api import PRN
import depthgen.utils.depth_image as DepthImage

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

prn = PRN(is_dlib = False, is_opencv = False) 

# path_image = './TestImages/0.jpg'

out_dir ='./data/processed/NUAA/'
out_option = ['train', 'test']

in_dir = './data/raw/NUAA/'


    
folders = {'client': 'ClientFace','imposter': 'ImposterFace'}
for type in folders:
    folder = folders[type]
    in_folder = in_dir+folder
    for opt in out_option:
        out_folder = out_dir + opt
        with open(in_dir+f'/{type}_{opt}_face.txt') as f:
            lines = f.readlines()
            paths = [line. split(" ")[0] for line in lines]

            for imfile in paths:

                image = imread(in_folder+f'/{imfile}')

                if os.path.exists(out_dir+f'/{imfile}'):
                    continue
                image_shape = [image.shape[0], image.shape[1]]
                imname = imfile.replace("\\",'_')
                if type == 'imposter':
                    # try:
                        # _, cropped_image = prn.process(image, None, None, image_shape)
                        cv2.imwrite(out_folder+f'/color/fake_{imname}',image)
                        cv2.imwrite(out_folder+f'/depth/fake_{imname}',np.zeros([256,256,3]))
                    # except:
                    #     os.remove(in_folder+f'/{imfile}')
                    #     print('remove '+in_folder+f'/{imfile}')
                    
                else:
                    # try:
                        pos, cropped_image = prn.process(image, np.array([0,image_shape[0],0,image_shape[1]]), None, image_shape)

                        kpt = prn.get_landmarks(pos)

                    # 3D vertices
                        vertices = prn.get_vertices(pos)

                        depth_scene_map = DepthImage.generate_depth_image(vertices, kpt, image.shape, isMedFilter=True)

                        cv2.imwrite(out_folder+f'/color/real_{imname}',image)
                        cv2.imwrite(out_folder+f'/depth/real_{imname}',depth_scene_map)
                        print(out_folder+f'/color/real_{imname}')


                    # except:
                    #     os.remove(in_folder+f'/{imfile}')
                    #     print('remove '+in_folder+f'/{imfile}')
    


