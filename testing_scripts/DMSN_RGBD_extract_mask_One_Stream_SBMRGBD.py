
"""
Created on Mon Jun 27 2018

@author: longang
"""

# coding: utf-8
#get_ipython().magic(u'load_ext autotime')
import numpy as np
import os, glob, sys
from keras.preprocessing import image as kImage
#from skimage.transform import pyramid_gaussian
from keras.models import load_model
from scipy.misc import imsave, imresize
import keras_contrib
import gc


# Optimize to avoid memory exploding. 
# For each video sequence, we pick only 1000frames where res > 400
# You may modify according to your memory/cpu spec.
def checkFrame(X_list):
    img = kImage.load_img(X_list[0], color_mode = 'rgb')
    img = kImage.img_to_array(img).shape # (480,720,3)
    num_frames = len(X_list) # 7000
    max_frames = 1000 # max frames to slice
    if(img[1]>=400 and len(X_list)>max_frames):
        print ('\t- Total Frames:' + str(num_frames))
        num_chunks = num_frames/max_frames
        num_chunks = int(np.ceil(num_chunks)) # 2.5 => 3 chunks
        start = 0
        end = max_frames
        m = [0]* num_chunks
        for i in range(num_chunks): # 5
            m[i] = range(start, end) # m[0,1500], m[1500, 3000], m[3000, 4500]
            start = end # 1500, 3000, 4500 
            if (num_frames - start > max_frames): # 1500, 500, 0
                end = start + max_frames # 3000
            else:
                end = start + (num_frames- start) # 2000 + 500, 2500+0
        print ('\t- Slice to:' + str(m))
        del img, X_list
        return [True, m]
    del img, X_list
    return [False, None]
    
# Load some frames (e.g. 1000) for segmentation
def generateData(scene_input_path, X_list, X_list_depth, scene, scene_input_path_depth):
    # read images
    X = []
    print ('\n\t- Loading frames:')
    for i in range(0, len(X_list)):
        img1 = kImage.load_img(X_list[i], color_mode = 'rgb')
        x1 = kImage.img_to_array(img1)
        img2 = kImage.load_img(X_list_depth[i], grayscale = True)
        x2 = kImage.img_to_array(img2)

        x = np.concatenate((x1, x2), axis=2)

        X.append(x)
        sys.stdout.write('\b' * len(str(i)))
        sys.stdout.write('\r')
        sys.stdout.write(str(i+1))
    
    del img1, img2, x, X_list
    X = np.asarray(X)
    print ('\nShape' + str(X.shape))

    return X #return for DMSN

def getFiles(scene_input_path):
    inlist = glob.glob(os.path.join(scene_input_path,'*.png'))
    return np.asarray(inlist)



# Extract all mask
dataset = {
            'S_1',
            'S_2',
            'S_3',
            'S_4',
            'S_5',
            'S_6',
}

# number of exp frame (25, 50, 200)
num_frames = 25

# 1. Raw RGB frame to extract foreground masks, downloaded from changedetection.net
raw_dataset_dir = os.path.join('datasets', 'SBMRGBD_dataset')

# 2. model dir
main_mdl_dir = os.path.join('DMSN_RGBD_One_Stream', 'SBMRGBD', 'models')

# 3. path to store results
results_dir = os.path.join('DMSN_RGBD_One_Stream', 'SBMRGBD', 'results', 'RGB')

for scene in dataset:
        # print ('\n->>> ' + category + ' / ' + scene)
        print ('\n->>> ' + scene)
        mdl_path = os.path.join('..', main_mdl_dir, 'DMSN_RGBD_One_Stream_mdl_SBMRGBD_RGB_' + scene + '.h5')
        
        mask_dir = os.path.join('..',results_dir, scene)
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)
        
        # path of dataset downloaded from CDNet
        scene_input_path = os.path.join('..', raw_dataset_dir, scene, 'test', 'RGB')
        scene_input_path_depth = os.path.join('..', raw_dataset_dir, scene, 'test', 'DEPTH')
        # path of ROI to exclude non-ROI
        # make sure that each scene contains ROI.bmp and have the same dimension as raw RGB frames
        ROI_file = os.path.join('..', raw_dataset_dir, 'ROI.bmp')
        #print (ROI_file)
        
        # refer to http://jacarini.dinf.usherbrooke.ca/datasetOverview/
        img = kImage.load_img(ROI_file, grayscale=True)
        img = kImage.img_to_array(img)
        img = img.reshape(-1) # to 1D
        idx = np.where(img == 0.)[0] # get the non-ROI, black area
        del img
        
        # load path of files
        X_list = getFiles(scene_input_path)
        X_list_depth = getFiles(scene_input_path_depth)
        if (X_list is None):
            raise ValueError('X_list is None')

        # slice frames
        results = checkFrame(X_list)
        
        # load model to segment
        model = load_model(mdl_path, compile=False, custom_objects={'InstanceNormalization':keras_contrib.layers.InstanceNormalization})

        # if large numbers of frames, slice it
        if(results[0]): 
            for rangeee in results[1]: # for each slice
                slice_X_list =  X_list[rangeee]
                slice_X_list_depth =  X_list_depth[rangeee]

                # load frames for each slice
                data = generateData(scene_input_path, slice_X_list, slice_X_list_depth, scene, scene_input_path_depth)
                
                # For DMSN
                Y_proba = model.predict(data, batch_size=1, verbose=1)
                del data

                # filter out
                shape = Y_proba.shape
                Y_proba = Y_proba.reshape([shape[0],-1])
                if (len(idx)>0): # if have non-ROI
                    for i in range(len(Y_proba)): # for each frames
                        Y_proba[i][idx] = 0. # set non-ROI pixel to black
                        
                Y_proba = Y_proba.reshape([shape[0], shape[1], shape[2]])

                prev = 0
                print ('\n- Saving frames:')
                for i in range(shape[0]):
                    fname = os.path.basename(slice_X_list[i]).replace('jpg','png')
                    x = Y_proba[i]
                    imsave(os.path.join(mask_dir, fname), x)
                    sys.stdout.write('\b' * prev)
                    sys.stdout.write('\r')
                    s = str(i+1)
                    sys.stdout.write(s)
                    prev = len(s)
                    
                del Y_proba, slice_X_list

        else: # otherwise, no need to slice
            data = generateData(scene_input_path, X_list, X_list_depth, scene, scene_input_path_depth)
                        
            # For DMSN
            Y_proba = model.predict(data, batch_size=1, verbose=1)
            
            del data
            shape = Y_proba.shape
            Y_proba = Y_proba.reshape([shape[0],-1])
            if (len(idx)>0): # if have non-ROI
                    for i in range(len(Y_proba)): # for each frames
                        Y_proba[i][idx] = 0. # set non-ROI pixel to black

            Y_proba = Y_proba.reshape([shape[0], shape[1], shape[2]])
            
            prev = 0
            print ('\n- Saving frames:')
            for i in range(shape[0]):
                fname = os.path.basename(X_list[i]).replace('jpg','png')
                x = Y_proba[i] 
                imsave(os.path.join(mask_dir, fname), x)
                sys.stdout.write('\b' * prev)
                sys.stdout.write('\r')
                s = str(i+1)
                sys.stdout.write(s)
                prev = len(s)
            del Y_proba
        del model, X_list, results

gc.collect()