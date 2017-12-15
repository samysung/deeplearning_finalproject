
from __future__ import print_function
import os
#import sklearn as sk

from sklearn.feature_extraction import image
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import imread
from skimage import data
from skimage import filters

from skimage.util.shape import view_as_windows
from skimage.util.shape import view_as_blocks
from scipy import stats

import math
import numpy as np
from keras.callbacks import TensorBoard,ModelCheckpoint
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization,Input
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model

import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from skimage import data
from skimage import filters
from skimage import exposure

train_data_dir_pos = "./data/train/cancer/"
train_data_dir_neg = "./data/train/non_cancer/"

val_data_dir = "./data/val/"
test_data_dir = "./data/test/"


def splitImage(img,patch_size=(80,80)):

    patch_height = patch_size[0]
    patch_width = patch_size[1]

    row = int(math.floor(img.shape[0] / patch_height))
    col = int(math.floor(img.shape[1] / patch_width))
    patch_grid = np.zeros((row,col,patch_height,patch_width,3))
    for h in range(0, row):
        for w in range(0, col):
            current_grid = img[h * patch_height:(h + 1) * patch_height, w * patch_width:(w + 1) * patch_width]
            patch_grid[h,w,:,:,:] = current_grid

    patch_grid = np.array(patch_grid)
    print(patch_grid.shape)
    return patch_grid,(row,col)


def VGG16():
    # path to the model weights files.
    weights_path = '../keras/examples/vgg16_weights.h5'
    #top_model_weights_path = 'bottleneck_fc_model.h5'

    # dimensions of our images.
    img_width, img_height = 80,80

    nb_train_samples = 2569
    nb_validation_samples = 550
    epochs = 200
    batch_size = 32

    # build the VGG19 network
    # input_tensor = Input(shape=(150, 150, 3))
    input_tensor = Input(shape=(80,80, 3))
    base_model = applications.VGG16(weights='imagenet', include_top= False,input_tensor = input_tensor)
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(320))
    top_model.add(BatchNormalization())
    top_model.add(Activation('relu'))
    top_model.add(Dropout(0.3))

    top_model.add(Dense(320))
    top_model.add(BatchNormalization())
    top_model.add(Activation('relu'))
    top_model.add(Dropout(0.2))
    top_model.add(Dense(1, activation='sigmoid'))
    #top_model.load_weights('bottleneck_fc_model.h5')

    model = Model(input=base_model.input, output=top_model(base_model.output))
    print('Model loaded.')

    # block the first 6 layers (up to the last conv block)
    for layer in model.layers[:3]:
        layer.trainable = False
    for layer in model.layers[3:]:
        layer.trainable = True

    # compile the model with a SGD/momentum optimizer
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-6, momentum= 0.7),
                  # optimizer="adam",
                  metrics=['accuracy'])
    return model


def threshChoose(H,percent):
    H_flatten = H.flatten()
    H_flatten = np.sort(H_flatten)
    #thresh_val = filters.threshold_otsu(H)
    bg = stats.mode(H_flatten)
    H_flatten = np.delete(H_flatten,np.where(H_flatten <= bg[0][0]))
    thresh = H_flatten[-math.floor(len(H_flatten)*percent)]
    return thresh,bg[0][0]

def loadData():
    imgs = []
    labels = []
    step_vis = 0
    for img_name in os.listdir(train_data_dir_pos):
        if (img_name.endswith(".png")):
            print("Loading pos img(total 82):", step_vis)
            step_vis = step_vis + 1
            img = imread(train_data_dir_pos + img_name)
            # samplewise normalization
            # img = (img - np.mean(img)) / np.std(img)
            imgs.append(img)
            labels.append(2)
    step_vis = 0
    for img_name in os.listdir(train_data_dir_neg):
        if (img_name.endswith(".png")):
            print("Loading neg img(total 208):", step_vis)
            step_vis = step_vis + 1
            img = imread(train_data_dir_neg + img_name)
            # samplewise normalization
            # img = (img - np.mean(img)) / np.std(img)
            imgs.append(img)
            labels.append(1)
    imgs = np.array(imgs)
    labels = np.array(labels)
    #shuffle the images set
    imgs, labels = shuffle(imgs, labels, random_state=0)
    return imgs,labels
k = 10
iter_num = 2
patch_size = (80,80)
H = []


# Define the training data generator
model = VGG16()
imgs,labels = loadData()
#train_datagen = ImageDataGenerator(
#    rescale=1. / 255,
#    featurewise_center=True,
#    featurewise_std_normalization=True,
#    shear_range=0.2,
#    zoom_range=0.2,
#    width_shift_range=0.2,
#    height_shift_range=0.2,
#    #zca_whitening=1,
#    horizontal_flip=True)
#train_generator = train_datagen.flow(imgs,batch_size=512)

# EM iteration until convergence
for iter in range(iter_num):
    ################@###############################
    # EM based CNN (2) M step :
    # 1 . Read in the image , and divide it to patches
    ################################################
    # for each patch initially assume H_i = 1
    img_num = 0
    m_step = 0
    for img in imgs:
        print("M-Step(330):",m_step)
        m_step = m_step+1
         # samplewise normalization
        img_norm = (img-np.mean(img))/np.std(img)
        patches,patch_num = splitImage(img, patch_size)
        if (iter == 0):
            H_i = np.ones((patch_num[0], patch_num[1]))
        else:
            H_i = H[img_num]
        patches_flatten = patches.reshape(patches.shape[0]*patches.shape[1],80,80,3)
        H_i_flatten = H_i.reshape(H_i.shape[0]*H_i.shape[1])
        model.fit(patches_flatten, H_i_flatten, batch_size=512, epochs=2)
        img_num = img_num + 1
    e_step = 0
    img_num = 0
    for img in imgs:
        print("E-Step(330):",e_step)
        e_step = e_step+1
        # samplewise normalization
        img_norm = (img-np.mean(img))/np.std(img)
        patches, patch_num = splitImage(img, patch_size)
        H_i = np.ones((patch_num[0], patch_num[1]))
        for i in range(patch_num[0]):
            for j in range(patch_num[1]):
                H_i[i,j] = model.predict(patches[i,j].reshape(1,80,80,3))

        #reconstruct the H matrix to 2D(for Gaussian smoothing)
################@###############################
# EM based CNN (2) E step :
################################################
        H_i = gaussian_filter(H_i,sigma=0.5)
        if(iter == iter_num-1):
            threshold,bg = threshChoose(H_i, 0.6)
            H_i1 = (H_i >= threshold) * 2
            H_i2 = (H_i != bg) & (H_i<threshold)
            H_i = H_i1 + H_i2
        else:
            threshold,bg = threshChoose(H_i,0.6)
            print("threshold:",threshold)
            H_i = H_i >= threshold
        if(iter == 0):
            H.append(H_i)
        else:
            H[img_num] = H_i
        img_num = img_num + 1
    H = np.array(H)

    # save the H and corresponding labels(each iteraton)
    np.save("H_patches.npy",H)
    np.save("H_labels.npy",labels)
    model.save_weights('model_weights.h5')
    #del model

