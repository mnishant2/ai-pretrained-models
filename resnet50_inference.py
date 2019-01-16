import os
import cv2
import numpy as np
import fnmatch
from PIL import Image, ImageChops, ImageEnhance
#from matplotlib import pyplot as plt
import os.path
from collections import namedtuple
import random
import imutils
from classification_models import ResNet18
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


data_path = '/home/archimedes/abs/data/forgery/copy_paste dataset/'
data_gen_path = '/home/archimedes/abs/data/forgery/30k_patches/'
original_filelist = []
duplicate_filelist = []


Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')


import math, json, os, sys

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np



TRAIN_DIR = "/home/jeniffer/abs/blurDetection/train"
VALID_DIR = "/home/jeniffer/abs/blurDetection/test"
SIZE = (224, 224)
BATCH_SIZE = 64

num_train_samples = sum([len(files) for r, d, files in os.walk(TRAIN_DIR)])
num_valid_samples = sum([len(files) for r, d, files in os.walk(VALID_DIR)])

num_train_steps = math.floor(num_train_samples/BATCH_SIZE)
num_valid_steps = math.floor(num_valid_samples/BATCH_SIZE)

gen = keras.preprocessing.image.ImageDataGenerator()
val_gen = keras.preprocessing.image.ImageDataGenerator()

batches = gen.flow_from_directory(TRAIN_DIR, target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)
val_batches = val_gen.flow_from_directory(VALID_DIR, target_size=SIZE, class_mode='categorical', shuffle=True, batch_size=BATCH_SIZE)


model = keras.applications.resnet50.ResNet50()
#model= keras.applications.MobileNetV2(include_top=True, weights='imagenet', input_shape=(224,224,3))
classes = list(iter(batches.class_indices))
#model = ResNet18(input_shape=(224,224,3), weights='imagenet', include_top=True)
# print(base_model.summary())
model.layers.pop()
#model.layers.pop()
# print(model.summary())
for layer in model.layers:
    layer.trainable=True
last = model.layers[-1].output
x = Dense(len(classes), activation="softmax")(last)
finetuned_model = Model(model.input, x)
# finetuned_model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# classes = list(iter(batches.class_indices))
# print (list(iter(batches.class_indices)))
# model.layers.pop()
# for layer in model.layers:
#     layer.trainable=True
# last = model.layers[-1].output
# x = Dense(len(classes), activation="softmax")(last)
# finetuned_model = Model(model.input, x)

finetuned_model.load_weights("/home/jeniffer/abs/saravanan-rnd2/forgery/custom/resnet50_blur_weights_08.h5")
 
def get_predict(image):
    image = cv2.resize(image,(224, 224))

    tmp_im = np.expand_dims(image, axis=0)
    tmp_im = preprocess_input(tmp_im)
    ynew = finetuned_model.predict(tmp_im)
    return ynew




# def convolve_custom(original_image):
#     cons = 256    
#     iH, iW = original_image.shape[:2]
#     total_area = iH * iW
#     blank_image = np.zeros((iH,iW,3), np.uint8)

#     for y in range(0, iH, 40):
#         for x in range (0, iW, 40):

#                 org_roi = original_image[y : y + cons, x: x + cons]
#                 orgH, orgW = org_roi.shape[:2]
#                 if orgH == orgW == 256:
#                     white_image = np.zeros((256,256,3), np.uint8)
#                     org_roi = imutils.resize(org_roi, height = 224)
#                     org_roi = np.array(org_roi)
#                     tmp_im = image.img_to_array(org_roi)z
#                     tmp_im = np.expand_dims(tmp_im, axis=0)
#                     tmp_im = preprocess_input(tmp_im)
#                     ynew = finetuned_model.predict(tmp_im)
#                     if ynew[0][0] > 0.5:
#                         print (ynew[0][0])
#                         print ("1",  end='')
#                         x_offset=x
#                         y_offset=y
#                         val = int((ynew[0][0] - 0.5)*100*2.25)
#                         print (val)
#                         white_image[:,:] = (val,val,val)
#                         blank_image[y_offset:y_offset+white_image.shape[0], x_offset:x_offset+white_image.shape[1]] = white_image

#                     else:
#                         print ("0",  end='')

#         print ("\n") 

#     return blank_image               
                    


def calc(filename):
        
        original_colour = cv2.imread(filename)
        # original_colour = cv2.resize(original_colour, (1600, 1200))
        # img = convolve_custom(original_colour)
        # cv2.imwrite(filename+"_out.jpg", img)
        return get_predict(original_colour)



if __name__ == "__main__":
    data_path = "/home/jeniffer/Downloads/blurid"
    count=0
    for path, subdirs, files in os.walk(data_path):
        for name in files:
            print (os.path.join(path, name))
            file = os.path.join(path, name)
            # filename = "/home/archimedes/abs/data/forgery/test/Pan_imGE_FORGE/Forge_image/individualPan_1533829932194.jpg"
            out=calc(file)
            print(out)
            if out[0][0]>=0.5:
                count+=1
    print(count)

