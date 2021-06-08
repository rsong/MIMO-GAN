# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 15:46:03 2020

@author: Ran Song
"""
#import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow import keras
import tensorflow_addons as tfa
import numpy as np
import scipy.io
import time

start_time = time.time()

test_data_dir = './3DVA/3DModels-Simplif-224-up'

nb_test_samples = 96
test_batch_size = 8
size_h = 224
size_w = 224

modelname = 'mimogan'
model = keras.models.load_model(modelname + '.h5')

model_sa = keras.Model(inputs = model.input[1], outputs = model.output[1])
model_sa.summary()
image_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_gen = image_datagen.flow_from_directory(test_data_dir, target_size=(size_h,size_w),
                                             batch_size=test_batch_size, shuffle=False,class_mode=None)

results = model_sa.predict_generator(test_gen,steps=int(np.floor(nb_test_samples/test_batch_size)), verbose=1)
print("--- %s seconds ---" % (time.time() - start_time))
scipy.io.savemat(modelname + '.mat', mdict={'results_sa': results})