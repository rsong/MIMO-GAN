# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 15:46:03 2020

@author: Ran Song
"""

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tensorflow import keras
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import numpy as np


train_data_dir = './salicon/images'
label_data_dir = './salicon/maps'
object_data_dir = './modelnet40v4'

nb_train_samples = 10000
batch_size = 8
size_h = 224
size_w = 224
alpha1 = 0.2
alpha2 = 1
alpha3 = 0.01
alpha4 = 0.01
alpha5 = 0.01


def discriminator():
    
     indis = keras.Input(shape=(size_h, size_w, 3))
     
     x = keras.layers.Conv2D(32, 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(indis)
     x = tfa.layers.InstanceNormalization()(x)
     x = keras.layers.LeakyReLU(alpha=0.2)(x)
     x = keras.layers.Dropout(0.3)(x)
     
     x = keras.layers.Conv2D(64, 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
     x = tfa.layers.InstanceNormalization()(x)
     x = keras.layers.LeakyReLU(alpha=0.2)(x)
     x = keras.layers.Dropout(0.3)(x)
     
     x = keras.layers.Conv2D(128, 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
     x = tfa.layers.InstanceNormalization()(x)
     x = keras.layers.LeakyReLU(alpha=0.2)(x)
     x = keras.layers.Dropout(0.3)(x)
     
     x = keras.layers.Conv2D(256, 3, strides=2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)
     x = tfa.layers.InstanceNormalization()(x)
     x = keras.layers.LeakyReLU(alpha=0.2)(x)
     x = keras.layers.Dropout(0.3)(x)
          
     x = keras.layers.Flatten()(x)
     outdis = keras.layers.Dense(1, activation='sigmoid')(x)
     dis = keras.Model(inputs = indis, outputs = outdis)
     return dis   
 

def little_unet():
 
    inp1 = keras.Input(shape=(size_h, size_w, 3),name='object_input')
    inp2 = keras.Input(shape=(size_h, size_w, 3),name='saliency_input')
    vgg16 = keras.applications.VGG16(include_top=False, weights='imagenet')
    model_vgg = keras.Model(inputs = vgg16.input, outputs = vgg16.layers[-2].output)

    y = model_vgg(inp1)
    
    x = model_vgg(inp2)
    expool1=keras.layers.MaxPooling2D(pool_size=(2, 2),name='saliency_pool1')

    x1 = expool1(x)
    y1 = expool1(y)

    conv4_layer = keras.Model(inputs = model_vgg.input, outputs = model_vgg.layers[-5].output)
    conv4 = conv4_layer(inp2)
    conv4y = conv4_layer(inp1)
    
    conv3_layer = keras.Model(inputs = model_vgg.input, outputs = model_vgg.layers[-9].output)
    conv3 = conv3_layer(inp2)
    conv3y = conv3_layer(inp1)
    
    conv2_layer = keras.Model(inputs = model_vgg.input, outputs = model_vgg.layers[-13].output)
    conv2 = conv2_layer(inp2)
    conv2y = conv2_layer(inp1)    
   
    conv1_layer = keras.Model(inputs = model_vgg.input, outputs = model_vgg.layers[-16].output)
    conv1 = conv1_layer(inp2)
    conv1y = conv1_layer(inp1)
   
    yyy = keras.layers.Flatten()(y1)
    yyy = keras.layers.Dense(256, activation='relu')(yyy)
    yyy = keras.layers.Dropout(0.3)(yyy)
    yyy = keras.layers.Dense(1024, activation='relu')(yyy)
    yyy = keras.layers.Dropout(0.3)(yyy)
    yyy = keras.layers.Dense(40, activation='softmax',name='category_output')(yyy)

    up3layer = keras.layers.Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    up3 = up3layer(keras.layers.UpSampling2D(size = (2,2))(x1))
    up3y = up3layer(keras.layers.UpSampling2D(size = (2,2))(y1))
    
    up4layer = keras.layers.Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    up4 = up4layer(keras.layers.UpSampling2D(size = (2,2))(up3))
    up4y = up4layer(keras.layers.UpSampling2D(size = (2,2))(up3y))
    
    lconv4layer = keras.layers.SeparableConv2D(128,2,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    lconv4 = lconv4layer(conv4)
    lconv4y = lconv4layer(conv4y)
    
    merge4 = keras.layers.concatenate([lconv4,up4], axis = 3)
    merge4y = keras.layers.concatenate([lconv4y,up4y], axis = 3)
   
    gconv4layer = keras.layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    gconv4 = gconv4layer(merge4)
    gconv4y = gconv4layer(merge4y)
   
    up5layer = keras.layers.Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    up5 = up5layer(keras.layers.UpSampling2D(size = (2,2))(gconv4))
    up5y = up5layer(keras.layers.UpSampling2D(size = (2,2))(gconv4y))
    
    lconv5layer = keras.layers.SeparableConv2D(64,2,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    lconv5 = lconv5layer(conv3)
    lconv5y = lconv5layer(conv3y)
    
    merge5 = keras.layers.concatenate([lconv5,up5], axis = 3)
    merge5y = keras.layers.concatenate([lconv5y,up5y], axis = 3)
    
    gconv5layer = keras.layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    gconv5 = gconv5layer(merge5)
    gconv5y = gconv5layer(merge5y)
    
    up6layer = keras.layers.Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    up6 = up6layer(keras.layers.UpSampling2D(size = (2,2))(gconv5))
    up6y = up6layer(keras.layers.UpSampling2D(size = (2,2))(gconv5y))
    
    lconv6layer = keras.layers.SeparableConv2D(32,2,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    lconv6 = lconv6layer(conv2)
    lconv6y = lconv6layer(conv2y)
    
    merge6 = keras.layers.concatenate([lconv6,up6], axis = 3)
    merge6y = keras.layers.concatenate([lconv6y,up6y], axis = 3)
   
    gconv6layer = keras.layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    gconv6 =  gconv6layer(merge6)
    gconv6y = gconv6layer(merge6y)
    
    up7layer = keras.layers.Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    up7 = up7layer(keras.layers.UpSampling2D(size = (2,2))(gconv6))
    up7y = up7layer(keras.layers.UpSampling2D(size = (2,2))(gconv6y))
    
    lconv7layer = keras.layers.SeparableConv2D(16,2,activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    lconv7 = lconv7layer(conv1)
    lconv7y = lconv7layer(conv1y)
   
    merge7 = keras.layers.concatenate([lconv7,up7], axis = 3)
    merge7y = keras.layers.concatenate([lconv7y,up7y], axis = 3)
    
    gconv7layer = keras.layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    gconv7 = gconv7layer(merge7)
    gconv7y = gconv7layer(merge7y)
    
    conv8layer = keras.layers.Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
    conv8 = conv8layer(gconv7)
    conv8y = conv8layer(gconv7y)
    
    conv9layer = keras.layers.Conv2D(1, 3, activation = 'linear', padding = 'same', kernel_initializer = 'Zeros')
    conv9 = conv9layer(conv8)
    conv9y = conv9layer(conv8y)
    
    conv10layer = keras.layers.Conv2D(3, 1, activation = None, padding = 'same', kernel_initializer = 'Ones',use_bias = False, name='saliency_output')
    conv10layer.trainable = False
    conv10 = conv10layer(conv9)
    conv10y = conv10layer(conv9y)
    
    dismodel = discriminator()
    dismodel._name = 'd_output'
    conv10xx = dismodel(conv10)
    conv10yy = dismodel(conv10y)

    dismodelf = keras.models.clone_model(dismodel)
    dismodelf.set_weights(keras.backend.batch_get_value(dismodel.weights))
    dismodelf._name = 'g_output'
    dismodelf.trainable = False
  
    conv10yyg = dismodelf(conv10y)
    
    model = keras.Model(inputs=[inp1,inp2], outputs=[yyy,conv10,conv10xx,conv10yy,conv10yyg]) 
    
    model.summary()
    return model


def scheduler(epoch):
    lr = max(0.001*0.1 ** (epoch // 200),0.0000002)
    return lr


image_datagen = keras.preprocessing.image.ImageDataGenerator(preprocessing_function = keras.applications.vgg16.preprocess_input)
image_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
 
model = little_unet()


def generate_generator(gen, dir1, dir2, dir3, b_size, img_height,img_width):
    
    genX1 = gen.flow_from_directory(
    dir1,
    target_size=(img_height, img_width),
    batch_size=b_size,
    shuffle=True,
    class_mode='categorical',
    seed=1)
    
    genX2 = gen.flow_from_directory(
    dir2,
    target_size=(img_height, img_width),
    batch_size=b_size,
    shuffle=True,
    class_mode=None,
    seed=5)
    
    genX3 = gen.flow_from_directory(
    dir3,
    target_size=(img_height, img_width),
    batch_size=b_size,
    shuffle=True,
    class_mode=None,
    seed=5)

    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            X3i = genX3.next()
            if len(X2i) == b_size:
                yield [X1i[0], X2i], [X1i[1], X3i, np.ones((b_size,1)), np.zeros((b_size,1)), np.ones((b_size,1))]
                
           
learning_rate = 0.001
momentum = 0.9
sgd = keras.optimizers.SGD(lr=learning_rate, momentum = momentum)

change_lr = keras.callbacks.LearningRateScheduler(scheduler)

losses = {"category_output": "categorical_crossentropy",
          "saliency_output": "mean_squared_error",
          "d_output": "binary_crossentropy",
          "d_output_1": "binary_crossentropy",
          "g_output": "binary_crossentropy"}
lossWeights = {"category_output": alpha1, "saliency_output": alpha2, "d_output":alpha3, "d_output_1":alpha4,"g_output":alpha5}
metrics = {"saliency_output": "mae",
           "d_output": "accuracy",
           "d_output_1": "accuracy",
           "g_output": "accuracy"}

# model = keras.utils.multi_gpu_model(smodel,gpus=2)    
model.compile(loss=losses, loss_weights=lossWeights, metrics=metrics, optimizer=sgd)
generator = generate_generator(image_datagen,object_data_dir,train_data_dir,label_data_dir,batch_size,size_h,size_w)
history = model.fit_generator(
    generator,
    steps_per_epoch=int(np.floor(nb_train_samples/batch_size)),
    epochs=100,
    verbose=2,
    callbacks=[change_lr])

print(history.history.keys())
model.save('mimogan.h5', include_optimizer=False)
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

plt.plot(history.history['saliency_output_mae'])
plt.title('model mae')
plt.ylabel('mse')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()