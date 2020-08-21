# -*- coding: utf-8 -*-

import os 

import numpy as np
import cv2

from keras import models

from keras.models import Input
from keras.models import Model

from keras.layers import Concatenate
from keras.layers import LeakyReLU
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import ReLU
from keras.layers import Conv2DTranspose
from keras.layers import Add
from keras.layers import Dropout

from keras.optimizers import Adam

from keras.initializers import RandomNormal

from keras import activations

from keras.applications import VGG19

import tensorflow as tf
from keras.engine.topology import Layer
from keras.engine import InputSpec
import keras.backend as K

from PIL import Image



class GAN_model():
    
    def __init__(self):        
        self.train_batch_size = 1
        self.test_batch_size = 1
        self.image_row = 256
        self.image_col = 256
        self.channel = 1
        self.image_shape = (self.image_row, self.image_col, self.channel)
        self.patch_row = self.image_row // 16
        self.patch_col = self.image_col // 16
        self.patch_shape = (self.train_batch_size, self.patch_row, self.patch_col, 1)
        
        self.train_file_name = 'train_image_pairs.npz'
        if not os.path.exists(self.train_file_name):
            self.imgs_to_train_npy()
        self.test_file_name = 'test_image_pairs.npz'
        if not os.path.exists(self.test_file_name):
            self.imgs_to_test_npy()
        
        self.iters_per_check = 2000
        
        self.my_vgg_models = self.define_my_vgg()
        self.D = self.build_d()
        self.D_light = self.build_d_light()
        self.G = self.build_g()
        self.GAN = self.build_gan()
        
        self.iterations = 300000
        self.real_patch_out = np.ones(self.patch_shape, dtype=np.float32)
        self.fake_patch_out = np.zeros(self.patch_shape, dtype=np.float32)
       
        self.train_data_generator = self.train_data_loader()
        self.test_data_generator = self.test_data_loader()
    def build_d(self):
        features = []
        init = RandomNormal(stddev=0.02)
        hdr = Input(shape=(None, None, self.channel))    
        ldr = Input(shape=(None, None, self.channel))
        x = Concatenate()([hdr, ldr])
        
        x = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(x) # kernel_initializer=init?
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        features.append(x)
        
        x = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        features.append(x)
        
        x = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        features.append(x)
        
        x = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        features.append(x)
        
        x = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)
        features.append(x)
        
        x = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(x)
        patch_out = Activation('sigmoid')(x)
        
        model = Model([hdr, ldr], [patch_out] + features)
        model.summary()
        return model
    
    def build_d_light(self):
                
        hdr = Input(shape=(None, None, self.channel))    
        ldr = Input(shape=(None, None, self.channel))
        
        x = self.D([hdr, ldr])[0]
        
        model = Model([hdr, ldr], x)
        opt = Adam(lr=0.0002, beta_1=0.5, beta_2 = 0.999)
        model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
        return model
    
    
    # define an encoder block
    def define_encoder_block(self, layer_in, n_filters, batchnorm=True):
     	# weight initialization
     	init = RandomNormal(stddev=0.02)
     	# add downsampling layer
     	g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
     	# conditionally add batch normalization
     	if batchnorm:
              g = BatchNormalization()(g, training=True)
     	# leaky relu activation
     	g = LeakyReLU(alpha=0.2)(g)
     	return g
    
    # define a decoder block
    def decoder_block(self, layer_in, skip_in, n_filters, dropout=True):
     	# weight initialization
     	init = RandomNormal(stddev=0.02)
     	# add upsampling layer
     	g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
     	# add batch normalization
     	g = BatchNormalization()(g, training=True)
     	# conditionally add dropout
     	if dropout:
     	    g = Dropout(0.5)(g, training=True)
     	# merge with skip connection
     	g = Concatenate()([g, skip_in])
     	# relu activation
     	g = Activation('relu')(g)
     	return g
    
    # define the standalone generator model
    def build_g(self):
     	# weight initialization
     	init = RandomNormal(stddev=0.02)
     	# image input
     	# in_image = Input(shape=self.image_shape)
     	in_image = Input(shape=(None, None, self.channel))
         
     	# encoder model
     	e1 = self.define_encoder_block(in_image, 64, batchnorm=False)
     	e2 = self.define_encoder_block(e1, 128)
     	e3 = self.define_encoder_block(e2, 256)
     	e4 = self.define_encoder_block(e3, 512)
     	e5 = self.define_encoder_block(e4, 512)
     	e6 = self.define_encoder_block(e5, 512)
     	e7 = self.define_encoder_block(e6, 512)
     	# bottleneck, no batch norm and relu
     	b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
     	b = Activation('relu')(b)
     	# decoder model
     	d1 = self.decoder_block(b, e7, 512)
     	d2 = self.decoder_block(d1, e6, 512)
     	d3 = self.decoder_block(d2, e5, 512)
     	d4 = self.decoder_block(d3, e4, 512, dropout=False)
     	d5 = self.decoder_block(d4, e3, 256, dropout=False)
     	d6 = self.decoder_block(d5, e2, 128, dropout=False)
     	d7 = self.decoder_block(d6, e1, 64, dropout=False)
     	# output
     	g = Conv2DTranspose(self.channel , (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
     	out_image = Activation('sigmoid')(g)
     	# define model
     	model = Model(in_image, out_image)
     	model.summary()
     	return model
    
    def build_gan(self):
        
        self.D.trainable = False
        
        hdr = Input(shape=(None, None, self.channel))
        ldr = Input(shape=(None, None, self.channel))
        
        def feature_matching_loss(real_ldr, fake_ldr, hdr=hdr, D=self.D):
            loss_feature_match = 0
            fake_features = D([hdr, fake_ldr])[1:]
            real_features = D([hdr, real_ldr])[1:]
            
            for i in range(len(real_features)):
                loss_feature_match += K.mean(K.abs(fake_features[i] - real_features[i]))
            return loss_feature_match
        
        def VGG19_loss(y_true, y_pred, vgg_models=self.my_vgg_models, channels = self.channel):
            loss_vgg19 = 0
            if channels == 1:
                y_true = Concatenate()([y_true, y_true, y_true])
                y_pred = Concatenate()([y_pred, y_pred, y_pred])
                    
            for model in vgg_models:
                model.trainable = False
                loss_vgg19 += K.mean(K.abs(model(y_true) - model(y_pred)))
            return loss_vgg19
        
        
        generated_ldr = self.G(hdr)        
        patch_out = self.D_light([hdr, generated_ldr])        
        
        model = Model(inputs=[hdr, ldr], outputs=[patch_out, generated_ldr, generated_ldr])
        # model = Model(inputs=[hdr, ldr], outputs=[patch_out, generated_ldr])
        model.summary()
        opt = Adam(lr=0.0002, beta_1=0.5, beta_2 = 0.999)
        # model.compile(loss=['binary_crossentropy', feature_matching_loss, VGG19_loss], optimizer=opt, loss_weights=[1, 10, 10])
        model.compile(loss=['binary_crossentropy', feature_matching_loss, VGG19_loss], optimizer=opt, loss_weights=[1, 10, 10])
        # , metrics = {"model_18":'binary_crossentropy', "model_19":[feature_matching_loss, VGG19_loss]}, 
        return model
    
    def define_my_vgg(self):
        vgg = VGG19(include_top=False, input_shape=(self.image_row, self.image_row, 3))
        vgg.trainable = False
        models = []
        for i in [1,2,4,5,7,8,9,10,12,13,14,15,17,18,19,20]:
            model = Model(inputs=vgg.input, outputs=vgg.layers[i].output)
            model.trainable = False
            models.append(model)
        return models
    
    def train(self):
        for iter_idx in range(self.iterations):
            print(iter_idx)
            hdr, ldr = next(self.train_data_generator)
            generated_ldr = self.G.predict(hdr)
            self.D_light.trainable = True
            self.G.trainable = False
            self.D_light.train_on_batch([hdr, ldr], self.real_patch_out)
            self.D_light.train_on_batch([hdr, generated_ldr], self.fake_patch_out)       
            self.D_light.trainable = False
            self.G.trainable = True
            self.GAN.train_on_batch([hdr, ldr], [self.real_patch_out, ldr, ldr])
            if (iter_idx + 1) % self.iters_per_check == 0:
                print(self.GAN.evaluate([hdr, ldr], [self.real_patch_out, ldr, ldr]))
                self.show_performance(iter_idx+1)
                
    def show_performance(self, iter_idx):
        hdr, ldr = next(self.test_data_generator)
        print(self.GAN.evaluate([hdr, ldr], [self.real_patch_out, ldr, ldr]))
        generated_ldr = self.G.predict(hdr)        
        hdr = np.reshape(hdr, self.image_shape)
        ldr = np.reshape(ldr, self.image_shape)
        generated_ldr = np.reshape(generated_ldr, self.image_shape)
        hdr *= 255
        hdr = hdr.astype(np.uint8)        
        hdr = np.reshape(hdr, (256,256))
        hdr = Image.fromarray(hdr, 'L')
        hdr.save('imgs/' + str(iter_idx) + 'test_hdr.jpg')
        
        
        ldr *= 255
        ldr = ldr.astype(np.uint8)
        ldr = np.reshape(ldr, (256,256))
        ldr = Image.fromarray(ldr, 'L')
        ldr.save('imgs/' + str(iter_idx) + 'test_ldr.jpg')        
        
        generated_ldr *= 255
        generated_ldr = generated_ldr.astype(np.uint8)
        generated_ldr = np.reshape(generated_ldr, (256,256))
        generated_ldr = Image.fromarray(generated_ldr, 'L')
        generated_ldr.save('imgs/' + str(iter_idx) + 'test_generated_ldr.jpg')
        
 
    def train_data_loader(self):
        batch_size = self.train_batch_size
        filename = self.train_file_name
        train_data = np.load(filename, allow_pickle=True)
        hdr, ldr = train_data['arr_0'], train_data['arr_1']
        size = len(hdr)
        ids = list(range(size))    
        batchs = size//batch_size
        while True:
            np.random.shuffle(ids)
            for i in range(batchs):
                ids_this_batch = ids[i*batch_size:(i+1)*batch_size]
                hdr_this_batch = [hdr[idx] for idx in ids_this_batch]
                ldr_this_batch = [ldr[idx] for idx in ids_this_batch]
                hdr_this_batch = np.reshape(hdr_this_batch, (batch_size, self.image_row, self.image_col, self.channel))
                ldr_this_batch = np.reshape(ldr_this_batch, (batch_size, self.image_row, self.image_col, self.channel))
                yield hdr_this_batch, ldr_this_batch
        
    def test_data_loader(self):
        batch_size = self.test_batch_size
        filename = self.test_file_name
        train_data = np.load(filename, allow_pickle=True)
        hdr, ldr = train_data['arr_0'], train_data['arr_1']
        size = len(hdr)
        ids = list(range(size))    
        batchs = size//batch_size
        while True:
            np.random.shuffle(ids)
            for i in range(batchs):
                ids_this_batch = ids[i*batch_size:(i+1)*batch_size]
                # print(ids_this_batch)
                hdr_this_batch = [hdr[idx] for idx in ids_this_batch]
                ldr_this_batch = [ldr[idx] for idx in ids_this_batch]
                hdr_this_batch = np.reshape(hdr_this_batch, (batch_size, self.image_row, self.image_col, self.channel))
                ldr_this_batch = np.reshape(ldr_this_batch, (batch_size, self.image_row, self.image_col, self.channel))
                yield hdr_this_batch, ldr_this_batch

    
    def save_model(self):
        
        def freeze(model):
            for layer in model.layers:
                layer.trainable = False
            
            if isinstance(layer, models.Model):
                freeze(layer)
                
        G = self.G
        
        
        D = self.D
        freeze(G)
        freeze(D)
        
        G.save('G_model')
        print('G saved')
        D.save('D_model')
        print('D saved')
        
        
gan = GAN_model()
gan.train()
gan.save_model()