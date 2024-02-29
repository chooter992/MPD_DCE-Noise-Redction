# 工具
from posixpath import split
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Concatenate, Activation, Conv2DTranspose, LeakyReLU, \
    MaxPool2D, BatchNormalization, DepthwiseConv2D, UpSampling2D
from tensorflow.keras import Input, Model


import numpy as np
# 自定義Loss
from utils import Losses
from utils.Activation import HardSwish
from utils.sub_sampler import sub_sampler
from utils.CBAM import CBAM_block


def dwconv(x, filter, kernel_size):
    x = DepthwiseConv2D(kernel_size, (1,1), 'same')(x)
    x = Conv2D(filter, 1, (1,1), 'same')(x)
    return x



def channel_shuffle(input_tensor, groups):
    # Get the input tensor shape
    b, h, w, c = tf.shape(input_tensor)
    # Calculate the number of channels per group
    channels_per_group = c // groups
    # Reshape the input tensor to [batch_size, height, width, groups, channels_per_group]
    x = tf.reshape(input_tensor, [-1, h, w, groups, channels_per_group])
    # Transpose the tensor to [batch_size, height, width, channels_per_group, groups]
    x = tf.transpose(x, [0, 1, 2, 4, 3])
    # Reshape the tensor back to [batch_size, height, width, channels]
    x = tf.reshape(x, [-1, h, w, c])
    return x

initializers = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
b_initalizer = tf.keras.initializers.Zeros()
def enhancement_net(inputs):
    x1 = Conv2D(32, 3, (1, 1), padding='same', kernel_initializer=initializers, bias_initializer=b_initalizer)(inputs)
    x1 = Activation('relu')(x1)
    x2 = Conv2D(32, 3, (1, 1), padding='same', kernel_initializer=initializers, bias_initializer=b_initalizer)(x1)
    x2 = Activation('relu')(x2)
    x3 = Conv2D(32, 3, (1, 1), padding='same', kernel_initializer=initializers, bias_initializer=b_initalizer)(x2)
    x3 = Activation('relu')(x3)
    x4 = Conv2D(32, 3, (1, 1), padding='same', kernel_initializer=initializers, bias_initializer=b_initalizer)(x3)
    x4 = Activation('relu')(x4)
    
    x5 = Concatenate()([x3,x4])
    x5 = Conv2D(32, 3, (1, 1), padding='same', kernel_initializer=initializers, bias_initializer=b_initalizer)(x5)
    x5 = Activation('relu')(x5)
    
    x6 = Concatenate()([x2,x5])
    x6 = Conv2D(32, 3, (1, 1), padding='same', kernel_initializer=initializers, bias_initializer=b_initalizer)(x6)
    x6 = Activation('relu')(x6)
    
    parameter_map = Concatenate()([x1,x6])
    parameter_map = Conv2D(3,  3, (1, 1), padding='same', kernel_initializer=initializers, bias_initializer=b_initalizer)(parameter_map)
    parameter_map = Activation('tanh')(parameter_map)
    
    return Model(inputs=inputs, outputs=parameter_map)

def MSP(inputs, filters):
    x1_2 = Conv2D(filters//2, 1, (1, 1), padding='same', kernel_initializer=initializers)(inputs)
    x1_2 = Activation('relu')(x1_2)
    
    x1 = Conv2D(filters//4 , 3, (1, 1), padding='same', kernel_initializer=initializers)(inputs)
    x1 = Activation('relu')(x1)
    x1_3 = Conv2D(filters//4, 1, (1, 1), padding='same', kernel_initializer=initializers)(x1)
    x1_3 = Activation('relu')(x1_3)
    x1 = Concatenate()([x1,x1_3])
    
    x1 = Concatenate()([x1,x1_2])
    return x1

def csp_enhancement_net(inputs,groups=4):
    

    #3x3 conv
    # x1 = Conv2D(16, 3, (1, 1), padding='same', kernel_initializer=initializers)(inputs)
    x1 = Conv2D(16, 3, (1, 1), padding='same', kernel_initializer=initializers)(inputs)
    x1 = Activation('relu')(x1)
    #1x1 conv
    x1_2 = Conv2D(16, 1, (1, 1), padding='same', kernel_initializer=initializers)(inputs)
    x1_2 = Activation('relu')(x1_2)
    #x1,x1_2 concat
    x1 = Concatenate()([x1,x1_2])
    #channel shuffle
    x1 = channel_shuffle(x1, groups=groups)
    


    #3x3 conv
    x2 = Conv2D(16, 3, (1, 1), padding='same', kernel_initializer=initializers)(x1)
    x2 = Activation('relu')(x2)
    #1x1 conv
    x2_2 = Conv2D(16, 1, (1, 1), padding='same', kernel_initializer=initializers)(x1)
    x2_2 = Activation('relu')(x2_2)
    #x2,x2_2 concat
    x2 = Concatenate()([x2,x2_2])
    #channel shuffle
    x2 = channel_shuffle(x2, groups=groups)

    #3x3 conv
    x3 = Conv2D(16, 3, (1, 1), padding='same', kernel_initializer=initializers)(x2)
    x3 = Activation('relu')(x3)
    #1x1 conv
    x3_2 = Conv2D(16, 1, (1, 1), padding='same', kernel_initializer=initializers)(x2)
    x3_2 = Activation('relu')(x3_2)
    #x3,x3_2 concat 
    x3 = Concatenate()([x3,x3_2])
    #channel shuffle
    x3 = channel_shuffle(x3, groups=groups)

    #3x3 conv
    x4 = Conv2D(16, 3, (1, 1), padding='same', kernel_initializer=initializers)(x3)
    x4 = Activation('relu')(x4)
    #1x1 conv
    x4_2 = Conv2D(16, 1, (1, 1), padding='same', kernel_initializer=initializers)(x3)
    x4_2 = Activation('relu')(x4_2)
    #x4,x4_2 concat
    x4 = Concatenate()([x4,x4_2])
    #channel shuffle
    x4 = channel_shuffle(x4, groups=groups)

    # x1 = CBAM_block(x1,name='x1_cbam')
    # x2 = CBAM_block(x2,name='x2_cbam')
    # x3 = CBAM_block(x3,name='x3_cbam')
    # x4 = CBAM_block(x4,name='x4_cbam')
    
    x4 = Concatenate()([x3,x4])
    x5 = Conv2D(16, 3, (1, 1), padding='same', kernel_initializer=initializers)(x4)
    x5 = Activation('relu')(x5)
    x5_2 = Conv2D(16, 1, (1, 1),  padding='same', kernel_initializer=initializers)(x4)
    x5_2 = Activation('relu')(x5_2)
    x5 = Concatenate()([x5,x5_2])
    x5 = channel_shuffle(x5, groups=groups)
    
    x5 = Concatenate()([x2,x5])
    x6 = Conv2D(16, 3, (1, 1), padding='same', kernel_initializer=initializers)(x5)
    x6 = Activation('relu')(x6)
    x6_2 = Conv2D(16, 1, (1, 1), padding='same', kernel_initializer=initializers)(x5)
    x6_2 = Activation('relu')(x6_2)
    x6 = Concatenate()([x6,x6_2])
    x6 = channel_shuffle(x6, groups=groups)

    parameter_map = Concatenate()([x1,x6])
    parameter_map = Conv2D(3, 1, (1, 1), padding='same', kernel_initializer=initializers)(parameter_map)
    parameter_map = Activation('tanh')(parameter_map)
    
    return Model(inputs=inputs, outputs=parameter_map)




def dil_enhancement_net(inputs,groups=4):
    

    #stage1 3x3 conv dialated conv
    x1_1 = Conv2D(8, 3, (1, 1), dilation_rate=2, padding='same', kernel_initializer=initializers)(inputs)
    x1_1 = Activation('relu')(x1_1)
    
    #1x1 conv
    x1_2 = Conv2D(8, 1, (1, 1), padding='same', kernel_initializer=initializers)(inputs)
    x1_2 = Activation('relu')(x1_2)

    #3x3 conv dialated conv
    x1_3 = Conv2D(16, 3, (1, 1),dilation_rate=2, padding='same', kernel_initializer=initializers)(inputs)
    x1_3 = Activation('relu')(x1_3)

    #addion version
    # x1_1 = x1_1 + x1_2
    # x1_2 = x1_1 + x1_2
    #x1,x1_2 concat
    #ver.1 3x3 dil 8 channel 、 3x3 dil 8 channel 、 1x1 concat 8 channel
    # x1 = Concatenate()([x1_1,x1_2,x1_3])
    #ver.2 3x3 dil 6 channel concat 3x3 dil 6 channel 、 24 1x1 add
    # x1 = Concatenate()([x1_1, x1_3])
    # ver.3 3x3 dil 、 1x1 concat
    x1 = Concatenate()([x1_1, x1_2])
    #channel shuffle
    x1 = channel_shuffle(x1, groups=groups)

    x1 = Conv2D(16, 1, (1, 1), padding='same', kernel_initializer=initializers)(x1)
    x1 = Activation('relu')(x1)
    x1 = x1 + x1_3

    #stage2 3x3 conv dialated conv
    x2_1 = Conv2D(8, 3, (1, 1),  dilation_rate=2, padding='same', kernel_initializer=initializers)(x1)
    x2_1 = Activation('relu')(x2_1)

    #1x1 conv
    x2_2 = Conv2D(8, 1, (1, 1), padding='same', kernel_initializer=initializers)(x1)
    x2_2 = Activation('relu')(x2_2)

    #3x3 conv dialated conv
    x2_3 = Conv2D(16, 3, (1, 1),dilation_rate=2, padding='same', kernel_initializer=initializers)(x1)
    x2_3 = Activation('relu')(x2_3)
    # x2_1 = x2_1 + x2_2
    # x2_2 = x2_1 + x2_2
    #x2,x2_2 concat
    x2 = Concatenate()([x2_1, x2_2])
    # x2 = Concatenate()([x2_1,x2_2,x2_3])
    #channel shuffle
    x2 = channel_shuffle(x2, groups=groups)

    x2 = Conv2D(16, 1, (1, 1), padding='same', kernel_initializer=initializers)(x2)
    x2 = Activation('relu')(x2)

    x2 = x2 + x2_3

    #stage3 3x3 conv dialated conv
    x3_1 = Conv2D(8, 3, (1, 1),  dilation_rate=2, padding='same', kernel_initializer=initializers)(x2)
    x3_1 = Activation('relu')(x3_1)
    #1x1 conv
    x3_2 = Conv2D(8, 1, (1, 1), padding='same', kernel_initializer=initializers)(x2)
    x3_2 = Activation('relu')(x3_2)

    #3x3 conv dialated conv
    x3_3 = Conv2D(16, 3, (1, 1),dilation_rate=2, padding='same', kernel_initializer=initializers)(x2)
    x3_3 = Activation('relu')(x3_3)

    # x3_1 = x3_1 + x3_2
    # x3_2 = x3_1 + x3_2
    #x3,x3_2 concat 
    x3 = Concatenate()([x3_1, x3_2])
    # x3 = Concatenate()([x3,x3_2, x3_3])

    #channel shuffle
    x3 = channel_shuffle(x3, groups=groups)

    x3 = Conv2D(16, 1, (1, 1), padding='same', kernel_initializer=initializers)(x3)
    x3 = Activation('relu')(x3)
    x3 = x3 + x3_3

    #stage4 3x3 conv dialated conv
    x4_1 = Conv2D(8, 3, (1, 1),  dilation_rate=2, padding='same', kernel_initializer=initializers)(x3)
    x4_1 = Activation('relu')(x4_1)

    #1x1 conv
    x4_2 = Conv2D(8, 1, (1, 1), padding='same', kernel_initializer=initializers)(x3)
    x4_2 = Activation('relu')(x4_2)
    
    #3x3 conv dialated conv
    x4_3 = Conv2D(16, 3, (1, 1),dilation_rate=2, padding='same', kernel_initializer=initializers)(x3)
    x4_3 = Activation('relu')(x4_3)
    # x4_1 = x4_1 + x4_2
    # x4_2 = x4_1 + x4_2
    #x4,x4_2 concat
    x4 = Concatenate()([x4_1, x4_2])
    # x4 = Concatenate()([x4_1, x4_2, x4_3])

    #channel shuffle
    x4 = channel_shuffle(x4, groups=groups)

    x4 = Conv2D(16, 1, (1, 1), padding='same', kernel_initializer=initializers)(x4)
    x4 = Activation('relu')(x4)
    x4 = x4 + x4_3
    # x1 = CBAM_block(x1,name='x1_cbam')
    # x2 = CBAM_block(x2,name='x2_cbam')
    # x3 = CBAM_block(x3,name='x3_cbam')
    # x4 = CBAM_block(x4,name='x4_cbam')
    
    x4 = Concatenate()([x3,x4])

    #3x3 conv
    x5_1 = Conv2D(8, 3, (1, 1),  dilation_rate=2, padding='same', kernel_initializer=initializers)(x4)
    x5_1 = Activation('relu')(x5_1)
    
    #1x1 conv
    x5_2 = Conv2D(8, 1, (1, 1),  padding='same', kernel_initializer=initializers)(x4)
    x5_2 = Activation('relu')(x5_2)

    #3x3 conv
    x5_3 = Conv2D(16, 3, (1, 1),  dilation_rate=2, padding='same', kernel_initializer=initializers)(x4)
    x5_3 = Activation('relu')(x5_3)

    # x5_1 = x5_1 + x5_2
    # x5_2 = x5_1 + x5_2
    # x5 = Concatenate()([x5_1, x5_2, x5_3])
    x5 = Concatenate()([x5_1,  x5_2])


    x5 = channel_shuffle(x5, groups=groups)
    x5 = Conv2D(16, 1, (1, 1), padding='same', kernel_initializer=initializers)(x5)
    x5 = Activation('relu')(x5)
    x5 = x5 + x5_3

    x5 = Concatenate()([x2,x5])
    #3x3 conv
    x6_1 = Conv2D(8, 3, (1, 1),  dilation_rate=2, padding='same', kernel_initializer=initializers)(x5)
    x6_1 = Activation('relu')(x6_1)

    x6_2 = Conv2D(8, 1, (1, 1), padding='same', kernel_initializer=initializers)(x5)
    x6_2 = Activation('relu')(x6_2)

    x6_3 = Conv2D(16, 3, (1, 1),  dilation_rate=2, padding='same', kernel_initializer=initializers)(x5)
    x6_3 = Activation('relu')(x6_3)

    # x6_1 = x6_1 + x6_2
    # x6_2 = x6_1 + x6_2

    # x6 = Concatenate()([x6,x6_2, x6_3])
    x6 = Concatenate()([x6_1, x6_2])

    x6 = channel_shuffle(x6, groups=groups)
    x6 = Conv2D(16, 1, (1, 1), padding='same', kernel_initializer=initializers)(x6)
    x6 = Activation('relu')(x6)
    x6 = x6 + x6_3

    parameter_map = Concatenate()([x1,x6])
    parameter_map = Conv2D(3, 1, (1, 1), padding='same', kernel_initializer=initializers)(parameter_map)
    parameter_map = Activation('tanh')(parameter_map)
    
    return Model(inputs=inputs, outputs=parameter_map)



def msp_enhancement_net(inputs):
    x1 = MSP(inputs, 32)
    x2 = MSP(x1, 32)
    x3 = MSP(x2, 32)
    x4 = MSP(x3, 32)
    
    x4 = Concatenate()([x3,x4])
    x5 = MSP(x4, 32)
    
    x5 = Concatenate()([x2,x5])
    x6 = MSP(x5, 32)
    
    parameter_map = Concatenate()([x1,x6])
    parameter_map = Conv2D(3,  3, (1, 1), padding='same', kernel_initializer=initializers)(parameter_map)
    parameter_map = Activation('tanh')(parameter_map)
    
    return Model(inputs=inputs, outputs=parameter_map)

def enhancement_net_lw(inputs):
    x1 = dwconv(inputs, 32, 3)
    x1 = HardSwish(name='hardswish1')(x1)
    
    x2 = dwconv(x1, 32, 3)
    x2 = HardSwish(name='hardswish2')(x2)
    
    x3 = dwconv(x2, 32, 3)
    x3 = HardSwish(name='hardswish3')(x3)
    
    x4 = dwconv(x3, 32, 3)
    x4 = HardSwish(name='hardswish4')(x4)
    
    x5 = Concatenate()([x3,x4])
    x5 = dwconv(x5, 32, 3)
    x5 = HardSwish(name='hardswish5')(x5)
    
    x6 = Concatenate()([x2,x5])
    x6 = dwconv(x6, 32, 3)
    x6 = HardSwish(name='hardswish6')(x6)
    
    parameter_map = Concatenate()([x1,x6])
    parameter_map = dwconv(parameter_map, 3, 3)
    parameter_map = Activation('tanh')(parameter_map)
    
    return Model(inputs=inputs, outputs=parameter_map)


def pad(img1,img2,img3,img4,pad=12):
    img1 = tf.concat([img1,img2[:,:,:pad,:]],axis=2)
    img2 = tf.concat([img1[:,:,-pad:,:],img2],axis=2)
    img3 = tf.concat([img3,img4[:,:,:pad,:]],axis=2)
    img4 = tf.concat([img3[:,:,-pad:,:],img4],axis=2)
    
    img1 = tf.concat([img1,img3[:,:pad,:,:]],axis=1)
    img2 = tf.concat([img2,img4[:,:pad,:,:]],axis=1)
    img3 = tf.concat([img1[:,-pad:,:,:],img3],axis=1)
    img4 = tf.concat([img2[:,-pad:,:,:],img4],axis=1)
    return img1,img2,img3,img4   
     
class enhance_net(tf.keras.Model):
    def __init__(self, input_shape=(None,None,3), model_name='', scale_factor=1, **kwargs):
        super(enhance_net, self).__init__(**kwargs)
        self.model_name = model_name
        self.scale_factor = scale_factor
        self.split = 4
        
        self.L_spa = Losses.L_spa()
        self.L_exp = Losses.L_exp(16)
        self.L_col = Losses.L_col()
        self.L_tv = Losses.L_tv()
        
        if self.model_name == 'DCE':
            self.enhancement_net = enhancement_net(Input(shape=input_shape))
        elif self.model_name == 'CSP_DCE':
            self.enhancement_net = csp_enhancement_net(Input(shape=input_shape))
        elif self.model_name == 'MSP_DCE':
            self.enhancement_net = msp_enhancement_net(Input(shape=input_shape))
        elif self.model_name == 'DCE++':
            self.enhancement_net = enhancement_net_lw(Input(shape=input_shape))
        elif self.model_name == 'Dil_DCE':
            self.enhancement_net = dil_enhancement_net(Input(shape=input_shape))
        
        self.build(input_shape=(None, input_shape[0], input_shape[1], input_shape[2]))
        self.call(Input(shape=input_shape))
         
    def enhance(self, x, x_r):
        for i in range(8):
            x = x + x_r*(x - tf.math.pow(x,2))
        return x 
    
    def call(self, inputs):
        # _,h,w,c = tf.shape(inputs)
        
        # if self.scale_factor != 1:
        #     down_inputs = tf.image.resize(inputs, [h//self.scale_factor, w//self.scale_factor])
            
        # parameter_map = self.enhancement_net(down_inputs)
        
        # if self.scale_factor != 1:
        #     inputs = tf.image.resize(inputs, [h, w])
            
        # enhance_image = self.enhance(inputs, parameter_map)
        # ----------------------------------------------------
        pad=12
        # if input.shape[1] %2 !=0 or inputs.shape[1] //2 !=0:
        #     pad = 11
        # if input.shape[2] %2 !=0 or inputs.shape[2] //2 !=0:
            # pad = 11
        if self.split == 2:
            img1,img2 = tf.split(inputs,2,axis= 1)
            
            img1 = tf.concat([img1,img2[:,:pad,:,:]],axis=1)
            img2 = tf.concat([img1[:,-pad:,:,:],img2],axis=1)
            
            img1 = self.enhancement_net(img1)
            img2 = self.enhancement_net(img2)
            
            img1 = img1[:,:-pad,:,:]
            img2 = img2[:,pad:,: ,:]

            parameter_map = tf.concat([img1,img2],axis=1)

        elif self.split == 4:
            
            img1,img2 = tf.split(inputs,2,axis= 2)
            img1,img3 = tf.split(img1,2,axis=1)
            img2,img4 = tf.split(img2,2,axis=1) 
            
            
            img1 = tf.concat([img1,img2[:,:,:pad,:]],axis=2)
            img2 = tf.concat([img1[:,:,-pad:,:],img2],axis=2)
            img3 = tf.concat([img3,img4[:,:,:pad,:]],axis=2)
            img4 = tf.concat([img3[:,:,-pad:,:],img4],axis=2)
            
            img1 = tf.concat([img1,img3[:,:pad,:,:]],axis=1)
            img2 = tf.concat([img2,img4[:,:pad,:,:]],axis=1)
            img3 = tf.concat([img1[:,-pad:,:,:],img3],axis=1)
            img4 = tf.concat([img2[:,-pad:,:,:],img4],axis=1)
            
            img1 = self.enhancement_net(img1)
            img2 = self.enhancement_net(img2)
            img3 = self.enhancement_net(img3)
            img4 = self.enhancement_net(img4)
            
            img1 = tf.concat([img1[:,:-pad,:-pad,:],img3[:,pad:,:-pad,:]],axis=1)
            img2 = tf.concat([img2[:,:-pad, pad:,:],img4[:,pad:, pad:,:]],axis=1)
            
            parameter_map = tf.concat([img1,img2],axis=2)
            del img1,img2,img3,img4
            
        elif self.split == 6:
            img1,img2 = tf.split(inputs,2,axis= 1)
            img1_1, img1_2, img1_3 = tf.split(img1,3,axis=2)
            del img1
            img2_1, img2_2, img2_3 = tf.split(img2,3,axis=2)
            del img2

            img1_1 = tf.concat([img1_1,img1_2[:,:,:pad,:]], axis=2)
            img1_2 = tf.concat([img1_1[:,:,-pad:,:],img1_2,img1_3[:,:,:pad,:]], axis=2)
            img1_3 = tf.concat([img1_2[:,:,-pad:,:],img1_3], axis=2)
            img2_1 = tf.concat([img2_1,img2_2[:,:,:pad,:]], axis=2)
            img2_2 = tf.concat([img2_1[:,:,-pad:,:],img2_2,img2_3[:,:,:pad,:]], axis=2)
            img2_3 = tf.concat([img2_2[:,:,-pad:,:],img2_3], axis=2)

            img1_1 = tf.concat([img1_1,img2_1[:,:pad,:,:]],axis=1)
            img1_2 = tf.concat([img1_2,img2_2[:,:pad,:,:]],axis=1)
            img1_3 = tf.concat([img1_3,img2_3[:,:pad,:,:]],axis=1)
            img2_1 = tf.concat([img1_1[:,-pad:,:,:],img2_1],axis=1)
            img2_2 = tf.concat([img1_2[:,-pad:,:,:],img2_2],axis=1)
            img2_3 = tf.concat([img1_3[:,-pad:,:,:],img2_3],axis=1)

            img1_1 = self.enhancement_net(img1_1)
            img1_2 = self.enhancement_net(img1_2)
            img1_3 = self.enhancement_net(img1_3)
            img2_1 = self.enhancement_net(img2_1)
            img2_2 = self.enhancement_net(img2_2)
            img2_3 = self.enhancement_net(img2_3)

            img1 = tf.concat([img1_1[:,:-pad,:-pad,:],img1_2[:,:-pad,pad:-pad,:],img1_3[:,:-pad,pad:,:]], axis=2)
            del img1_1,img1_2,img1_3
            img2 = tf.concat([img2_1[:, pad:,:-pad,:],img2_2[:, pad:,pad:-pad,:],img2_3[:, pad:,pad:,:]], axis=2)
            del img2_1,img2_2,img2_3

            
            parameter_map = tf.concat([img1,img2],axis=1)
        
        elif self.split == 8:
            img1, img2 = tf.split(inputs, 2, axis=1)
            img1_1, img1_2, img1_3, img1_4 = tf.split(img1, 4, axis=2)
            img2_1, img2_2, img2_3, img2_4 = tf.split(img2, 4, axis=2)
            
            img1_1 = tf.concat([img1_1, img1_2[:, :, :pad, :]], axis=2)
            img1_2 = tf.concat([img1_1[:, :, -pad:, :], img1_2, img1_3[:, :, :pad, :]], axis=2)
            img1_3 = tf.concat([img1_2[:, :, -pad:, :], img1_3], axis=2)
            img1_4 = tf.concat([img1_2[:, :, -pad:, :], img1_4], axis=2)
            
            img2_1 = tf.concat([img2_1, img2_2[:, :, :pad, :]], axis=2)
            img2_2 = tf.concat([img2_1[:, :, -pad:, :], img2_2, img2_3[:, :, :pad, :]], axis=2)
            img2_3 = tf.concat([img2_2[:, :, -pad:, :], img2_3], axis=2)
            img2_4 = tf.concat([img2_2[:, :, -pad:, :], img2_4], axis=2)
            
            img1_1 = tf.concat([img1_1, img2_1[:, :pad, :, :]], axis=1)
            img1_2 = tf.concat([img1_2, img2_2[:, :pad, :, :]], axis=1)
            img1_3 = tf.concat([img1_3, img2_3[:, :pad, :, :]], axis=1)
            img1_4 = tf.concat([img1_4, img2_4[:, :pad, :, :]], axis=1)
            
            img2_1 = tf.concat([img1_1[:, -pad:, :, :], img2_1], axis=1)
            img2_2 = tf.concat([img1_2[:, -pad:, :, :], img2_2], axis=1)
            img2_3 = tf.concat([img1_3[:, -pad:, :, :], img2_3], axis=1)
            img2_4 = tf.concat([img1_4[:, -pad:, :, :], img2_4], axis=1)
            
            img1_1 = self.enhancement_net(img1_1)
            img1_2 = self.enhancement_net(img1_2)
            img1_3 = self.enhancement_net(img1_3)
            img1_4 = self.enhancement_net(img1_4)
            img2_1 = self.enhancement_net(img2_1)
            img2_2 = self.enhancement_net(img2_2)
            img2_3 = self.enhancement_net(img2_3)
            img2_4 = self.enhancement_net(img2_4)
            
            img1 = tf.concat([img1_1[:, :-pad, :-pad, :], img1_2[:, :-pad, pad:-pad, :], 
                            img1_3[:, :-pad, pad:, :], img1_4[:, -pad:, pad:, :]], axis=2)
            
            img2 = tf.concat([img2_1[:, pad:, :-pad, :], img2_2[:, pad:, pad:-pad, :], 
                            img2_3[:, pad:, pad:, :], img2_4[:, pad:, pad:, :]], axis=2)
            
            parameter_map = tf.concat([img1, img2], axis=1)
                    
        enhance_image = self.enhance(inputs, parameter_map)


        return enhance_image, parameter_map
        
    @tf.function
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        
        with tf.GradientTape() as tape:
            # Forward pass
            enhance_image, parameter_map = self(data)
            
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            
            # enhancemnet_net Loss
            l_spa = self.L_spa(enhance_image, data)
            l_exp = self.L_exp(enhance_image, 0.6)
            l_col = self.L_col(enhance_image)
            l_tv  = self.L_tv(parameter_map)
            
            # 1, 10, 5, 1600 best
            # 1, 20, 10, 1600
            w = [1, 10, 5, 1600]
            
            # Total loss
            enhancement_loss = w[0]*l_spa + w[1]*l_exp + w[2]*l_col + w[3]*l_tv
            
            loss = enhancement_loss
           
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        return {'train_loss': loss, 'l_spa': w[0]*l_spa, 'l_exp': w[1]*l_exp, 'l_col': w[2]*l_col, 'l_tv': w[3]*l_tv}
    
    @tf.function
    def validation_step(self, validation_data, validation_label):
        parameter_map, inputs = self(validation_data)
        enhance_img = inputs
        enhance_image, parameter_map = self(validation_data)
        
        #計算 val loss
        l_spa = self.L_spa(enhance_image, validation_label)
        l_exp = self.L_exp(enhance_image, 0.6)
        l_col = self.L_col(enhance_image)
        l_tv = self.L_tv(parameter_map)

        w = [1, 10, 5, 1600]

        enhancement_val_loss = w[0]*l_spa + w[1]*l_exp + w[2]*l_col + w[3]*l_tv
        
        val_loss = enhancement_val_loss

        parameter_map, inputs = self(validation_data)
        enhance_img = inputs
        for i in range(4):
            enhance_img = enhance_img + parameter_map*(enhance_img - tf.math.pow(enhance_img, 2))
        ssim = tf.reduce_mean(tf.image.ssim(enhance_img, validation_label, max_val=1))
        psnr = tf.reduce_mean(tf.image.psnr(enhance_img, validation_label, max_val=1))
 

        return ssim, psnr, val_loss
    
    def model_save(self, epoch, path):
        # self.enhancement_net.save(path + self.model_name +'/model/epoch{0}'.format(epoch+1))
        self.enhancement_net.save_weights(path + self.model_name + '/weights/epoch{0}/'.format(epoch+1))
        
        # # Convert the model.
        # converter = tf.lite.TFLiteConverter.from_keras_model(self.enhancement_net)
        # tflite_model = converter.convert()
        # # Save the model.
        # with open('model.tflite', 'wb') as f:
        #     f.write(tflite_model)