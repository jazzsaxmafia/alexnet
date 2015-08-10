#-*- coding: utf-8 -*-
import os
import ipdb
import theano
import theano.tensor as T

from theano.sandbox.cuda import dnn
from pylearn2.expr.normalize import CrossChannelNormalization
import numpy as np

rng = np.random.RandomState(23455)

class Weight(object):
    def __init__(self, w_shape, mean=0, std=0.01):
        super(Weight, self).__init__()

        if std != 0:
            self.np_values = np.asarray(
                    rng.normal(mean, std, w_shape), dtype=theano.config.floatX)
        else:
            self.np_values = np.array(mean*np.ones(w_shape, dtype=theano.config.floatX)).astype(theano.config.floatX)

        self.val = theano.shared(value=self.np_values)

    def save_weight(self, dir, name):
        print 'weight saved: ' + name
        np.save(os.path.join(dir, name), self.val.get_value())

    def load_weight(self, dir, name):
        print 'weight loaded: ' + name
        self.np_values = np.load(os.path.join(dir, name+'.npy'))
        self.val.set_value(value=self.np_values)



class DataLayer():
    '''
    input: (batch_size, height, width, channel)
    '''
    def __init__(self,
                 input,
                 image_shape,
                 cropsize,
                 rand,
                 mirror,
                 flag_rand):

        mirror = input[:,:,::-1,:]
        input = T.concatenate([input, mirror], axis=0)

        # center_margin 기준으로 자르면 가운데가 잘림
        center_margin = (image_shape[2] - cropsize) / 2

        if flag_rand:
            # mirror_rand: 0 또는 1. 0이면 원본이미지, 1이면 좌우반전
            mirror_rand = T.cast(rand[2], 'int32')
            # 0*center_margin: 왼쪽 상단
            # 2*center_margin: 오른쪽 하단
            crop_xs = T.cast(rand[0] * center_margin * 2, 'int32')
            crop_ys = T.cast(rand[1] * center_margin * 2, 'int32')
        else:
            mirror_rand = 0
            crop_xs = center_margin
            crop_ys = center_margin

        self.output = input[mirror_rand*3:(mirror_rand+1)*3,:,:,:] # 원본과 좌우 반전 중 하나 선택
        self.output = self.output[:,crop_xs:crop_xs+cropsize, crop_ys:crop_ys+cropsize,:]

class ConvPoolLayer():
    def __init__(self,
                 input,
                 image_shape=[1,3,227,227], # should be [batch_size, n_input_kernel, row, col]
                 filter_shape=[10,3,11,11], # should be [n_output_kern, n_input_kern, kern_row, kern_col]
                 convstride=2,
                 padsize='full',
                 group=1,
                 poolsize=2,
                 poolstride=2,
                 bias_init=np.array(0).astype(theano.config.floatX),
                 lrn=False,
                 lib_conv='cudnn'):

        self.filter_shape = filter_shape
        self.convstride = convstride
        self.padsize = padsize
        self.poolsize = poolsize
        self.poolstride = poolstride
        self.channel = image_shape[0]
        self.lrn = lrn
        self.lib_conv = lib_conv

        assert group in [1,2]

        self.filter_shape = np.asarray(filter_shape)
        self.image_shape = np.asarray(image_shape)

        if self.lrn:
            self.lrn_func = CrossChannelNormalization()

        if group == 1: # 일반
            self.W = Weight(self.filter_shape)
            self.b = Weight(self.filter_shape[0], bias_init, std=0)
        else:
            self.filter_shape[0] = self.filter_shape[0] / 2 # output kernel 개수
            self.filter_shape[1] = self.filter_shape[1] / 2 # input kernel 개수

            self.image_shape[0] = self.image_shape[0] / 2 # batch_size
            self.image_shape[1] = self.image_shape[1] / 2 # input kernel 개수 ( 위랑 같음 )

            self.W0 = Weight(self.filter_shape)
            self.W1 = Weight(self.filter_shape)

            self.b0 = Weight(self.filter_shape[0], bias_init, std=0)
            self.b1 = Weight(self.filter_shape[0], bias_init, std=0)

        if lib_conv == 'cudnn':
            if group == 1:
                conv_out = dnn.dnn_conv(img=input,
                                        kerns=self.W.val,
                                        subsample=(convstride, convstride),
                                        border_mode=padsize)
                conv_out = conv_out + self.b.val.dimshuffle('x',0,'x','x')

            else:
                conv_out0 = dnn.dnn_conv(img=input[:, :self.channel/2, :, :],
                                         kerns=self.W0.val,
                                         subsample=(convstride,convstride),
                                         border_mode=padsize)

                conv_out0 = conv_out0 + self.b0.val.dimshuffle('x',0,'x','x')

                conv_out1 = dnn.dnn_conv(img=input[:, :self.channel/2, :, :],
                                         kerns=self.W1.val,
                                         subsample=(convstride,convstride),
                                         border_mode=padsize)

                conv_out1 = conv_out1 + self.b1.val.dimshuffle('x',0,'x','x')

            self.output = T.maximum(conv_out, 0)

            if self.poolsize != 1:
                self.output = dnn.dnn_pool(self.output,
                                           ws=(poolsize,poolsize),
                                           stride=(poolstride,poolstride))

        else:
            NotImplementedError("lib_conv can only be cudaconvnet or cudnn")

        if self.lrn:
            self.output = self.lrn_func(self.output)

        if group == 1:
            self.params = [self.W.val, self.b.val]
            self.weight_type = ['W', 'b']
        else:
            self.params = [self.W0.val, self.b0.val, self.W1.val, self.b1.val]
            self.weight_type = ['W','b','W','b']

