#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import argparse

import sys
import numpy as np
from PIL import Image
import math
import random
import six
#import cPickle as pickle
import six.moves.cPickle as pickle
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import serializers

import network
from network import imageModel

parser = argparse.ArgumentParser(description='predict object number in a picture')
parser.add_argument('path', help='Path to image file')
args = parser.parse_args()


#mean_image = pickle.load(open("mean.npy", 'rb'))
#sigma_image = pickle.load(open("sigma.npy",'rb'))
mean_image = pickle.load(open("mean.npy", 'rb'),encoding='latin-1')
sigma_image = pickle.load(open("sigma.npy",'rb'),encoding='latin-1')



model = network.imageModel()
serializers.load_hdf5("modelhdf5", model)
cropwidth = 128 - model.insize
model.to_cpu()



def read_image(path):

	image = np.asarray(Image.open(path))
	#top = random.randint(0, cropwidth - 1)
	#left = random.randint(0, cropwidth - 1)
	top = left = cropwidth / 2
	bottom = model.insize + top
	right = model.insize + left
	image = image[top:bottom, left:right].astype(np.float32)
	#正規化
	image -= mean_image[top:bottom, left:right]
	image/=sigma_image

	return image



def predict(path):
	
	img = read_image(path)
	x = np.ndarray((1,1, model.insize, model.insize), dtype=np.float32)
	x[0]=img
	x = chainer.Variable(np.asarray(x), volatile='on')
	number = imageModel.predict(model,x)

	return number.data[0][0]

if __name__ == '__main__': 
	num=predict(args.path)
	print(round(num,0))






