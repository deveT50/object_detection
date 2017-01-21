#!/usr/bin/env python
# -*- coding: utf-8 -*-

import chainer
import chainer.functions as F
import chainer.links as L
import chainer.initializers
import math


class imageModel(chainer.Chain):

	#insize = 125
	insize = 250


	def __init__(self):
		initializer = chainer.initializers.HeNormal()
		w = math.sqrt(2)  # MSRA scaling
		super(imageModel, self).__init__(
			#入力チャネル,出力チャネル, フィルタサイズpx

			#60.9%モデル--------------------------------------
			#conv1=L.Convolution2D(3, 8, 7),
			#conv2=L.Convolution2D(8, 16, 5),
			#conv3=L.Convolution2D(16, 32, 3),
			#conv4=L.Convolution2D(32, 48, 3),


			conv1=L.Convolution2D(3, 8, 7,wscale=w),
			conv2=L.Convolution2D(8, 16, 3,wscale=w),
			conv3=L.Convolution2D(16, 24, 3,wscale=w),
			conv4=L.Convolution2D(24, 32, 3,wscale=w),
			conv5=L.Convolution2D(32, 40, 3,wscale=w),#add
			conv6=L.Convolution2D(40, 48, 3,wscale=w),#add
			conv7=L.Convolution2D(48, 56, 3,wscale=w),#add
			fc1=L.Linear(768,100,wscale=w),
			fc2=L.Linear(100,4,wscale=w),
			fc3=L.Linear(72,5,wscale=w),
			

		)
		self.train = True
	def __call__(self, x, t, train):


		h = self.conv1(x)
		h = F.relu(h)
		h = F.max_pooling_2d(h, 3, stride=2)
		#print(h.shape)	
		h = self.conv2(h)
		h = F.relu(h)
		h = F.average_pooling_2d(h, 3, stride=2)
		#print(h.shape)	
		h = self.conv3(h)
		h = F.relu(h)
		h = F.average_pooling_2d(h, 3, stride=2)
		#print(h.shape)	
		h = self.conv4(h)
		#h = F.relu(F.dropout(h, ratio=0.1,train=train))#0.5
		h = F.relu(h)#0.5
		h = F.average_pooling_2d(h, 3, stride=2)
		#h=F.average_pooling_2d(h, 6)	
		#print(h.shape)	
		h = self.conv5(h)
		h = F.relu(F.dropout(h, ratio=0.1,train=train))
		h = F.average_pooling_2d(h, 3, stride=1)
		#print(h.shape)	
		h = self.conv6(h)
		h = F.relu(F.dropout(h, ratio=0.1,train=train))
		h = F.average_pooling_2d(h, 3, stride=1)
		#print(h.shape)	
		#--h = self.conv7(h)
		#--h = F.relu(F.dropout(h, ratio=0.1,train=train))
		#--h = F.average_pooling_2d(h, 3, stride=1)
		#print(h.shape)	
		#y = F.reshape(h, (x.data.shape[0],192))
		#y = F.reshape(F.average_pooling_2d(h, 3), (8,1))
		#h=self.fc1(h)
		#h=F.relu(F.dropout(h, ratio=0.5,train=train))
		h=self.fc1(h)
		h=F.relu(F.dropout(h, ratio=0.1,train=train))
		y=self.fc2(h)
		#h=F.relu(F.dropout(h, ratio=0.3,train=train))
		#y=self.fc3(h)

		if train:
			return F.mean_squared_error(y, t)
		else:
			return F.mean_squared_error(y, t)



	def predict(self, x_data):
		x=x_data
		h = self.conv1(x)
		h = F.relu(h)
		h = F.max_pooling_2d(h, 3, stride=2)

		h = self.conv2(h)
		h = F.relu(h)
		h = F.average_pooling_2d(h, 3, stride=2)

		h = self.conv3(h)
		h = F.relu(h)
		h = F.average_pooling_2d(h, 3, stride=2)
		
		h = self.conv4(h)
		h = F.relu(F.dropout(h, ratio=0.5,train=False))
		h = F.average_pooling_2d(h, 3, stride=2)
		
		y=self.fc1(h)

		return y

		






