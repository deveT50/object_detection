
# -*- coding: utf-8 -*-

import argparse
import datetime
import json
import random
import sys
import time

import numpy as np
from PIL import Image
import six
import six.moves.cPickle as pickle

import chainer
from chainer import cuda
from chainer import optimizers
from chainer import serializers

import math


parser = argparse.ArgumentParser(
    description='Learning convnet from ILSVRC2012 dataset')
parser.add_argument('train', help='Path to training image-answer list file')
parser.add_argument('val', help='Path to validation image-answer list file')
parser.add_argument('--mean', '-m', default='mean.npy',
                    help='Path to the mean file (computed by compute_mean.py)')
parser.add_argument('--batchsize', '-B', type=int, default=8,
                    help='Learning minibatch size')
parser.add_argument('--val_batchsize', '-b', type=int, default=8,
                    help='Validation minibatch size')
parser.add_argument('--epoch', '-E', default=50, type=int,
                    help='Number of epochs to learn')
parser.add_argument('--gpu', '-g', default=0, type=int,
		    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--out', '-o', default='model',
                    help='Path to save model on each validation')
args = parser.parse_args()


pathList=[]
logArray=[]
logArray2=[]

# 準備
# リスト pathList[(tuples[path],[classNo.])]を作成
def load_image_list(path):
	tuples = []
	for line in open(path):
		pair = line.strip().split()
		tuples.append((pair[0], np.float32(pair[1:])))
	pathList.append(pair[0])
	return tuples


#txtのロード
train_list = load_image_list(args.train)
val_list = load_image_list(args.val)
mean_image = pickle.load(open(args.mean, 'rb'))

# モデルロード
#import network
#model = network.imageModel()

#import resnet
#model = resnet.ResNet(resnet.ResBlock)

from resnet import ResNet

model = ResNet()


if args.gpu >= 0:
	cuda.init(args.gpu)
	print("gpu")
	model.to_gpu()
	xp=cuda.cupy
else:
	xp=np

# Setup optimizer
#optimizer = optimizers.MomentumSGD(lr=0.08, momentum=0.9)
optimizer = optimizers.Adam()
optimizer.setup(model)
#これらは正則化パラメータ。過学習しているなら正則化、データ数を増やす、
#http://lab.synergy-marketing.co.jp/blog/tech/machine-learning-stanford-3
#	過学習に有効
#		学習データを増やす
#		変数を減らす
#		λ（正則化パラメータ）を大きくする
#	高バイアスに有効
#		変数増やす
#		多項式にする（モデルを複雑にする）
#		λ減らす

#重み上限(+dropoutが有効)
grad_clip=8.75#8.75 #10
optimizer.add_hook(chainer.optimizer.GradientClipping(grad_clip))
#重み減衰
optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))


# Data loading routine
cropwidth = 256 - model.insize
#cropwidth = 128 - model.insize

#学習打ち切り
g_end=False


g_accum_loss=0

#標準偏差を計算する
def compute_stdDeviation(model, path, center=True, flip=False): #if center==true then fix image extract area
	pwArray=np.zeros((3,model.insize,model.insize))
	#pwArray=np.zeros((model.insize,model.insize))
	for i in path:
		
		#imageを開いて範囲を決める
		image = np.asarray(Image.open(i)).transpose(2, 0, 1) #カラー用
		#image = np.asarray(Image.open(i))
		
		if center:
			top = left = cropwidth / 2
		else:
			#between 0~(256-223)::0~23, move objective resion slightly for each images
			top = random.randint(0, cropwidth - 1)
			left = random.randint(0, cropwidth - 1) 
		bottom = model.insize + top
		right = model.insize + left

		image = image[:, top:bottom, left:right].astype(xp.float32)#カラー用
		#image = image[top:bottom, left:right].astype(xp.float32)
		#平均を引く
		image -= mean_image[:, top:bottom, left:right]#カラー用
		#image -= mean_image[top:bottom, left:right]
		#image-meanの2乗を足していく
		pwArray += np.power(image,2)

	#imageの差の2乗をimage数でわってルートしたもの
	#sqrt(pow(image-mean image)/image num)
	pwArray=np.sqrt(pwArray/len(i))  #データ数-1で割る
	return pwArray



#imageを読み込んで平均を引く、標準偏差で割る
def read_image(path, center=True, flip=False):
	#row, col, color -> color, row, col
	image = np.asarray(Image.open(path)).transpose(2, 0, 1)#カラー用
	#image = np.asarray(Image.open(path))
	#範囲を決める
	if center:
		top = left = cropwidth / 2
	else:
		top = random.randint(0, cropwidth - 1)
		left = random.randint(0, cropwidth - 1)
	bottom = model.insize + top
	right = model.insize + left

	image = image[:, top:bottom, left:right].astype(np.float32)
	#image = image[top:bottom, left:right].astype(np.float32)
	#平均を引く
	image -= mean_image[:, top:bottom, left:right]
	#image -= mean_image[top:bottom, left:right]
	
	#標準偏差で割る
	#devide by stdDeviation instead of by 256
	image/=g_stdDev
	#image/=255.0
	
	#左右反転
	if flip and random.randint(0, 1) == 0:
		return image[:, :, ::-1]
		#return image[:, ::-1]  
	else:
		return image



# 訓練ループ
def train_loop():
	# 画像は（RGB3チャネル、Xサイズ、Yサイズ）のarray
	#ミニバッチの入れ物を用意する
	#カラー画像
	x_batch = xp.ndarray((args.batchsize, 3, model.insize, model.insize), dtype=np.float32)
	y_batch = xp.ndarray((args.batchsize,), dtype=np.int32)
	#モノクロ画像
	#x_batch = xp.ndarray((args.batchsize, model.insize, model.insize), dtype=np.float32)
	#y_batch = xp.ndarray((args.batchsize,), dtype=np.float32)
	



	#validation用ミニバッチカラー用
	val_x_batch = xp.ndarray((args.val_batchsize, 3, model.insize, model.insize), dtype=np.float32)
	val_y_batch = xp.ndarray((args.val_batchsize,), dtype=np.int32)
	#モノクロ画像
	#val_x_batch = xp.ndarray((args.val_batchsize, model.insize, model.insize), dtype=np.float32)
	#val_y_batch = xp.ndarray((args.val_batchsize,), dtype=np.float32)
	



	#trainListをシャッフルするためランダムなインデックスを作る
	perm = np.random.permutation(len(train_list))

	#誤差平均値
	loss_mean=0
	acc_mean=0
	val_acc_mean=0
	val_times=0


	start = time.time()
	for epoch in range(args.epoch):
		#学習打ち切り
		if g_end:
			break

		x_batch_list=[]
		y_batch_list=[]
		val_x_batch_list=[]
		val_y_batch_list=[]

		
		loss_mean=0
		acc_mean=0
		

		#10回ごとにvalidation用バッチを作る
		if (epoch % 10 == 0 and epoch!=0) or epoch==args.epoch-1: 
			val_acc_mean=0
			j = 0

			for idx in range(len(val_list)):#182 images

				#imagePathとanswerをセット
				path, answer = val_list[idx]
				val_x_batch_list.append(read_image(path, True, False))
				val_y_batch_list.append(answer)
				j += 1
	
				#バッチサイズまで到達したら
				if j == args.val_batchsize or idx==len(train_list):
					
					#val_times+=1
					train=False

					x_batch=xp.asarray(val_x_batch_list,dtype=np.float32)
					y_batch=xp.asarray(val_y_batch_list,dtype=np.float32)
					j = 0
					accuracy=perform(x_batch,y_batch,train)
					val_x_batch_list=[]
					val_y_batch_list=[]
					val_acc_mean+=accuracy


		i=0
		#cnt=0	
		#作成したインデックスの中で、
		for idx in range(len(train_list)):
			#imagePathとラベルをセット
			path, answer = train_list[perm[idx]]
			x_batch_list.append(read_image(path, True, True))
			y_batch_list.append(answer)
			#バッチサイズまでカウントを繰り返す変数i
			i += 1

			#バッチサイズまで到達したら
			if i == args.batchsize or idx==len(train_list)-1:
				
				train=True
				#ｘバッチを作成する
				x_batch=xp.asarray(x_batch_list,dtype=xp.float32)
				y_batch=xp.asarray(y_batch_list,dtype=xp.float32)
				i = 0
				loss=perform(x_batch,y_batch,train)
				loss_mean+=loss
				#acc_mean+=accuracy
				x_batch_list=[]
				y_batch_list=[]

		print('epoch', epoch)
		#１エポックごとのlossとaccuracyを表示
		#print('learning rate', optimizer.lr)
		print("mean loss:",loss_mean/(len(train_list)/args.batchsize))
		print("mean accuracy:",acc_mean/(len(train_list)/args.batchsize))

		#validation のaccuracy
		print("test mean accuracy:",val_acc_mean/(len(val_list)/args.val_batchsize))

		elapsed_time = time.time() - start
		print(("elapsed_time:{0}".format(elapsed_time)) + "[sec]")


def perform(x_batch,y_batch,train):

	#学習率を下げる
	#if optimizer.lr<0.0001:
	#	g_end=True
	#else:
	#	optimizer.lr *= 0.97

	#color
	x = chainer.Variable(x_batch.reshape((len(x_batch), 3, model.insize, model.insize)), volatile="off")
	t = chainer.Variable(y_batch.reshape((len(y_batch),4)), volatile="off")
	#mono
	#x = chainer.Variable(x_batch.reshape((len(x_batch), 1, model.insize, model.insize)), volatile="off")
	#t = chainer.Variable(y_batch.reshape(len(y_batch),1), volatile="off")


	if train :
		optimizer.zero_grads()
		#loss= model(x, t, True)
		loss= model(x, t)
		loss.backward()
		optimizer.update()
		return loss.data
	else:
		#loss= model(x, t, True)
		loss= model(x, t)
		return loss.data






#標準偏差を計算する
g_stdDev=compute_stdDeviation(model, pathList)
#学習開始
train_loop()
#write stdDeviation array
pickle.dump(g_stdDev, open('sigma.npy', 'wb'), -1)


#modelの保存
serializers.save_hdf5('modelhdf5', model)

