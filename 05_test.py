
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
parser.add_argument('test', help='Path to validation image-answer list file')
parser.add_argument('--batchsize', '-B', type=int, default=8,
                    help='Learning minibatch size')
parser.add_argument('--gpu', '-g', default=0, type=int,
		    help='GPU ID (negative value indicates CPU)')
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
		tuples.append((pair[0], np.float32(pair[1])))
	pathList.append(pair[0])
	return tuples


#txtのロード
test_list = load_image_list(args.test)

mean_image = pickle.load(open("mean.npy", 'rb'),encoding='latin-1')
sigma_image = pickle.load(open("sigma.npy",'rb'),encoding='latin-1')

# モデルロード
#import network
#model = network.imageModel()


from resnet import ResNet
model = ResNet()

serializers.load_hdf5("modelhdf5", model)


if args.gpu >= 0:
	cuda.init(args.gpu)
	model.to_gpu()
	xp=cuda.cupy
else:
	xp=np



# Data loading routine
cropwidth = 256 - model.insize

#学習打ち切り
g_end=False


g_accum_loss=0


#imageを読み込んで平均を引く、標準偏差で割る
def read_image(path, center=True, flip=False):
	#row, col, color -> color, row, col
	#image = np.asarray(Image.open(path)).transpose(2, 0, 1)#カラー用
	image = np.asarray(Image.open(path))
	#範囲を決める
	top = left = cropwidth / 2
	
	bottom = model.insize + top
	right = model.insize + left

	#image = image[:, top:bottom, left:right].astype(np.float32)
	image = image[top:bottom, left:right].astype(np.float32)
	#平均を引く
	#image -= mean_image[:, top:bottom, left:right]
	image -= mean_image[top:bottom, left:right]
	
	#標準偏差で割る
	#devide by stdDeviation instead of by 256
	image/=sigma_image
	#image/=255.0
	
	#左右反転
	return image



# ループ
def test_loop():

	#x_batch = xp.ndarray((args.batchsize, 3, model.insize, model.insize), dtype=np.float32)
	#y_batch = xp.ndarray((args.batchsize,), dtype=np.int32)
	#モノクロ画像
	x_batch = xp.ndarray((args.batchsize, model.insize, model.insize), dtype=np.float32)
	y_batch = xp.ndarray((args.batchsize,), dtype=np.float32)
	

	#trainListをシャッフルするためランダムなインデックスを作る
	#perm = np.random.permutation(len(test_list))


	#誤差平均値
	loss_mean=0
	failed_num=0
	loss_max=0


	start = time.time()

	x_batch_list=[]
	y_batch_list=[]
	
	loss_mean=0
	acc_mean=0
	
	i=0

	#作成したインデックスの中で、
	for idx in range(len(test_list)):
		#imagePathとラベルをセット
		#path, answer = test_list[perm[idx]]
		path, answer = test_list[idx]
		x_batch_list.append(read_image(path, True, False))
		y_batch_list.append(answer)
		#バッチサイズまでカウントを繰り返す変数i
		i += 1

		#バッチサイズまで到達したら
		if i == args.batchsize or idx==len(test_list)-1:
			
			train=False
			#ｘバッチを作成する
			x_batch=xp.asarray(x_batch_list,dtype=xp.float32)
			y_batch=xp.asarray(y_batch_list,dtype=xp.float32)
			i = 0
			loss=perform(x_batch,y_batch,train)
			loss_mean+=loss
			if loss>1.0:
				failed_num+=1
			if loss>loss_max:
				loss_max=loss

			x_batch_list=[]
			y_batch_list=[]

	
	print("mean loss:",loss_mean/(len(test_list)/args.batchsize))
	print("max loss:",loss_max)
	print("failed times:",failed_num)


	elapsed_time = time.time() - start
	print(("elapsed_time:{0}".format(elapsed_time)) + "[sec]")


def perform(x_batch,y_batch,train):
	x = chainer.Variable(x_batch.reshape((len(x_batch), 1, model.insize, model.insize)), volatile="off")
	t = chainer.Variable(y_batch.reshape(len(y_batch),1), volatile="off")

	loss= model(x, t, False)
	return loss.data






#test開始
test_loop()







