# 物体位置認識
画像内の物体位置を検出して、物体を囲む矩形の  
x座標, y座標, width, height  
を返すニューラルネットワーク(回帰)。  

環境：ubuntu 16.04, python 3.5.2, Chainer v1.19.0 


##使い方  
このファイル（README.md）と同じ階層で、  

平均画像作成  
`$　python3 03_compute_mean.py train.txt`  
mean.npyが出力されます。  

訓練開始  
`$ python3 04_train_net.py -B 8 -g 0 -E 5 train.txt test.txt`  
-B はミニバッチあたりの枚数です。  
-g 0　はGPU（ID=0）を使用することを意味します。-1がCPUです。  
-E はエポック数です。  
modelhdf5 が出力されます。重みとバイアスを保存しています。  
sigma.npy が作成されます。画像の標準偏差（シグマ）を保存しています。  


以下は未完成。  
テスト  
`$ python3 05_test.py test.txt`  
推定  
`$ python3 06_predict.py someimage.jpg`  




##フォルダ構成  
###学習用  
* 02_crop_128.py  ・・・・・・・・・・・・・・・・・・・・ 画像リサイズ(opencv必要)  
* 03_compute_mean.py ・・・・・・・・・・・・・・・・・ 平均画像を計算して出力する  
* 04_train_net.py ・・・・・・・・・ ニューラルネットを訓練してパラメータファイルを出力する  
* 05_test.py ・・・・・・・・・・・・・・・・・・・・・・・・・ （未完成）  
* 06_predict.py ・・・・・・・・・・・・・・・・・・・・・・ （未完成）  
* network.py ・・・・・・・・・・・・・・・・・・・・・・・・・ CNNの定義(ResNet導入前に使っていたもの)  
* resnet.py ・・・・・・・・・・・・・・・・・・・・・・・・・ ResNetの定義  
* mean.npy ・・・・・・・・・・・・・・・・・・・・・・・・・・・ 平均画像ファイル  
* sigma.npy ・・・・・・・・・・・・・・・・・・・・・・・・・・ 標準偏差ファイル  


