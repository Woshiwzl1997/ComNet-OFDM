# ComNet
Implemention of ComNet with Keras：Combination of Deep Learning and Expert Knowledge in OFDM Receivers


# ComNet-OFDM
Using deep model to implement channel estimation and demodulation based on classcial OFDM receiver. 


# Envs:
Windows


tensorflow:1.8.0


Keras:2.1.6


python:3.6


You can re-train this model on Ubuntu system.


# Get Start
1.Download data from：


Link：https://pan.baidu.com/s/1N_OLJYWlyDukM3-tx2BgZQ 


password：ep86 

Move dataset to ./datas folder


2.Train


python main_comnet.py --train_flag True 


3.Test and Plot:


python main_comnet.py --train_flag False --plot True


# Tips:


Every 128 bits datas are set as training label and predicted by 8 deep models.


The first model predicts [0:16] bits,the second model predicts [16:32], etc.


This code only includes the first model.You can also try to convert the size 16 to other numbers.


If you think this work is helpful to you, click "Star' to let me know.

