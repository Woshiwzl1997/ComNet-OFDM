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
Train:


python main_comnet.py --train_flag True 


Test and Plot:


python main_comnet.py --train_flag False --plot True


# Tips:


Every 128 bits datas are predicted by 8 deep models.The first model predict [0:16] bits,the second model predict [16:32], etc.This code only includes the first model.You can also try to convert the size 16 to other numbers.


If you think this work is helpful to you, click "Star' to let me know.

