from tensorflow.python.keras import *
from tensorflow.python.keras.layers import *
from generations_comnet import *
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import backend as K
import os
import scipy.io as sio
import argparse


def bit_err(y_true, y_pred):
    err = 1 - tf.reduce_mean(
        tf.reduce_mean(
            tf.to_float(
                tf.equal(
                    tf.sign(
                        y_pred - 0.5),
                    tf.cast(
                        tf.sign(
                            y_true - 0.5),
                        tf.float32))),
            1))
    return err

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true'):
        return True
    elif v.lower() in ('false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def model(args):

    input_LS = Input(shape=(payloadBits_per_OFDM,))  # Channel LS estimation
    input_DATA = Input(shape=(int(payloadBits_per_OFDM / 2), 2,))  # Data after passing through the channel

    temp = BatchNormalization()(input_LS)
    temp = Dense(n_hidden_1, activation='relu')(temp)  # 'relu->liner'
    temp = BatchNormalization()(temp)
    out_put1 = Dense(n_h, activation='relu')(temp)  # Channel estimation output
    out_put1 = BatchNormalization(name='h_predict_output')(out_put1)

    # out_put1=Reshape((int(payloadBits_per_OFDM/2),2))(out_put1)

    out_put1 = Reshape((2, int(payloadBits_per_OFDM / 2)))(
        out_put1)  # The way of 'Reshape' works is row first,but we need column first way
    out_put1 = Permute((2, 1))(out_put1)

    def complex_division(h_ls):
        "complex vector division: y_data/h_ls"
        real_input_Data = input_DATA[:, :, 0]
        imag_input_Data = input_DATA[:, :, 1]
        real_input_hls = h_ls[:, :, 0]
        imag_input_hls = h_ls[:, :, 1]
        real_result = (real_input_Data * real_input_hls + imag_input_Data * imag_input_hls) / (
                    real_input_hls ** 2 + imag_input_hls ** 2)
        imag_result = (imag_input_Data * real_input_hls - real_input_Data * imag_input_hls) / (
                    real_input_hls ** 2 + imag_input_hls ** 2)
        return concatenate([real_result, imag_result])

    x_zf = Lambda(complex_division)(out_put1)
    x_zf = Reshape((2, int(payloadBits_per_OFDM / 2)))(x_zf)
    x_zf = Permute((2, 1))(x_zf)



    combined = concatenate([input_DATA, out_put1], axis=2)
    combined = concatenate([combined, x_zf], axis=2)

    # Build multi-layer BiLSTM Network
    temp_lstm = Bidirectional(LSTM(n_BiLSTM_1, return_sequences=True))(combined)

    temp_lstm = Bidirectional(LSTM(n_BiLSTM_2, return_sequences=True))(temp_lstm)

    temp_lstm = Bidirectional(LSTM(n_BiLSTM_3))(temp_lstm)

    final_output = Dense(n_output, activation='sigmoid',name='model_output')(temp_lstm)
    model = Model(inputs=[input_LS, input_DATA], outputs=final_output)
    model.compile(optimizer='adam', loss='mse', metrics=[bit_err])
    model.summary()
    if args.train_flag == True:
        checkpoint = callbacks.ModelCheckpoint('./model/temp_trained_25.h5', monitor='val_bit_err',
                                               verbose=0, save_best_only=True, mode='min', save_weights_only=True)
        history = model.fit_generator(
            training_gen(1000, 25),
            steps_per_epoch=50,
            epochs=500,
            validation_data=validation_gen(1000, 25),
            validation_steps=1,
            callbacks=[checkpoint],
            verbose=2)
    else:
        model.load_weights('./model/temp_trained_25.h5')
        BER = []  # Bit-error-rate

        # save h_predict
        if args.save_h_pred == True:
            h_layer = Model(inputs=model.input,
                            outputs=model.get_layer('h_predict_output').output)  # output result of h_pred
            h_output = h_layer.predict_generator(test_gen(100, 10, save_data=True), steps=1)
            sio.savemat('./matlab_draw_plot/h_predict_freq.mat', {'h_p': h_output})

        # The performance of differ SNRS
        for SNR in range(5, 30, 5):
            y = model.evaluate_generator(  # model.evaluate_generator return loss and metrics you choose
                test_gen(10000, SNR, save_data=False),  # The network metrics including loss and bit_err
                steps=1)
            BER.append(y[1])  # y[0]:loss y[1]:bit_error
            print(y)
        print(BER)

        if args.plot == True:
            plt.plot(np.arange(5, 30, 5), BER)
            plt.xlabel('SNR')
            plt.ylabel('BER')
            plt.title('BER-SNR')
        BER_matlab = np.array(BER)
        sio.savemat('BER1.mat', {'BER': BER_matlab})
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ComNet')
    parser.add_argument('--train_flag',default=False,type=str2bool,help="whether to train")
    parser.add_argument('--save_h_pred', default=False, type=str2bool, help="whether save prediction of channel response")
    parser.add_argument('--plot', default=True, type=str2bool, help="whether to plot bit error-SNR curve")
    args=parser.parse_args()
    model(args)
