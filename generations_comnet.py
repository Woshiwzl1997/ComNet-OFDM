from Global_parameters_comnet import *
import scipy.io as sio

channel_train = np.load(r'.\datas\channel_train.npy')#shape(100w,16)
train_size = channel_train.shape[1]
channel_test = np.load(r'.\datas\channel_train.npy')
test_size = channel_test.shape[1]#shape(39w,16)


def training_gen(bs, SNRdb = 20):
    while True:
        index = np.random.choice(np.arange(train_size), size=bs)#choose bs daa from trainsets
        H_total = channel_train[index]
        input_hls = []
        input_data = []
        input_labels = []
        for H in H_total:
            s_bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))#randomly generate s_bits with length 128 bits
            h_ls,signal_output, para = ofdm_simulate(s_bits, H, SNRdb)
            input_labels.append(s_bits[0:16])#set formers_16bits of s_bits as label
            input_hls.append(h_ls)#add h_ls to input_1
            signal_output=signal_output.reshape((-1,2),order='F')#reshape(128,1) to (64,2) with column first
            input_data.append(signal_output)#add data passed through channel to input_2
        yield ({'input_1':np.asarray(input_hls),'input_2':np.asarray(input_data)}, {'model_output':np.asarray(input_labels)})


def validation_gen(bs, SNRdb = 20):
    while True:
        index = np.random.choice(np.arange(train_size), size=bs)
        H_total = channel_train[index]
        input_hls = []
        input_data = []
        input_labels = []
        all_bits=[]
        for H in H_total:
            bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))
            all_bits.append(bits)
            h_ls,signal_output, para = ofdm_simulate(bits, H, SNRdb)
            input_labels.append(bits[0:16])
            input_hls.append(h_ls)
            signal_output=signal_output.reshape((-1, 2),order='F')  # reshape(128,1) to (64,2),column first
            input_data.append(signal_output)# data passed through channel

        # sio.savemat('y_real+imag_{}.mat'.format(str(SNRdb)),{'y_SNR_{}'.format(str(SNRdb)):input_data})
        # sio.savemat('init_bits_{}.mat'.format(str(SNRdb)), {'bit_SNR_{}'.format(str(SNRdb)):all_bits})
        # sio.savemat('h_ls_real+imag_{}.mat'.format(str(SNRdb)), {'h_SNR_{}'.format(str(SNRdb)): input_hls})

        yield ({'input_1':np.asarray(input_hls),'input_2':np.asarray(input_data)}, {'model_output':np.asarray(input_labels)})

def test_gen(bs, SNRdb = 20,save_data=False):
    while True:
        index = np.random.choice(np.arange(test_size), size=bs)
        H_total = channel_train[index]
        if save_data==True:
            sio.savemat('./matlab_draw_plot/h_true_time.mat', {'h_t':H_total}) #save true h to compare
        input_hls = []
        input_data = []
        input_labels = []
        all_bits = []
        for H in H_total:
            bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))
            all_bits.append(bits)
            h_ls, signal_output, para = ofdm_simulate(bits, H, SNRdb)
            input_labels.append(bits[0:16])
            input_hls.append(h_ls)
            signal_output = signal_output.reshape((-1, 2),order='F')  # reshape(128,1) to (64,2)
            input_data.append(signal_output)
        if save_data == True:
            sio.savemat('./matlab_draw_plot/h_ls_draw.mat', {'h_ls': input_hls})
            sio.savemat('./matlab_draw_plot/y_real+imag_draw',{'y':input_data})
            sio.savemat('./matlab_draw_plot/init_bits_draw.mat', {'bit':all_bits})
        yield ({'input_1': np.asarray(input_hls), 'input_2': np.asarray(input_data)}, {'model_output': np.asarray(input_labels)})
