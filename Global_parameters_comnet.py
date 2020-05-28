import os
import scipy.io as sio
import numpy as np
K = 64#length of symbol after modulation
CP = K // 4
P = 64#num of pilots
allCarriers = np.arange(K)  # indices of all subcarriers position([0, 1, ... K-1])
pilotCarriers = allCarriers # one pilots after subcarriers

mu = 2
payloadBits_per_OFDM = K * mu
SNRdb = 25
H_folder_train = '../H_dataset/Train/'
H_folder_test = '../H_dataset/Test/'
n_hidden_1 = 256#1st layer num features
n_hidden_2 = 250  # 2st layer num features
n_hidden_3 = 120  # 3nd layer num features
n_output = 16  # every 16 bit are predicted by a model
n_h=128 # length of the output h_^

n_BiLSTM_1=30
n_BiLSTM_2=20
n_BiLSTM_3=16


def Modulation(bits):
    bit_r = bits.reshape((int(len(bits) / mu), mu))
    # This is just for QPSK modulation
    return np.sqrt(2)*(2 * bit_r[:, 0] - 1) + np.sqrt(2)*1j * (2 * bit_r[:, 1] - 1)

def OFDM_symbol(Data, pilot_flag):
    symbol = np.zeros(K, dtype=complex)  # the overall K subcarriers
    #symbol = np.zeros(K)
    symbol[pilotCarriers] = pilotValue  # allocate the pilot subcarriers
    return symbol


def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)


def addCP(OFDM_time):
    cp = OFDM_time[-CP:]               # take the last CP samples ...
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning


def channel(signal, channelResponse, SNRdb):
    convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(abs(convolved**2))
    sigma2 = signal_power * 10**(-SNRdb / 10)
    noise = np.sqrt(sigma2 / 2) * (np.random.randn(*
                                                   convolved.shape) + 1j * np.random.randn(*convolved.shape))
    return convolved + noise


def removeCP(signal):
    return signal[CP:(CP + K)]


def DFT(OFDM_RX):
    return np.fft.fft(OFDM_RX)


def ofdm_simulate(codeword, channelResponse, SNRdb):
    """
    :param codeword: data to modulation
    """
    # ----- Calculate h_ls using pilots   ---
    OFDM_data = np.zeros(K, dtype=complex)
    OFDM_data[pilotCarriers] = pilotValue#put pilot at position [0:64]
    OFDM_time = IDFT(OFDM_data)
    OFDM_withCP = addCP(OFDM_time)
    OFDM_TX = OFDM_withCP
    OFDM_RX = channel(OFDM_TX, channelResponse, SNRdb)
    OFDM_RX_noCP = removeCP(OFDM_RX)
    OFDM_RX_noCP = DFT(OFDM_RX_noCP)
    h_ls=OFDM_RX_noCP/pilotValue#LS estimation

    # ----- target inputs ---
    symbol = np.zeros(K, dtype=complex)
    codeword_qam = Modulation(codeword)#modualte 128 data bits data into 64 symbol
    symbol[np.arange(K)] = codeword_qam
    OFDM_data_codeword = symbol
    OFDM_time_codeword = np.fft.ifft(OFDM_data_codeword)
    OFDM_withCP_cordword = addCP(OFDM_time_codeword)
    OFDM_RX_codeword = channel(OFDM_withCP_cordword, channelResponse, SNRdb)
    OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword)
    OFDM_RX_noCP_codeword = DFT(OFDM_RX_noCP_codeword)
    return (np.concatenate((np.real(h_ls), np.imag(h_ls))),
            np.concatenate((np.real(OFDM_RX_noCP_codeword), np.imag(OFDM_RX_noCP_codeword)))
            , abs(channelResponse))


Pilot_file_name = 'Pilot_' + str(P)
if os.path.isfile(Pilot_file_name):
    print('Load Training Pilots txt')
    # load file
    bits = np.loadtxt(Pilot_file_name, delimiter=',')
else:
    # write file
    bits = np.random.binomial(n=1, p=0.5, size=(K * mu, ))#randomly generate data via binomal distribution
    np.savetxt(Pilot_file_name, bits, delimiter=',')


pilotValue = Modulation(bits)#modulate data into pilots
