import math
import torch
import numpy as np
from utils import *
from typing import *

bit_per_symbol = 4 # bits per symbol (64QAM)
mapping_table = {
    (1,0,1,0) : -3-3j,
    (1,0,1,1) : -3-1j,
    (1,0,0,1) : -3+1j,
    (1,0,0,0) : -3+3j,
    (1,1,1,0) : -1-3j,
    (1,1,1,1) : -1-1j,
    (1,1,0,1) : -1+1j,
    (1,1,0,0) : -1+3j,
    (0,1,1,0) : 1-3j,
    (0,1,1,1) : 1-1j,
    (0,1,0,1) : 1+1j,
    (0,1,0,0) : 1+3j,
    (0,0,1,0) : 3-3j,
    (0,0,1,1) : 3-1j,
    (0,0,0,1) : 3+1j,
    (0,0,0,0) : 3+3j,
}
demapping_table = {v : k for k, v in mapping_table.items()}


def split(word): 
    return [char for char in word]

def group_bits(bitc):
    bity = []
    x = 0
    for i in range((len(bitc)//bit_per_symbol)):
        bity.append(bitc[x:x+bit_per_symbol])
        x = x+bit_per_symbol
    return bity

def tensor_real2complex(input: torch.Tensor, method: Literal['view', 'concat']) -> torch.Tensor:
    """
        Args:
            input: real tensor with shape (*, N), where N must be even. only supports floating precision tensors
            mode: the method of which the tensors are converted.
                  For example, given tensor [a, b, c, d]:
                  - view: [a + bi, c + di] (i.e., torch.view_as_complex())
                  - concat: [a + ci, b + di] (i.e., input = [Re(return), Im(return)])

        Returns:
            complex tensor with shape (*, N/2)
            will return a copy
    """
    if method == 'view':
        size = (*input.size()[:-1], input.size()[-1] // 2, 2)
        input = input.reshape(size).contiguous()
        ret = torch.view_as_complex_copy(input)
    else:   # concat
        half = input.shape[-1] // 2
        real_part = input[..., :half]
        imag_part = input[..., half:]
        ret = torch.complex(real_part, imag_part)
    return ret

def tensor_complex2real(input: torch.Tensor, method: Literal['view', 'concat']) -> torch.Tensor:
    """
        Args:
            input: complex tensor with shape (*, N)
            mode: the method of which the tensors are converted.
                  For example, given tensor [a + bi, c + di]:
                  - view: [a, b, c, d] (i.e., torch.view_as_complex())
                  - concat: [a, c, b, d] (i.e., return = [Re(input), Im(input)])

        Returns:
            real tensor with shape (*, N*2)
            will return a copy!
    """ 
    real_part = input.real      # size (*, N)
    imag_part = input.imag
    
    if method == 'view':
        # (*, N, 2) -> (*, N*2)
        return torch.stack([real_part, imag_part], dim=-1).view(*input.shape[:-1], -1)
    elif method == 'concat':
        return torch.cat([real_part, imag_part], dim=-1)

def channel(signal, SNRdb, ouput_power=False):
    signal_power = np.mean(abs(signal**2))
    sigma2 = signal_power * 10**(-SNRdb/10) # calculate noise power based on signal power and SNR
    if ouput_power:
        print ("RX Signal power: %.4f. Noise power: %.4f" % (signal_power, sigma2))     
    # Generate complex noise with given variance
    noise = np.sqrt(sigma2/2) * (np.random.randn(*signal.shape)+1j*np.random.randn(*signal.shape))
    return signal + noise
    
def channel_Rayleigh(signal, SNRdb, ouput_power=False):
    shape = signal.shape
    sigma = math.sqrt(1/2)
    H = np.random.normal(0.0, sigma , size=[1]) + 1j*np.random.normal(0.0, sigma, size=[1])
    Tx_sig = signal* H
  
    Rx_sig = channel(Tx_sig, SNRdb, ouput_power=False)
    # Channel estimation
    Rx_sig = Rx_sig / H
    return Rx_sig
  
def channel_Rician(signal, SNRdb, ouput_power=False,K=1):
    shape = signal.shape
    mean = math.sqrt(K / (K + 1))
    std = math.sqrt(1 / (K + 1))
    H = np.random.normal(mean, std , size=[1]) + 1j*np.random.normal(mean, std, size=[1])
    Tx_sig = signal* H
  
    Rx_sig = channel(Tx_sig, SNRdb, ouput_power=False)
    # Channel estimation
    Rx_sig = Rx_sig / H
    return Rx_sig
    
    
def Demapping(QAM):
    # array of possible constellation points
    constellation = np.array([x for x in demapping_table.keys()])

    # calculate distance of each RX point to each possible point
    dists = abs(QAM.reshape((-1,1)) - constellation.reshape((1,-1)))
    
    # for each element in QAM, choose the index in constellation that belongs to the nearest constellation point
    const_index = dists.argmin(axis=1)
    
    # get back the real constellation point
    hardDecision = constellation[const_index]
    
    # transform the constellation point into the bit groups
    return np.vstack([demapping_table[C] for C in hardDecision]), hardDecision


def transmit(data, SNRdb, bits_per_digit):
    TX_signal = data[:].cpu().numpy().flatten()
    Tx_data_binary = []

    # Produce binary data
    for i in TX_signal:
        Tx_data_binary.append('{0:b}'.format(i).zfill(bits_per_digit))
    Tx_data = []
    Tx_data_ready = []   
    for i in Tx_data_binary:
        Tx_data.append(split(i))
    img_for_trans1 = np.vstack(Tx_data)
    for i in img_for_trans1:
        for j in range(bits_per_digit):
            Tx_data_ready.append(int(i[j]))
    Tx_data_ready = np.array(Tx_data_ready)

    ori_len = len(Tx_data_ready)
    padding_len = ori_len

    if ori_len % 4 != 0: 
        padding_len = ori_len + (bit_per_symbol - (ori_len % bit_per_symbol))

    Whole_tx_data = np.zeros(padding_len,dtype=int)
    Whole_tx_data[:ori_len] = Tx_data_ready

    bit_group = group_bits(Whole_tx_data)
    bit_group = np.array(bit_group)

    # bit is mapped into the QAM symbols
    QAM_symbols = []
    for bits in bit_group:
        symbol = mapping_table[tuple(bits)]
        QAM_symbols.append(symbol)
    QAM_symbols = np.array(QAM_symbols) 

    Rx_symbols = channel(QAM_symbols, SNRdb)    #Pass the Guassian Channel 
    Rx_bits, hardDecision = Demapping(Rx_symbols)

    # Reconstruct the tx data by using the Rx bits
    data_rea = []
    Rx_long = Rx_bits.reshape(-1,)[0:ori_len]
    k = 0
    for i in range(Rx_long.shape[0]//bits_per_digit):
        data_rea.append(Rx_long[k:k+bits_per_digit])
        k+=bits_per_digit
    data_done = []
    for i in data_rea[:]:
        x = []
        for j in range(len(i)):
            x.append(str(i[j]))
        data_done.append(x)
    sep = ''
    data_fin = []
    for i in data_done:
        data_fin.append(sep.join(i))
    data_dec = []
    for i in data_fin:
        data_dec.append(i[0:bits_per_digit])
    data_dec = np.array(data_dec)
    data_back = []
    for i in range(len(Tx_data_binary)):
        data_back.append(int(data_dec[i],2))
    data_back = np.array(data_back)
    return data_back



def power_norm_batchwise(signal: torch.Tensor, power:float=1.0):
    """
            For forward().
            Args:
                signal: real tensor of shape (batch_size, *dim, symbol_dim)
                power_constraint: integer
            Returns:
                The same as signal, but power normalized.
            
            NOTE: 
                MUST USE ON REAL TENSOR!!!!!! it can't handle complex tensor
    """
    signal_shape = signal.shape
    dim = tuple(signal.size()[1:-1])
    signal = torch.flatten(signal, start_dim=1)
    batchsize , num_elements = signal.shape[0], signal.size()[-1]
    
    power_constraint = torch.full((batchsize, 1), power).to(signal.device) # (batch_size, 1)
    K = num_elements // 2 # number of complex symbols
    
    
    
    # # make as complex signals (batch_size, num_complex, 2)
    # signal = signal.view(batchsize, K, 2)
    # signal_power = torch.sum((signal[:,:,0]**2 + signal[:,:,1]**2), dim=-1) / K
    
    # signal = signal * math.sqrt(power) / torch.sqrt(signal_power.unsqueeze(-1).unsqueeze(-1))
    # # print("Signal after power normalize: " + str_type(signal))
    # signal = signal.view(signal_shape)
    
    power = torch.sqrt(K * power_constraint / torch.sum(signal ** 2, dim=1, keepdim=True))
    signal = signal * power
    signal = torch.reshape(signal, (-1, *dim, signal_shape[-1]))
    
    return signal, power