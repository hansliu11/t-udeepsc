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
        ret = torch.view_as_complex(input)
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
    
    signal_power = torch.sum(torch.abs(signal ** 2), dim=-1, keepdim=True)
    power = torch.sqrt(K * power_constraint / signal_power)
    signal = signal * power
    signal = torch.reshape(signal, (-1, *dim, signal_shape[-1]))
    
    return signal, power

def power_normlize_stack(signal: torch.Tensor, power_constraint_per_complex_symbol: torch.FloatTensor=torch.FloatTensor([1])):
    """
            For signal that stack before normalized
            Args:
                signal: real tensor of shape (batch_size, n_user, *dim, symbol_dim)
                power_constraint: float
            Returns:
                The same as signal, but power normalized.
            
    """
    signal_shape = signal.shape
    dim = tuple(signal.size()[2:-1])
    batchsize, n_user, num_elements = signal_shape[0], signal_shape[1], signal_shape[-1]
    signal = torch.flatten(signal, start_dim=2)
    
    power_constraint = power_constraint_per_complex_symbol.clone().detach().to(signal.device)  # (user,)
    power_constraint = power_constraint.view(1, n_user, 1)
    power_constraint = power_constraint.expand(batchsize, n_user, 1)
    
    K = num_elements // 2 # number of complex symbols
    
    signal_power = torch.sum(torch.abs(signal ** 2), dim=-1, keepdim=True)
    power = torch.sqrt(K * power_constraint / signal_power)
    signal = signal * power
    signal = torch.reshape(signal, (-1, n_user, *dim, signal_shape[-1]))
    
    return signal

def signal_power(signal: torch.Tensor):
    """
        Args:
            signal: real or complex tensor of shape (*batch_size, signal_length)
        Returns:
            real tensor of shape (*batch_size)
    """
    return torch.sum(torch.abs(signal ** 2), dim=-1)

class SingleChannel():
    def interfere(Tx_sig: torch.Tensor, SNRdb: float) -> torch.Tensor:
        """
            Args:
                signal: a complex tensor with any shape,
                        will treat the last dimension as one signal (i.e., a list of complex numbers)
            Return:
                the signal with interference
        """
        raise NotImplementedError()
    
class AWGNSingleChannel(SingleChannel):
    @staticmethod
    def make_awgn_noise(Tx_sig: torch.Tensor, SNRdb: float, ouput_power=False):
        '''
            make AWGN noise for a signal (without applying the signal on it)
            the noise for a signal will be n ~ CN(0, sigma^2 * I),
            where CN is circularly symmetric complex Gaussian distribution

            Args:
                Tx_sig: complex tensor with any shape
                SNRdb: signal-to-noise ratio in decibel
                        float (same shape and device as signal)
            Return:
                the complex noise that can be applied onto it
        '''
        device = Tx_sig.device
        signal_power = torch.mean(Tx_sig.abs().detach()**2, dim=-1, keepdim=True)
        snr_linear = 10 ** (SNRdb / 10.0)
        sigma2 = signal_power / snr_linear # calculate noise power based on signal power and SNR
        if ouput_power:
            print ("TX Signal power: %.4f. Noise power: %.4f" % (signal_power, sigma2))     

        noise_real = torch.randn_like(Tx_sig.real) * torch.sqrt(sigma2 / 2)
        noise_imag = torch.randn_like(Tx_sig.imag) * torch.sqrt(sigma2 / 2)
        noise = torch.complex(noise_real, noise_imag).to(device)
        return noise
    
    def interfere(self, Tx_sig: torch.Tensor, SNRdb: float) -> torch.Tensor:
        """
            Args:
                Tx_sig: complex tensor with any shape
                SNRdb: signal-to-noise ratio in decibel
                        can be float (same shape and device as signal)
            Return:
                the signal with interference
        """
        return Tx_sig + AWGNSingleChannel.make_awgn_noise(Tx_sig, SNRdb)
    
class RayleighFadingSingleChannel(SingleChannel):
    @staticmethod
    def make_channel_gain_from_var(signal: torch.Tensor, channel_gain_var: float):
        channel_gain = torch.complex(
            torch.randn_like(signal.real) * math.sqrt(channel_gain_var / 2),
            torch.randn_like(signal.real) * math.sqrt(channel_gain_var / 2)
        )
        return channel_gain
    
    @staticmethod
    def make_channel_gain_from_var_tensor(channel_gain_var: torch.Tensor):
        channel_gain = torch.complex(
            torch.randn_like(channel_gain_var.float()) * torch.sqrt(channel_gain_var / 2),
            torch.randn_like(channel_gain_var.float()) * torch.sqrt(channel_gain_var / 2)
        )
        return channel_gain
    
    def interfere(self, signal, SNRdb, channel_gain_var: float | torch.Tensor = None):
        # slow fading -> all symbol use same channel gain
        """
            Args:
                signal: a complex tensor with any shape,
                        will treat the last dimension as one signal (i.e., a list of complex numbers)
            Return:
                the signal with interference
                if self.divide_gain, the return signal will be divided by channel gain
        """
        # channel_gain = CN(0, channel_gain_var)
        channel_gain = RayleighFadingSingleChannel.make_channel_gain_from_var(signal, channel_gain_var)
        signal = signal * channel_gain

        noise = AWGNSingleChannel.make_awgn_noise(signal, SNRdb)
        signal = signal + noise
        signal /= channel_gain

        return signal

class MultiChannel:
    def interfere(signal: torch.Tensor, user_dim_index: int) -> torch.Tensor:
        """
            Args:
                signal: a complex tensor representing the signal from multiple transmitters.
                        with shape (*dim1, user_dim, *dim2, symbol_dim):
                        - user_dim: for users, indexed by user_dim_index
                        - symbol_dim: for signal symbols
            Return:
                the signal with interference for receiver.
                will have dimension (*dim2, user_dim, *dim2, symbol_dim)
                - user_dim: for users, indexed by user_dim_index
        """
        raise NotImplementedError()

class AWGNMultiChannel(MultiChannel):
    """
        AWGN channel
    """
    
    def interfere(self, Tx_sig: torch.Tensor, SNRdb: float, user_dim_index: int) -> torch.Tensor:
        """
            AWGN channel for multiple users
            Superimpose signal before adding noise
        """
        dim = tuple(Tx_sig.size()[user_dim_index + 1:-1])
        batch_size = Tx_sig.size()[0]
        symbol_dim = Tx_sig.size()[-1]
        
        # make superimposed signal
        Tx_sig = torch.sum(Tx_sig, dim=user_dim_index) 
        Tx_sig = Tx_sig.view(batch_size, 1, *dim, symbol_dim)
        noise = AWGNSingleChannel.make_awgn_noise(Tx_sig, SNRdb)

        # add noise (wrt receiver)
        Tx_sig = Tx_sig + noise

        return Tx_sig

class FadingMultiChannel():
    @staticmethod
    def _get_diagonal(input: torch.Tensor, first_index: int):
        """
            Get diagonal while maintaining position
            
            Args:
                tensor: shape (*dim1, a_1, a_2, *dim2), a_1 is indexed by `first_index`, a_1 == a_2 := a
            Returns:
                tensor: shape (*dim1, a, *dim2), the diagonal of dimension a_1 and a_2
        """
        dim1 = tuple(input.size()[:first_index])
        # get channel gain: (*dim1, *dim2, a)
        input = input.diagonal(0, first_index, first_index+1)
        # change it back to (*dim1, a, *dim2)
        dim_len = len(input.size())
        dims = list(range(len(dim1))) + [dim_len - 1] + list(range(len(dim1), dim_len-1))
        input = torch.permute(input, dims)

        return input

    def _make_channel_gain(self, signal: torch.Tensor, user_dim_index: int, channel_gain_var: float | torch.Tensor = None) -> torch.Tensor:
        """
            Make channel gain for the given signal.
            
            Args:
                signal: a complex tensor representing the signals from multiple transmitter (user).
                        with shape (*dim1, transmitter_dim = n_tx, *dim2, symbol_dim).
                        dim1 and dim2 is treated as batch dimensions.
                user_dim_index: the index of transmitter_dim in the signal tensor
            Return:
                channel_gain: the channel gain for each transmitter to the receiver.
                a complex tensor with size (*dim1, transmitter_dim = n_tx, receiver_dim = n_rx, *dim2, symbol_dim)
        """
        raise NotImplementedError()
    
    def _make_noise(self, signal: torch.Tensor, user_dim_index: int, SNRdb: float | torch.Tensor) -> torch.Tensor:
        """
            Make noise for the given signal.
            The signal is already multiplied by the channel gain (h*x)
            
            Args:
                signal: a complex tensor representing the signals from multiple transmitter (user).
                        with shape (*dim1, transmitter_dim = n_tx, *dim2, symbol_dim).
                        dim1 and dim2 is treated as batch dimensions.
                user_dim_index: the index of transmitter_dim in the signal tensor
            Return:
                noise: the noise for the received signal for each transmitter.
                a complex tensor with size (*dim1, receiver_dim = n_rx, *dim2, symbol_dim)

                this is defined as:
                noise[*dim1, r, *dim2, k] := n^k_{r}

                do NOT do anything to the signal: it is here to give information about what we're
                transmitting (e.g., the signal symbol count, etc)
            
            NOTE: you can get the noise n used from _get_noise()
        """
        raise NotImplementedError()

    def interfere(self, signal: torch.Tensor, user_dim_index: int, SNRdb: float | torch.Tensor, noise_tensor: torch.Tensor = None, channel_gain_var: list[list[float]] | torch.Tensor = None, channel_gain_tensor: torch.Tensor = None, interfere_mode: Literal['all', 'self'] = 'all') -> torch.Tensor:
        """
            Args:
                signal: a complex tensor representing the signals from multiple transmitter (user).
                        with shape (*dim1, n_tx, *dim2, symbol_dim):
                        - n_tx: for transmitters, indexed by user_dim_index
                        - symbol_dim: for signal symbols
                user_dim_index: the index of transmitter_dim in the signal tensor
                noise_tensor: the noise tensor to be used in the interference,
                              for experiment reproducibility purposes: in case if you have previous
                              noise tensor that you wanna apply this time.
                              if None (default), will generate random noise tensor with the parameters
                              of this channel.
                channel_gain_tensor: same way as noise_tensor works
                
                For interference mode:
                    'all': superimpose all signals, every receiver gets the same signal (before channel estimation)
                        this is the default for all cases (even including n_tx != n_rx)
                    'self': only works when n_tx == n_rx
                            receiver r only gets signal from transmitter r (effectively orthogonal multiple access, e.g., TDMA)
                            i.e., y^k_{r} = h^k_{r, r} * x^k_{r} + n^k_{r} 

            Return:
                the signal with interference and superposition for the receiver
                will have dimension (*dim1, 1, *dim2, symbol_dim)
        """

        self.n_tx = signal.size()[user_dim_index]

        if isinstance(channel_gain_var, torch.Tensor):
            channel_gain_var = channel_gain_var.clone().detach()
        elif isinstance(channel_gain_var, List):
            channel_gain_var = torch.tensor(channel_gain_var)
        else:
            channel_gain_var = torch.full((self.n_tx, 1), 1.0)

        if channel_gain_var.size() != (self.n_tx, 1):
            raise Exception(f'{channel_gain_var.size() = } invalid ({self.n_tx = })')
        
        dim1 = tuple(signal.size()[:user_dim_index])
        dim2 = tuple(signal.size()[user_dim_index+1:-1])
        symbol_dim = signal.size()[-1]

        # make channel gain: (*dim1, transmitter_dim, receiver_dim, *dim2, symbol_dim)
        if channel_gain_tensor is not None:
            channel_gain = channel_gain_tensor.clone().detach().to(signal.device)
        else:
            channel_gain = self._make_channel_gain(signal, user_dim_index, channel_gain_var).to(signal.device)

        self.channel_gain = channel_gain.clone().detach().to('cpu')
        # expand signal to (*dim1, transmitter_dim, receiver_dim, *dim2, symbol_dim)
        signal = signal.view(*dim1, self.n_tx, 1, *dim2, symbol_dim)
        signal = signal.expand_as(channel_gain)

        # make superimposed signal
        signal = signal * channel_gain

        if interfere_mode == 'all':
            signal = torch.sum(signal, dim=user_dim_index)
        elif interfere_mode == 'self':
            signal = self._get_diagonal(signal, user_dim_index)

        # add noise        
        if noise_tensor is not None:
            noise = noise_tensor.clone().detach().to(signal.device)
        else:
            noise = self._make_noise(signal, user_dim_index, SNRdb).to(signal.device)

        self.noise = noise.clone().detach().to('cpu')
        signal = signal + noise # prevent using +=, which could cause inplace error

        if interfere_mode == 'self':
            if self.n_tx == 1:
                channel_gain = channel_gain.squeeze(dim=user_dim_index)
            else:
                channel_gain = channel_gain.squeeze(dim=user_dim_index+1)

            signal = signal / channel_gain
        
        return signal
    
    def get_channel_gain(self) -> torch.Tensor:
        """
            Return the random channel gain used in the last interfere()
            Note that if you give channel_gain_tensor parameter in the last interfere(), 
            then this function will return that tensor
            Do note that this tensor is on CPU due to not wanting to occupy GPU memory
            
            Return:
                complex tensor of size (*dim1, self.n_tx, self.n_rx, *dim2, symbol_dim)
                where dim1, dim2, symbol_dim is determined by the signal used in the last interfere()

                if we only focus on the dimension (self.n_tx, self.n_rx, symbol_dim),
                get_channel_gain()[i, j, k] := h^k_{i, j}
        """
        return self.channel_gain

    def get_noise(self) -> torch.Tensor:
        """
            Roughly the same as get_channel_gain()
        """
        return self.noise
    

class RayleighFadingMultiChannel(FadingMultiChannel):
    """
        Rayleigh channel implementation
        h^k_{t, r} (C^{1x1}) = path_loss_factor_{r, t} * CN(0, channel_gain_var)
        where:
            path_loss_factor_{r, t}: the path loss factor between Transmitter t and Receiver r
                        - in U-DeepSC with SIC, this is sqrt(PL(d^s_i)
                        - in U-DeepSC, this is 1
    """
    def __init__(self, 
                 noise_power_density_dBm: torch.Tensor | float, 
                 reference_distance: float,
                 reference_path_loss: float,
                 path_loss_exponent: float,
                 distance: torch.Tensor | float
                 ):
        """
            Args:
                For the below arguments, if specified as `torch.Tensor | type`,
                it means you can specify each element by giving a tensor or fill every element with the same number.

                noise_power_density_dBm (torch.Tensor | float):
                    The power density N_0 of the noise, in dBm. a float tensor with shape (n_rx,)
                reference_distance (float): 
                    The reference distance of the reference path loss d_0
                reference_path_loss (float): 
                    The reference path loss \rho_0, equals PL(d_0) by definition
                path_loss_exponent (float): 
                    the exponent for the path loss distance ratio l.
                distance (torch.Tensor | float): 
                    the distance of each transmitter-receiver pair. a float tensor with shape (n_tx, n_rx)
                    unit same as reference_distance
        """
        def fill_default_value(name, value, size):
            if isinstance(value, torch.Tensor):
                if value.size() != size:
                    raise Exception(f'{name}: size {value.size() = } invalid (desired size = {size})')
                return value.clone().detach().to('cpu')
            return torch.full(size, value)
        
        self.n_rx = 1
        self.noise_power_density_dBm = fill_default_value('noise_power_density_dBm', noise_power_density_dBm, (self.n_rx,))
        # self.channel_gain_var = fill_default_value('channel_gain_var', channel_gain_var, (n_tx, n_rx))
        self.reference_distance = reference_distance
        self.reference_path_loss = reference_path_loss
        self.distance = distance
        self.path_loss_exponent = path_loss_exponent

    def get_signal_power_constraint(self, signal: torch.Tensor, SNRdB: torch.Tensor, user_dim_index: int, channel_gain_var: list[list[float]] | torch.Tensor = None)-> torch.Tensor:
        '''
            Get signal power constraint given SNRdB and noise power
            Final transmitted signal power = P_i * |h_i|^2 = SNR * noise power
             
            Args:
                signal: a complex tensor representing the signals from multiple transmitter (user).
                SNRdB: the SNR in dB
                user_dim_index: the index of transmitter_dim in the signal tensor
            
            Return:
                The power constraint for each transmitter will have dimension (n_tx, )
        '''
        
        # SNRdB = torch.tensor(SNRdB)

        dim1 = tuple(signal.size()[:user_dim_index])
        dim2 = tuple(signal.size()[user_dim_index+1:-1])
        dim1_1 = (0,) * len(dim1)
        dim2_1 = (0,) * len(dim2)

        # Convert noise power from dBm to watts
        noise_power_density = 10 ** (self.noise_power_density_dBm / 10) * 1e-3 
        
        snr_linear = 10 ** (SNRdB / 10.0)

        # Get signal power with channel gain from noise power and SNR
        signal_power_with_h = (snr_linear * noise_power_density).to(signal.device)
        channel_gain = self._make_channel_gain(signal, user_dim_index, channel_gain_var).to(signal.device)

        # save channel gain for later use
        self.channel_gain = channel_gain.clone().detach().to('cpu')

        # channel gain: (*dim1, transmitter_dim, receiver_dim, *dim2, symbol_dim)
        # reduce channel gain dimension to (n_tx, n_rx)
        index = (
            *dim1_1,            # dim1 singleton indices
            slice(None),        # :
            slice(None),        # :
            *dim2_1,            # dim2 singleton indices
            0                   # symbol dim
        )
        reduced_channel_gain = channel_gain[index]
        
        # make signal power constraint from channel gain
        h2 = reduced_channel_gain.abs().detach() ** 2
        power_constraint = signal_power_with_h / h2
        power_constraint = power_constraint.view(self.n_tx,)

        return power_constraint

    def _make_channel_gain(self, signal: torch.Tensor, user_dim_index: int, channel_gain_var: list[list[float]] | torch.Tensor = None) -> torch.Tensor:
        # slow fading -> all symbol use same channel gain
        self.n_tx = signal.size()[user_dim_index]
        dim1 = tuple(signal.size()[:user_dim_index])
        dim2 = tuple(signal.size()[user_dim_index+1:-1])
        symbol_dim = signal.size()[-1]
        dim1_1 = (1,) * len(dim1)
        dim2_1 = (1,) * len(dim2)

        # make channel gain by expanding the given gain variance in user dimension
        # channel gain: (*dim1, transmitter_dim, receiver_dim, *dim2, symbol_dim)
        channel_gain_var = torch.tensor(channel_gain_var)
        cg_dim = channel_gain_var.size()  # (n_tx, n_rx)
        channel_gain_var = channel_gain_var.clone().detach().to(signal.device)
        channel_gain_var = channel_gain_var.view(*dim1_1, *cg_dim, *dim2_1, 1)
        channel_gain_var = channel_gain_var.expand(*dim1, *cg_dim, *dim2, 1)
        
        # make channel gain and then expand symbol_dim to make each symbol have the same channel gain
        NLOS_channel_gain = RayleighFadingSingleChannel.make_channel_gain_from_var_tensor(channel_gain_var)
        NLOS_channel_gain = NLOS_channel_gain.expand(*dim1, *cg_dim, *dim2, symbol_dim)

        # make path loss factor sqrt(PL(d_i))
        path_loss_factor = torch.sqrt(
            self.reference_path_loss * torch.pow(self.distance / self.reference_distance, -self.path_loss_exponent)
        ).to(signal.device)
        path_loss_factor = path_loss_factor.view(*dim1_1, self.n_tx, self.n_rx, *dim2_1, 1)
        path_loss_factor = path_loss_factor.expand(*dim1, self.n_tx, self.n_rx, *dim2, symbol_dim)

        # combine all together
        channel_gain = path_loss_factor * NLOS_channel_gain
        
        return channel_gain
    
    def _make_noise(self, signal: torch.Tensor, user_dim_index: int, SNRdb: list[float] | torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(SNRdb):
            SNRdb = torch.Tensor(SNRdb)

        dim1 = tuple(signal.size()[:user_dim_index])
        dim2 = tuple(signal.size()[user_dim_index+1:-1])
        symbol_dim = signal.size()[-1]
        dim1_1 = (1,) * len(dim1)
        dim2_1 = (1,) * len(dim2)

        # Convert noise power from dBm to watts
        noise_power_density = 10 ** (self.noise_power_density_dBm / 10) * 1e-3  

        # convert it to the same size as signal
        noise_power_density = noise_power_density.view(*dim1_1, self.n_rx, *dim2_1, 1)
        noise_power_density = noise_power_density.expand(*dim1, self.n_rx, *dim2, symbol_dim)

        # Generate complex Gaussian noise: CN(0, sigma^2 * I)
        noise_real = torch.randn_like(noise_power_density) * torch.sqrt(noise_power_density / 2)
        noise_imag = torch.randn_like(noise_power_density) * torch.sqrt(noise_power_density / 2)
        noise = torch.complex(noise_real, noise_imag)

        # snr_dim = SNRdb.size()    # == (n_rx,)
        # snr_db = SNRdb.clone().detach().to(signal.device)
        # snr_db = snr_db.view(*dim1_1, *snr_dim, *dim2_1, 1)
        # snr_db = snr_db.expand(*dim1, *snr_dim, *dim2, symbol_dim)
        # noise = AWGNSingleChannel.make_awgn_noise(signal, SNRdb)

        return noise


def test_channel_class():
    settings = []
    for n_user in [3]:
        for interference_mode in ['all', 'self']:
            settings.append((n_user, interference_mode))

    def Uplink():
        for n_user, interference_mode in settings:
            print(toColor(f'{n_user = }, {interference_mode = }', 'yellow'))
            try:  
                channel = RayleighFadingMultiChannel()
                input = torch.randn(10, 20, n_user, 30, 40, dtype=torch.complex64)
                print(f'    input = {str_type(input)}')
                output = channel.interfere(input, 2, 10, None, [[1.0]] * n_user, None, interference_mode)
                print(f'    output = {str_type(output)}')
                print(f'    channel_gain = {str_type(channel.get_channel_gain())}')
            except Exception as e:
                print(toColor(f'    Exception: {e}', 'cyan'))

    Uplink()

if __name__ == '__main__':
    test_channel_class()