import librosa
import numpy as np
import scipy.signal as signal
from scipy.fftpack import fft
import bz2
import pickle

def powerlaw_quant(x, mu):
    return np.sign(x)*np.log(1 + mu*np.abs(x))/np.log(1 + mu)

def inv_powerlaw_quant(y, mu):
    return np.sign(y)*((mu + 1)**np.abs(y) - 1)/mu

def ath(f):
    return 3.64*np.float_power(f/1000, -0.8) - \
        6.5*np.exp(-0.6*np.power(f/1000 - 3.3, 2)) + \
        np.float_power(10, -3)*np.power(f/1000, 4)

def bark(f):
    return 13*np.arctan(0.00076*f) + 3.5*np.arctan((f/7500)**2)

class QuantizedFrame(object):
    def __init__(self, data=np.array([], dtype='float16')):
        self.data = data
        
class EncodedSignal(object):
    def __init__(self):
        self.data = None
        self.bit_allocation_table = None
        self.num_blocks = 0
    
npoints = 1152
nbands = 32
window_length = 64
window_function = np.sin(np.pi / 2 * \
                       np.power(np.sin(np.pi / window_length * \
                       np.arange(0.5, window_length + 0.5)), 2))

scale_table = np.array([19, 19, 19, 19, 19, 21, 21, 21, 32, 20, 23, \
                   23, 23, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, \
                   32, 32, 32, 32, 32, 32], dtype='uint8')


def encode(x):
    
    ### DISCLAIMER: Implementation based on paper written by JOEBERT S. JACABA, Dept. Mathematics, College of Science, 
    ### University of The Phillipines, Diliman, Quezon City: AUDIO COMPRESSION USING MODIFIED DISCRETE COSINE TRANSFORM:
    ### THE MP3CODING STANDARD. Available at: https://www.mp3-tech.org/programmer/docs/jacaba_main.pdf
    def psyacoustic_analysis(frame):
        wl = window_length
        x_fr_stft = librosa.stft(frame.astype('float'), wl, wl//2, window=signal.hanning)
        x_fr_stft2 = x_fr_stft**2
        P_k = 2.0 * x_fr_stft2[:(wl // 2 + 1)] / wl ** 2
        P_k[0] = 0.5 * P_k[0]
        if (wl % 2 == 0):
            P_k[-1] = 0.5 * P_k[-1]
        P_k = 10.0 ** (96.0 / 10.0) * P_k
        P_k = 10.0 * np.log10(P_k)

        sr = 44100
        dt = 1.0 / sr
        k = np.arange(wl // 2 + 1)
        f_k = k * sr / wl
        b_k = bark(f_k)
        ath_k = 10 ** (ath(f_k)/10)

        ath_k.shape = (len(ath_k),1)
        coeffs_bool = (P_k - np.repeat(ath_k, P_k.shape[1], axis=1) <= 0)

        P_tonal = np.zeros(P_k.shape)
        k_tonal = []
        js = np.array([-2, +2])
        P_nontonal = np.copy(P_k)
        for col in range(0, P_k.shape[1]):
            for row in range(3, P_k.shape[0]-3):
                if P_k[row,col] >= P_k[row-1,col] and P_k[row,col] >= P_k[row+1,col]:
                    if np.all(P_k[row,col] - P_k[row+js,col] >= 7):
                        P_tonal[row,col] = P_k[row-1,col] + P_k[row,col] + P_k[row+1,col]
                        k_tonal.append((row, col))
                        P_nontonal[row-1,col] = P_nontonal[row,col] = P_nontonal[row+1,col] = 0.0

        decimated_tonal_k = np.zeros(P_k.shape)
        for i in range(len(k_tonal)-1):
            row, col = k_tonal[i]
            next_row, next_col = k_tonal[i+1]
            currP = P_tonal[row, col]
            nextP = P_tonal[next_row, next_col]

            if abs(b_k[row] - b_k[next_row]) < 1:
                if currP >= nextP:
                    coeffs_bool[next_row, next_col] = False
                else:
                    coeffs_bool[row, col] = False

        cb_k = np.array([int(b) for b in np.floor(b_k)])
        crit_bands = {}
        for col in range(0, P_k.shape[0]):
            crit_bands.clear
            for row in range(0, P_k.shape[0]):
                if cb_k[row] not in crit_bands:
                    crit_bands[cb_k[row]] = [(P_nontonal[row, col], row)]
                else:
                    crit_bands[cb_k[row]].append((P_nontonal[row, col], row))

            for band in crit_bands.keys():
                P_sum = sum(i for i, _ in crit_bands[band])
                avg_row = int(round(sum(i for _, i in crit_bands[band])/len(crit_bands[band])))
                coeffs_bool[avg_row, col] = False


        return coeffs_bool[1:,:]
    
    def filter_banks(x):
        coeff_bool = psyacoustic_analysis(x)

        x_bands = mdct(x, window_function)
        coeff_bool = np.pad(coeff_bool, [(0,0), (0, x_bands.shape[1] - coeff_bool.shape[1])], \
                            mode='constant', constant_values=1)
        x_bands[coeff_bool] = 0

        return x_bands

    x_bands = filter_banks(x)
    num_blocks = x_bands.shape[1]
    
    def quantization(x_bands):
        max_bits = np.max(scale_table)
        quantized_signal = []

        quantfr = QuantizedFrame()
        for j in range(0, x_bands.shape[0]):
            bit_alloc = scale_table[j]
            quantized_mdct = powerlaw_quant(x_bands[j], 255)
            quantized_mdct = quantized_mdct/(2**bit_alloc)
            quantized_mdct = quantized_mdct.astype('float16')
            quantfr.data = np.append(quantfr.data, quantized_mdct)

        quantized_signal.append(quantfr)

        return quantized_signal
    
    quantized_signal = quantization(x_bands)
    

    def encoding(quantized_signal):
        encoded_signal = EncodedSignal()
        encoded_signal.data = quantized_signal
        encoded_signal.scale_table = scale_table
        encoded_signal.num_blocks = num_blocks
        i_str = pickle.dumps(encoded_signal)
        compressed_signal = bz2.compress(i_str)

        return compressed_signal

    compressed_signal = encoding(quantized_signal)
    
    return compressed_signal


def decode(encoded_signal):
    decompressed_signal = pickle.loads(bz2.decompress(encoded_signal))
    mdct_per_frame = decompressed_signal.num_blocks
    decoded_signal = np.array([])

    max_bits = np.max(decompressed_signal.scale_table)

    for i, frame in enumerate(decompressed_signal.data):
        decoded_mdct = np.zeros((nbands, mdct_per_frame))
        for j in range(0, len(decompressed_signal.scale_table)):
            scale_factor = decompressed_signal.scale_table[j]
            start = j*mdct_per_frame
            end = (j+1)*mdct_per_frame
            encoded_mdct = frame.data[start:end]*(2**scale_factor)
            decoded_mdct[j] = inv_powerlaw_quant(encoded_mdct, 255)
        decoded_frame = imdct(decoded_mdct, window_function)
        decoded_signal = np.append(decoded_signal, decoded_frame)

    return decoded_signal.flatten().astype('int16')


### DISCLAIMER: MDCT4 & IMDCT4 implementation by Zafar Rafii, Ph.D. in Electrical Engineering and Computer Science from Northwestern University, Evanston, IL, USA. Available at: https://github.com/zafarrafii/Z/blob/master/z.py

def mdct(audio_signal, window_function):
    # Number of samples and window length
    number_samples = len(audio_signal)
    window_length = len(window_function)

    # Number of time frames
    number_times = int(np.ceil(2 * number_samples / window_length) + 1)

    # Pre and post zero-padding of the signal
    pre_pad = int(window_length / 2)
    post_pad = int((number_times + 1) * window_length / 2 - number_samples)
    audio_signal = np.pad(audio_signal, (pre_pad, post_pad), 'constant', constant_values=0)
    

    # Initialize the MDCT
    audio_mdct = np.zeros((int(window_length / 2), number_times))

    # Pre and post-processing arrays
    preprocessing_array = np.exp(-1j * np.pi / window_length * np.arange(0, window_length))
    postprocessing_array = np.exp(-1j * np.pi / window_length * (window_length / 2 + 1)
                                  * np.arange(0.5, window_length / 2 + 0.5))

    # Loop over the time frames
    for time_index in range(0, number_times):

        # Window the signal
        sample_index = time_index * int(window_length / 2)
        audio_segment = audio_signal[sample_index:sample_index + window_length] * window_function

        # FFT of the audio segment after pre-processing
        audio_segment = np.fft.fft(audio_segment * preprocessing_array)

        # Truncate to the first half before post-processing
        audio_mdct[:, time_index] = np.real(audio_segment[0:int(window_length / 2)] * postprocessing_array)

    return audio_mdct


def imdct(audio_mdct, window_function):
    # Number of frequency channels and time frames
    number_frequencies, number_times = np.shape(audio_mdct)

    # Number of samples for the signal
    number_samples = number_frequencies * (number_times + 1)

    # Initialize the audio signal
    audio_signal = np.zeros(number_samples)

    # Pre and post-processing arrays
    preprocessing_array = np.exp(-1j * np.pi / (2 * number_frequencies)
                                 * (number_frequencies + 1) * np.arange(0, number_frequencies))
    postprocessing_array = np.exp(-1j * np.pi / (2 * number_frequencies)
                                  * np.arange(0.5 + number_frequencies / 2,
                                              2 * number_frequencies + number_frequencies / 2 + 0.5)) \
        / number_frequencies

    # FFT of the frames after pre-processing
    audio_mdct = np.fft.fft(audio_mdct.T * preprocessing_array, n=2 * number_frequencies, axis=1)

    # Apply the window to the frames after post-processing
    audio_mdct = 2 * (np.real(audio_mdct * postprocessing_array) * window_function).T

    # Loop over the time frames
    for time_index in range(0, number_times):

        # Recover the signal thanks to the time-domain aliasing cancellation (TDAC) principle
        sample_index = time_index * number_frequencies
        audio_signal[sample_index:sample_index + 2 * number_frequencies] \
            = audio_signal[sample_index:sample_index + 2 * number_frequencies] + audio_mdct[:, time_index]

    # Remove the pre and post zero-padding
    audio_signal = audio_signal[number_frequencies:-number_frequencies - 1]

    return audio_signal