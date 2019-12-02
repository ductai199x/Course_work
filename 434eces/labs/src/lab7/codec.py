import librosa
import numpy as np
import scipy.signal as signal
from scipy.fftpack import fft
from PQMFB import *
from dahuffman import HuffmanCodec

class EncodedFrame(object):
    def __init__(self, scaling_factors=np.array([]), data=np.array([])):
        self.scaling_factors = scaling_factors
        self.data = data
        
class Classifier(object):
    def __init__(self):
        small  = np.array([-2, +2])
        medium = np.array([-3, -2, +2, +3]) 
        large  = np.array([-6, -5, -4, -3, -2, +2, +3, +4, +5, +6])
        xlarge  = np.linspace(-12, 12, 25, dtype='int')
        self.neighbourhood = 512 * [None]
        for _k in range(2, 63):
            self.neighbourhood[_k] = small
        for _k in range(63, 127):
            self.neighbourhood[_k] = medium
        for _k in range(127, 255):
            self.neighbourhood[_k] = large
        for _k in range(255, 501):
            self.neighbourhood[_k] = xlarge
    def __call__(self, k, P):
        k_t = []
        P_t = []
        for _k in np.arange(3, 501):
            if (P[_k-1] <= P[_k] and P[_k+1] <= P[_k]):
                js = self.neighbourhood[_k]
                if np.all(P[_k] - P[_k+js] >= 7):
                    k_t.append(_k)
                    P_t.append(P[_k-1] + P[_k] + P[_k+1])
                    P[_k-1] = P[_k] = P[_k+1] = 0.0
        return (np.array(k_t, dtype='uint16'), np.array(P_t)), (k, P)

    
npoints = 1152
nbands = 32
nsubbands = 18
window_length = 96
window_function = np.sin(np.pi / 2 * \
                       np.power(np.sin(np.pi / window_length * \
                       np.arange(0.5, window_length + 0.5)), 2))
num_blocks = 38

def absolute_threshold_hearing(f):
    return 3.64*np.float_power(f/1000, -0.8) - \
        6.5*np.exp(-0.6*np.power(f/1000 - 3.3, 2)) + \
        np.float_power(10, -3)*np.power(f/1000, 4)

def bark(f):
    return 13*np.arctan(0.00076*f) + 3.5*np.arctan((f/7500)**2)

sr = 44100
dt = 1.0 / sr
k = np.arange(npoints // 2 + 1)
f_k = k * sr / npoints
b_k = bark(f_k)

k_i = np.r_[1:49, 49:97:2, 97:255:4, 255:513:8]
f_i = k_i * sr / npoints
b_i = bark(f_i)
ATH_i = absolute_threshold_hearing(f_i)
subband_i = np.array([int(s) for s in np.round(f_i * nbands / (0.45 * sr) - 0.5)])
bit_allocation_table = np.array([10, 10, 10, 10, 9, 9, 9, 8, 8, 8, 8, \
                   8, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, \
                   4, 4, 4, 4, 4, 4])


def encode(x):

    
    def divide_into_frames(signal, npoints):
        # Find out how many frames are needed
        nwholeframes = np.floor(len(signal)/npoints).astype('int')
        nframes = np.ceil(len(signal)/npoints).astype('int')
        # Pad the signal so that its length is divisible by nframes
        padded = np.pad(signal, (0, npoints-len(signal)%nwholeframes), 'constant', constant_values=0.0)
        # Reshape padded signal into nframes rows, each row is one frame
        frames = padded.reshape((nframes, len(padded)//nframes))
        return frames
    x_frames = divide_into_frames(x, npoints)
    
    def filter_banks(frames, nbands, num_blocks):
        x_bands = np.zeros((nbands, frames.shape[0], num_blocks))
        for i in range(0, frames.shape[0]):
            x_bands[:,i,:] = PQMFBana(frames[i], nbands, window_function)

        return x_bands
    x_bands = filter_banks(x_frames, nbands, num_blocks)
    
    def SMR_from_frame(frame):
        frame = np.array(frame, copy=False)

        mask_i = 10.0 ** (ATH_i / 10.0)

        (k_t, P_t), (k_nt, P_nt) = maskers(frame, k)

        for masker_index in np.arange(len(k_t)):
            _b, _P = b_k[ k_t[masker_index] ], P_t[ masker_index ]
            ep = excitation_pattern(b_i, b_m=_b, I_m=10.0*np.log10(_P), tonal=True)
            mask_i += 10.0 ** (ep / 10.0)

        for masker_index in np.arange(len(k_nt)):
            _b, _P = b_k[k_nt[masker_index]], P_nt[masker_index]
            ep = excitation_pattern(b_i, b_m=_b, I_m=10.0*np.log10(_P), tonal=False) 
            mask_i += 10.0 ** (ep / 10.0)

        mask_i = 10.0 * np.log10(mask_i)

        subband_masks = [[] for _ in range(32)] 
        subband_MMT = np.zeros(32) # MMT = Minimum Masking Threshold
        subband_LSB = np.zeros(32) # LSB = Sound Pressure Level
        subband_SMR = np.zeros(32) # SMR = Signal To Mask Ratio

        for i, _mask_i in enumerate(mask_i):
            subband_masks[subband_i[i]].append(_mask_i)

        for i, _masks in enumerate(subband_masks):
            subband_MMT[i] = np.amin(_masks)

        P_k = P_k_from_frame(frame, signal.windows.hann, 1152)
        for i, _masks in enumerate(subband_MMT):
            max_scalefactor = 20*np.log10((2**bit_allocation_table[i]-1)*32768)-10
            subband_LSB[i] = np.maximum(np.max(P_k), max_scalefactor)

        subband_SMR = subband_LSB - subband_MMT

        return np.clip(subband_SMR, 0, None)
    
    def psychoacoustic_analysis(x_frames, nbands):
        SMR = np.zeros((x_frames.shape[0], nbands))
        for i in range(0, x_frames.shape[0]):
            if np.any(x_frames[i]):
                SMR[i,:] = SMR_from_frame(x_frames[i])
            else:
                SMR[i,:] = np.zeros(nbands)

        return SMR
    x_SMR = psychoacoustic_analysis(x_frames, nbands)
    
    
    
    def quantization(x_bands, x_SMR, nbands, nsubbands):
        x_SMR = x_SMR.astype('int')
        max_bits = np.amax(bit_allocation_table)
        SMR_arr = np.arange(np.amin(x_SMR), np.amax(x_SMR)+1)
        SMR_to_bits = np.round(SMR_arr/(np.amax(x_SMR) / max_bits)).astype('uint8')
        SMR_to_mu = SMR_arr*(-255/np.amax(x_SMR)) + 255

        quantized_signal = []
        enc_tmp = []
        scale_factor = []
        frame_separator = 2**(max_bits+1)
        for i in range(0, x_bands.shape[1]):
            encfr = EncodedFrame()
            for j in range(0, nbands):
                band_smr = x_SMR[i,j]
                bit_alloc = SMR_to_bits[band_smr]
                if bit_alloc > 7:
                    quantized_mdct = powerlaw_quant(x_bands[j,i,:], 255)/band_smr
        #             print(signal_to_noise(x_subbands[j,i,:], inv_powerlaw_quant(quantized_mdct*band_smr,255)))
                    quantized_mdct = quantized_mdct.astype('float16')
                    enc_tmp.extend(quantized_mdct)
                    encfr.data = np.append(encfr.data, quantized_mdct)
                    encfr.scaling_factors = np.append(encfr.scaling_factors, band_smr)
            quantized_signal.append(encfr)

        return (quantized_signal, enc_tmp)
    (quantized_signal, enc_tmp) = quantization(x_bands, x_SMR, nbands, nsubbands)
    

    def huffman_encoding(quantized_signal, enc_tmp):
        encoded_signal = EncodedSignal()
        codec = HuffmanCodec.from_data(enc_tmp)
        encoded_signal.huffman_table = codec.get_code_table()
        for i, frame in enumerate(quantized_signal):
            frame.data = codec.encode(frame.data)
        encoded_signal.data = quantized_signal
        return encoded_signal
    encoded_signal = huffman_encoding(quantized_signal, enc_tmp)
    
    return encoded_signal

class EncodedSignal(object):
    def __init__(self):
        self.huffman_table = None
        self.data = None


def decode(encoded_signal):
    huffman_decoder = HuffmanCodec(code_table=encoded_signal.huffman_table, check=False)
    empty_frame = [0]*npoints
    mdct_per_frame = num_blocks
    decoded_signal = np.zeros((len(encoded_signal.data), npoints))

    for i, frame in enumerate(encoded_signal.data):
        decoded_mdct = np.zeros((nbands, num_blocks))
        frame_data = np.array(huffman_decoder.decode(frame.data))
        for j in range(0, len(frame.scaling_factors)):
            scale_factor = frame.scaling_factors[j]
            start = j*mdct_per_frame
            end = (j+1)*mdct_per_frame
            encoded_mdct = frame_data[start:end]*scale_factor
            decoded_mdct[j] = inv_powerlaw_quant(encoded_mdct, 255)
        decoded_frame = PQMFBsyn(decoded_mdct, window_function)

        diff = 1152 - len(decoded_frame)
        if diff < 0:
            decoded_signal[i,:] = decoded_frame[:diff]
        if diff > 0:
            decoded_signal[i,:] = np.append(decoded_frame, np.zeros((diff,1)))

    return decoded_signal.flatten().astype('int16')








def group_by_critical_band(k, P):
    b_k = bark(f_k)
    cb_k = np.array([int(b) for b in np.floor(b_k)])
    bands = [[[], []] for _ in np.arange(np.amax(cb_k) + 1)]
    for _k, _P in zip(k, P):
        band = bands[cb_k[_k]]
        band[0].append(_k)
        band[1].append(_P)
    for b, band in enumerate(bands):
        bands[b] = np.array(band)
    return bands

def merge_tonals(k_t, P_t):
    bands = group_by_critical_band(k_t, P_t)
    k_t_out, P_t_out = [], []
    for band, k_P_s in enumerate(bands):
        if len(k_P_s[0]):
            k_max = None
            P_max = - np.inf 
            for _k, _P in zip(*k_P_s):
               if _P > P_max:
                   k_max = _k
                   P_max = _P
            k_t_out.append(k_max)
            P_t_out.append(P_max)
    return np.array(k_t_out, dtype='uint16'), np.array(P_t_out)

def merge_non_tonals(k_nt, P_nt):
    bands = group_by_critical_band(k_nt, P_nt)
    k_nt_out = np.zeros(len(bands), dtype='uint8')
    P_nt_out = np.zeros(len(bands))
    for band, k_P_s in enumerate(bands):
        if len(k_P_s[0]):
            k, P = k_P_s
            P_sum = sum(P)
            if P_sum == 0.0:
                P = np.ones_like(P)
            k_mean = int(round(np.average(k, weights=P))) 
            k_nt_out[band] = k_mean
            P_nt_out[band] = P_sum
    return k_nt_out, P_nt_out

def threshold(k, P):
    ATH_k = 10 ** (absolute_threshold_hearing(f_k) / 10.0)
    k_out, P_out = [], []
    for (_k, _P) in zip(k, P):
        if _P > ATH_k[_k]:
            k_out.append(_k)
            P_out.append(_P)
    return np.array(k_out), np.array(P_out)

def excitation_pattern(b, b_m, I_m, tonal):
    db = b - b_m
    db_1 = np.minimum(db + 1.0, 0.0)
    db_2 = np.minimum(db      , 0.0)
    db_3 = np.maximum(db      , 0.0)
    db_4 = np.maximum(db - 1.0, 0.0)    
    mask  = I_m \
          + (11.0 - 0.40 * I_m) * db_1 \
          + ( 6.0 + 0.40 * I_m) * db_2 \
          - (17.0             ) * db_3 \
          + (       0.15 * I_m) * db_4
    if tonal:
        mask += -1.525 - 0.275 * b - 4.5
    else:
        mask += -1.525 - 0.175 * b - 0.5
    return mask

def maskers(frame, k, merge=True, ATH_threshold=True):
    P_k = P_k_from_frame(frame, signal.windows.hann, 1152)
    classify = Classifier()
    (k_t, P_t), (k_nt, P_nt) = classify(k, P_k)
    if merge:
        k_t, P_t = merge_tonals(k_t, P_t)
        k_nt, P_nt = merge_non_tonals(k_nt, P_nt)
    if ATH_threshold:
        k_t, P_t = threshold(k_t, P_t)
        k_nt, P_nt = threshold(k_nt, P_nt)
    return (k_t, P_t), (k_nt, P_nt)

def P_k_from_frame(frame, window, N, dB=True):
    alpha = 1.0 / np.sqrt(sum(window(N)**2) / N)
    frame = alpha * window(N) * frame

    frame_fft_2 = abs(fft(frame)) ** 2

    P_k = 2.0 * frame_fft_2[:(N // 2 + 1)] / N ** 2
    P_k[0] = 0.5 * P_k[0]
    if (N % 2 == 0):
        P_k[-1] = 0.5 * P_k[-1]

    P_k = 10.0 ** (96.0 / 10.0) * P_k
    if dB:
        P_k = 10.0 * np.log10(P_k)  
    return P_k



def powerlaw_quant(x, mu):
    return np.sign(x)*np.log(1 + mu*np.abs(x))/np.log(1 + mu)

def inv_powerlaw_quant(y, mu):
    return np.sign(y)*((mu + 1)**np.abs(y) - 1)/mu