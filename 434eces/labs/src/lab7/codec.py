import librosa

def encode(signal):
    sr = 44100
    e_sig = librosa.resample(signal.astype('float_'), sr, sr//2)
    return e_sig.astype('int16')

def decode(signal):
    sr = 44100
    d_sig = librosa.resample(signal.astype('float_'), sr//2, sr)
    return d_sig.astype('int16')