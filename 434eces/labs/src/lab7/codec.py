import librosa

def encode(signal):
    fs = 44100
    
    
    
    
    e_sig = librosa.resample(signal.astype('float_'), sr, sr//2)
    return e_sig.astype('int16')

def decode(signal):
    sr = 44100
    d_sig = librosa.resample(signal.astype('float_'), sr//2, sr)
    return d_sig.astype('int16')










def powerlaw_quantizer(x):
    return sgn(x)*np.log(1 + 255*np.abs(x))/np.log(1 + 255)

def sgn(x):
    a = np.array(x, copy=True)
    a[a < 0] = -1.0
    a[a >= 0] = 1.0
    return np.asarray(a)