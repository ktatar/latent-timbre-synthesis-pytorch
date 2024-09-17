import librosa
import numpy as np
import torch

s, fs = librosa.load(librosa.ex('trumpet'))


bins_per_octave = 48
num_octaves = 8
cqt_fmin = 'A1'

n_iter=32
sr=44100
hop_length=128
fmin='A1'
bins_per_octave=48
tuning=0.0
filter_scale=1
norm=1
sparsity=0.01
window="hann"
scale=True
pad_mode="reflect"
res_type="kaiser_fast"
dtype=None
length=None
momentum=0.99
init="random"
random_state=None
n_bins = num_octaves * bins_per_octave

# Get the CQT magnitude
C_complex = librosa.cqt(y=s, sr=sr, hop_length= hop_length, bins_per_octave=bins_per_octave, n_bins=n_bins)
C = np.abs(C_complex)

y_inv_32 = librosa.griffinlim_cqt(C, sr=sr, n_iter=1, hop_length=128, bins_per_octave=48, dtype=np.float32)