
from __future__ import absolute_import, division, print_function, unicode_literals

#import tensorflow as tf
#from tensorflow.keras import layers

import random
import numpy as np

import os, sys, argparse, time
from multiprocessing import Pool

import librosa
from pathlib import Path 
import configparser
import pdb

#Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default ='./default.ini' , help='path to the config file')
args = parser.parse_args()

#Get configs
config_path = args.config
config = configparser.ConfigParser(allow_no_value=True)
config.read(config_path)

#audio configs
sampling_rate = config['audio'].getint('sampling_rate')
hop_length = config['audio'].getint('hop_length')
bins_per_octave = config['audio'].getint('bins_per_octave')
num_octaves = config['audio'].getint('num_octaves')
n_bins = num_octaves * bins_per_octave
n_iter=config['audio'].getint('n_iter')
cqt_bit_depth = config['audio'].get('cqt_bit_depth')

#dataset
dataset = Path(config['dataset'].get('datapath'))
if not dataset.exists():
    raise FileNotFoundError(dataset.resolve())

cqt_dataset = config['dataset'].get('cqt_dataset')
my_cqt = dataset / cqt_dataset

os.makedirs(my_cqt,exist_ok=True)
my_audio_folder = dataset / 'audio'


audio_files = [f for f in my_audio_folder.glob('*.wav')]

print('TOTAL FILES: {}'.format(len(audio_files)))

config_path = my_cqt / 'config.ini'
with open( config_path, 'w') as configfile:
    config.write(configfile)

def calculate_cqt(f):

    outfile = my_cqt.joinpath(f.stem + '.npy')
   
    try:
        s, fs = librosa.load(f, sr=None)
        if fs != sampling_rate:
            print("Resampling {}".format(f))
            s, fs = librosa.load(f, sr=sampling_rate)

        # Get the CQT magnitude
        C_complex = librosa.cqt(y=s, sr=sampling_rate, hop_length= hop_length, bins_per_octave=bins_per_octave, n_bins=n_bins)
        C = np.abs(C_complex)
        # pytorch expects the transpose of librosa's output
        C = np.transpose(C)

        # Choose the datatype
        if cqt_bit_depth == 'float32':
            C = C.astype('float32')
        elif cqt_bit_depth == 'float64':
            pass
        else:
            raise TypeError('cqt_bit_depth datatype is unknown. Choose either float32 or float64')
        print('writing: {}'.format(outfile))
        np.save(outfile, C)
    
    except:
        print('There was an issue with '+f)    
        return

if __name__ == '__main__': 

    pool = Pool()
    pool.map(calculate_cqt, audio_files)
    pool.close()
    pool.join()