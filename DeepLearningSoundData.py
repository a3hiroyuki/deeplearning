'''
Created on 2017/09/04

@author: Abe
'''

import glob
import os
import librosa
import numpy as np

class DeepLearningSoundData:
    
    def __init__(self, sub_tr_dirs, sub_ts_dirs):
        self.parent_dir = os.path.dirname(os.path.abspath(__file__))
        self.sub_tr_dirs = sub_tr_dirs
        self.sub_ts_dirs = sub_ts_dirs
        

    def extract_feature(self, file_name):
        X, sample_rate = librosa.load(file_name)
        stft = np.abs(librosa.stft(X))
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),
        sr=sample_rate).T,axis=0)
        return mfccs,chroma,mel,contrast,tonnetz
    
    def parse_audio_files(self, sub_dirs,file_ext="*.wav"):
        features, labels = np.empty((0,193)), np.empty(0)
        for label, sub_dir in enumerate(sub_dirs):
            for fn in glob.glob(os.path.join(self.parent_dir, sub_dir, file_ext)):
                print (fn)
                try:
                    mfccs, chroma, mel, contrast,tonnetz = self.extract_feature(fn)
                except Exception as e:
                    print ("Error encountered while parsing file: ", fn)
                    print (e.args)
                    continue
                ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
                features = np.vstack([features,ext_features])
                names = fn.split('/')
                labels = np.append(labels, names[len(names) - 1].split('-')[1])
        return np.array(features), np.array(labels, dtype = np.int)
    
    def one_hot_encode(self, labels):
        n_labels = len(labels)
        n_unique_labels = len(np.unique(labels))
        one_hot_encode = np.zeros((n_labels,n_unique_labels))
        one_hot_encode[np.arange(n_labels), labels] = 1
        return n_unique_labels, one_hot_encode
    
    def create_sound_data(self):
        tr_features, tr_labels = self.parse_audio_files(self.sub_tr_dirs)
        ts_features, ts_labels = self.parse_audio_files(self.sub_ts_dirs)
        n_unique_labels, tr_labels = self.one_hot_encode(tr_labels)
        n_unique_labels, ts_labels = self.one_hot_encode(ts_labels)
        return tr_features, tr_labels, ts_features, ts_labels, n_unique_labels
