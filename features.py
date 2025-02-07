# Author: Axel Mukwena
# ECG Biometric Authentication using CNN

import os
import pickle
import random

import librosa
import numpy as np
import pandas as pd
import warnings
from biosppy.signals import ecg
from scipy import signal
from scipy.signal import filtfilt, find_peaks
from matplotlib import pyplot as plt


def resamp(array, times):
    return signal.resample(array, times)


# def mel(y, sr):
#     spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
#     return librosa.power_to_db(spectrogram, ref=np.max)


# def filters(array, n):
#     # the larger n is, the smoother curve will be
#     b = [1.0 / n] * n
#     a = 1
#     array = filtfilt(b, a, array)
#     return array


def refine_r_peaks(sig, r_peaks):
    r_peaks2 = np.array(r_peaks)  # make a copy
    for i in range(len(r_peaks)):
        r = r_peaks[i]  # current R-peak
        small_segment = sig[max(0, r - 100):min(len(sig), r + 100)]  # consider the neighboring segment of R-peak
        r_peaks2[i] = np.argmax(small_segment) - 100 + r_peaks[i]  # picking the highest point
        r_peaks2[i] = min(r_peaks2[i], len(sig))  # the detected R-peak shouldn't be outside the signal
        r_peaks2[i] = max(r_peaks2[i], 0)  # checking if it goes before zero
    return r_peaks2  # returning the refined r-peak list


def segment_signals(sig, r_peaks_annot,fs, bmd=True, normalization=True):
    # segmented_signals = []
    # r_peaks = np.array(r_peaks_annot)
    # # r_peaks = refine_r_peaks(sig, r_peaks)
    # if bmd:
    #     win_len = 300
    # else:
    #     win_len = 256
    # left = fs*32//100
    # right = fs*48//100
    # # prev_r_peak = -1
    # for i in range(len(r_peaks) - 1):
    #     r = r_peaks[i]
    #     next_r_peak = r_peaks[i + 1]
    #     if ((r - left) < 0) or ((r + right) >= len(sig)):  # not enough signal to segment
    #         continue
    #     # if (r + 50) >= next_r_peak:
    #     #     continue
    #     segmented_signal = np.array(sig[r - left:r + right])  # segmenting a heartbeat

    #     if normalization:  # Z-score normalization
    #         if abs(np.std(segmented_signal)) < 1e-6:  # flat line ECG, will cause zero division error
    #             continue
    #         segmented_signal = (segmented_signal - np.mean(segmented_signal)) / np.std(segmented_signal)

    #     if not np.isnan(segmented_signal).any():  # checking for nan, this will never happen
    #         segmented_signals.append(segmented_signal)
    # return segmented_signals, r_peaks

    segmented_signals = []
    r_peaks = np.array(r_peaks_annot)
    # r_peaks = refine_r_peaks(sig, r_peaks)

    left = fs*32//100
    right = fs*48//100
    for r in r_peaks:
        if ((r - left) < 0) or ((r + right) >= len(sig)):  # not enough signal to segment
            continue
        segmented_signal = np.array(sig[r - left:r + right])  # segmenting a heartbeat

        if normalization:  # Z-score normalization
            if abs(np.std(segmented_signal)) < 1e-6:  # flat line ECG, will cause zero division error
                continue
            segmented_signal = (segmented_signal - np.mean(segmented_signal)) / np.std(segmented_signal)

        if not np.isnan(segmented_signal).any():  # checking for nan, this will never happen
            segmented_signals.append(segmented_signal)

    return segmented_signals, r_peaks


class GetFeatures:
    def __init__(self):
        self.dir = os.path.expanduser("data/ready/")
        self.age_labels = []
        self.gender_labels = []
        self.signals = []
        self.where = ""
        self.all = []
        self.person_waves = []
        self.person = ""

    # Extracts features from csv file of each person | mit database
    def features(self, where, people,fs):
        self.where = where
        for person in people:
            self.signals = []  # reset signal array
            self.person = person
            if self.where == "ecgid":
                try:
                    folder = os.path.expanduser("data/raw/ecgid/Person_" + person + "/")
                    files = [os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.csv')]
                except FileNotFoundError:
                    continue
            elif self.where == "mit":
                files = ["data/raw/mit/" + person + ".csv"]
            elif self.where == "nsrdb":
                files = ["data/raw/nsrdb/" + person + ".csv"]
            elif self.where == "CYBHi_short":
                files = ["data/raw/CYBHi_short/" + person + ".csv"]
            elif self.where == "CYBHi_long":
                files = ["data/raw/CYBHi_long/" + person + ".csv"]
            elif self.where == "AM":
                files = ["data/raw/AM/" + person + ".csv"]
            elif self.where == "now":
                files = ["data/raw/CYBHi/data/long-term/now/" + person + ".csv"]
            elif self.where == "later":
                files = ["data/raw/CYBHi/data/long-term/later/" + person + ".csv"]
            elif self.where == "85":
                files = ["data/raw/CYBHi/data/short-term/85/" + person + ".csv"]
            elif self.where == "8B":
                files = ["data/raw/CYBHi/data/short-term/8B/" + person + ".csv"]
            elif self.where == "lowB":
                files = ["data/raw/CYBHi/data/short-term/lowB/" + person + ".csv"]
            elif self.where == "low5":
                files = ["data/raw/CYBHi/data/short-term/low5/" + person + ".csv"]
            elif self.where == "highB":
                files = ["data/raw/CYBHi/data/short-term/highB/" + person + ".csv"]
            elif self.where == "CI":
                files = ["data/raw/CYBHi/data/short-term/CI/" + person + ".csv"]
            else:  # bmd
                files = ["data/raw/bmd101/csv/" + person + ".csv"]

            sgs = []
            for file in files:
                with open(file, 'r') as f:
                    features = pd.read_csv(f)
                filtered = features['0'].values
                sgs = np.concatenate((sgs, filtered))

            self.segment(np.array(sgs),fs)

            length = len(self.signals)
            num = random.randint(0, length)
            random.seed(num)
            random.shuffle(self.signals)

            self.dump_pickle(self.signals, person)

        # self.dump_pickle(self.all, 'all')
        print("Feature extraction complete.")

    # ECG R-peak segmentation
    def segment(self, array,fs):
        count = 1

        array = np.array(array, dtype="float16")
        array = (array - array.min()) / (array.max() - array.min())
        peaks = ecg.christov_segmenter(signal=array, sampling_rate=fs)[0]
        waves, pks = segment_signals(array, peaks,fs, False, True)
        how_many = []
        self.person_waves = []
        length = len(waves)

        for k in range(length):
            wave = waves[k]
            plt.title(self.person)
            self.augment(wave, len(wave),fs)
            how_many.append(len(wave))
            count += 1

        # plt.show()
        # print("Len per Wave", how_many)
        # print("Mean per Wave", np.mean(how_many))
        # print("How many", len(how_many))
        # print("Total", len(how_many) * 9)

    # Augment each signal and convert call function to convert it to image
    def augment(self, array, times,fs):
        array = resamp(array, times)
        self.signals.append(array)
        plt.plot(array)

        # Noise addition using normal distribution with mean = 0 and std =1
        # Permissible noise factor value = x > 0.009
        if self.where[:4] == "live":
            one = 0.09
            two = 0.07
            three = 0.05
        else:
            one = 0.09
            two = 0.07
            three = 0.05

        # self.help(array, times, "noise", one,fs)
        # self.help(array, times, "noise", two,fs)
        # self.help(array, times, "noise", three,fs)
        # plt.show()

        # Permissible factor values = samplingRate / 100
        # self.help(array, times, "time_shifting", 100,fs)
        # self.help(array, times, "time_shifting", 120,fs)
        # self.help(array, times, "time_shifting", 150,fs)

        # Permissible factor values = -5 <= x <= 5
        # self.help(array, times, "pitch_shifting", -0.3,fs)
        # self.help(array, times, "pitch_shifting", -0.2,fs)
        # self.help(array, times, "pitch_shifting", -0.1,fs)

    def help(self, array, times, which, factor,fs):
        if which == "noise":
            noise = array + factor * np.random.normal(0, 1, len(array))
            noise = resamp(noise, times)
            self.signals.append(noise)
            plt.plot(noise)

        if which == "time_shifting":
            time_shifting = np.roll(array, int(fs / factor))
            time_shifting = resamp(time_shifting, times)
            self.signals.append(time_shifting)
            plt.plot(time_shifting)

        if which == "pitch_shifting":
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                pitch_shifting = librosa.effects.pitch_shift(array, sr=fs, n_steps=float(factor))
            pitch_shifting = resamp(pitch_shifting, times)
            self.signals.append(pitch_shifting)
            plt.plot(pitch_shifting)

        if which == "time_stretching":
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                time_stretching = librosa.effects.time_stretch(array, factor)
            time_stretching = resamp(time_stretching, times)
            self.signals.append(time_stretching)
            plt.plot(time_stretching)

    def dump_pickle(self, signals, basename):
        folder = self.dir + 'signals/'
        if not os.path.exists(folder):
            os.makedirs(folder)

        filename = folder + basename + '.pickle'
        pickle_out = open(filename, 'wb')
        pickle.dump(signals, pickle_out)
        pickle_out.close()
        print('Person ' + basename, "\n")
