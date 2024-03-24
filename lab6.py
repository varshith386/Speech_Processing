import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.fft import fft
from scipy.signal import spectrogram
import os
# Function to load the recorded speech signal
def load_recorded_signal(file_path):
    return librosa.load(file_path)
# Function to perform FFT on a signal snippet and plot the amplitude spectrum
def analyze_fft(signal_snippet, sr, title):
    fft_result = fft(signal_snippet)
    freq_axis = np.fft.fftfreq(len(signal_snippet), d=1/sr)
    plt.plot(freq_axis, np.abs(fft_result))
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.show()
    


def generate_spectrogram(signal, sr):
    f, t, Sxx = spectrogram(signal, sr)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.colorbar(label='Intensity (dB)')
    plt.show()
# Function to iterate over files in a directory and analyze each file
def analyze_files_in_directory(directory_path, title_prefix):
    # Iterate over each file in the directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        # Load the audio file
        signal, sr = load_recorded_signal(file_path)
        # Perform FFT analysis and plot the amplitude spectrum
        analyze_fft(signal, sr, f'{title_prefix}: {filename}')
        # Generate and plot spectrogram
        generate_spectrogram(signal, sr)
consonants_dir = r"C:\Users\hp\OneDrive\Desktop\Pyhton\Speech_Processing\Speech_Processing\consonants"
# Analyze consonant files
analyze_files_in_directory(consonants_dir, "Consonant Sound")


vowels_dir = r"C:\Users\hp\OneDrive\Desktop\Pyhton\Speech_Processing\Speech_Processing\vowels"
# Analyze vowel files
analyze_files_in_directory(vowels_dir, "Vowel Sound")

non_voiced_dir = r"C:\Users\hp\OneDrive\Desktop\Pyhton\Speech_Processing\Speech_Processing\non_voiced"
analyze_files_in_directory(non_voiced_dir, "Non-Voiced Portion")

