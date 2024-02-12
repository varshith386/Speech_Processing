#Varshith M
#BL.EN.U4AIE21078

import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

#Q1) -------------------------------------------


def finite_difference(signal, sampling_rate):
    dt = 1.0 / sampling_rate
    derivative = np.diff(signal) / dt
    return np.concatenate(([0], derivative))


file_path = r"C:\Users\hp\OneDrive\Desktop\Pyhton\Speech_Processing\Audio.wav"
sampling_rate, signal = wavfile.read(file_path)


derivative = finite_difference(signal, sampling_rate)
time = np.arange(0, len(signal)) / sampling_rate

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(time, signal)
plt.title('Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
time_derivative = np.arange(0, len(derivative)) / sampling_rate
plt.plot(time_derivative, derivative)
plt.title('First Derivative')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.show()

#Q2) -------------------------------------------


def zero_crossings(derivative):
    return np.where(np.diff(np.sign(derivative)))[0]

def calculate_average_length(zero_crossings):
    return np.mean(np.diff(zero_crossings))


zero_crossings_indices = zero_crossings(derivative)


time = np.arange(0, len(signal)) / sampling_rate

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(time, signal)
plt.title('Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 2)
time_derivative = np.arange(0, len(derivative)) / sampling_rate
plt.plot(time_derivative, derivative)
plt.title('First Derivative with Zero Crossings')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.scatter(zero_crossings_indices / sampling_rate, np.zeros_like(zero_crossings_indices), color='red', marker='o', label='Zero Crossing')


speech_zero_crossings = zero_crossings_indices[zero_crossings_indices < len(signal) / 2]  
silence_zero_crossings = zero_crossings_indices[zero_crossings_indices >= len(signal) / 2] 


average_length_speech = calculate_average_length(speech_zero_crossings)
average_length_silence = calculate_average_length(silence_zero_crossings)


print(f"Average length between zero crossings for speech: {average_length_speech} samples")
print(f"Average length between zero crossings for silence: {average_length_silence} samples")
a=speech_zero_crossings / sampling_rate
b=silence_zero_crossings / sampling_rate
plt.subplot(3, 1, 3)
plt.plot(time_derivative, derivative)
plt.title('First Derivative with Zero Crossings for speech and silence regions')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.scatter(a, np.zeros_like(speech_zero_crossings), color='green', marker='o', label='Speech Zero Crossing')
plt.scatter(b, np.zeros_like(silence_zero_crossings), color='blue', marker='o', label='Silence Zero Crossing')
plt.legend()

plt.tight_layout()
plt.show()


#Q3)
import librosa
import librosa.display


audio_file1 = r"C:\Users\hp\OneDrive\Desktop\Pyhton\Speech_Processing\amal.wav"
audio_file2 = r"C:\Users\hp\OneDrive\Desktop\Pyhton\Speech_Processing\varshit.wav"



y1, sr1 = librosa.load(audio_file1)
y2, sr2 = librosa.load(audio_file2)


duration1 = librosa.get_duration(y=y1, sr=sr1)
duration2 = librosa.get_duration(y=y2, sr=sr2)

print("The length Duration of first audio file:", duration1, "seconds")
print("The length Duration of Second audio file:", duration2, "seconds")


def remove_silence(y, sr, threshold=0.01):
    yt = librosa.effects.trim(y, top_db=threshold)
    return yt[0]


audio_trimmed1 = remove_silence(y1, sr1)
audio_trimmed2 = remove_silence(y2, sr2)


time1 = np.linspace(0, len(audio_trimmed1) / sr1, len(audio_trimmed1))
time2 = np.linspace(0, len(audio_trimmed2) / sr2, len(audio_trimmed2))


plt.figure(figsize=(18, 6))
plt.plot(time1, audio_trimmed1, label='Amal audio')
plt.plot(time2, audio_trimmed2, label='Varshit audio')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Overlapped Audio Files')
plt.show()