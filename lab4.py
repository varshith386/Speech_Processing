import librosa
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio
from scipy.signal import spectrogram


y,sr = librosa.load("Audio.wav")
plt.figure(figsize=(10,4))
plt.plot(y)

fft_ = np.fft.fft(y)
print(fft_)

plt.figure(figsize=(10,4))
plt.plot(abs(fft_))

ifft_ = np.fft.ifft(fft_)
plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
plt.plot(ifft_)
plt.title('IFFT')
plt.subplot(1,2,2)
plt.plot(y)
plt.title(' original signal')

Audio("Audio.wav")
len(y)
Audio(y[14000:22000],rate = sr)

word = y[14000:22000]
word_fft = np.fft.fft(word)
plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
plt.plot(np.abs(fft_))
plt.title('FFT')
plt.subplot(1,2,2)
plt.plot(np.abs(word_fft))
plt.title('word')

y_22500,r = librosa.load("Audio.wav",sr = 22500)
Audio(y_22500,rate = r)

fft_2 = np.fft.fft(y_22500[10000:10450])
plt.figure(figsize=(10,4))
plt.plot(abs(fft_2))

y = librosa.stft(y_22500,n_fft=450,hop_length=225)
print(y)

freq_spectogram, times_spectogram, spectrogram_matrix = spectrogram(y,22500)
plt.figure(figsize=(14, 5))
plt.imshow(np.log(spectrogram_matrix), aspect='auto', origin='lower', cmap='viridis')
plt.colorbar()
plt.title("Spectrogram using scipy.signal.spectrogram()")
plt.xlabel("Time")
plt.ylabel("Frequency Bin")
plt.show()