#BL.EN.U4AIE21078
#Varshith M

import librosa
import IPython.display as ipd
import matplotlib.pylab as plt
import librosa.display
from IPython.display import Audio, display

audio =  r"C:\Users\hp\OneDrive\Desktop\Pyhton\Speech_Processing\Audio.wav"
ipd.Audio(audio)
y, sr = librosa.load(audio) 
librosa.display.waveshow(y)


#Plotting The Waveform                  -----(A1)
plt.title('Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

#Calculating Duration & Magnitude Range   ---(A2)
duration = len(y) / sr
print(f"Duration: {duration:.2f} seconds")
print(f"Magnitude range: {y.min()} to {y.max()}")


#Getting a segment of the audio file      ---(A3)     
y1, sr1 = librosa.load(audio , offset=0.0, duration=0.5)
audio1 = Audio(y1, rate=sr1)
librosa.display.waveshow(y1)

plt.title('Waveform-1')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

#Altering the settings with audio file   ---(A4)
y2, sr2 = librosa.load(audio , offset=0.0, duration=0.1)
audio1 = Audio(y1, rate=sr1)
librosa.display.waveshow(y1)

plt.title('Waveform-2')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()