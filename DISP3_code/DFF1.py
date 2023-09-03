import numpy as np
import matplotlib.pyplot as plt
import wave

wav_file = wave.open('output.wav', 'rb')

sample_rate = wav_file.getframerate()
audio_data = wav_file.readframes(-1)

wav_file.close()

audio_np = np.frombuffer(audio_data, dtype=np.int16)

fft_result = np.fft.fft(audio_np)

freq_axis = np.fft.fftfreq(len(fft_result), 1.0 / sample_rate)

plt.figure()
plt.plot(freq_axis, np.abs(fft_result))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('Spectrum of Audio')
plt.grid(True)
plt.show()