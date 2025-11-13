"""Exibir as frequências dominantes de EEG de um canal EDF usando FFT."""


import mne
import numpy as np
import matplotlib.pyplot as plt

edf_path = "/Users/crisly/Desktop/testandoaplicacao/Pessoa_1.edf"


raw = mne.io.read_raw_edf(edf_path, preload=True)


canal = 'F8'  
data, times = raw[canal]


fs = int(raw.info['sfreq'])  
signal = data[0] 


fft_vals = np.fft.rfft(signal)
fft_freqs = np.fft.rfftfreq(len(signal), d=1/fs)
fft_amplitude = np.abs(fft_vals)


plt.figure(figsize=(10, 4))
plt.plot(fft_freqs, fft_amplitude)
plt.title(f"Espectro de Frequência do Canal {canal}")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, 20)  
plt.grid(True)
plt.show()
