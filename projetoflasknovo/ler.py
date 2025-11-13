#arquivo que mostra a frequencia das ondas usando fft


import mne
import numpy as np
import matplotlib.pyplot as plt

# Caminho do seu arquivo EDF
edf_path = "/Users/crisly/Desktop/testandoaplicacao/Pessoa_1.edf"

# === 1. Carregar o arquivo EDF ===
# Coloque o caminho para seu arquivo .edf real


raw = mne.io.read_raw_edf(edf_path, preload=True)

# === 2. Escolher o canal (ex: 'FP2') ===
canal = 'F8'  # ou o canal que quiser analisar
data, times = raw[canal]

# === 3. Parâmetros da FFT ===
fs = int(raw.info['sfreq'])  # Frequência de amostragem do arquivo
signal = data[0]  # Extrai o vetor do canal

# === 4. Aplicar FFT ===
fft_vals = np.fft.rfft(signal)
fft_freqs = np.fft.rfftfreq(len(signal), d=1/fs)
fft_amplitude = np.abs(fft_vals)

# === 5. Plotar espectro de frequência ===
plt.figure(figsize=(10, 4))
plt.plot(fft_freqs, fft_amplitude)
plt.title(f"Espectro de Frequência do Canal {canal}")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, 20)  # Foco em frequências até 20 Hz
plt.grid(True)
plt.show()
