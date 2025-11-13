#arquivo que realiza a plotagem de um arquivo edf mostrando a onda predominante usando FFT
import mne
import numpy as np
import matplotlib.pyplot as plt

# Caminho do seu arquivo EDF
edf_path = "/Users/crisly/Desktop/testandoaplicacao/EEG_Elisvane_Oliveira_da_Silva_beta_20251028_1555.edf"


raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
data, sfreq = raw.get_data(), raw.info['sfreq']


signal = np.mean(data, axis=0)


frequencies = np.fft.rfftfreq(len(signal), d=1/sfreq)
fft_values = np.abs(np.fft.rfft(signal))**2


bands = {
    'Delta': (0.5, 4),
    'Teta': (4, 8),
    'Alfa': (8, 13),
    'Beta': (13, 30),
    'Gama': (30, 100)
}


band_powers = {}
for band, (low, high) in bands.items():
    idx = np.where((frequencies >= low) & (frequencies <= high))[0]
    band_power = np.sum(fft_values[idx])
    band_powers[band] = band_power


plt.figure(figsize=(8, 5))
plt.bar(band_powers.keys(), band_powers.values(), color='orchid')
plt.title('Potência por Banda de Frequência (via FFT)')
plt.xlabel('Banda')
plt.ylabel('Potência (a.u.)')
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
