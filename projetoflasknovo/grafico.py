import pyedflib
import matplotlib.pyplot as plt
import numpy as np

def plot_eeg_segment(edf_path, duration_sec=20, output_path='segmento.png'):

    f = pyedflib.EdfReader(edf_path)
    n_channels = f.signals_in_file
    signal_labels = f.getSignalLabels()
    sample_rate = f.getSampleFrequency(0)  


    n_samples = int(duration_sec * sample_rate)


    plt.figure(figsize=(12, n_channels * 2))

    for i in range(n_channels):
        signal = f.readSignal(i)
        signal_segment = signal[:n_samples]  
        time = np.arange(n_samples) / sample_rate


        plt.subplot(n_channels, 1, i + 1)
        plt.plot(time, signal_segment, linewidth=0.8)
        plt.title(signal_labels[i])
        plt.ylabel("Amplitude")
        if i == n_channels - 1:
            plt.xlabel("Tempo (s)")
        else:
            plt.xticks([])

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    f.close()
    print(f"Gráfico salvo em '{output_path}'.")


edf_file= "/Users/crisly/Desktop/testandoaplicacao/EEG_Elisvane_Oliveira_da_Silva_beta_20251028_1555.edf"
  # Substitua pelo caminho do seu arquivo
plot_eeg_segment(edf_file)
