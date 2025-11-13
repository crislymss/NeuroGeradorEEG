import io
import base64
import matplotlib.pyplot as plt
import pyedflib
import numpy as np

def generate_eeg_plot_base64(edf_path, duration_sec=20):
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
        plt.tight_layout()

    # Salvar o gráfico em memória
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    base64_img = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()
    return base64_img
