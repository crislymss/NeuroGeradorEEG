#codigo antigo, so da pra escolher apenas um canal no edf


from flask import Blueprint, render_template, request, send_file
import os
import numpy as np
import pyedflib
import torch
from datetime import datetime
import tempfile
from pathlib import Path
import sys
from app.generator import Generator
# Adicione esta linha para corrigir o problema de importação
sys.path.append(str(Path(__file__).parent.parent))

bp = Blueprint('main', __name__)

# Configurações com caminhos absolutos
BASE_DIR = Path(__file__).parent.parent
MODELS_FOLDER = BASE_DIR / "app" / "models"
EEG_CHANNELS = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
                'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
SAMPLE_RATE = 250
LATENT_DIM = 100
AMPLITUDE_TARGET = 100  # µV



def load_model(wave_type):
    """Versão corrigida que carrega o modelo corretamente"""
    model_path = MODELS_FOLDER / f"{wave_type}_generator.pth"
    
    if not model_path.exists():
        print(f"ERRO: Arquivo não encontrado em {model_path}")
        return None
    
    try:
        # 1. Carrega o state_dict
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(model_path, map_location=device)
        
        # 2. Cria uma instância do modelo
        model = Generator(LATENT_DIM, SAMPLE_RATE * 10).to(device)
        
        # 3. Carrega os pesos
        model.load_state_dict(state_dict)
        
        # 4. Coloca em modo de avaliação
        model.eval()
        
        print(f"Modelo {wave_type} carregado com sucesso!")
        return model
        
    except Exception as e:
        print(f"FALHA ao carregar modelo {wave_type}: {str(e)}")
        return None
    
def adjust_amplitude(signal, target=AMPLITUDE_TARGET):
    """Ajusta a amplitude do sinal para valores realistas de EEG"""
    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) * 2 - 1
    return signal * target


def generate_eeg_signal(generator, duration_min, channels):
    """Gera dados EEG com dimensões compatíveis"""
    try:
        total_samples = int(duration_min * 60 * SAMPLE_RATE)
        num_channels = len(channels)
        
        # Gera ruído base com dimensão correta
        noise = np.random.randn(total_samples, num_channels) * 5  # Ruído reduzido
        
        # Gera a onda principal com o tamanho exato necessário
        with torch.no_grad():
            z = torch.randn(1, LATENT_DIM)
            synthetic_wave = generator(z).cpu().numpy()
            
            # Redimensiona para o tamanho correto
            if synthetic_wave.size < total_samples:
                # Se o modelo gera menos amostras que o necessário, repete
                repeats = int(np.ceil(total_samples / synthetic_wave.size))
                synthetic_wave = np.tile(synthetic_wave, repeats)[:total_samples]
            else:
                # Se gera mais, corta
                synthetic_wave = synthetic_wave[:total_samples]
            
            # Ajusta amplitude e formata
            synthetic_wave = adjust_amplitude(synthetic_wave) * 0.8
            synthetic_wave = synthetic_wave.reshape(-1, 1)  # Garante formato (N,1)
        
        # Combina ruído + onda para todos os canais
        eeg_data = noise + np.tile(synthetic_wave, (1, num_channels))
        
        return eeg_data
    
    except Exception as e:
        print(f"Erro na geração do EEG: {str(e)}")
        return None
    
    
def create_edf(eeg_data, channels, patient_info, wave_type):
    """Cria arquivo EDF sem warnings e com metadados válidos"""
    try:
        # Cria arquivo temporário
        temp_file = tempfile.NamedTemporaryFile(suffix='.edf', delete=False)
        temp_path = temp_file.name
        temp_file.close()

        # Configuração do writer
        with pyedflib.EdfWriter(temp_path, len(channels), file_type=pyedflib.FILETYPE_EDFPLUS) as f:
            # Prepara os cabeçalhos dos canais
            channel_headers = []
            for ch in channels:
                channel_headers.append({
                    'label': ch[:16],  # Limita a 16 caracteres (padrão EDF)
                    'dimension': 'uV',
                    'sample_frequency': SAMPLE_RATE,
                    'physical_max': AMPLITUDE_TARGET * 1.2,
                    'physical_min': -AMPLITUDE_TARGET * 1.2,
                    'digital_max': 32767,
                    'digital_min': -32768,
                    'transducer': '',
                    'prefilter': ''
                })
            
            # Define os cabeçalhos
            f.setSignalHeaders(channel_headers)
            
            # Metadados do paciente (formatados corretamente)
            f.setPatientName(patient_info['name'][:80])  # Limita a 80 caracteres
            
            # Remove espaços e caracteres especiais dos metadados
            recording_info = f"SYNTH_{wave_type.upper()}".replace(" ", "_")
            f.setRecordingAdditional(recording_info[:80])
            
            # Garante ordem C nos dados (resolve warning Fortran order)
            eeg_data = np.ascontiguousarray(eeg_data)
            
            # Escreve os dados canal por canal
            for i in range(len(channels)):
                f.writeSamples([eeg_data[:, i]])
        
        return temp_path
    except Exception as e:
        print(f"Erro ao criar EDF: {str(e)}")
        return None

@bp.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Dados do formulário
        patient_info = {
            'name': request.form.get('patient_name', 'Anonymous'),
            'birth_date': request.form.get('birth_date', '')
        }
        duration_min = int(request.form.get('duration_min', 5))
        selected_channels = request.form.getlist('channels')
        wave_type = request.form.get('dominant_wave')
        
        # Validações SIMPLES
        if not selected_channels:
            return render_template('index.html', error="Selecione pelo menos um canal", channels=EEG_CHANNELS)
        
        # Carrega o modelo
        generator = load_model(wave_type)
        if not generator:
            return render_template('index.html', 
                                error=f"Modelo {wave_type} não carregado. Verifique o terminal.",
                                channels=EEG_CHANNELS)
        
        # Gera os dados
        eeg_data = generate_eeg_signal(generator, duration_min, selected_channels)
        if eeg_data is None:
            return render_template('index.html', 
                                error="Falha na geração dos dados",
                                channels=EEG_CHANNELS)
        
        # Cria EDF
        edf_path = create_edf(eeg_data, selected_channels, patient_info, wave_type)
        if not edf_path:
            return render_template('index.html', 
                                error="Falha ao criar arquivo EDF",
                                channels=EEG_CHANNELS)
        
        # Download do arquivo
        filename = f"EEG_{patient_info['name']}_{wave_type}_{datetime.now().strftime('%Y%m%d_%H%M')}.edf"
        return send_file(
            edf_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/edf'
        )
    
    # GET request - mostra o formulário
    return render_template('index.html', channels=EEG_CHANNELS)