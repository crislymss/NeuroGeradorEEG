from flask import Blueprint, render_template, request, send_file, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import io
import sys
import base64
import zipfile
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import pyedflib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from faker import Faker

from app.generator import Generator


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'edf'}


sys.path.append(str(Path(__file__).parent.parent))
bp = Blueprint('main', __name__)

fake = Faker()


@bp.route('/gerador2')
def gerador2_page():
    channels = [
        'F8', 'T4', 'T6', 'Fp2', 'F4', 'C4', 'P4', 'O2', 'Iz', 'Oz',
        'Pz', 'Cz', 'Fz', 'Fpz', 'Nz', 'Fp1', 'F3', 'C3', 'P3', 'O1',
        'T5', 'T3', 'F7'
    ]
    
    return render_template('gerador2.html', channels=channels)



@bp.route('/pagina02')
def pagina02_page():
    return render_template('pagina02.html')


@bp.route('/')
def index():
    return render_template('home.html')



BASE_DIR = Path(__file__).parent.parent
MODELS_FOLDER = BASE_DIR / "app" / "models"
EEG_CHANNELS = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
                'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']
SAMPLE_RATE = 250
LATENT_DIM = 100
AMPLITUDE_TARGET = 100  # µV


def load_model(wave_type):
    model_path = MODELS_FOLDER / f"{wave_type}_generator.pth"
    if not model_path.exists():
        print(f"ERRO: Arquivo não encontrado em {model_path}")
        return None

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(model_path, map_location=device)
        model = Generator(LATENT_DIM, SAMPLE_RATE * 10).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Modelo {wave_type} carregado com sucesso!")
        return model
    except Exception as e:
        print(f"FALHA ao carregar modelo {wave_type}: {str(e)}")
        return None


def adjust_amplitude(signal, target=AMPLITUDE_TARGET):
    signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) * 2 - 1
    return signal * target

def generate_eeg_signal(generator, duration_min, channels):
    try:
        total_samples = int(duration_min * 60 * SAMPLE_RATE)
        samples_per_segment = SAMPLE_RATE * 10  
        
     
        num_segments = int(np.ceil(total_samples / samples_per_segment))

        all_channels_signals = []
   
        for _ in channels:
            channel_segments = []
            
          
            for _ in range(num_segments):
                with torch.no_grad():
                    z = torch.randn(1, LATENT_DIM)
                    synthetic_wave_segment = generator(z).cpu().numpy().flatten()
                    
                    adjusted_segment = adjust_amplitude(synthetic_wave_segment) * 0.8
                    channel_segments.append(adjusted_segment)

            full_channel_signal = np.concatenate(channel_segments)
            
            full_channel_signal = full_channel_signal[:total_samples]
            
            all_channels_signals.append(full_channel_signal)

  
        eeg_data = np.stack(all_channels_signals, axis=1)
        print(f"Shape final do EEG gerado (com variabilidade): {eeg_data.shape}")
        return eeg_data

    except Exception as e:
        print(f"Erro na geração do EEG: {str(e)}")
        return None
    
def create_edf(eeg_data, channels, patient_info, wave_type):
    try:
        temp_file = tempfile.NamedTemporaryFile(suffix='.edf', delete=False)
        temp_path = temp_file.name
        temp_file.close()

        with pyedflib.EdfWriter(temp_path, len(channels), file_type=pyedflib.FILETYPE_EDFPLUS) as f:
            channel_headers = []
            for ch in channels:
                channel_headers.append({
                    'label': ch[:16],
                    'dimension': 'uV',
                    'sample_frequency': SAMPLE_RATE,
                    'physical_max': AMPLITUDE_TARGET * 1.2,
                    'physical_min': -AMPLITUDE_TARGET * 1.2,
                    'digital_max': 32767,
                    'digital_min': -32768,
                    'transducer': '',
                    'prefilter': ''
                })

            f.setSignalHeaders(channel_headers)
            f.setPatientName(patient_info.get('name', '')[:80])
            f.setRecordingAdditional(f"SYNTH_{wave_type.upper()}".replace(" ", "_")[:80])

            additional_info = f"BirthDate: {patient_info.get('birth_date', '')}; Gender: {patient_info.get('gender', 'N/A')}"
            f.setPatientAdditional(additional_info[:80])

            eeg_data = np.ascontiguousarray(eeg_data)
            f.writeSamples([eeg_data[:, i] for i in range(len(channels))])

        return temp_path

    except Exception as e:
        print(f"Erro ao criar EDF: {str(e)}")
        return None



@bp.route('/gerador1', methods=['GET', 'POST'])
def gerador1():
    if request.method == 'POST':
        patient_info = {
            'name': request.form.get('patient_name', 'Anonymous'),
            'birth_date': request.form.get('birth_date', ''),
            'gender': request.form.get('gender', 'Não informado') 
        }
        duration_min = int(request.form.get('duration_min', 5))
        selected_channels = request.form.getlist('channels')
        wave_type = request.form.get('dominant_wave')

        if not selected_channels:
            return render_template('gerador1.html', error="Selecione pelo menos um canal", channels=EEG_CHANNELS)

        generator = load_model(wave_type)
        if not generator:
            return render_template('gerador1.html', error=f"Modelo {wave_type} não carregado.", channels=EEG_CHANNELS)

        eeg_data = generate_eeg_signal(generator, duration_min, selected_channels)
        if eeg_data is None:
            return render_template('gerador1.html', error="Falha na geração dos dados", channels=EEG_CHANNELS)

        edf_path = create_edf(eeg_data, selected_channels, patient_info, wave_type)
        if not edf_path:
            return render_template('gerador1.html', error="Falha ao criar arquivo EDF", channels=EEG_CHANNELS)

        filename = f"EEG_{patient_info['name']}_{wave_type}_{datetime.now().strftime('%Y%m%d_%H%M')}.edf"
        return send_file(
            edf_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/edf'
        )

    return render_template('gerador1.html', channels=EEG_CHANNELS)




#implementado a parte de grupo logo abaixo
@bp.route('/generate_group', methods=['POST'])
def generate_group():
    if request.method == 'POST':
        num_people = int(request.form.get('num_people'))
        age_min = int(request.form.get('age_min'))
        age_max = int(request.form.get('age_max'))
        duration_minutes = int(request.form.get('duration_minutes'))
        selected_channels = request.form.getlist('channels')
        wave_type = request.form.get('waves')  

        if not selected_channels:
            return "Por favor, selecione pelo menos um canal.", 400

        if not wave_type:
            return "Por favor, selecione uma onda predominante.", 400


        generator = load_model(wave_type)
        if not generator:
            return f"Modelo {wave_type} não carregado.", 500

        edf_files = []

        for _ in range(num_people):
          
            name = fake.name()
            birthdate = fake.date_of_birth(minimum_age=age_min, maximum_age=age_max)

      
            eeg_data = generate_eeg_signal(generator, duration_minutes, selected_channels)
            if eeg_data is None:
                return "Erro na geração dos dados EEG.", 500

            patient_info = {
                'name': name,
                'birth_date': birthdate.strftime("%Y-%m-%d"),
            }

          
            edf_path = create_edf(eeg_data, selected_channels, patient_info, wave_type)
            if not edf_path:
                return "Erro ao criar arquivo EDF.", 500

            edf_files.append(edf_path)


        zip_filename = f'eeg_group_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip'
        zip_path = os.path.join(os.getcwd(), 'app', 'static', 'generated_files', zip_filename)

        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for index, edf_file in enumerate(edf_files):
                zipf.write(edf_file, f'Pessoa_{index+1}.edf')


        return send_file(zip_path, as_attachment=True)

    return render_template('gerador2.html')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    base64_img = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()
    return base64_img

@bp.route('/abrir_edf', methods=['GET', 'POST'])
def abrir_edf():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Nenhum arquivo foi enviado.')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('Nenhum arquivo selecionado.')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            try:
                # Gerar gráfico em base64
                base64_img = generate_eeg_plot_base64(filepath)

                f = pyedflib.EdfReader(filepath)
                info = {
                    'filename': filename,
                    'duration_sec': int(f.getNSamples()[0] / f.getSampleFrequency(0)),
                    'n_channels': f.signals_in_file,
                    'channels': f.getSignalLabels()
                }

                return render_template('mostrar_edf.html', info=info, eeg_plot=base64_img)

            except Exception as e:
                flash(f'Erro ao ler arquivo EDF: {e}')
                return redirect(request.url)

        else:
            flash('Formato de arquivo inválido. Envie um arquivo EDF.')
            return redirect(request.url)

    return render_template('abrir_edf.html')
