"""Ponto de entrada para executar a aplicação Flask do gerador de EEG em modo debug."""

from app import create_app
import os


app = create_app()

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
