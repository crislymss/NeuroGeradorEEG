"""Fábrica da aplicação Flask e registro do blueprint do gerador de EEG."""

from flask import Flask
from .routes import bp


def create_app():
    """Construir e configurar a instância principal da aplicação Flask."""
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.register_blueprint(bp)
    return app