# NeuroGeradorEEG: Gerador de Sinais de EEG Sintéticos com WGAN-GP

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Flask-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

Uma aplicação web para gerar dados de eletroencefalograma (EEG) sintéticos e realistas utilizando um modelo WGAN-GP (Wasserstein Generative Adversarial Network with Gradient Penalty). Ideal para pesquisadores, estudantes e desenvolvedores que precisam de grandes volumes de dados de EEG para treinar e testar algoritmos sem se preocupar com a privacidade e o acesso a dados reais.

## 🧠 Sobre o Projeto

A obtenção de dados de EEG de qualidade é um desafio comum na área de neurociência e Interface Cérebro-Computador (BCI). Este projeto visa solucionar esse problema fornecendo uma ferramenta de fácil uso que gera arquivos de EEG no formato padrão europeu (EDF).

O coração do NeuroGeradorEEG é uma **WGAN-GP implementada em PyTorch**. O diferencial deste modelo é que ele foi **treinado com um conjunto de dados de sinais de EEG humanos reais**. Ao aprender os padrões, ritmos e complexidades diretamente da fonte, o gerador se torna capaz de produzir dados sintéticos com alta fidelidade e características dinâmicas que espelham as de uma gravação autêntica.

## ✨ Principais Funcionalidades

* **Treinado com Dados Reais:** O modelo aprendeu a gerar ondas cerebrais a partir de dados reais de amostras de EEG reais, garantindo que os sinais sintéticos sejam estruturalmente realistas e úteis para análise.
* **Gerador Individual:** Crie um único arquivo `.edf` com informações de um paciente simulado.
* **Gerador em Grupo:** Gere um lote de arquivos `.edf` para um grupo de pacientes simulados, com faixas etárias personalizadas, e baixe tudo em um único arquivo `.zip`.
* **Personalização:** Escolha a duração do sinal, os canais de EEG (seguindo o sistema 10-20) e a onda cerebral predominante.
* **Realismo Dinâmico:** A geração de sinal não é repetitiva, garantindo que longas gravações sejam variadas, evitando padrões artificiais.
* **Visualizador de EDF:** Faça o upload e visualize graficamente os sinais de qualquer arquivo `.edf`, seja ele gerado pela ferramenta ou um arquivo real.

## 🛠️ Tecnologias Utilizadas

* **Backend:** Flask
* **Deep Learning:** PyTorch
* **Manipulação de Dados:** NumPy
* **Manipulação de Arquivos EDF:** pyEDFlib
* **Visualização:** Matplotlib
* **Dados de Pacientes Fictícios:** Faker

## 🚀 Instalação e Execução

Siga os passos abaixo para executar o projeto localmente.

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/crislymss/geradorEEGGAN.git](https://github.com/crislymss/geradorEEGGAN.git)
    cd geradorEEGGAN
    ```

2.  **Crie e ative um ambiente virtual** (recomendado):
    * No Windows:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    * No macOS/Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Instale as dependências:**
    (Se o arquivo `requirements.txt` não existir, crie-o primeiro com `pip freeze > requirements.txt`)
    ```bash
    pip install -r requirements.txt
    ```

4.  **Execute a aplicação:**
    (Assumindo que seu arquivo principal se chama `run.py`)
    ```bash
    python run.py
    ```

5.  Abra seu navegador e acesse `http://127.0.0.1:5000`.

## 💻 Como Usar

* Acesse **`/gerador1`** para usar o gerador de EEG individual.
* Acesse **`/gerador2`** para usar o gerador em lote para grupos.
* Acesse **`/abrir_edf`** para fazer o upload e visualizar um arquivo `.edf` existente.

## 📄 Licença

Este projeto é disponibilizado sob a licença MIT. Seu uso é livre para fins educacionais e científicos. Veja o arquivo `LICENSE` para mais detalhes, se aplicável.

## 👤 Autor e Contato

**Crisly Santos**

* **Instituição:** Universidade Federal do Piauí (UFPI)
* **Email:** crisly.santos@ufpi.edu.br
* **GitHub:** [@crislymss](https://github.com/crislymss)
