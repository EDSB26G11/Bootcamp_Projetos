"""Ponto de entrada da app no Hugging Face Spaces (T4.3).

O Space corre `python app.py` a partir da raiz. A app está em src/app.py; este
ficheiro só a carrega com o modelo gravado em models/v1/. Localmente, o mesmo
que `python -m src.app`.
"""

import gradio as gr

from src.app import CSS, FORCE_LIGHT, _theme, build_app
from src.models.train import MODEL_DIR, load_model

demo = build_app(load_model(MODEL_DIR))

if __name__ == "__main__":
    demo.launch(css=CSS, theme=_theme(gr), js=FORCE_LIGHT)
