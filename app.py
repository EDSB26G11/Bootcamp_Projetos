"""Ponto de entrada da app no Hugging Face Spaces (T4.3).

O Space corre `python app.py` a partir da raiz. A app está em src/app.py; este
ficheiro só a carrega com o modelo gravado em models/v1/. Localmente, o mesmo
que `python -m src.app`.

ZeroGPU: o Space só tem hardware ZeroGPU disponível, e um Space ZeroGPU só
arranca se encontrar pelo menos uma função com @spaces.GPU. O modelo
(HistGradientBoosting) não precisa de GPU: a app corre toda em CPU, como num
Space normal, e a função marcada abaixo nunca é chamada, por isso não gasta
quota de GPU. Fora do Space (sem o pacote `spaces`) o decorador não se aplica.
"""

import gradio as gr

from src.app import CSS, FORCE_LIGHT, _theme, build_app
from src.models.train import MODEL_DIR, load_model

try:
    import spaces
except ImportError:  # localmente e no CI: não há ZeroGPU
    spaces = None

if spaces is not None:
    @spaces.GPU
    def _zerogpu_startup_check():
        """Só existe para o ZeroGPU deixar o Space arrancar; nunca é chamada."""
        return None

demo = build_app(load_model(MODEL_DIR))

if __name__ == "__main__":
    demo.launch(css=CSS, theme=_theme(gr), js=FORCE_LIGHT)
