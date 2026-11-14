"""Publica a app no Hugging Face Spaces (T4.3).

Envia para o Space só o que a app precisa para correr: app.py, o código de
src/, o modelo de models/v1/ (que não está no Git), requirements-space.txt
(como requirements.txt) e o README do Space (docs/space_README.md, que tem a
configuração do Space no cabeçalho). Os dados não vão para o Space.

Antes de correr:
1. gravar o modelo (notebooks/06_deployment.ipynb), para existir models/v1/;
2. autenticar com um token com permissão de escrita no Space:
   `hf auth login` (ou a variável de ambiente HF_TOKEN).

Correr a partir da raiz do repositório:
    python scripts/deploy_space.py
    python scripts/deploy_space.py --space outro-utilizador/outro-space
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
SPACE = "SandraLurdes/Bootcampprojectodiabetes"
MODEL_FILES = ["model.skops", "config.json", "README.md"]


def main(space: str) -> None:
    missing = [f for f in MODEL_FILES if not (ROOT / "models" / "v1" / f).exists()]
    if missing:
        raise SystemExit(f"Falta models/v1/{missing}: correr primeiro o notebooks/06_deployment.ipynb.")

    api = HfApi()
    print(f"A publicar em https://huggingface.co/spaces/{space} como {api.whoami()['name']}…")
    api.upload_folder(
        repo_id=space, repo_type="space", folder_path=ROOT,
        allow_patterns=["app.py", "src/**/*.py", "models/v1/*"],
        ignore_patterns=["**/__pycache__/**", "src/models/kumo.py", "src/data_preparation.py"],
        commit_message="Publica a app e o modelo v1",
    )
    api.upload_file(repo_id=space, repo_type="space", path_or_fileobj=ROOT / "requirements-space.txt",
                    path_in_repo="requirements.txt", commit_message="Dependências da app")
    api.upload_file(repo_id=space, repo_type="space", path_or_fileobj=ROOT / "docs" / "space_README.md",
                    path_in_repo="README.md", commit_message="README do Space")
    print("Feito. O Space reconstrói-se sozinho; acompanhar o separador Logs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--space", default=SPACE, help="id do Space (utilizador/nome)")
    main(parser.parse_args().space)
