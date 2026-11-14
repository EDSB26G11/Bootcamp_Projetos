"""Publica a app no Hugging Face Spaces (T4.3).

Envia para o Space só o que a app precisa para correr: app.py, o código de
src/, o modelo de models/v1/ (que não está no Git), requirements-space.txt
(como requirements.txt) e o README do Space (docs/space_README.md, que tem a
configuração do Space no cabeçalho). Os dados não vão para o Space.

Normalmente corre sozinho no GitHub Actions, em cada push para a main, depois
dos testes (.github/workflows/ci.yml, com o secret HF_TOKEN). Para correr à mão:
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
    try:
        user = api.whoami()["name"]
    except Exception as e:
        raise SystemExit(f"Sem autenticação no Hugging Face ({e.__class__.__name__}). No GitHub: confirmar o secret TAFS "
                         "(Settings → Secrets and variables → Actions) com um token Write. "
                         "À mão: `hf auth login`.")
    if not api.repo_exists(space, repo_type="space"):
        raise SystemExit(f"O Space {space} não existe ou a conta {user} não o vê.")
    print(f"A publicar em https://huggingface.co/spaces/{space} como {user}…")
    try:
        _upload(api, space)
    except Exception as e:
        raise SystemExit(f"O envio falhou ({e}). Se for 401/403: a conta {user} não tem escrita no Space "
                         "(o token tem de ser Write e de uma conta com acesso ao Space).")
    print("Feito. O Space reconstrói-se sozinho; acompanhar o separador Logs.")


def _upload(api: HfApi, space: str) -> None:
    api.upload_folder(
        repo_id=space, repo_type="space", folder_path=ROOT,
        # "src/*.py" e "src/**/*.py": o segundo só apanha ficheiros dentro de subpastas de src/.
        allow_patterns=["app.py", "src/*.py", "src/**/*.py", "models/v1/*"],
        ignore_patterns=["**/__pycache__/**", "src/models/kumo.py", "src/data_preparation.py"],
        commit_message="Publica a app e o modelo v1",
    )
    api.upload_file(repo_id=space, repo_type="space", path_or_fileobj=ROOT / "requirements-space.txt",
                    path_in_repo="requirements.txt", commit_message="Dependências da app")
    api.upload_file(repo_id=space, repo_type="space", path_or_fileobj=ROOT / "docs" / "space_README.md",
                    path_in_repo="README.md", commit_message="README do Space")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--space", default=SPACE, help="id do Space (utilizador/nome)")
    main(parser.parse_args().space)
