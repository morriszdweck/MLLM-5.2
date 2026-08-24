#!/usr/bin/env python3
"""Extract embedded corpora from the MLLM-5.2 model files into editor/corpora/.

The web editor fetches these plain-text copies at runtime; they are generated
artifacts (gitignored). Run from the repo root:

    python3 editor/extract_corpora.py

Abyss is skipped on purpose — it is the bring-your-own-corpus model.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "editor" / "corpora"

MODELS = {
    "MLLM-5.2-Muir-26P.py": "muir",
    "MLLM-5.2-Monterey-71P.py": "monterey",
    "MLLM-5.2-Tahoe-309P.py": "tahoe",
    "MLLM-5.2-Whitney-1042P.py": "whitney",
    "MLLM-5.2-Golden-3981P.py": "golden",
}


def load_corpus(path: Path) -> str:
    spec = importlib.util.spec_from_file_location("mllm_mod", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # needed by @dataclass on Python 3.14
    spec.loader.exec_module(module)
    return module.BUILT_IN_CORPUS


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for filename, short in MODELS.items():
        corpus = load_corpus(ROOT / filename)
        dest = OUT / f"{short}.txt"
        dest.write_text(corpus, encoding="utf-8")
        print(f"{short:10s} {len(corpus):>10,} chars -> {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
