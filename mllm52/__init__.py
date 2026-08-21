"""mllm52 — lightweight autocomplete micro LM (causal n-gram)."""

from .engine import AutocompleteEngine, CompleteResult
from .terminal import Term
from .topology import CausalTopology, tokenize

__version__ = "5.2"

__all__ = [
    "AutocompleteEngine",
    "CausalTopology",
    "CompleteResult",
    "Term",
    "tokenize",
    "__version__",
]
