"""Tests for mllm52.cli — corpus-required."""

import builtins
import pytest
from pathlib import Path

from mllm52.cli import CORPUS_ERROR, main
from mllm52.topology import CausalTopology

CORPUS = (
    "the cat sat on the mat. the cat ate the fish. the dog sat on the mat. "
    "a cat is an animal. the dog is an animal. animals eat food and sleep."
)


@pytest.fixture
def corpus_file(tmp_path):
    p = tmp_path / "corpus.txt"
    p.write_text(CORPUS, encoding="utf-8")
    return p


def test_corpus_required_no_arg_fails(capsys):
    rc = main([])
    assert rc == 2
    err = capsys.readouterr().err
    assert "No corpus found" in err or "corpus" in err.lower()


def test_corpus_required_missing_file_fails(tmp_path, capsys):
    rc = main(["--corpus", str(tmp_path / "nope.txt"), "autocomplete", "hello", "--plain"])
    assert rc == 2
    assert "corpus" in capsys.readouterr().err.lower()


def test_autocomplete_generates_with_corpus(corpus_file, capsys):
    rc = main(["--corpus", str(corpus_file), "--seed", "42", "--plain", "autocomplete", "the cat", "--steps", "5"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "the cat" in out.lower()


def test_bare_prefix_mode(corpus_file, capsys):
    rc = main(["--corpus", str(corpus_file), "--seed", "42", "--plain", "the cat"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "the cat" in out.lower()


def test_chat_eof_exits(corpus_file, monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda prompt="": (_ for _ in ()).throw(EOFError))
    rc = main(["--corpus", str(corpus_file), "--seed", "1", "chat", "--steps", "4"])
    assert rc == 0


def test_topology_causal_only(corpus_file):
    # Ensure left_counts built, no right_counts attribute
    topo = CausalTopology.from_text(CORPUS)
    assert hasattr(topo, "left_counts")
    assert not hasattr(topo, "right_counts")
    assert topo.left_counts[1][("the",)]["cat"] > 0
