"""The byte-pair tokenizer."""
import numpy as np
import pytest
from mitsu_engine import BPETokenizer
from mitsu_engine.tokenizer import _PIECES

TEXT = ("The quick brown fox jumps over the lazy dog. The dog, being lazy, does not jump back!\n"
        "Numbers like 12345 and 3.14159; under_scores, UPPER and lower, tabs\tand   runs of   spaces.\n\n") * 30

TRICKY = ["", " ", "\n\n\n", "a", "Ünïcödé çhârs", "emoji 🎉🎉 and 三つ巴", "MITSU_c = s(s−1)/(s+2)", "  leading and trailing  ",
          "tabs\t\tand\r\nwindows lines", "x" * 500, "__dunder__ and snake_case_names", "100,000.50 + 7e-3", " non-breaking spaces"]


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer.train(TEXT, 330)


def test_piece_pattern_loses_nothing():
    r = np.random.default_rng(0)
    alphabet = list("abcXYZ 019_.,!?\n\t-é三🎉 ")
    for _ in range(300):
        s = "".join(r.choice(alphabet, size=int(r.integers(0, 40))))
        assert "".join(_PIECES.findall(s)) == s


@pytest.mark.parametrize("s", TRICKY + [TEXT])
def test_round_trip_is_lossless(tok, s):
    ids = tok.encode(s)
    assert ids.dtype == np.int64 and tok.decode(ids) == s
    assert len(ids) == 0 or (ids.min() >= 0 and ids.max() < tok.vocab_size)


def test_training_reaches_the_size_compresses_and_is_deterministic(tok):
    assert tok.vocab_size == 330 and len(tok.merges) == 74
    assert len(tok.encode(TEXT)) < 0.5 * len(TEXT.encode("utf-8"))
    assert BPETokenizer.train(TEXT, 330).merges == tok.merges
    assert BPETokenizer.train(TEXT, 256).merges == [] and BPETokenizer.train(TEXT, 256).decode(BPETokenizer.train(TEXT, 256).encode("héllo")) == "héllo"


def test_training_stops_when_nothing_repeats():
    t = BPETokenizer.train("abcdefg", 1000)
    assert t.merges == [] and t.vocab_size == 256
    t = BPETokenizer.train("ab ab ab cd", 1000)
    assert len(t.merges) == 2 and t.token(257) == b" ab"


def test_merges_stay_inside_pieces(tok):
    for i in range(256, tok.vocab_size):
        piece = tok.token(i).decode("utf-8", errors="replace")
        assert len(_PIECES.findall(piece)) == 1


def test_special_tokens():
    eot = "<|endoftext|>"
    t = BPETokenizer.train(TEXT + eot + TEXT, 300, special_tokens=[eot])
    assert t.vocab_size == 300 and t.special_ids[eot] == 299
    ids = t.encode("first doc" + eot + "second doc")
    assert (ids == 299).sum() == 1 and t.decode(ids) == "first doc" + eot + "second doc"
    plain = t.encode("first doc" + eot, allow_special=False)
    assert 299 not in plain and t.decode(plain) == "first doc" + eot
    assert all(b"<|" not in t.token(i) for i in range(256, 299))
    with pytest.raises(ValueError):
        BPETokenizer([], ["a", "a"])
    with pytest.raises(ValueError):
        BPETokenizer.train(TEXT, 256, special_tokens=["<x>"])


def test_save_and_load(tok, tmp_path):
    path = str(tmp_path / "tokenizer.json")
    tok.save(path)
    again = BPETokenizer.load(path)
    assert again.merges == tok.merges and np.array_equal(again.encode(TEXT), tok.encode(TEXT))
    other = tmp_path / "other.json"
    other.write_text('{"format": "something else"}')
    with pytest.raises(ValueError):
        BPETokenizer.load(str(other))


def test_cut_multibyte_character_decodes_without_error(tok):
    ids = tok.encode("é")
    assert "�" in tok.decode(ids[:1]) or tok.decode(ids[:1]) == "é"
