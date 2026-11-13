"""Token files on disk and the corpus preparation."""
import json, os
import numpy as np
import pytest
import mitsu_engine as me

PARAGRAPH = ("In the beginning the signal was silent, and the three seats rested at one.\n"
             "Then x rose, y fell, and z — ever the third force — kept the balance.  \n\n"
             "\tÜber-long words, numbers 1234567890, emoji 🎉, and 三つ巴 follow.\n")


@pytest.fixture()
def text_file(tmp_path):
    path = tmp_path / "book.txt"
    path.write_text(PARAGRAPH * 400, encoding="utf-8")
    return str(path)


def test_token_file_round_trip(tmp_path):
    assert me.token_dtype(65536) == np.uint16 and me.token_dtype(65537) == np.uint32
    for vocab in (1000, 100_000):
        ids = np.random.default_rng(0).integers(0, vocab, 5000)
        path = str(tmp_path / f"t{vocab}.bin")
        me.write_tokens(path, ids, vocab)
        back = me.open_tokens(path, vocab)
        assert isinstance(back, np.memmap) and np.array_equal(back, ids)
        assert os.path.getsize(path) == 5000 * (2 if vocab == 1000 else 4)


@pytest.mark.parametrize("chunk", [64, 257, 5000, 1 << 22])
def test_chunked_reading_and_encoding_match_the_whole_text(text_file, tmp_path, chunk):
    text = open(text_file, encoding="utf-8").read()
    assert "".join(me.iter_text(text_file, chunk_bytes=chunk)) == text
    tok = me.BPETokenizer.train(text[:20000], 400)
    out = str(tmp_path / "all.bin")
    n = me.encode_file(text_file, tok, out, chunk_bytes=chunk)
    whole = tok.encode(text)
    assert n == len(whole) and np.array_equal(me.open_tokens(out, tok.vocab_size), whole)


def test_prepare_corpus_and_load(text_file, tmp_path):
    out = str(tmp_path / "data")
    meta = me.prepare_corpus(text_file, out, vocab_size=350, val_fraction=0.1, log=None)
    assert sorted(os.listdir(out)) == ["meta.json", "tokenizer.json", "train.bin", "val.bin"]
    assert meta == json.load(open(os.path.join(out, "meta.json")))
    assert meta["train_bytes"] + meta["val_bytes"] == os.path.getsize(text_file) and 0.05 < meta["val_bytes"] / meta["bytes"] < 0.15
    tok, train, val, meta2 = me.load_corpus(out)
    assert meta2 == meta and len(train) == meta["train_tokens"] and len(val) == meta["val_tokens"] and tok.vocab_size == 350
    text = open(text_file, encoding="utf-8").read()
    assert tok.decode(np.concatenate([train, val])) == text
    raw = open(text_file, "rb").read()
    assert raw[meta["train_bytes"] - 1:meta["train_bytes"]] == b"\n"          # the split falls on a line boundary


def test_prepare_with_existing_tokenizer_and_no_validation(text_file, tmp_path):
    tok = me.BPETokenizer.train(PARAGRAPH * 5, 300)
    meta = me.prepare_corpus(text_file, str(tmp_path / "d"), val_fraction=0.0, tokenizer=tok, log=None)
    assert meta["val_tokens"] == 0 and meta["vocab_size"] == 300
    _, train, val, _ = me.load_corpus(str(tmp_path / "d"))
    assert val is None and len(train) == meta["train_tokens"]


def test_batcher_reads_from_a_memory_mapped_file(tmp_path):
    ids = np.arange(3000) % 500
    path = str(tmp_path / "t.bin")
    me.write_tokens(path, ids, 500)
    batcher = me.TextBatcher(me.open_tokens(path, 500), context=16, batch_size=8, rng=0)
    x, y = batcher()
    assert x.dtype == y.dtype == np.int64 and x.shape == (8, 16) and np.array_equal(x[:, 1:], y[:, :-1])
    assert sum(len(a) for a, _ in batcher.windows(batch_size=32)) == (3000 - 1) // 16
    limited = list(batcher.windows(batch_size=4, max_windows=10))
    assert sum(len(a) for a, _ in limited) == 10 and limited[0][0][0, 0] == ids[0]
    with pytest.raises(ValueError):
        me.TextBatcher(ids[:10], context=16, batch_size=2)
