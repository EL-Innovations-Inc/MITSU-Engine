"""Turning a text file into token files on disk, and reading them back without loading them.

    prepare_corpus("book.txt", "data/book", vocab_size=4096)
        data/book/tokenizer.json   the trained tokenizer
        data/book/train.bin        training tokens, raw unsigned integers
        data/book/val.bin          held-out tokens (the last part of the file)
        data/book/meta.json        sizes and settings

    train = open_tokens("data/book/train.bin", vocab_size)     # numpy memmap: stays on disk
    batcher = TextBatcher(train, context=256, batch_size=32)

The text is read and encoded in pieces, so files far larger than memory can be prepared.
"""
import json, os
import numpy as np
from .tokenizer import BPETokenizer

_WS = b" \n\t\r"


def token_dtype(vocab_size):
    """Smallest unsigned integer type that holds every token id."""
    return np.uint16 if vocab_size <= 1 << 16 else np.uint32


def write_tokens(path, ids, vocab_size):
    np.asarray(ids).astype(token_dtype(vocab_size)).tofile(path)


def open_tokens(path, vocab_size):
    """Memory-map a token file for reading. Indexing it reads only the parts you ask for."""
    return np.memmap(path, dtype=token_dtype(vocab_size), mode="r")


def _split_offset(path, val_fraction):
    """Byte offset where the validation part starts: the first line start at or after the target."""
    size = os.path.getsize(path)
    if val_fraction <= 0:
        return size, size
    with open(path, "rb") as f:
        f.seek(int(size * (1.0 - val_fraction)))
        f.readline()
        return f.tell(), size


def iter_text(path, start=0, end=None, chunk_bytes=1 << 22):
    """Yield the file's text between byte offsets start and end in chunks.

    A chunk always ends just before an ASCII whitespace byte that follows a non-whitespace
    ASCII byte. Tokenizer pieces never span such a point, so encoding the chunks one by
    one gives exactly the tokens that encoding the whole text at once would give."""
    end = os.path.getsize(path) if end is None else end
    with open(path, "rb") as f:
        f.seek(start)
        remaining, buf = end - start, b""
        while remaining > 0 or buf:
            block = f.read(min(chunk_bytes, remaining)) if remaining > 0 else b""
            remaining -= len(block)
            buf += block
            if remaining <= 0:
                yield buf.decode("utf-8", errors="replace")
                return
            cut = None
            for i in range(len(buf) - 1, max(len(buf) - 65536, 0), -1):
                if buf[i] in _WS and buf[i - 1] < 0x80 and buf[i - 1] not in _WS:
                    cut = i
                    break
            if cut is None:
                continue                        # no safe cut yet: read more
            yield buf[:cut].decode("utf-8", errors="replace")
            buf = buf[cut:]


def encode_file(path, tokenizer, out_path, start=0, end=None, chunk_bytes=1 << 22, log=None):
    """Encode part of a text file to a token file, piece by piece. Returns the token count."""
    dtype, total = token_dtype(tokenizer.vocab_size), 0
    with open(out_path, "wb") as out:
        for text in iter_text(path, start, end, chunk_bytes):
            ids = tokenizer.encode(text)
            ids.astype(dtype).tofile(out)
            total += len(ids)
            if log:
                log(f"  encoded {total:,} tokens")
    return total


def prepare_corpus(text_path, out_dir, vocab_size=1024, val_fraction=0.1, special_tokens=(), tokenizer=None,
                   tokenizer_bytes=64 << 20, chunk_bytes=1 << 22, log=print):
    """Train a tokenizer on the training part of `text_path` and write train.bin / val.bin.

    The validation part is the last `val_fraction` of the file and is never seen by the
    tokenizer training. For very large files the tokenizer is trained on the first
    `tokenizer_bytes` of the training part. Pass `tokenizer` to reuse an existing one.
    Returns the metadata dict that is also written to meta.json."""
    log = log or (lambda *a: None)
    os.makedirs(out_dir, exist_ok=True)
    split, size = _split_offset(text_path, val_fraction)
    if tokenizer is None:
        log(f"training a {vocab_size}-token tokenizer on {min(split, tokenizer_bytes):,} bytes")
        sample = "".join(iter_text(text_path, 0, min(split, tokenizer_bytes), chunk_bytes))
        tokenizer = BPETokenizer.train(sample, vocab_size, special_tokens, log=log)
    tokenizer.save(os.path.join(out_dir, "tokenizer.json"))
    n_train = encode_file(text_path, tokenizer, os.path.join(out_dir, "train.bin"), 0, split, chunk_bytes)
    n_val = encode_file(text_path, tokenizer, os.path.join(out_dir, "val.bin"), split, size, chunk_bytes)
    meta = {"source": os.path.basename(text_path), "bytes": size, "vocab_size": tokenizer.vocab_size,
            "dtype": np.dtype(token_dtype(tokenizer.vocab_size)).name, "train_tokens": n_train, "val_tokens": n_val,
            "train_bytes": split, "val_bytes": size - split, "val_fraction": val_fraction}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    log(f"train {n_train:,} tokens ({split / max(n_train, 1):.2f} bytes per token)   val {n_val:,} tokens   vocabulary {tokenizer.vocab_size}")
    return meta


def load_corpus(data_dir):
    """Open a folder written by prepare_corpus. Returns (tokenizer, train tokens, val tokens, meta)."""
    with open(os.path.join(data_dir, "meta.json")) as f:
        meta = json.load(f)
    tokenizer = BPETokenizer.load(os.path.join(data_dir, "tokenizer.json"))
    train = open_tokens(os.path.join(data_dir, "train.bin"), meta["vocab_size"])
    val = open_tokens(os.path.join(data_dir, "val.bin"), meta["vocab_size"]) if meta["val_tokens"] else None
    return tokenizer, train, val, meta
