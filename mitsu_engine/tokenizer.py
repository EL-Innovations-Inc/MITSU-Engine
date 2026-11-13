"""Byte-pair-encoding tokenizer, trained from text. Pure Python.

The base vocabulary is the 256 byte values, so any text in any language can be encoded
and decoding always gives the original text back. Training repeatedly merges the most
frequent adjacent pair of tokens into a new token until the vocabulary reaches the
requested size.

    tok = BPETokenizer.train(text, vocab_size=1024)
    ids = tok.encode("To be, or not to be")
    tok.decode(ids) == "To be, or not to be"
    tok.save("tokenizer.json");  tok = BPETokenizer.load("tokenizer.json")
"""
import json, re
from collections import Counter, defaultdict
import numpy as np

# Text is cut into pieces before merging, and merges never cross a piece boundary:
# a word with its leading space, a number, a run of punctuation, or a run of whitespace.
_PIECES = re.compile(r" ?[^\W\d_]+| ?\d+| ?(?:[^\s\w]|_)+|\s+")


class BPETokenizer:
    def __init__(self, merges=(), special_tokens=()):
        self.merges = [(int(a), int(b)) for a, b in merges]
        self.special_tokens = list(special_tokens)
        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("special tokens must be distinct")
        self.vocab = [bytes([i]) for i in range(256)]
        self._rank = {}
        for a, b in self.merges:
            self._rank[(a, b)] = len(self.vocab)
            self.vocab.append(self.vocab[a] + self.vocab[b])
        self.special_ids = {}
        for s in self.special_tokens:
            self.special_ids[s] = len(self.vocab)
            self.vocab.append(s.encode("utf-8"))
        self._special_re = (re.compile("(" + "|".join(re.escape(s) for s in sorted(self.special_tokens, key=len, reverse=True)) + ")")
                            if self.special_tokens else None)
        self._cache = {}

    # ---- training ---------------------------------------------------------
    @classmethod
    def train(cls, text, vocab_size, special_tokens=(), min_frequency=2, log=None):
        """Learn merges from `text` until the vocabulary has `vocab_size` tokens (256 bytes +
        merges + special tokens), or no pair occurs at least `min_frequency` times."""
        special_tokens = list(special_tokens)
        n_merges = vocab_size - 256 - len(special_tokens)
        if n_merges < 0:
            raise ValueError("vocab_size must be at least 256 plus the number of special tokens")
        if special_tokens:                       # special tokens never take part in merges
            text = re.sub("|".join(re.escape(s) for s in sorted(special_tokens, key=len, reverse=True)), "\n", text)
        counts = Counter(_PIECES.findall(text))
        words = [list(w.encode("utf-8")) for w in counts]
        freqs = list(counts.values())
        pair_count, where = Counter(), defaultdict(set)
        for wi, w in enumerate(words):
            for pair in zip(w, w[1:]):
                pair_count[pair] += freqs[wi]
                where[pair].add(wi)
        merges = []
        for step in range(n_merges):
            if not pair_count:
                break
            # most frequent pair; ties go to the pair with the smallest token ids (deterministic)
            pair, count = max(pair_count.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))
            if count < min_frequency:
                break
            new_id = 256 + len(merges)
            merges.append(pair)
            for wi in list(where[pair]):
                w, f = words[wi], freqs[wi]
                for p in zip(w, w[1:]):
                    pair_count[p] -= f
                    if pair_count[p] <= 0:
                        del pair_count[p]
                    where[p].discard(wi)
                merged, i = [], 0
                while i < len(w):
                    if i + 1 < len(w) and w[i] == pair[0] and w[i + 1] == pair[1]:
                        merged.append(new_id)
                        i += 2
                    else:
                        merged.append(w[i])
                        i += 1
                words[wi] = merged
                for p in zip(merged, merged[1:]):
                    pair_count[p] += f
                    where[p].add(wi)
            where.pop(pair, None)
            if log and (step + 1) % 500 == 0:
                log(f"  bpe: {step + 1} / {n_merges} merges")
        return cls(merges, special_tokens)

    # ---- encoding and decoding -------------------------------------------
    def _encode_piece(self, piece):
        ids = self._cache.get(piece)
        if ids is None:
            ids = list(piece.encode("utf-8"))
            while len(ids) > 1:
                rank, pair = min(((self._rank.get(p, 1 << 62), p) for p in zip(ids, ids[1:])))
                if rank == 1 << 62:
                    break
                merged, i = [], 0
                while i < len(ids):
                    if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                        merged.append(rank)
                        i += 2
                    else:
                        merged.append(ids[i])
                        i += 1
                ids = merged
            if len(self._cache) < 2_000_000:
                self._cache[piece] = ids
        return ids

    def _encode_plain(self, text, out):
        for piece in _PIECES.findall(text):
            out.extend(self._encode_piece(piece))

    def encode(self, text, allow_special=True):
        """Text -> array of token ids. Special tokens written in the text become their single
        id when allow_special is True; otherwise they are encoded as ordinary characters."""
        out = []
        if self._special_re is None or not allow_special:
            self._encode_plain(text, out)
        else:
            for part in self._special_re.split(text):
                if part in self.special_ids:
                    out.append(self.special_ids[part])
                elif part:
                    self._encode_plain(part, out)
        return np.array(out, dtype=np.int64)

    def decode(self, ids):
        """Token ids -> text. Bytes that do not form valid UTF-8 (a token sequence cut
        mid-character) are shown as the replacement character."""
        return b"".join(self.vocab[int(i)] for i in np.asarray(ids).reshape(-1)).decode("utf-8", errors="replace")

    def token(self, i):
        """The bytes of token i, for inspection."""
        return self.vocab[int(i)]

    @property
    def vocab_size(self):
        return len(self.vocab)

    # ---- persistence -----------------------------------------------------
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"format": "mitsu-engine-bpe", "version": 1, "merges": self.merges, "special_tokens": self.special_tokens}, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("format") != "mitsu-engine-bpe":
            raise ValueError(f"{path} is not a mitsu-engine tokenizer file")
        return cls(d["merges"], d["special_tokens"])

    def __repr__(self):
        return f"BPETokenizer(vocab_size={self.vocab_size}, merges={len(self.merges)}, special_tokens={self.special_tokens})"
