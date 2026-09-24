"""
Word-level tokenizer for avatar captions.

Vocabulary is built ONLY from the training split's captions.
"""
import json
import re
from pathlib import Path
from typing import List, Dict

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]

_word_re = re.compile(r"[a-zA-Z]+")


def simple_word_tokenize(text: str) -> List[str]:
    return _word_re.findall(text.lower())


class Tokenizer:
    def __init__(self, max_len: int = 24):
        self.max_len = max_len
        self.token2id: Dict[str, int] = {}
        self.id2token: Dict[int, str] = {}
        self._fitted = False

    def build_vocab(self, train_captions: List[str]):
        """Build vocabulary strictly from the given (training-split) captions."""
        vocab = set()
        for cap in train_captions:
            vocab.update(simple_word_tokenize(cap))
        tokens = SPECIAL_TOKENS + sorted(vocab)
        self.token2id = {t: i for i, t in enumerate(tokens)}
        self.id2token = {i: t for t, i in self.token2id.items()}
        self._fitted = True

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    @property
    def pad_id(self) -> int:
        return self.token2id[PAD]

    def encode(self, text: str) -> List[int]:
        assert self._fitted, "Call build_vocab() on the TRAIN split before encoding."
        words = simple_word_tokenize(text)
        ids = [self.token2id[BOS]]
        for w in words[: self.max_len - 2]:
            ids.append(self.token2id.get(w, self.token2id[UNK]))
        ids.append(self.token2id[EOS])
        while len(ids) < self.max_len:
            ids.append(self.token2id[PAD])
        return ids[: self.max_len]

    def decode(self, ids: List[int]) -> str:
        words = [self.id2token.get(i, UNK) for i in ids]
        words = [w for w in words if w not in (PAD, BOS, EOS)]
        return " ".join(words)

    # --- persistence, so the exact vocab used for training is reproducible ---
    def save(self, path: str):
        Path(path).write_text(json.dumps({
            "max_len": self.max_len,
            "token2id": self.token2id,
        }, indent=2))

    @classmethod
    def load(cls, path: str) -> "Tokenizer":
        data = json.loads(Path(path).read_text())
        tok = cls(max_len=data["max_len"])
        tok.token2id = {k: int(v) for k, v in data["token2id"].items()}
        tok.id2token = {v: k for k, v in tok.token2id.items()}
        tok._fitted = True
        return tok