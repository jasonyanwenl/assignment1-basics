from collections import defaultdict
from functools import lru_cache
import json
import logging
import os
import regex as re
from tests.common import gpt2_bytes_to_unicode
from typing import Iterable, Iterator


logger = logging.getLogger(__name__)

BYTE1: tuple[bytes] = tuple(bytes([b]) for b in range(256))

@lru_cache(maxsize=8192)
def _pretoken_to_bytes_tuple(pretoken: str) -> tuple[bytes, ...]:
    return tuple(BYTE1[i] for i in pretoken.encode("utf-8"))

class Tokenizer:
    def __init__(
        self, 
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str] | None =None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        self.PAT: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.merge2rank: dict[tuple[bytes, bytes], int] = {merge: rank for rank, merge in enumerate(merges)}
        escaped = [re.escape(t) for t in sorted(self.special_tokens, key=len, reverse=True)]
        self.split_delimiter = '(' + "|".join(escaped) + ')'

        self.vocab2id: dict[bytes, int] = defaultdict(int)
        next_id = float('-inf')
        for k, v in vocab.items():
            self.vocab2id[v] = k
            next_id = max(next_id, k)

        logger.info("next id: %s", next_id)

        for special_token in self.special_tokens:
            if special_token not in self.vocab2id:
                special_token_bytes = special_token.encode('utf-8')
                self.vocab[next_id] = special_token_bytes
                self.vocab2id[special_token_bytes] = next_id
                next_id += 1
            logger.info("special token: %s, id: %s", special_token, self.vocab2id[special_token_bytes])

    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        logger.info("input text: %s", text)
        encoded = []
        text_iterator = (
            re.splititer(self.split_delimiter, text) 
            if self.special_tokens else (text,)
        )
        encoded = []
        for i, doc in enumerate(text_iterator):
            if doc in self.special_tokens:
                encoded.append(self.vocab2id[doc.encode('utf-8')])
                continue
            for pretoken in re.finditer(self.PAT, doc):
                encoded.extend(self._encode_pretoken(pretoken.group()))
        logger.info("encoded: %s", encoded)
        return encoded

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        decoded: bytes = b"".join(self.vocab[id] for id in ids)
        logger.info("decoded: %s", decoded)
        return decoded.decode('utf-8', errors='replace')

    def _encode_pretoken(self, pretoken: str) -> list[int]:
        logger.info("input pretoken: %s", pretoken)
        pretoken_bytes = _pretoken_to_bytes_tuple(pretoken)
        logger.info("pretoken_bytes: %s", pretoken_bytes)
        while len(pretoken_bytes) > 1:
            logger.info("len(pretoken_bytes): %s, pretoken_bytes: %s", len(pretoken_bytes), pretoken_bytes)
            curr_rank = len(self.merges)
            picked_bp = None
            for bp in zip(pretoken_bytes[:-1], pretoken_bytes[1:]):
                if bp in self.merge2rank and self.merge2rank[bp] < curr_rank:
                    curr_rank = self.merge2rank[bp]
                    picked_bp = bp
            logger.info("picked_bp: %s", picked_bp)
            if not picked_bp:
                break
            
            pretoken_bytes_new_list: list[bytes] = []
            i = 0
            while i < len(pretoken_bytes):
                if pretoken_bytes[i:i+2] == picked_bp:
                    pretoken_bytes_new_list.append(b"".join(picked_bp))
                    i += 2
                else:
                    pretoken_bytes_new_list.append(pretoken_bytes[i])
                    i += 1
            pretoken_bytes: tuple[bytes, ...] = tuple(pretoken_bytes_new_list)
        logger.info("merged pretoken_bytes: %s", pretoken_bytes)
        encoded = []
        for curr_vocab in pretoken_bytes:
            encoded.append(self.vocab2id[curr_vocab])
            logger.info("curr_vocab: %s, id: %s", curr_vocab, encoded[-1])
        return encoded


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    def get_tokenizer_from_vocab_merges_path(
        vocab_path: str | os.PathLike,
        merges_path: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ):
        gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
        with open(vocab_path) as vocab_f:
            gpt2_vocab = json.load(vocab_f)
        gpt2_bpe_merges = []
        with open(merges_path) as f:
            for line in f:
                cleaned_line = line.rstrip()
                if cleaned_line and len(cleaned_line.split(" ")) == 2:
                    gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))
        # The GPT-2 tokenizer uses a remapped unicode encoding for bytes. Let's
        # just return the original bytes, so we don't force students to use
        # any particular encoding scheme.
        vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
        }
        # If any of the special tokens don't exist in the vocab, append them to the vocab.
        if special_tokens:
            for special_token in special_tokens:
                byte_encoded_special_token = special_token.encode("utf-8")
                if byte_encoded_special_token not in set(vocab.values()):
                    vocab[len(vocab)] = byte_encoded_special_token

        merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_bpe_merges
        ]
        return Tokenizer(vocab, merges, special_tokens)

    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path= "tests/fixtures/gpt2_vocab.json",
        merges_path= "tests/fixtures/gpt2_merges.txt"
    )
    test_string = "Hello, how are you?"
    encoded_ids = tokenizer.encode(test_string)
    logger.info("encoded_ids: %s", encoded_ids)
    decoded_string = tokenizer.decode(encoded_ids)
    logger.info("decoded_string: %s", decoded_string)