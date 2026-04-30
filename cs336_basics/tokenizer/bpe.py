from collections import Counter, defaultdict
from dataclasses import dataclass, field
import logging
import os
from pprint import pformat
import regex as re

from cs336_basics.tokenizer.pretokenization import find_chunk_boundaries


logger = logging.getLogger(__name__)

class BpeContext:
    def __init__(self, input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str],) -> None:
        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.num_processes = 4
        self.PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pass


@dataclass
class BPFreqVal:
    freq: int = 0
    in_pretoken_bytes: set[tuple[bytes]] = field(default_factory=set)


class BPE:
    def __init__(self, input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str],) -> None:
        self.context = BpeContext(input_path, vocab_size, special_tokens)

    def train(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        freq_tables: list[dict[tuple[bytes], int]] = []

        with open(self.context.input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, self.context.num_processes, self.context.special_tokens[0].encode("utf-8"))

            for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                chunk_processor = ChunkProcessor(self.context, i, chunk, start, end)
                freq_tables.append(chunk_processor.process_chunk())

        self.freq_table: dict[tuple[bytes], int] = dict[tuple[bytes], int](sum([Counter(t) for t in freq_tables], Counter()))

        logger.debug(f"Final pretoken freq table:\n{pformat(self.freq_table, width=160)}")

        self.bp_freq_table: dict[tuple[bytes], BPFreqVal] = self._build_bp_freq_table()

        logger.debug(f"Byte-pair freq table:\n{pformat(self.bp_freq_table, width=160)}")

        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.vocab |= {(len(self.vocab) + idx): t.encode('utf-8') for idx, t in enumerate(self.context.special_tokens)}
        self.merges: list[tuple[bytes, bytes]] = []

        while len(self.vocab) < self.context.vocab_size:
            picked_bp: tuple[bytes] = max(self.bp_freq_table, key=lambda x: (self.bp_freq_table[x].freq, x))

            self.vocab[len(self.vocab)] = b"".join(picked_bp)
            self.merges.append(picked_bp)

            logger.debug(f"Picked byte-pair: {picked_bp}")

            self._update_bp_freq_table(picked_bp)
        
        logger.debug(f"Final vocab: {self.vocab}\nFinal merges: {self.merges}")

        return (self.vocab, self.merges)
    
    def _build_bp_freq_table(self) -> dict[tuple[bytes], BPFreqVal]:
        bp_freq_table: dict[tuple[bytes], BPFreqVal] = defaultdict(BPFreqVal)
        for pretoken_bytes, pretoken_freq in self.freq_table.items():
            for left, right in zip(pretoken_bytes[:-1], pretoken_bytes[1:]):
                bp_freq_table[(left, right)].freq += pretoken_freq
                bp_freq_table[(left, right)].in_pretoken_bytes.add(pretoken_bytes)
        return bp_freq_table
    
    def _update_bp_freq_table(self, picked_bp: tuple[bytes]):
        in_pretoken_bytes: set[tuple[bytes]] = set(self.bp_freq_table[picked_bp].in_pretoken_bytes)
        for pretoken_bytes in in_pretoken_bytes:
            logger.debug(f"Merging with pretoken {pretoken_bytes} and {picked_bp}")
            for left, right in zip(pretoken_bytes[:-1], pretoken_bytes[1:]):
                self.bp_freq_table[(left, right)].freq -= self.freq_table[pretoken_bytes]
                self.bp_freq_table[(left, right)].in_pretoken_bytes.remove(pretoken_bytes)

            logger.debug(f"Byte-pair freq table (during merge, removal):\n{pformat(self.bp_freq_table, width=160)}")
            pretoken_bytes_new_list: list[bytes] = []
            i = 0
            while i < len(pretoken_bytes):
                if pretoken_bytes[i:i+2] == picked_bp:
                    pretoken_bytes_new_list.append(b"".join(picked_bp))
                    i += 2
                else:
                    pretoken_bytes_new_list.append(pretoken_bytes[i])
                    i += 1
            pretoken_bytes_new: tuple[bytes] = tuple(pretoken_bytes_new_list)
            logger.debug(f"pretoken before: {pretoken_bytes} after: {pretoken_bytes_new}")

            assert pretoken_bytes_new not in self.freq_table, f"{pretoken_bytes_new} should not exist in freq_table"
            assert len(pretoken_bytes) - len(pretoken_bytes_new) >= 1

            self.freq_table[pretoken_bytes_new] = self.freq_table.pop(pretoken_bytes)
            for left, right in zip(pretoken_bytes_new[:-1], pretoken_bytes_new[1:]):
                self.bp_freq_table[(left, right)].freq += self.freq_table[pretoken_bytes_new]
                self.bp_freq_table[(left, right)].in_pretoken_bytes.add(pretoken_bytes_new)
            logger.debug(f"Byte-pair freq table (after merge):\n{pformat(self.bp_freq_table, width=160)}")


# TODO: parallelize worker
class ChunkProcessor:
    def __init__(self, bpe_context: BpeContext, processor_id: str, chunk: str, start: int, end: int, ) -> None:
        self.bpe_context = bpe_context
        self.processor_id = processor_id
        self.chunk = chunk
        self.start, self.end = start, end
        self.freq_table: dict[tuple[bytes], int] = defaultdict[tuple[bytes], int](int)

    def process_chunk(self):
        logger.debug(f"\n[{self.processor_id}] chunk:\n{self.chunk}\n===========================")
        docs = re.split("|".join([re.escape(t) for t in self.bpe_context.special_tokens]), self.chunk)
        docs = [d for d in docs if d.strip()]
        for i, doc in enumerate(docs):
            self._update_freq_table_with_doc(f"{self.processor_id}-{i}", doc)
        return self.freq_table

    def _update_freq_table_with_doc(self, doc_id, doc) -> dict[tuple[bytes], int]:
        logger.debug(f"[{doc_id}] doc: {doc}\n++++++++")
        # for pretoken in re.finditer(self.bpe_context.PAT, doc):
        for pretoken in doc.split():
            logger.debug(f"[{doc_id}] pretoken: {pretoken}\n-----")
            # pretoken_bytes = pretoken.group().encode("utf-8")
            pretoken_bytes = pretoken.encode("utf-8")
            key = tuple[bytes](bytes([b]) for b in pretoken_bytes)
            self.freq_table[key] += 1
        logger.debug(f"[{doc_id}] Freq table: \n{self.freq_table}")
        return self.freq_table


def _show_bytes_freq(freq_table: dict[tuple[bytes], int]) -> str:
    lines = []
    lines.append("======= Freq Table =======")
    for k, v in freq_table.items():
        lines.append(f"key: {bytes(k)} / {k} , freq: {v}")
    lines.append("=========================")
    return "\n".join(lines)
 