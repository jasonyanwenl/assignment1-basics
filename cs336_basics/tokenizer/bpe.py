from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
import heapq
import logging
from multiprocessing import Pool
import os
from pprint import pformat
import regex as re

from cs336_basics.tokenizer.pretokenization import find_chunk_boundaries


logger = logging.getLogger(__name__)

@dataclass
class BPEContext:
    input_path: str | os.PathLike
    vocab_size: int
    special_tokens: list[str]
    num_processes: int = 16
    PAT: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


@dataclass
class BPFreqVal:
    freq: int = 0
    in_pretoken_bytes: set[tuple[bytes]] = field(default_factory=set)


@dataclass
class HeapEntry:
    freq: int
    bp: tuple[bytes, ...]

    def __lt__(self, other):
        if not isinstance(other, HeapEntry):
            return NotImplemented
        if self.freq != other.freq:
            return self.freq > other.freq
        return self.bp > other.bp


class BPE:
    def __init__(self, input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str],) -> None:
        self.context = BPEContext(input_path, vocab_size, special_tokens)

    def train(self) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        freq_tables: list[dict[tuple[bytes], int]] = []

        chunk_processor_inputs: list[tuple] = []

        with open(self.context.input_path, "rb") as f:
            boundaries = find_chunk_boundaries(f, self.context.num_processes, self.context.special_tokens[0].encode("utf-8"))
            for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
                chunk_processor_inputs.append((self.context, i, start, end))

        # for chunk_input in chunk_processor_inputs:
        #     freq_tables.append(chunk_processor(*chunk_input))
        with Pool(self.context.num_processes) as pool:
            freq_tables = pool.starmap(chunk_processor, chunk_processor_inputs)
        assert len(freq_tables) <= self.context.num_processes, \
            f"Pretoken freq table list size: {len(freq_tables)} > num processes {self.context.num_processes}"

        self.freq_table: Counter[tuple[bytes]] = sum(freq_tables, Counter())

        # logger.debug(f"Final pretoken freq table:\n{pformat(self.freq_table, width=160)}")

        self.bp_freq_table: dict[tuple[bytes], BPFreqVal] = self._build_bp_freq_table()

        # logger.debug(f"Byte-pair freq table:\n{pformat(self.bp_freq_table, width=160)}")

        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.vocab |= {(len(self.vocab) + idx): t.encode('utf-8') for idx, t in enumerate(self.context.special_tokens)}
        self.merges: list[tuple[bytes, bytes]] = []

        heap: list[HeapEntry] = [HeapEntry(v.freq, k) for k, v in self.bp_freq_table.items()]
        heapq.heapify(heap)

        while len(self.vocab) < self.context.vocab_size:
            picked_bp = None
            while heap:
                entry = heap[0]
                if entry.freq == self.bp_freq_table[entry.bp].freq:
                    picked_bp = entry.bp
                    break
                heapq.heappop(heap)

            if __debug__:
                picked_bp2: tuple[bytes] = max(self.bp_freq_table, key=lambda x: (self.bp_freq_table[x].freq, x))
                assert picked_bp2 == picked_bp, f"picked_bp: {picked_bp} != picked_bp2: {picked_bp2}"

            self.vocab[len(self.vocab)] = b"".join(picked_bp)
            self.merges.append(picked_bp)

            logger.info(f"Picked the {len(self.vocab)} byte-pair: {picked_bp} / {b''.join(picked_bp)}, freq: {self.bp_freq_table[picked_bp].freq}")

            self._update_bp_freq_table(picked_bp, heap)

            assert all(v.freq >= 0 for _, v in self.bp_freq_table.items()), "BP Frequency is negative"

            if __debug__:
                if not any(v.freq != 0 for _, v in self.bp_freq_table.items()):
                    logger.warning(f"No more byte-pair found for the {len(self.vocab)} byte-pair!")
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Final vocab: {self.vocab}\nFinal merges: {self.merges}")

        return (self.vocab, self.merges)
    
    def _build_bp_freq_table(self) -> dict[tuple[bytes], BPFreqVal]:
        bp_freq_table: dict[tuple[bytes], BPFreqVal] = defaultdict(BPFreqVal)
        for pretoken_bytes, pretoken_freq in self.freq_table.items():
            for left, right in zip(pretoken_bytes[:-1], pretoken_bytes[1:]):
                bp_freq_table[(left, right)].freq += pretoken_freq
                bp_freq_table[(left, right)].in_pretoken_bytes.add(pretoken_bytes)
        return bp_freq_table
    
    def _update_bp_freq_table(self, picked_bp: tuple[bytes], heap: list[tuple[int, tuple[bytes], BPFreqVal]]):
        in_pretoken_bytes: set[tuple[bytes]] = set(self.bp_freq_table[picked_bp].in_pretoken_bytes)
        for pretoken_bytes in in_pretoken_bytes:
            logger.debug(f"Merging the pretoken {pretoken_bytes} using {picked_bp}")

            for bp in zip(pretoken_bytes[:-1], pretoken_bytes[1:]):
                # logger.debug(f"Removing the pretoken from {bp}")
                entry = self.bp_freq_table[bp]
                entry.freq -= self.freq_table[pretoken_bytes]
                entry.in_pretoken_bytes.discard(pretoken_bytes)
                heapq.heappush(heap, HeapEntry(entry.freq, bp))

            # logger.debug(f"Byte-pair freq table (during merge, removal):\n{pformat(self.bp_freq_table, width=160)}")
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
            # logger.debug(f"pretoken before: {pretoken_bytes} after: {pretoken_bytes_new}")

            assert pretoken_bytes_new not in self.freq_table, f"{pretoken_bytes_new} should not exist in freq_table"
            assert len(pretoken_bytes) - len(pretoken_bytes_new) >= 1

            self.freq_table[pretoken_bytes_new] = self.freq_table.pop(pretoken_bytes)
            for bp in zip(pretoken_bytes_new[:-1], pretoken_bytes_new[1:]):
                entry = self.bp_freq_table[bp]
                entry.freq += self.freq_table[pretoken_bytes_new]
                entry.in_pretoken_bytes.add(pretoken_bytes_new)
                heapq.heappush(heap, HeapEntry(entry.freq, bp))
            # logger.debug(f"Byte-pair freq table (after merge):\n{pformat(self.bp_freq_table, width=160)}")


BYTE1: tuple[bytes] = tuple(bytes([b]) for b in range(256))

@lru_cache(maxsize=8192)
def _pretoken_to_bytes_tuple(pretoken: str) -> tuple[bytes]:
    return tuple(BYTE1[i] for i in pretoken.encode("utf-8"))


def chunk_processor(*args, **kwargs) -> Counter[tuple[bytes]]:
    chunk_processor = ChunkProcessor(*args, **kwargs)
    return chunk_processor.process_chunk()


class ChunkProcessor:

    def __init__(self, bpe_context: BPEContext, processor_id: str, start: int, end: int) -> None:
        self.bpe_context = bpe_context
        self.processor_id = processor_id
        self.start, self.end = start, end
        self.freq_table: Counter[tuple[bytes], int] = Counter()

        with open(self.bpe_context.input_path, "rb") as f:
            f.seek(start)
            self.chunk = f.read(end - start).decode("utf-8", errors="ignore")

    def process_chunk(self) -> Counter[tuple[bytes], int]:
        # logger.debug(f"\n[{self.processor_id}] chunk:\n{self.chunk}\n===========================")
        docs = re.split("|".join([re.escape(t) for t in self.bpe_context.special_tokens]), self.chunk)
        docs = [d for d in docs if d.strip()]
        counter: Counter[str, int] = Counter()
        for i, doc in enumerate(docs):
            # doc_id = f"{self.processor_id}-{i}"
            counter.update(pretoken.group() for pretoken in re.finditer(self.bpe_context.PAT, doc))

        for k, v in counter.items():
            self.freq_table[_pretoken_to_bytes_tuple(k)] = v

        return self.freq_table
 