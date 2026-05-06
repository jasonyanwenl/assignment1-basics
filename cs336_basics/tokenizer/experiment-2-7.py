import logging
from statistics import median
import time
import regex as re

from cs336_basics.tokenizer.pretokenization import find_chunk_boundaries
from cs336_basics.tokenizer.tokenizer import Tokenizer


logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(funcName)s:%(lineno)d %(message)s"
    )
    input_file = "data/TinyStoriesV2-GPT4-train.txt"
    # input_file = "data/owt_train.txt"
    special_tokens=["<|endoftext|>"]
    sampled_boundaries = []
    with open(input_file, "rb") as f:
        boundaries = find_chunk_boundaries(f, 10000, special_tokens[0].encode("utf-8"))
        for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            if start % 2 == 0:
                sampled_boundaries.append((start, end))
            if len(sampled_boundaries) == 10:
                break
    
    logger.info("Sampled boundaries: %s", sampled_boundaries)

    tokenizer = Tokenizer.from_files(
        vocab_filepath="vocab_tinystories.json",
        merges_filepath="merges_tinystories.txt",
        # vocab_filepath="vocab_owt.json",
        # merges_filepath="merges_owt.txt",
        special_tokens=special_tokens
    )

    doc_id = 0
    throughputs = []
    with open(input_file, "rb") as f:
        for i, (start, end) in enumerate(sampled_boundaries):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            for j, doc in enumerate(re.splititer("|".join([re.escape(t) for t in special_tokens]), chunk)):
                if not doc.strip():
                    continue
                if j % 2 == 1:
                    logger.info("doc: %s...", doc[:10])
                    t0 = time.perf_counter()
                    encoded_ids = tokenizer.encode(doc)
                    elapsed = time.perf_counter() - t0
                    logger.info("encoded_ids: %s", encoded_ids[:10])
                    total_bytes = len(doc.encode('utf-8'))
                    compression_ratio = total_bytes / len(encoded_ids)
                    throughput = total_bytes / elapsed
                    logger.info("doc: %s, compression ratio: %s, throughput: %s bytes/sec", doc_id, compression_ratio, throughput)
                    throughputs.append(throughput)
                    doc_id += 1
                    break
    median_throughput = median(throughputs)
    pile_bytes = 825e9
    seconds = pile_bytes / median_throughput
    hours = seconds / 3600
    logger.info(f"Estimated time for Pile: {hours:.2f} hours")
