import os
import asyncio
import math
import tempfile
from math import sqrt
from typing import Dict, List, Optional, Tuple

import numpy as np

from parsers import (
    ABIParser,
    BAMParser,
    BEDParser,
    BigBedParser,
    BigWigParser,
    CRAMParser,
    FASTAParser,
    FASTQParser,
    FSAParser,
    GFFParser,
    GTFParser,
    SAMParser,
    SEQParser,
    VCFParser,
    WIGParser,
)
from base import BaseComponent, BaseParser




# ===========================================================
# PARSER FACTORY 
# ===========================================================
class ParserFactory(BaseComponent):
    """Dosya uzantısına göre uygun parser döndürür."""

    SEQUENCE_FORMATS = {"fa", "fasta", "fq", "fastq", "sam", "bam", "cram", "seq"}
    _PARSER_MAP = {
        "fa": FASTAParser,
        "fasta": FASTAParser,
        "fsa": FSAParser,
        "fq": FASTQParser,
        "fastq": FASTQParser,
        "seq": SEQParser,
        "abi": ABIParser,
        "ab1": ABIParser,
        "sam": SAMParser,
        "bam": BAMParser,
        "cram": CRAMParser,
        "vcf": VCFParser,
        "bed": BEDParser,
        "bb": BigBedParser,
        "bigbed": BigBedParser,
        "bw": BigWigParser,
        "bigwig": BigWigParser,
        "wig": WIGParser,
        "gff": GFFParser,
        "gff3": GFFParser,
        "gtf": GTFParser,
    }

    @staticmethod
    def get_parser(filepath: str) -> BaseParser:
        ext = filepath.lower().split('.')[-1]
        cls = ParserFactory._PARSER_MAP.get(ext)
        if not cls:
            raise ValueError(f"Unsupported file format: {filepath}")
        return cls()

    @staticmethod
    def get_parsers_with_normalization(filepaths: List[str]) -> List[BaseParser]:
        """
        Entry point for multi-file handling:
          - If all formats are identical: return raw parsers (no normalization).
          - If mixed formats and any is non-sequence: raise error.
          - If mixed formats and all are sequence: wrap non-FASTA parsers with SequenceNormalizer.
        """
        exts = [fp.lower().split('.')[-1] for fp in filepaths]
        unique_exts = set(exts)
        parsers = [ParserFactory.get_parser(fp) for fp in filepaths]

        # all same type -> pass through
        if len(unique_exts) == 1:
            return parsers

        # mixed: check for non-sequence formats
        if any(ext not in ParserFactory.SEQUENCE_FORMATS for ext in unique_exts):
            raise ValueError(f"Files {unique_exts} can only be compared with each other.")

        # mixed but all sequence -> normalize non-FASTA
        normalizer = SequenceNormalizationManager()
        return normalizer.wrap_if_mixed(parsers, filepaths)


class SequenceNormalizer(BaseParser):
    """
    Wrapper to normalize non-FASTA sequence formats into FASTA-comparable payloads.
    Delegates to the inner parser, preserving its logging/behavior, and ensures a
    consistent dict payload with 'sequence' and 'n_positions'.
    """

    def __init__(self, inner: BaseParser):
        self.inner = inner
        self._logged_once = False
        self._length_logged = False
        self._total_len = 0
        self._seq_type: Optional[str] = None

    async def read(self, filepath: str):
        self._log_state("STARTED", f"Normalizing read for {filepath}")
        result = await self.inner.read(filepath)
        normalized = self._normalize(result)
        self._log_state("FINISHED", f"Normalizing read for {filepath}")
        return normalized

    async def read_batches(self, filepath: str, batch_size: int):
        self._log_state("STARTED", f"Normalizing batches for {filepath}")
        async for batch in self.inner.read_batches(filepath, batch_size):
            yield self._normalize(batch)
        self._log_state("FINISHED", f"Normalizing batches for {filepath}")

    def _normalize(self, payload):
        """
        Enforce a FASTA-like payload: sequence only, unknown bases stripped,
        metadata dropped (to avoid header/aux fields in downstream math).
        """
        excluded: List[str] = []
        excluded_path: Optional[str] = None

        if isinstance(payload, dict) and "sequence" in payload:
            seq = payload.get("sequence")
            metadata = payload.get("metadata", {})
            if metadata:
                excluded.extend([f"{k}={v}" for k, v in metadata.items()])
            cleaned, npos = self._strip_unknowns(seq)
            self._total_len += len(cleaned)
            if self._seq_type is None:
                self._seq_type = type(cleaned).__name__
            if excluded:
                excluded_path = self._memmap_excluded(excluded)
            if not self._logged_once:
                if metadata:
                    keys_types = [f"{k}:{type(v).__name__}" for k, v in metadata.items()]
                    self._log_state("META_FOUND", f"Metadata fields detected: {keys_types}")
                else:
                    self._log_state("META_FOUND", "No metadata detected; sequence-only payload")
                self._log_state(
                    "NORMALIZED",
                    f"Removed {len(npos)} positions; excluded {len(excluded)} metadata entries"
                    + (f"; excluded_info_memmap={excluded_path}" if excluded_path else "")
                )
                self._logged_once = True
            return {"sequence": cleaned, "n_positions": npos, "metadata": {}, "excluded_info_path": excluded_path}

        cleaned, npos = self._strip_unknowns(payload)
        self._total_len += len(cleaned)
        if self._seq_type is None:
            self._seq_type = type(cleaned).__name__
        if excluded:
            excluded_path = self._memmap_excluded(excluded)
        if not self._logged_once:
            self._log_state(
                "NORMALIZED",
                f"Removed {len(npos)} positions; excluded {len(excluded)} metadata entries"
                + (f"; excluded_info_memmap={excluded_path}" if excluded_path else "")
            )
            self._logged_once = True
        return {"sequence": cleaned, "n_positions": npos, "metadata": {}, "excluded_info_path": excluded_path}

    async def read(self, filepath: str):
        """Normalize full read: delegate to inner parser then normalize payload."""
        self._log_state("STARTED", f"Normalizing read for {filepath}")
        payload = await self.inner.read(filepath)
        normalized = self._normalize(payload)
        self._log_total_once()
        self._log_state("FINISHED", f"Normalizing read for {filepath}")
        return normalized

    async def read_batches(self, filepath: str, batch_size: int):
        """Normalize streamed batches: delegate to inner parser and normalize each batch."""
        self._log_state("STARTED", f"Normalizing batches for {filepath}")
        try:
            async for batch in self.inner.read_batches(filepath, batch_size):
                yield self._normalize(batch)
        finally:
            self._log_total_once()
            self._log_state("FINISHED", f"Normalizing batches for {filepath}")

    def _log_total_once(self):
        if not self._length_logged:
            self._log_state(
                "SEQUENCE_INFO",
                f"type={self._seq_type}, total_length={self._total_len}"
            )
            self._length_logged = True

    def _memmap_excluded(self, entries: List[str]) -> str:
        """Persist excluded info to a tiny memmap to avoid keeping it in RAM."""
        data = "\n".join(entries).encode("utf-8")
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".excluded.mm")
        tmp.write(data)
        tmp.flush()
        tmp.close()
        # map to ensure consistency; caller can delete if desired
        np.memmap(tmp.name, dtype=np.uint8, mode="r", shape=(len(data),))
        return tmp.name

    # --- Abstract methods passthrough (BaseParser requirements) ---
    async def _read_file(self, filepath: str):
        # Delegate to inner parser; not used directly because read/read_batches
        # already proxy, but required to satisfy abstract base.
        return await self.inner._read_file(filepath)

    def _extract_sequence(self, raw_data):
        return self.inner._extract_sequence(raw_data)


class SequenceNormalizationManager(BaseComponent):
    """
    Called only when different file types are present and all are sequence formats.
    Returns parsers as-is for FASTA, wraps non-FASTA sequence parsers with
    SequenceNormalizer to emit FASTA-equivalent payloads.
    """

    def __init__(self):
        self.sequence_formats = {"fa", "fasta", "fq", "fastq", "sam", "bam", "cram", "seq"}
        self.feature_formats = {"big", "bigwig", "bw", "bed", "bb", "bigbed", "gtf", "gff", "gff3", "vcf"}

    def wrap_if_mixed(self, parsers: List[BaseParser], filepaths: List[str]) -> List[BaseParser]:
        self._log_state("STARTED", f"Normalizing mixed sequence formats for: {filepaths}")
        exts = [fp.lower().split('.')[-1] for fp in filepaths]
        unique_exts = set(exts)

        # Preconditions (enforced by caller): mixed types AND all sequence formats
        wrapped: List[BaseParser] = []
        for parser, ext in zip(parsers, exts):
            if ext in ('fa', 'fasta'):
                wrapped.append(parser)
            else:
                self._log_state("NORMALIZE_FILE", f"{parser.__class__.__name__} for {ext} -> SequenceNormalizer")
                wrapped.append(SequenceNormalizer(parser))
        self._log_state("FINISHED", "Normalization applied to non-FASTA parsers")
        return wrapped


# ===========================================================
# STR SEARCHER
# ===========================================================
class STRSearcher(BaseComponent):
    """
    Searches for the most similar occurrence of a user-defined STR pattern across one
    or multiple sequence files using sliding window + cosine similarity (no batching).
    Returns the best-matching substring, its file, and position.
    """

    BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': -1}

    async def search(self, pattern: str, filepaths: List[str]) -> Optional[Dict[str, object]]:
        if not pattern:
            raise ValueError("Pattern must be non-empty.")

        pat = pattern.upper()
        m = len(pat)

        best = None  # (similarity, file, pos, substring)

        def _read_fasta_raw(fp: str) -> str:
            seqs = []
            with open(fp, "r") as fh:
                for line in fh:
                    if line.startswith(">"):
                        continue
                    seqs.append(line.strip())
            return "".join(seqs)

        for fp in filepaths:
            ext = fp.lower().split(".")[-1]
            if ext in ("fa", "fasta"):
                seq_str = _read_fasta_raw(fp)
            else:
                parser = ParserFactory.get_parser(fp)
                payload = await parser.read(fp)
                seq = payload.get("sequence", "")
                seq_str = self._to_string(seq)
            seq_up = seq_str.upper()
            sim, pos = self._best_window_string(pat, seq_up)
            if sim is None:
                continue
            substring = seq_up[pos:pos + m]
            if (best is None) or (sim > best[0]):
                best = (sim, fp, pos, substring)

        if not best:
            return None

        return {
            "file": best[1],
            "position": best[2],
            "substring": best[3],
            "similarity": best[0],
        }

    def _to_string(self, seq) -> str:
        if isinstance(seq, str):
            return seq
        if isinstance(seq, np.ndarray):
            return ''.join(chr(int(c)) for c in seq.tolist())
        return str(seq)

    def _best_window_string(self, pat: str, seq: str) -> Tuple[Optional[float], Optional[int]]:
        m = len(pat)
        n = len(seq)
        if n < m:
            return None, None
        best_sim = None
        best_pos = None
        for i in range(0, n - m + 1):
            window = seq[i:i + m]
            matches = sum(1 for a, b in zip(pat, window) if a == b)
            sim = matches / m
            if (best_sim is None) or (sim > best_sim):
                best_sim = sim
                best_pos = i
        return best_sim, best_pos

# ===========================================================
# DNA SAMPLE
# ===========================================================
class DNASample:
    """Represents the DNA sample vectorized presentation."""
    def __init__(
        self,
        sample_id: str,
        sequence: np.ndarray,
        file_format: str,
        n_positions=None,
        metadata=None,
        memmap_path: Optional[str] = None,
    ):
        self.id = sample_id
        self.sequence = sequence
        self.format = file_format
        self.n_positions = n_positions or []
        self.metadata = metadata or {}
        self.memmap_path = memmap_path  # allows downstream cleanup if needed


# ===========================================================
# SAMPLE LOADER
# ===========================================================
class SampleLoader(BaseComponent):
    """Loads DNA files batch-by-batch, vectorizing each batch into DNASample objects."""

    async def load_samples(
        self,
        filepaths: List[str],
        batch_size: int,
        *,
        aggregate: bool = True,
        use_memmap: bool = True,
        memmap_dir: Optional[str] = None,
    ) -> Dict[str, List[DNASample]]:
        """
        Asynchronously load and vectorize batches for each file.

        Returns a mapping of filepath -> ordered list of DNASample batches.
        When `aggregate=True` (default), each file is represented by a single
        DNASample backed by a memory-mapped array to minimize RAM usage.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        self._log_state("STARTED", f"Loading DNA samples in batches of {batch_size}")

        # choose parsers with normalization if mixed sequence formats
        parsers = ParserFactory.get_parsers_with_normalization(filepaths)

        tasks = [
            self._load_single_file_with_parser(
                parser=parser,
                filepath=fp,
                batch_size=batch_size,
                aggregate=aggregate,
                use_memmap=use_memmap,
                memmap_dir=memmap_dir,
            )
            for parser, fp in zip(parsers, filepaths)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        grouped: Dict[str, List[DNASample]] = {}
        total_batches = 0

        for fp, result in zip(filepaths, results):
            if isinstance(result, Exception):
                self._log_state("FAILED", fp, error=str(result))
                continue
            grouped[fp] = result
            total_batches += len(result)

        self._log_state("FINISHED", f"Loaded {total_batches} batch samples across {len(grouped)} files.")
        return grouped

    async def stream_samples(
        self,
        filepaths: List[str],
        batch_size: int,
        *,
        use_memmap: bool = False,
        memmap_dir: Optional[str] = None,
    ):
        """
        Async generator that yields (filepath, DNASample) per batch without
        keeping anything in memory. Useful for true streaming comparisons.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        for fp in filepaths:
            parser = ParserFactory.get_parser(fp)
            fmt = os.path.splitext(fp)[1].replace('.', '')
            idx = 0
            async for raw_batch in parser.read_batches(fp, batch_size=batch_size):
                seq, n_positions, metadata = self._unpack_batch(raw_batch)
                vectorized = self._vectorize_batch(seq)
                memmap_path = None
                data = vectorized
                if use_memmap:
                    temp_dir = memmap_dir or tempfile.gettempdir()
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir, suffix=".mm")
                    temp_file.write(vectorized.tobytes())
                    temp_file.flush()
                    temp_file.close()
                    memmap_path = temp_file.name
                    data = np.memmap(memmap_path, dtype=np.int8, mode="r", shape=(vectorized.size,))

                sample = DNASample(
                    sample_id=self._format_sample_id(fp, idx),
                    sequence=data,
                    file_format=fmt,
                    n_positions=n_positions,
                    metadata=metadata,
                    memmap_path=memmap_path,
                )
                idx += 1
                yield fp, sample

    async def stream_paired_batches(
        self,
        file1: str,
        file2: str,
        batch_size: int,
        *,
        use_memmap: bool = False,
        memmap_dir: Optional[str] = None,
    ):
        """
        Yield paired DNASample batches from two files in order, preserving batch
        alignment. Only one pair is kept at a time.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        parser1 = ParserFactory.get_parser(file1)
        parser2 = ParserFactory.get_parser(file2)

        g1 = parser1.read_batches(file1, batch_size=batch_size)
        g2 = parser2.read_batches(file2, batch_size=batch_size)

        async def anext(gen):
            try:
                return await gen.__anext__()
            except StopAsyncIteration:
                return None

        idx = 0
        while True:
            b1 = await anext(g1)
            b2 = await anext(g2)
            if b1 is None or b2 is None:
                break

            seq1, npos1, meta1 = self._unpack_batch(b1)
            seq2, npos2, meta2 = self._unpack_batch(b2)
            v1 = self._vectorize_batch(seq1)
            v2 = self._vectorize_batch(seq2)

            memmap_path1 = memmap_path2 = None
            data1, data2 = v1, v2
            if use_memmap:
                temp_dir = memmap_dir or tempfile.gettempdir()
                os.makedirs(temp_dir, exist_ok=True)
                tf1 = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir, suffix=".mm")
                tf1.write(v1.tobytes()); tf1.flush(); tf1.close()
                memmap_path1 = tf1.name
                data1 = np.memmap(memmap_path1, dtype=np.int8, mode="r", shape=(v1.size,))

                tf2 = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir, suffix=".mm")
                tf2.write(v2.tobytes()); tf2.flush(); tf2.close()
                memmap_path2 = tf2.name
                data2 = np.memmap(memmap_path2, dtype=np.int8, mode="r", shape=(v2.size,))

            s1 = DNASample(
                sample_id=self._format_sample_id(file1, idx),
                sequence=data1,
                file_format=os.path.splitext(file1)[1].replace('.', ''),
                n_positions=npos1,
                metadata=meta1,
                memmap_path=memmap_path1,
            )
            s2 = DNASample(
                sample_id=self._format_sample_id(file2, idx),
                sequence=data2,
                file_format=os.path.splitext(file2)[1].replace('.', ''),
                n_positions=npos2,
                metadata=meta2,
                memmap_path=memmap_path2,
            )
            idx += 1
            yield s1, s2

    async def stream_similarity(self, file1: str, file2: str, batch_size: int) -> float:
        """
        Stream batches from two files and compute similarity without holding
        the full sequences or DNASample objects.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        parser1 = ParserFactory.get_parser(file1)
        parser2 = ParserFactory.get_parser(file2)
        g1 = parser1.read_batches(file1, batch_size=batch_size)
        g2 = parser2.read_batches(file2, batch_size=batch_size)

        async def anext(gen):
            try:
                return await gen.__anext__()
            except StopAsyncIteration:
                return None

        matches = 0
        valid1 = 0
        valid2 = 0

        while True:
            b1 = await anext(g1)
            b2 = await anext(g2)
            if b1 is None or b2 is None:
                break

            s1 = self._vectorize_sequence(b1.get("sequence", ""))
            s2 = self._vectorize_sequence(b2.get("sequence", ""))
            min_len = min(len(s1), len(s2))
            if min_len == 0:
                continue
            s1 = s1[:min_len]
            s2 = s2[:min_len]
            valid_mask = (s1 != -1) & (s2 != -1)
            matches += int(np.sum(s1[valid_mask] == s2[valid_mask]))
            valid1 += int(np.sum(s1 != -1))
            valid2 += int(np.sum(s2 != -1))
            await asyncio.sleep(0)

        if valid1 == 0 or valid2 == 0:
            return 0.0
        return matches / float(np.sqrt(valid1 * valid2))

    def sample_positions(self, sample: DNASample, min_trials: int) -> np.ndarray:
        """Select deterministic positions on a target sample for re-use."""
        seq_len = len(sample.sequence)
        size = min(seq_len, min_trials)
        # without replacement when possible; otherwise allow replacement
        replace = seq_len < size
        return np.random.default_rng().choice(seq_len, size=size, replace=replace)

    async def stream_similarity_at_positions(
        self,
        target: DNASample,
        candidate_file: str,
        positions: np.ndarray,
        batch_size: int,
    ) -> float:
        """
        Stream a candidate file and compare only the provided positions against
        a pre-vectorized target sample.
        """
        parser = ParserFactory.get_parser(candidate_file)
        ext = candidate_file.lower().split(".")[-1]
        target_fmt = getattr(target, "format", "").lower()
        seq_formats = ParserFactory.SEQUENCE_FORMATS
        # Only normalize when formats are mixed but both are sequence formats.
        if ext in seq_formats and target_fmt in seq_formats and ext != target_fmt:
            self._log_state("NORMALIZE", f"Wrapping {candidate_file} with SequenceNormalizer for streaming compare")
            parser = SequenceNormalizer(parser)
        g = parser.read_batches(candidate_file, batch_size=batch_size)

        matches = 0
        valid = 0

        # sort positions to allow single pass
        positions_sorted = np.sort(positions)
        pos_idx = 0
        offset = 0

        async for batch in g:
            seq = batch.get("sequence", "")
            cand_vec = self._vectorize_batch(seq)
            if pos_idx >= len(positions_sorted):
                break
            # find positions that fall into this batch window
            start = offset
            end = offset + len(cand_vec)
            while pos_idx < len(positions_sorted) and positions_sorted[pos_idx] < end:
                pos = positions_sorted[pos_idx]
                target_val = target.sequence[pos] if pos < len(target.sequence) else None
                cand_val = cand_vec[pos - start]
                if target_val is not None and target_val != -1 and cand_val != -1:
                    valid += 1
                    if target_val == cand_val:
                        matches += 1
                pos_idx += 1
            offset += len(cand_vec)
            await asyncio.sleep(0)

        if valid == 0:
            return 0.0
        return matches / valid

    async def _load_single_file_with_parser(
        self,
        parser: BaseParser,
        filepath: str,
        batch_size: int,
        *,
        aggregate: bool,
        use_memmap: bool,
        memmap_dir: Optional[str],
    ) -> List[DNASample]:
        fmt = os.path.splitext(filepath)[1].replace('.', '')

        if aggregate:
            return [await self._load_aggregated_file(parser, filepath, fmt, batch_size, use_memmap, memmap_dir)]
        return await self._load_per_batch(parser, filepath, fmt, batch_size, use_memmap, memmap_dir)

    def _format_sample_id(self, filepath: str, batch_index: int) -> str:
        name = os.path.basename(filepath)
        return f"{name}#batch{batch_index}"

    def _vectorize_batch(self, raw_batch) -> np.ndarray:
        """Convert incoming batch payload to numeric np.ndarray."""
        if isinstance(raw_batch, np.ndarray):
            return raw_batch.astype(np.int8, copy=False)

        if isinstance(raw_batch, (list, tuple)):
            if len(raw_batch) == 0:
                return np.array([], dtype=np.int8)
            if isinstance(raw_batch[0], (int, np.integer)):
                return np.array(raw_batch, dtype=np.int8)
            raw_batch = "".join(str(part) for part in raw_batch)

        if isinstance(raw_batch, str):
            # pass-through to numeric byte values; no hardcoded base map
            return np.frombuffer(raw_batch.encode("utf-8"), dtype=np.int8)

        raise TypeError(f"Unsupported batch type for vectorization: {type(raw_batch)}")

    def _vectorize_sequence(self, seq) -> np.ndarray:
        """Vectorize arbitrary sequence payload to int8 np.ndarray."""
        if isinstance(seq, np.ndarray):
            return seq.astype(np.int8, copy=False)
        if isinstance(seq, str):
            return np.frombuffer(seq.encode("utf-8"), dtype=np.int8)
        return np.array(seq, dtype=np.int8)

    def _unpack_batch(self, raw_batch):
        """Handle parser payloads that may include N positions."""
        if isinstance(raw_batch, dict) and "sequence" in raw_batch:
            seq = raw_batch.get("sequence")
            n_positions = raw_batch.get("n_positions", [])
            metadata = raw_batch.get("metadata", {})
            if isinstance(seq, str):
                seq_array = np.frombuffer(seq.encode("utf-8"), dtype=np.int8)
            elif isinstance(seq, np.ndarray):
                seq_array = seq.astype(np.int8, copy=False)
            else:
                seq_array = np.array(seq, dtype=np.int8)
            return seq_array, n_positions, metadata
        return raw_batch, [], {}

    async def _load_aggregated_file(
        self,
        parser: BaseParser,
        filepath: str,
        fmt: str,
        batch_size: int,
        use_memmap: bool,
        memmap_dir: Optional[str],
    ) -> DNASample:
        """
        Streams batches from parser, writes them into a single memory-mapped
        buffer on disk, and returns one DNASample per file.
        """
        temp_dir = memmap_dir or tempfile.gettempdir()
        os.makedirs(temp_dir, exist_ok=True)
        temp_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir, suffix=".mm")
        total_len = 0
        n_positions_agg: List[int] = []
        metadata_agg: Dict = {}

        try:
            async for raw_batch in parser.read_batches(filepath, batch_size=batch_size):
                sequence_payload, n_positions, metadata = self._unpack_batch(raw_batch)
                vectorized = self._vectorize_batch(sequence_payload)
                temp_file.write(vectorized.tobytes())
                total_len += vectorized.size
                if not n_positions_agg:
                    # positions refer to the full sequence, only need to capture once
                    n_positions_agg = list(n_positions)
                if metadata and not metadata_agg:
                    metadata_agg = metadata
                self._log_state("LOADED_BATCH", f"{filepath} (+{vectorized.size} bases)")
        except Exception as e:
            self._log_state("FAILED", filepath, error=str(e))
            temp_file.close()
            os.unlink(temp_file.name)
            raise
        finally:
            temp_file.flush()
            temp_file.close()

        if not use_memmap:
            with open(temp_file.name, "rb") as f:
                data = np.frombuffer(f.read(), dtype=np.int8)
            os.unlink(temp_file.name)
            return DNASample(
                sample_id=self._format_sample_id(filepath, 0),
                sequence=data,
                file_format=fmt,
                n_positions=n_positions_agg,
                metadata=metadata_agg,
                memmap_path=None,
            )

        mm = np.memmap(temp_file.name, dtype=np.int8, mode="r", shape=(total_len,))
        return DNASample(
            sample_id=self._format_sample_id(filepath, 0),
            sequence=mm,
            file_format=fmt,
            n_positions=n_positions_agg,
            metadata=metadata_agg,
            memmap_path=temp_file.name,
        )

    async def _load_per_batch(
        self,
        parser: BaseParser,
        filepath: str,
        fmt: str,
        batch_size: int,
        use_memmap: bool,
        memmap_dir: Optional[str],
    ) -> List[DNASample]:
        """
        Legacy per-batch loading; still memory-mapped for sequences to reduce RAM
        footprint when desired.
        """
        batches: List[DNASample] = []
        temp_dir = memmap_dir or tempfile.gettempdir()
        os.makedirs(temp_dir, exist_ok=True)

        try:
            index = 0
            async for raw_batch in parser.read_batches(filepath, batch_size=batch_size):
                sequence_payload, n_positions, metadata = self._unpack_batch(raw_batch)
                vectorized = self._vectorize_batch(sequence_payload)

                memmap_path = None
                data = vectorized
                if use_memmap:
                    temp_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir, suffix=".mm")
                    temp_file.write(vectorized.tobytes())
                    temp_file.flush()
                    temp_file.close()
                    memmap_path = temp_file.name
                    data = np.memmap(memmap_path, dtype=np.int8, mode="r", shape=(vectorized.size,))

                sample_id = self._format_sample_id(filepath, index)
                batches.append(
                    DNASample(
                        sample_id=sample_id,
                        sequence=data,
                        file_format=fmt,
                        n_positions=n_positions,
                        metadata=metadata,
                        memmap_path=memmap_path,
                    )
                )
                #self._log_state("LOADED_BATCH", f"{sample_id}")
                index += 1
        except Exception as e:
            self._log_state("FAILED", filepath, error=str(e))
            raise

        return batches

    async def select_top_two(
        self,
        target: DNASample,
        pool: List[DNASample],
        batch_size: int,
        seed: Optional[int] = None
    ) -> Tuple[Optional[DNASample], Optional[DNASample]]:
        """
        Approximate the two most similar specimens (target + best candidate) using
        random sampling, extrapolation via Wilson interval, and ordered processing.
        Returns (target, best_match) where best_match can be None if pool empty.
        """
        # Align unknown-base positions: remove N positions from all samples where target had N
        if target.n_positions:
            mask = np.ones(len(target.sequence), dtype=bool)
            mask[target.n_positions] = False
            target.sequence = target.sequence[mask]
            for s in pool:
                if len(s.sequence) >= len(mask):
                    s.sequence = s.sequence[mask]

        matcher = SequenceMatcher(target, pool, seed=seed)
        result = await matcher.match(batch_size=batch_size)
        if not result:
            return target, None

        best_id = result["best_match"]
        best_sample = next((s for s in pool if s.id == best_id), None)
        return target, best_sample


# ===========================================================
# SEQUENCE MATCHER
# ===========================================================
class SequenceMatcher(BaseComponent):
    """Hedef DNA ile havuzdaki örnekleri karşılaştırır (loglama, sıralı işleme).

    Improvements added:
    - deterministic sampling via optional `seed`
    - Bernoulli-error elimination using extrapolated error counts
    - Wilson lower bound guard for robust pruning
    - preserves ordering and remains async-friendly (yields to loop)
    - no internal threading (can be added externally if needed)
    """

    def __init__(
        self,
        target: DNASample,
        pool: Optional[List[DNASample]] = None,
        seed: Optional[int] = None,
        min_trials: int = 2500,
        error_threshold: float = 0.02,
        p_tolerance: float = 0.02,
        confidence_z: float = 1.96,
    ):
        self.target = target
        self.pool = pool or []
        self.seed = seed
        self.min_trials = min_trials
        self.error_threshold = error_threshold  # threshold on expected errors (fraction of length)
        self.p_tolerance = p_tolerance          # tolerance for Wilson lower bound
        self.confidence_z = confidence_z
        # use a Generator for reproducible sampling when seed provided
        self.rng = np.random.default_rng(seed)
        self.stats: Dict[str, Dict[str, float]] = {s.id: {"matches": 0, "compared": 0, "errors": 0} for s in self.pool}

    # Wilson interval (private)
    def _wilson_interval(self, successes: int, n: int, confidence: float = 0.95) -> Tuple[float, float]:
        if n == 0:
            return (0.0, 0.0)
        z = 1.96 if confidence == 0.95 else sqrt(2) * confidence
        phat = successes / n
        denominator = 1 + (z ** 2 / n)
        center = (phat + (z ** 2) / (2 * n)) / denominator
        margin = (z * sqrt((phat * (1 - phat) / n) + (z ** 2) / (4 * n ** 2))) / denominator
        return (center - margin, center + margin)

    def _wilson_lower_bound_error(self, errors: int, trials: int) -> float:
        """Lower bound on error rate using Wilson score interval."""
        if trials == 0:
            return 0.0
        p_hat = errors / trials
        z = self.confidence_z
        numerator = (2 * trials * p_hat + z ** 2 - z * math.sqrt(
            z ** 2 - (1 / trials) + 4 * trials * p_hat * (1 - p_hat) + ((4 * p_hat) - 2) / trials
        ))
        denominator = 2 * (trials + z ** 2)
        return numerator / denominator

    # Tek örnek karşılaştırma
    def _compare_single(self, sample: DNASample, positions: np.ndarray) -> Dict[str, float]:
        target_seq = self.target.sequence
        sample_seq = sample.sequence

        # guard against length mismatch by restricting positions to available indices
        max_len = min(len(target_seq), len(sample_seq))
        if max_len == 0:
            return {"id": sample.id, "matches": 0, "compared": 0, "errors": 0}
        safe_positions = positions[positions < max_len]
        if safe_positions.size == 0:
            return {"id": sample.id, "matches": 0, "compared": 0, "errors": 0}

        valid_mask = (target_seq[safe_positions] != -1) & (sample_seq[safe_positions] != -1)
        matches = int(np.sum(target_seq[safe_positions][valid_mask] == sample_seq[safe_positions][valid_mask]))
        compared = int(np.sum(valid_mask))
        errors = compared - matches
        return {"id": sample.id, "matches": matches, "compared": compared, "errors": errors}

    # Ana karşılaştırma
    async def match(self, batch_size: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self._log_state(
            "STARTED",
            f"Comparing target {self.target.id} with {len(self.pool)} samples using statistical elimination...",
        )
        target_seq = self.target.sequence
        seq_len = len(target_seq)

        # handle trivial cases explicitly
        if len(self.pool) == 0:
            self._log_state("FINISHED", "No samples in pool to compare.")
            return None

        if len(self.pool) == 1:
            # compare fully against the single candidate
            all_pos = np.arange(seq_len, dtype=int)
            r = self._compare_single(self.pool[0], all_pos)
            low, high = self._wilson_interval(r["matches"], r["compared"]) if r["compared"] > 0 else (0.0, 0.0)
            similarity = (r["matches"] / r["compared"]) if r["compared"] > 0 else 0.0
            self._log_state("FINISHED", f"Best match found: {self.pool[0].id}")
            return {
                "best_match": self.pool[0].id,
                "similarity": similarity,
                "interval": (low, high),
                "error_rate": (r["errors"] / r["compared"]) if r["compared"] > 0 else 0.0,
            }

        remaining = self.pool.copy()
        confidences: Dict[str, Tuple[float, float, float]] = {s.id: (0.0, 0.0, 0.0) for s in remaining}

        while len(remaining) > 1:
            sample_size = min(self.min_trials, seq_len)
            positions = self.rng.choice(seq_len, size=sample_size, replace=seq_len < sample_size)
            self._log_state("ROUND_START", f"Sampling {sample_size} positions for {len(remaining)} candidates")

            results = [self._compare_single(s, positions) for s in remaining]

            for r in results:
                sid = r["id"]
                self.stats[sid]["matches"] += r["matches"]
                self.stats[sid]["compared"] += r["compared"]
                self.stats[sid]["errors"] += r["errors"]
            self._log_state("ROUND_STATS", f"Updated stats for {len(results)} candidates")

            survivors: List[DNASample] = []
            for s in remaining:
                m = int(self.stats[s.id]["matches"])
                c = int(self.stats[s.id]["compared"])
                e = int(self.stats[s.id]["errors"])
                low, high = self._wilson_interval(m, c)
                confidences[s.id] = ((m / c) if c > 0 else 0.0, low, high)

                if c >= self.min_trials:
                    p_hat = (e / c) if c > 0 else 0.0
                    expected_errors = p_hat * seq_len
                    p_low = self._wilson_lower_bound_error(e, c)

                    if (expected_errors > self.error_threshold * seq_len) or (p_low > self.p_tolerance):
                        self._log_state(
                            "ELIMINATED",
                            f"{s.id} after {c} trials "
                            f"(p_hat={p_hat:.4f}, expected_errors={expected_errors:.2f}, "
                            f"wilson=({low:.4f},{high:.4f}))"
                        )
                        continue

                survivors.append(s)

            if not survivors:
                self._log_state("STOP", "All candidates eliminated; no survivors")
                break
            remaining = survivors

            # keep candidates whose upper bound overlaps the best lower bound
            best_id = max(confidences, key=lambda k: confidences[k][0])
            best_low = confidences[best_id][1]
            remaining = [s for s in remaining if confidences.get(s.id, (0.0, 0.0, 0.0))[2] >= best_low]

            self._log_state("ROUND_COMPLETE", f"Remaining candidates: {len(remaining)}")
            await asyncio.sleep(0)

        # finalize result
        # finalize result (covers cases where loop exited due to convergence or stall)
        if not remaining:
            remaining = self.pool

        def _score(sample: DNASample):
            m = int(self.stats[sample.id]["matches"])
            c = int(self.stats[sample.id]["compared"])
            sim = (m / c) if c > 0 else 0.0
            low, high = self._wilson_interval(m, c) if c > 0 else (0.0, 0.0)
            return sim, low, high

        ordered = sorted(remaining, key=lambda s: _score(s)[0], reverse=True)
        top = ordered[:2]
        ids = [s.id for s in top]
        stats_out = [_score(s) for s in top]
        self._log_state("FINISHED", f"Top candidates: {ids}")
        return {
            "candidates": ids,
            "similarities": [st[0] for st in stats_out],
            "intervals": [(st[1], st[2]) for st in stats_out],
            "error_rates": [1 - st[0] for st in stats_out],
        }

    async def match_streaming_paths(
        self,
        target_path: str,
        candidate_paths: List[str],
        batch_size: int,
        loader: Optional["SampleLoader"] = None,
    ):
        """
        Streaming elimination without repeatedly reading the target. The target is
        loaded once, positions are sampled once (min_trials), and each candidate is
        streamed at those positions for elimination. Returns only the best match
        (path and similarity).
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if not candidate_paths:
            return None

        loader = loader or SampleLoader()

        # If we have mixed sequence formats, emit normalization logs up-front so
        # users can see that cross-format handling is active.
        all_paths = [target_path] + candidate_paths
        exts = [p.lower().split(".")[-1] for p in all_paths]
        if len(set(exts)) > 1 and all(ext in ParserFactory.SEQUENCE_FORMATS for ext in exts):
            mgr = SequenceNormalizationManager()
            parsers = [ParserFactory.get_parser(p) for p in all_paths]
            # We only need the side effect (logging); target still loaded separately.
            mgr.wrap_if_mixed(parsers, all_paths)

        self._log_state("STARTED", f"Streaming elimination for target {target_path} across {len(candidate_paths)} candidates")
        target_batches = await loader.load_samples([target_path], batch_size=batch_size, aggregate=True, use_memmap=True)
        target_sample = target_batches[target_path][0]
        positions = loader.sample_positions(target_sample, self.min_trials)

        scores = []
        for cand in candidate_paths:
            sim = await loader.stream_similarity_at_positions(target_sample, cand, positions, batch_size)
            self._log_state("CANDIDATE_SCORE", f"{cand} similarity={sim:.6f} trials={len(positions)}")
            scores.append((cand, sim))
            await asyncio.sleep(0)

        # simple Wilson-style ranking: higher similarity -> lower error
        scores.sort(key=lambda x: x[1], reverse=True)
        best = scores[0]
        self._log_state("FINISHED", f"Best streaming match: {best[0]} (similarity={best[1]:.6f}, trials={len(positions)})")
        return {"best_match": best[0], "similarity": best[1]}

    async def match_stream(self, batch_stream, batch_size: int):
        """
        Streaming matcher: iterate over a stream of (filepath, DNASample) pairs,
        comparing each to the target and keeping only the best. Non-winning
        memmaps are unlinked immediately to reduce memory.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        comp = DoubleSampleComparator()
        best_sample: Optional[DNASample] = None
        best_sim = -1.0

        async for _, sample in batch_stream:
            sim = await comp.compare(
                self.target,
                sample,
                batch_size=min(batch_size, len(self.target.sequence), len(sample.sequence)),
            )
            if sim > best_sim:
                if best_sample and getattr(best_sample, "memmap_path", None):
                    try:
                        os.unlink(best_sample.memmap_path)
                    except OSError:
                        pass
                best_sim = sim
                best_sample = sample
            else:
                if getattr(sample, "memmap_path", None):
                    try:
                        os.unlink(sample.memmap_path)
                    except OSError:
                        pass

            await asyncio.sleep(0)

        if not best_sample:
            return None

        self._log_state("FINISHED", f"Best streaming match: {best_sample.id}")
        return {"candidates": [best_sample.id], "similarities": [best_sim], "best_sample": best_sample}



class DoubleSampleComparator(BaseComponent):
    """
    İki DNASample nesnesi arasında cosine-benzeri benzerlik
    hesaplayan yüksek performanslı sınıf.

    ❗ NOT:
       - Sadece 2 örnek karşılaştırması için kullanılabilir.
       - Çoklu karşılaştırmalar için SequenceMatcher kullanılır.
       - FASTA/FASTQ ayrımı SampleLoader & Parser seviyesinde zaten yapılmış olmalı.
    """

    async def compare(self, s1, s2, batch_size: int) -> float:
        """
        Similarity = matches / sqrt(len1 * len2)
        """

        # =====================================================
        # Basic checks
        # =====================================================
        fmt = self._get_format(s1, s2)
        self._log_state("STARTED", f"Comparing: {fmt} batches (chunked, batch_size={batch_size})")

        # =====================================================
        # Vectorized computation
        # =====================================================
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        matches = 0
        valid1 = 0
        valid2 = 0

        stream1 = self._chunk_stream(s1, batch_size)
        stream2 = self._chunk_stream(s2, batch_size)

        while True:
            try:
                chunk1 = next(stream1)
                chunk2 = next(stream2)
            except StopIteration:
                break

            if len(chunk1) == 0 or len(chunk2) == 0:
                continue

            # align chunk lengths to avoid shape mismatch
            min_len = min(len(chunk1), len(chunk2))
            chunk1 = chunk1[:min_len]
            chunk2 = chunk2[:min_len]

            valid_mask = (chunk1 != -1) & (chunk2 != -1)
            matches += int(np.sum(chunk1[valid_mask] == chunk2[valid_mask]))
            valid1 += int(np.sum(chunk1 != -1))
            valid2 += int(np.sum(chunk2 != -1))

            # yield to event loop to keep async responsive
            await asyncio.sleep(0)

        if valid1 == 0 or valid2 == 0:
            self._log_state("FAILED", "No valid bases in one of the samples")
            return 0.0

        similarity = matches / math.sqrt(valid1 * valid2)

        # =====================================================
        # Logging and return
        # =====================================================
        label1 = s1.id if isinstance(s1, DNASample) else "batches_1"
        label2 = s2.id if isinstance(s2, DNASample) else "batches_2"
        self._log_state("FINISHED", f"{label1} vs {label2} similarity = {similarity:.15f}")
        return similarity

    async def compare_stream(self, file1: str, file2: str, batch_size: int, use_memmap: bool = True) -> float:
        """
        Streaming comparison directly from file paths (no pre-loaded DNASample).
        """
        return await self.compare_stream_samples(file1, file2, batch_size, use_memmap=use_memmap)

    async def compare_stream_samples(self, file1: str, file2: str, batch_size: int, use_memmap: bool = True) -> float:
        """
        Streaming comparison that routes batches through SampleLoader so each batch
        is a DNASample. Only one pair of batches is kept at a time.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")

        loader = SampleLoader()
        paired = loader.stream_paired_batches(file1, file2, batch_size=batch_size, use_memmap=use_memmap)

        matches = 0
        valid1 = 0
        valid2 = 0

        async for s1, s2 in paired:
            stream1 = self._chunk_stream(s1, batch_size)
            stream2 = self._chunk_stream(s2, batch_size)

            while True:
                try:
                    c1 = next(stream1)
                    c2 = next(stream2)
                except StopIteration:
                    break
                if len(c1) == 0 or len(c2) == 0:
                    continue
                min_len = min(len(c1), len(c2))
                c1 = c1[:min_len]
                c2 = c2[:min_len]
                valid_mask = (c1 != -1) & (c2 != -1)
                matches += int(np.sum(c1[valid_mask] == c2[valid_mask]))
                valid1 += int(np.sum(c1 != -1))
                valid2 += int(np.sum(c2 != -1))
                await asyncio.sleep(0)

            for sample in (s1, s2):
                if getattr(sample, "memmap_path", None):
                    try:
                        os.unlink(sample.memmap_path)
                    except OSError:
                        pass

        if valid1 == 0 or valid2 == 0:
            self._log_state("FAILED", "No valid bases in one of the streams")
            return 0.0

        similarity = matches / math.sqrt(valid1 * valid2)
        self._log_state("FINISHED", f"{file1} vs {file2} similarity(stream) = {similarity:.15f}")
        return similarity

    # ============================================================
    # Private validation helper
    # ============================================================
    def _assert_valid_samples(self, s1: DNASample, s2: DNASample):
        if not isinstance(s1, DNASample) or not isinstance(s2, DNASample):
            raise TypeError("DoubleSampleComparator works only with DNASample objects.")
        if s1.format != s2.format:
            raise ValueError(
                f"DoubleSampleComparator requires SAME file format. "
                f"Got: {s1.format} vs {s2.format}"
            )

    def _get_format(self, s1, s2) -> str:
        def fmt_from(obj):
            if isinstance(obj, list):
                if not obj:
                    raise ValueError("Empty batch list provided.")
                fmt_local = obj[0].format
                for b in obj:
                    if b.format != fmt_local:
                        raise ValueError("Mixed formats within batches are not allowed.")
                return fmt_local
            if isinstance(obj, DNASample):
                return obj.format
            raise TypeError("DoubleSampleComparator accepts DNASample or list of DNASample.")

        fmt1 = fmt_from(s1)
        fmt2 = fmt_from(s2)
        if fmt1 != fmt2:
            raise ValueError(f"DoubleSampleComparator requires SAME file format. Got: {fmt1} vs {fmt2}")
        return fmt1

    def _chunk_stream(self, obj, batch_size: int):
        """
        Yield sequence chunks of at most `batch_size`, preserving order, without
        concatenating the entire payload into memory.
        """
        if isinstance(obj, DNASample):
            seq = obj.sequence
            for start in range(0, len(seq), batch_size):
                yield seq[start : start + batch_size]
            return

        if isinstance(obj, list):
            if not obj:
                return
            for b in obj:
                seq = b.sequence
                for start in range(0, len(seq), batch_size):
                    yield seq[start : start + batch_size]
            return

        raise TypeError("DoubleSampleComparator accepts DNASample or list of DNASample.")

    
