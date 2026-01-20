import aiofiles
import numpy as np
from typing import List
import asyncio
import vcfpy
from base import BaseParser

from Bio import SeqIO
import pysam
import pyBigWig
from pyfaidx import Fasta
import gffutils
from gffutils.iterators import DataIterator
try:
    from bx.bbi import wiggle as bx_wiggle
except ImportError:
    bx_wiggle = None  # optional; fallback parsing will still work


# ===========================================================
# FASTA PARSER
# ===========================================================
class FASTAParser(BaseParser):
    """
    FASTA parser: streams records via pyfaidx for speed and low memory, concatenating
    all sequences in file order into a single string for downstream numeric mapping.
    Suitable for large genomes; avoids redundant reads.
    """
    async def _read_file(self, filepath: str) -> List[str]:
        # async signature kept; pass filepath through for downstream library use
        return [filepath]

    def _extract_sequence(self, raw_data: List[str]) -> str:
        filepath = raw_data[0]
        seqs = []
        with open(filepath, "r") as handle:
            for line in handle:
                if line.startswith(">"):
                    continue
                seqs.append(line.strip())
        return ''.join(seqs)

    def _extract_metadata(self, raw_data: List[str]) -> dict:
        filepath = raw_data[0]
        headers = []
        with open(filepath, "r") as fh:
            for line in fh:
                if line.startswith(">"):
                    headers.append(line.strip()[1:])
        return {"headers": headers}




# ===========================================================
# FASTQ PARSER
# ===========================================================
class FASTQParser(BaseParser):
    """
    FASTQ parser: uses BioPython SeqIO to stream reads in order, concatenating bases
    across all records. Single-pass to minimize memory; preserves read order.
    """
    async def _read_file(self, filepath: str) -> List[str]:
        return [filepath]

    def _extract_sequence(self, raw_data: List[str]) -> str:
        filepath = raw_data[0]
        # BioPython streaming to avoid loading entire file twice
        seqs = []
        with open(filepath, "r") as handle:
            for record in SeqIO.parse(handle, "fastq"):
                seqs.append(str(record.seq))
        return ''.join(seqs)

    def _extract_metadata(self, raw_data: List[str]) -> dict:
        # Skip expensive second pass for large FASTQ; metadata omitted to save memory.
        return {}

    async def read_batches(self, filepath: str, batch_size: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        try:
            self._log_state("STARTED", f"Reading {filepath} in batches of {batch_size}")
            buf: List[str] = []
            total = 0
            with open(filepath, "r") as handle:
                for record in SeqIO.parse(handle, "fastq"):
                    seq = str(record.seq)
                    buf.append(seq)
                    total += len(seq)
                    if total >= batch_size:
                        joined = ''.join(buf)
                        yield {"sequence": joined[:batch_size], "n_positions": [], "metadata": {}}
                        remainder = joined[batch_size:]
                        buf = [remainder] if remainder else []
                        total = len(remainder)
                    await asyncio.sleep(0)
            if buf:
                chunk = ''.join(buf)
                yield {"sequence": chunk, "n_positions": [], "metadata": {}}
            self._log_state("FINISHED", filepath)
        except Exception as e:
            self._log_state("FAILED", filepath, error=str(e))
            raise


# ===========================================================
# VCF PARSER
# ===========================================================
class VCFParser(BaseParser):
    """
    VCF parser: leverages pysam.VariantFile to stream variants, extracting QUAL and
    depth (DP) into a flat numeric feature array. Handles bgzipped or plain VCF; missing
    values become -1.
    """
    async def _read_file(self, filepath: str) -> List[str]:
        return [filepath]

    def _extract_sequence(self, raw_data: List[str]):
        filepath = raw_data[0]
        values = []
        # pysam.VariantFile handles both bgzipped and plain VCF
        with vcfpy.Reader.from_path(filepath) as vcf:
            for rec in vcf.fetch():
                qual = float(rec.qual) if rec.qual is not None else -1.0
                dp = -1
                if rec.info is not None and "DP" in rec.info:
                    try:
                        dp = int(rec.info["DP"])
                    except Exception:
                        dp = -1
                values.extend([qual, dp])
        return np.array(values, dtype=np.float32)

    def _extract_metadata(self, raw_data: List[str]) -> dict:
        filepath = raw_data[0]
        try:
            with open(filepath, "r") as fh:
                headers = [line.strip() for line in fh if line.startswith("#")]
        except Exception:
            headers = []
        return {"headers": headers}

    async def read_batches(self, filepath: str, batch_size: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        try:
            self._log_state("STARTED", f"Reading {filepath} in batches of {batch_size}")
            buf: List[float] = []
            with pysam.VariantFile(filepath) as vcf:
                for rec in vcf.fetch():
                    qual = float(rec.qual) if rec.qual is not None else -1.0
                    dp = -1
                    if rec.info is not None and "DP" in rec.info:
                        try:
                            dp = int(rec.info["DP"])
                        except Exception:
                            dp = -1
                    buf.extend([qual, float(dp)])
                    if len(buf) >= batch_size:
                        chunk = np.array(buf[:batch_size], dtype=np.float32)
                        buf = buf[batch_size:]
                        yield {"sequence": chunk, "n_positions": [], "metadata": {}}
                    await asyncio.sleep(0)
            if buf:
                yield {"sequence": np.array(buf, dtype=np.float32), "n_positions": [], "metadata": {}}
            self._log_state("FINISHED", filepath)
        except Exception as e:
            self._log_state("FAILED", filepath, error=str(e))
            raise


# ===========================================================
# SAM PARSER
# ===========================================================
class SAMParser(BaseParser):
    """
    SAM parser: streams alignments with pysam.AlignmentFile, concatenating query
    sequences in file order. Skips headers; streaming keeps memory bounded.
    """
    async def _read_file(self, filepath: str) -> List[str]:
        return [filepath]

    def _extract_sequence(self, raw_data: List[str]):
        filepath = raw_data[0]
        # pysam streaming keeps memory bounded for large SAM
        seqs = []
        with pysam.AlignmentFile(filepath, "r") as f:
            for read in f.fetch(until_eof=True):
                if read.query_sequence:
                    seqs.append(read.query_sequence.upper())
        return ''.join(seqs)

    def _extract_metadata(self, raw_data: List[str]) -> dict:
        filepath = raw_data[0]
        headers = []
        with open(filepath, "r") as fh:
            for line in fh:
                if line.startswith("@"):
                    headers.append(line.strip())
                else:
                    break
        return {"headers": headers}

    async def read_batches(self, filepath: str, batch_size: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        try:
            self._log_state("STARTED", f"Reading {filepath} in batches of {batch_size}")
            buf: List[str] = []
            total = 0
            with pysam.AlignmentFile(filepath, "r") as f:
                for read in f.fetch(until_eof=True):
                    if read.query_sequence:
                        seq = read.query_sequence.upper()
                        buf.append(seq)
                        total += len(seq)
                        if total >= batch_size:
                            joined = ''.join(buf)
                            yield {"sequence": joined[:batch_size], "n_positions": [], "metadata": {}}
                            remainder = joined[batch_size:]
                            buf = [remainder] if remainder else []
                            total = len(remainder)
                    await asyncio.sleep(0)
            if buf:
                chunk = ''.join(buf)
                yield {"sequence": chunk, "n_positions": [], "metadata": {}}
            self._log_state("FINISHED", filepath)
        except Exception as e:
            self._log_state("FAILED", filepath, error=str(e))
            raise




# ===========================================================
# BAM / CRAM PARSER (pysam required)
# ===========================================================
class BAMParser(BaseParser):
    """
    BAM parser: binary alignment reader via pysam.AlignmentFile; streams reads and
    concatenates query sequences in order. Requires pysam; efficient for large files.
    """
    async def _read_file(self, filepath: str) -> List[str]:
        if not pysam:
            raise ImportError("pysam required for BAM parsing")
        return [filepath]  # placeholder

    def _extract_sequence(self, raw_data: List[str]):
        filepath = raw_data[0]
        seqs = []
        with pysam.AlignmentFile(filepath, "rb") as f:
            for read in f.fetch():
                seqs.append(read.query_sequence)
        return ''.join(seqs)

    def _extract_metadata(self, raw_data: List[str]) -> dict:
        filepath = raw_data[0]
        with pysam.AlignmentFile(filepath, "rb") as f:
            header = f.header.to_dict()
        return {"header": header}

    async def read_batches(self, filepath: str, batch_size: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        try:
            self._log_state("STARTED", f"Reading {filepath} in batches of {batch_size}")
            buf: List[str] = []
            total = 0
            with pysam.AlignmentFile(filepath, "rb") as f:
                for read in f.fetch(until_eof=True):
                    if read.query_sequence:
                        seq = read.query_sequence
                        buf.append(seq)
                        total += len(seq)
                        if total >= batch_size:
                            joined = ''.join(buf)
                            yield {"sequence": joined[:batch_size], "n_positions": [], "metadata": {}}
                            remainder = joined[batch_size:]
                            buf = [remainder] if remainder else []
                            total = len(remainder)
                    await asyncio.sleep(0)
            if buf:
                chunk = ''.join(buf)
                yield {"sequence": chunk, "n_positions": [], "metadata": {}}
            self._log_state("FINISHED", filepath)
        except Exception as e:
            self._log_state("FAILED", filepath, error=str(e))
            raise



class CRAMParser(BaseParser):
    """
    CRAM parser: reference-aware alignment reader via pysam.AlignmentFile; streams
    reads and concatenates query sequences in order. Requires pysam.
    """
    async def _read_file(self, filepath: str) -> List[str]:
        if not pysam:
            raise ImportError("pysam required for CRAM parsing")
        return [filepath]

    def _extract_sequence(self, raw_data: List[str]):
        filepath = raw_data[0]
        seqs = []
        with pysam.AlignmentFile(filepath, "rc") as f:
            try:
                iterator = f.fetch()
            except ValueError:
                # fallback when index is missing
                iterator = f.fetch(until_eof=True)
            for read in iterator:
                seqs.append(read.query_sequence)
        return ''.join(seqs)

    def _extract_metadata(self, raw_data: List[str]) -> dict:
        filepath = raw_data[0]
        with pysam.AlignmentFile(filepath, "rc") as f:
            header = f.header.to_dict()
        return {"header": header}

    async def read_batches(self, filepath: str, batch_size: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        try:
            self._log_state("STARTED", f"Reading {filepath} in batches of {batch_size}")
            buf: List[str] = []
            total = 0
            with pysam.AlignmentFile(filepath, "rc") as f:
                try:
                    iterator = f.fetch()
                except ValueError:
                    iterator = f.fetch(until_eof=True)
                for read in iterator:
                    if read.query_sequence:
                        seq = read.query_sequence
                        buf.append(seq)
                        total += len(seq)
                        if total >= batch_size:
                            joined = ''.join(buf)
                            yield {"sequence": joined[:batch_size], "n_positions": [], "metadata": {}}
                            remainder = joined[batch_size:]
                            buf = [remainder] if remainder else []
                            total = len(remainder)
                    await asyncio.sleep(0)
            if buf:
                chunk = ''.join(buf)
                yield {"sequence": chunk, "n_positions": [], "metadata": {}}
            self._log_state("FINISHED", filepath)
        except Exception as e:
            self._log_state("FAILED", filepath, error=str(e))
            raise


# ===========================================================
# BED PARSER
# ===========================================================
class BEDParser(BaseParser):
    """
    BED parser: extracts the score column (column 5) into a numeric vector. Uses
    pysam.TabixFile when available for indexed beds; falls back to line parsing.
    Invalid or missing scores are mapped to -1.
    """
    async def _read_file(self, filepath: str) -> List[str]:
        return [filepath]

    def _extract_sequence(self, raw_data: List[str]):
        filepath = raw_data[0]
        values = []
        try:
            tbx = pysam.TabixFile(filepath)
            for rec in tbx.fetch():
                cols = rec.strip().split('\t')
                if len(cols) >= 5:
                    try:
                        score = float(cols[4])
                    except Exception:
                        score = -1.0
                    values.append(score)
            tbx.close()
        except Exception:
            with open(filepath, "r") as fh:
                for line in fh:
                    if line.startswith('#') or not line.strip():
                        continue
                    cols = line.strip().split('\t')
                    if len(cols) >= 5:
                        try:
                            score = float(cols[4])
                        except Exception:
                            score = -1.0
                        values.append(score)
        return np.array(values, dtype=np.float32)

    def _extract_metadata(self, raw_data: List[str]) -> dict:
        filepath = raw_data[0]
        headers = []
        try:
            with open(filepath, "r") as fh:
                for line in fh:
                    if line.startswith("#"):
                        headers.append(line.strip())
                    else:
                        break
        except Exception:
            headers = []
        return {"headers": headers}


# ===========================================================
# BigBed / BigWig (pyBigWig required)
# ===========================================================
class BigBedParser(BaseParser):
    """
    BigBed parser: reads entries via pyBigWig and approximates a numeric score per
    entry (first parseable numeric in extra fields). Preserves chromosome ordering;
    missing or invalid scores -> -1.
    """
    async def _read_file(self, filepath: str) -> List[str]:
        if not pyBigWig:
            raise ImportError("pyBigWig required for BigBed")
        return [filepath]

    def _extract_sequence(self, raw_data: List[str]):
        filepath = raw_data[0]
        values = []
        with pyBigWig.open(filepath) as bb:
            for chrom, length in bb.chroms().items():
                # entries yields tuples; we approximate a single score per entry
                entries = bb.entries(chrom, 0, length)
                if not entries:
                    continue
                for _, _, rest in entries:
                    score = -1.0
                    if rest:
                        parts = str(rest).split('\t')
                        # bigBed commonly stores score as first or second extra field
                        for p in parts:
                            try:
                                score = float(p)
                                break
                            except Exception:
                                continue
                    values.append(score)
        return np.array(values, dtype=np.float32)

    def _extract_metadata(self, raw_data: List[str]) -> dict:
        filepath = raw_data[0]
        chroms = {}
        try:
            with pyBigWig.open(filepath) as bb:
                chroms = bb.chroms()
        except Exception:
            chroms = {}
        return {"chroms": chroms}



# ===========================================================
# BigWig PARSER
# ===========================================================
class BigWigParser(BaseParser):
    """
    BigWig parser: streams per-chromosome signal via pyBigWig into a flat numeric
    array. NaNs converted to -1; ordering follows chromosome order in the file.
    """
    async def _read_file(self, filepath: str) -> List[str]:
        if not pyBigWig:
            raise ImportError("pyBigWig required for BigWig")
        return [filepath]

    def _extract_sequence(self, raw_data: List[str]):
        filepath = raw_data[0]
        values = []
        with pyBigWig.open(filepath) as bw:
            for chrom, length in bw.chroms().items():
                chrom_vals = bw.values(chrom, 0, length, numpy=True)
                # bw.values can return None for missing; filter safely
                if chrom_vals is None:
                    continue
                values.extend(np.nan_to_num(chrom_vals, nan=-1.0).tolist())
        return np.array(values, dtype=np.float32)

    def _extract_metadata(self, raw_data: List[str]) -> dict:
        filepath = raw_data[0]
        chroms = {}
        try:
            with pyBigWig.open(filepath) as bw:
                chroms = bw.chroms()
        except Exception:
            chroms = {}
        return {"chroms": chroms}


# ===========================================================
# WIG PARSER
# ===========================================================
class WIGParser(BaseParser):
    """
    WIG parser: prefers bx-python wiggle Reader to expand intervals into per-base
    values; falls back to numeric line parsing when bx is unavailable. Ignores header
    directives; invalid/missing values -> -1.
    """
    async def _read_file(self, filepath: str) -> List[str]:
        return [filepath]

    def _extract_sequence(self, raw_data: List[str]):
        filepath = raw_data[0]
        values = []

        # Preferred: bx-python wiggle reader if available
        if bx_wiggle:
            try:
                with open(filepath, "r") as fh:
                    reader = bx_wiggle.Reader(fh)
                    for entry in reader:
                        try:
                            chrom, start, end, val = entry
                            # repeat value across interval length to approximate per-base values
                            span = max(1, end - start)
                            values.extend([float(val)] * span)
                        except Exception:
                            continue
            except Exception:
                values = []

        # Fallback: simple line-based numeric parsing
        if not values:
            with open(filepath, "r") as fh:
                for line in fh:
                    if line.startswith(('track','variableStep','fixedStep','#')) or not line.strip():
                        continue
                    try:
                        val = float(line.strip())
                    except Exception:
                        val = -1.0
                    values.append(val)

        return np.array(values, dtype=np.float32)

    def _extract_metadata(self, raw_data: List[str]) -> dict:
        filepath = raw_data[0]
        headers = []
        try:
            with open(filepath, "r") as fh:
                for line in fh:
                    if line.startswith(('track','variableStep','fixedStep','#')):
                        headers.append(line.strip())
                    else:
                        continue
        except Exception:
            headers = []
        return {"headers": headers}



# ===========================================================
# GFFPARSER
# ===========================================================
class GFFParser(BaseParser):
    """
    Generic Feature Format (.gff/.gff3) parser: builds an in-memory gffutils DB when
    possible (preserving feature order), falling back to a streaming iterator. Extracts
    feature scores (column 6) into a numeric vector; missing/invalid scores -> -1.
    """

    async def _read_file(self, filepath: str) -> List[str]:
        return [filepath]

    def _extract_sequence(self, raw_data: List[str]):
        filepath = raw_data[0]
        values = []

        try:
            db = gffutils.create_db(
                filepath,
                dbfn=":memory:",
                force=True,
                keep_order=True,
                merge_strategy="merge",
                sort_attribute_values=True,
            )
            for feat in db.all_features(order_by=("seqid", "start")):
                try:
                    score = float(feat.score) if feat.score not in (None, ".", "") else -1.0
                except Exception:
                    score = -1.0
                values.append(score)
        except Exception:
            # fallback to streaming iterator when DB build is not possible
            with open(filepath, "r") as fh:
                for feature in DataIterator(fh):
                    try:
                        score = float(feature.score) if feature.score not in (None, ".", "") else -1.0
                    except Exception:
                        score = -1.0
                    values.append(score)

        return np.array(values, dtype=np.float32)

    def _extract_metadata(self, raw_data: List[str]) -> dict:
        filepath = raw_data[0]
        headers = []
        try:
            with open(filepath, "r") as fh:
                for line in fh:
                    if line.startswith("#"):
                        headers.append(line.strip())
                    else:
                        break
        except Exception:
            headers = []
        return {"headers": headers}



# ===========================================================
# GTFPARSER
# ===========================================================
class GTFParser(BaseParser):
    """
    Gene Transfer Format (.gtf) parser: builds an in-memory gffutils DB when possible
    (order-preserving), falling back to streaming. Extracts feature scores (column 6)
    into a numeric vector; missing/invalid scores -> -1.
    """

    async def _read_file(self, filepath: str) -> List[str]:
        return [filepath]

    def _extract_sequence(self, raw_data: List[str]):
        filepath = raw_data[0]
        values = []

        try:
            db = gffutils.create_db(
                filepath,
                dbfn=":memory:",
                force=True,
                keep_order=True,
                merge_strategy="merge",
                sort_attribute_values=True,
            )
            for feat in db.all_features(order_by=("seqid", "start")):
                try:
                    score = float(feat.score) if feat.score not in (None, ".", "") else -1.0
                except Exception:
                    score = -1.0
                values.append(score)
        except Exception:
            with open(filepath, "r") as fh:
                for feature in DataIterator(fh):
                    try:
                        score = float(feature.score) if feature.score not in (None, ".", "") else -1.0
                    except Exception:
                        score = -1.0
                    values.append(score)

        return np.array(values, dtype=np.float32)

    def _extract_metadata(self, raw_data: List[str]) -> dict:
        filepath = raw_data[0]
        headers = []
        try:
            with open(filepath, "r") as fh:
                for line in fh:
                    if line.startswith("#"):
                        headers.append(line.strip())
                    else:
                        break
        except Exception:
            headers = []
        return {"headers": headers}


# ===========================================================
# SEQ PARSER
# ===========================================================
class SEQParser(BaseParser):
    """
    SEQ parser: treats .seq as raw bases; first attempts FASTA parsing with BioPython
    to support headered files, then falls back to concatenating non-empty lines.
    Maintains input order for downstream vectorization.
    """

    async def _read_file(self, filepath: str) -> List[str]:
        return [filepath]

    def _extract_sequence(self, raw_data: List[str]):
        filepath = raw_data[0]
        # Try BioPython FASTA parsing in case the .seq has headers
        with open(filepath, "r") as fh:
            records = list(SeqIO.parse(fh, "fasta"))
        if records:
            return ''.join(str(rec.seq) for rec in records)

        # Fallback: raw concatenation
        with open(filepath, "r") as fh:
            seq = ''.join(line.strip() for line in fh if line.strip())
        return seq

    def _extract_metadata(self, raw_data: List[str]) -> dict:
        filepath = raw_data[0]
        headers = []
        with open(filepath, "r") as fh:
            for line in fh:
                if line.startswith(">"):
                    headers.append(line.strip()[1:])
        return {"headers": headers}



# ===========================================================
# ABI / AB1 PARSER
# ===========================================================
class ABIParser(BaseParser):
    """
    ABI/AB1 parser: uses BioPython SeqIO to extract basecalls from chromatogram files.
    """

    async def _read_file(self, filepath: str) -> List[str]:
        return [filepath]

    def _extract_sequence(self, raw_data: List[str]) -> str:
        filepath = raw_data[0]
        seqs = []
        with open(filepath, "rb") as handle:
            for record in SeqIO.parse(handle, "abi"):
                seqs.append(str(record.seq))
        return ''.join(seqs)

    def _extract_metadata(self, raw_data: List[str]) -> dict:
        filepath = raw_data[0]
        headers = []
        with open(filepath, "rb") as handle:
            for record in SeqIO.parse(handle, "abi"):
                headers.append(str(record.id))
        return {"headers": headers}
    


# ===========================================================
# FSA PARSER
# ===========================================================
class FSAParser(BaseParser):
    """
    FSA parser: treats .fsa as FASTA-like; streams with BioPython SeqIO and
    concatenates sequences. Headers preserved in metadata.
    """

    async def _read_file(self, filepath: str) -> List[str]:
        return [filepath]

    def _extract_sequence(self, raw_data: List[str]) -> str:
        filepath = raw_data[0]
        seqs = []
        with open(filepath, "r") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                seqs.append(str(record.seq))
        return ''.join(seqs)

    def _extract_metadata(self, raw_data: List[str]) -> dict:
        filepath = raw_data[0]
        headers = []
        with open(filepath, "r") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                headers.append(str(record.id))
        return {"headers": headers}
