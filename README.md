<p align="center">
  <img src="images/banner.png" alt="Biotronics Ai">
</p>
# Forensight

# *ForenSight*

An open-source toolkit for forensic genomic comparison and data storage, built by Biotronics AI. It streams large files, minimizes RAM with memory mapping, surfaces rich logging, and runs locally for forensic/genomic analysis.

**What it delivers in practice?**

- **Multi-DNA comparison within seconds:** stream batches, eliminate weak candidates fast, then score the top match with cosine similarity.
- **STR pattern detection in a second across multiple DNA samples:** sliding-window STR search scans whole genomes without loading everything into memory.
- **HID/ABI/FSA to CSV in one pass:** extract peaks, traces, and metadata for downstream mixture/QC without proprietary viewers.
- **Multi-format handling:**  Normalize the sequence formats to be compatible and comparable to each other.

## Features

**Supported Genomic Data File Formats**:

* **Sequence:**  Data types including sequence information FASTA/FASTQ/SEQ/SAM/BAM/CRAM files.
* **Annotation:**  VCF/BED/BigBed/BigWig/WIG/GFF/GTF files.
* **Electrophoregram:**  FSA/ABI/AB1/HID with async batch reading and N-stripping.

**Normalization**: Cross-format sequence normalization **(SequenceNormalizationManager)** for mixed sequence inputs with async batch reading and N-stripping.

**Core engines**:

- `SampleLoader` (batch/memmap streaming)
- `SequenceMatcher` (statistical elimination with Wilson intervals)
- `DoubleSampleComparator` (cosine-like similarity, chunked)
- `STRSearcher` (sliding-window similarity search)

**Visualizers → CSV**: HID/ABI/FSA emit multiple CSVs (main peaks, trace/basecall, excluded fields, APrX sidecars). Band visualizer for sequence/feature pairs.

**Memory-aware**: Streaming, per-batch processing, optional memmap buffers, explicit cleanup paths.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install forensight
```

Dependencies are defined in `pyproject.toml` / `requirements.txt`.

## Project layout

- `models.py` — parsers factory, loader, matchers, comparators, STR search.
- `parsers.py` — file-format parsers (async batch, N-stripping).
- `visualize.py` — CSV visualizers (HID/ABI/FSA), band visualizer.
- `test.py` — runnable scenarios (pipelines, visualizers, STR search).
- `base.py` — base components/logging helpers.
- `data_samples/` — sample data paths expected by `test.py` (adjust as needed).
- `mem_map/`, `logs/` — runtime outputs (ignored by git).

## Quick start (scenarios)

Edit `test.py` and uncomment the scenarios you want, then:

```bash
source .venv/bin/activate
python test.py
```

Scenarios (toggle in `main()`):

- `scenario1..5` — sequence comparison pipelines (SampleLoader → SequenceMatcher/DoubleSampleComparator).
- `scenario_hid` — HID → CSVs.
- `scenario_abi` — ABI/AB1 → CSVs.
- `scenario_fsa` — FSA → CSVs.
- `scenario_str_searcher` — sliding-window STR search across FASTA files.

Outputs land in `logs/`:

- HID/ABI/FSA: `{name}.csv` (peaks), `{name}.trace.csv` (PLOC/DATA/PBAS), `{name}.excluded.csv` (other fields), optional `{name}.aprx.csv/.xml` (APrX1).
- Band visualizer: WEBP tiles (if used) and optional metadata CSV.

## Core usage examples (and real-world analogues)

### 1) Dual-sample comparison

Use when you have two specimens of the same format (e.g., two FASTA genomes, or two FASTQ readsets) and want a similarity score.

```python
from forensight import SampleLoader, DoubleSampleComparator

loader = SampleLoader()
batches = await loader.load_samples(
    ["sample1.fa", "sample2.fa"],
    batch_size=16384,
    memmap_dir="mem_map",
)
s1, s2 = batches["sample1.fa"], batches["sample2.fa"]
comp = DoubleSampleComparator()
sim = await comp.compare(s1, s2, batch_size=16384)
print("cosine-like similarity:", sim)
```

_Real life_: Basic “are these two references the same?” QC, or comparing two assemblies of the same chromosome.

### 2) Multi-sample elimination to find the closest match (scenario2/3/4)

Use when you have one target and a pool of candidates (all sequence formats). The matcher samples loci, eliminates weak candidates with Wilson intervals, and returns the best; then you can stream a full comparison against the winner. Keep `memmap_dir` set for all loaders.

```python
from forensight import SampleLoader, SequenceMatcher, DoubleSampleComparator

loader = SampleLoader()
matcher = SequenceMatcher(target=None, pool=[])
best = await matcher.match_streaming_paths(
    "target.fa",
    ["cand1.fa", "cand2.fa", "cand3.fa"],
    batch_size=16384,
    loader=loader,
    memmap_dir="mem_map",
)
winner = best["best_match"]
comp = DoubleSampleComparator()
sim = await comp.compare_stream("target.fa", winner, batch_size=16384)
print("best:", winner, "similarity:", sim)
```

_Real life_: Pick the closest specimen in a large archive to a query genome/contig, without loading everything into RAM.

### 2b) Mixed sequence formats handled seamlessly

When the target and candidates are different sequence file types (e.g., FASTA + FASTQ + SEQ), the ParserFactory + SequenceNormalizationManager normalize non-FASTA inputs to a common sequence form before comparison. Ordering is preserved batch-by-batch. Use memmaps consistently.

```python
from forensight import SampleLoader, SequenceMatcher

files = ["target.fa", "reads.fastq", "sample.seq"]
matcher = SequenceMatcher(target=None, pool=[])
best = await matcher.match_streaming_paths(
    files[0],
    files[1:],
    batch_size=16384,
    loader=SampleLoader(),
    memmap_dir="mem_map",
)
print("best match across mixed formats:", best)
```

_Real life_: Compare a reference contig (FASTA) against sequencing reads (FASTQ) and a legacy SEQ file in one pass, without pre-conversion steps.

### 3) STR sliding-window search (scenario_str_searcher)

Use when you need to find the most similar occurrence of a short STR pattern across multiple sequences.

```python
from forensight import STRSearcher

searcher = STRSearcher()
result = await searcher.search("ATGCTAGCTA", ["genome1.fa", "genome2.fa"])
print(result)  # file, position, substring, similarity
```

_Real life_: Forensic STR probe search across multiple chromosomes/assemblies; finds best match even with minor mismatches.

### 4) HID/ABI/FSA to CSV (and optional WEBP traces)

Use when you have capillary electrophoresis outputs and need structured CSVs of signals/metadata. You can toggle plotting of DATA9–12 traces with `visualize=True/False`.

```python
from forensight import HIDVisualizer, ABIChromatogramVisualizer, FSAElectropherogramVisualizer

# HID: CSVs + WEBP
await HIDVisualizer().visualize("sample.hid", output_path="hid_output.csv", visualize=True)
# HID: CSVs only (skip WEBP)
# await HIDVisualizer().visualize("sample.hid", output_path="hid_output.csv", visualize=False)

# ABI: CSVs + WEBP (or set visualize=False to skip)
await ABIChromatogramVisualizer().visualize("sample.abi", output_path="abi_output.csv", visualize=True)

# FSA: CSVs + WEBP (or set visualize=False to skip)
await FSAElectropherogramVisualizer().visualize("sample.fsa", output_path="fsa_output.csv", visualize=True)

# Sidecars: .trace.csv, .excluded.csv, .aprx.csv/.xml (if present)
# WEBP legend: A=DATA9 (blue), C=DATA10 (green), G=DATA11 (yellow), T=DATA12 (magenta)
```

_Real life_: Extract instrument settings, traces/basecalls, and metadata from CE runs for downstream mixture/trace analysis or QC.

### 5) Trace visualization from HID/ABI/FSA traces

Use when you want a quick look at DATA9–12 traces without rerunning the extractor. Pass `visualize=False` to the main visualizers to skip auto-WEBP, then render later:

```python
import csv
from forensight import DataBandVisualizer

with open("hid_output.trace.csv", newline="") as fh:
    trace_rows = list(csv.reader(fh))

DataBandVisualizer().render_from_trace_rows(trace_rows, "hid_output.trace.webp")
# Legend: A=DATA9 (blue), C=DATA10 (green), G=DATA11 (yellow), T=DATA12 (magenta)
```

_Real life_: Inspect electropherogram channel intensities quickly without heavy GUIs.

### 6) Kernel matrix (memmap) with and without saved vectors/ids

Use when you want a reusable kernel over many samples. Always set `memmap_dir` and `logs_dir`.

```python
from forensight import KernelMatrix, DNASample
import numpy as np

# Synthetic samples
samples = [
    DNASample(f"sample_{i}", np.random.rand(1024).astype(np.float32), "synthetic")
    for i in range(100)
]

# Case A: no vector save, conditional off
km_a = KernelMatrix(
    samples,
    memmap_dir="mem_map",
    logs_dir="logs",
    conditional=False,
    save_vectors_path=None,   # nothing persisted
)
best_a = km_a.best_match("sample_0")
km_a.cleanup()

# Case B: save vectors and ids, conditional on
km_b = KernelMatrix(
    samples,
    memmap_dir="mem_map",
    logs_dir="logs",
    conditional=True,                     # optional, additional security layer before encryption
    save_vectors_path="logs/kernel.npy",  # persists stacked vectors
)
# ids are written to logs/kernel_vectors_ids.txt
best_b = km_b.best_match("sample_0")
km_b.cleanup()
```

⚠️ **Hardware & data quality**: Kernel builds are memory-heavy (O(n²) for n samples). Ensure `memmap_dir` has disk space and your machine has enough RAM for the chosen sample count/length. Poor-quality or inconsistent sequences will degrade similarity results—verify inputs before kernelizing.

### 7) Creating `DNASample` objects directly

`SampleLoader` usually creates `DNASample` objects for you (batch-by-batch, with N-stripping and optional memmaps). If you need to construct them manually (e.g., synthetic tests), use:

```python
import numpy as np
from forensight import DNASample

vec = np.array([0, 1, 2, 3], dtype=np.int8)  # your vectorized sequence
sample = DNASample(
    sample_id="sample_0",
    sequence=vec,
    file_format="fq",   # e.g., 'fa', 'fq', etc.
    metadata={"note": "example"},  # optional: headers/fields from parser
    memmap_path=None           # optional: path if sequence is a memmap
)
```

Note: In normal use, `SampleLoader` handles parsing, normalization, vectorization, N-removal, metadata attachment, and optional memmap creation automatically. The above is for users who need to craft `DNASample` objects by hand for custom pipelines or tests.

## Notes on visualizers

- HID/ABI/FSA readers expect ABIF traces (PLOC*/DATA*). If traces are missing, peak CSVs may be empty but metadata sidecars still export.
- ABI/FSA/HID visualizers partition outputs:
  - **main metadata**: curated fields (run/sample/dye/trace pointers).
  - **trace/basecall**: PLOC1/2, DATA1–12, PBAS1/2.
  - **excluded**: everything else.
  - **APrX1**: parsed parameters + raw XML when present.
  - **optional WEBP**: DATA9-12 line plot (A=DATA9, C=DATA10, G=DATA11, T=DATA12). Toggle with `visualize=True/False`.

## Memory & performance tips

- Prefer streaming APIs: `SampleLoader.stream_samples`, `compare_stream`, `match_streaming_paths`.
- Use `memmap_dir` to offload large batches.
- Keep `batch_size` aligned across components; adjust for RAM via `utils.calculate_effort` (helper).
- Clean up memmaps after use (see `cleanup_memmaps` in `test.py`).

## Testing

- Minimal scenarios live in `test.py`. Provide small sample files under `data_samples/`.
- Suggested additions: `pytest` + `pytest-asyncio` for unit coverage of parsers, matchers, visualizers.

## CI (suggested)

- Add a GitHub Actions workflow to run `python -m compileall`, `ruff` (optional), and `pytest`.

## Contributing

We value all kinds of contributions to the projects but the

- Standard PR/issue workflow.
- Keep new parsers async-friendly and streaming-capable.
- Preserve logging via `BaseComponent._log_state`.
- Development of ForenSight for other programming languages.
- Improve the system to overcome known limitations.
- Increase the number of supported file formats.
- Implement progress bar to the models properly.

## Known limitations

- HID/ABI/FSA peak CSVs rely on available traces; when absent, only metadata is emitted.
- Band visualizer image output is tiled to respect WEBP size limits; text labels are minimal by design.

<p align="center">
  <a href="https://biotronics.ai">
    <img src="images/logo.png" alt="Biotronics Ai Logo" width="180">
  </a>
</p>
