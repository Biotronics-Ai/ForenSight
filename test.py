import asyncio
import time
from pathlib import Path
import contextlib
import zipfile
import csv
import numpy as np
import sys
import os

from models import SampleLoader, DoubleSampleComparator, ParserFactory, SequenceMatcher, STRSearcher, DNASample
from kernel import KernelMatrix
from visualize import (
    HIDVisualizer,
    ABIChromatogramVisualizer,
    FSAElectropherogramVisualizer,
    DataBandVisualizer,
)
import os

DATA_DIR = Path("data_samples")
MEMMAP_DIR = Path("mem_map")
LOG_DIR = Path("logs")
MEMMAP_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


def cleanup_memmaps(samples):
    """Remove temporary memmap files to reclaim disk/memory mappings."""
    for s in samples:
        path = getattr(s, "memmap_path", None)
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass


def log_msg(scenario: str, message: str):
    """Append a log line to the per-scenario log file."""
    log_file = LOG_DIR / f"{scenario}.logs"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a") as fh:
        fh.write(f"{message}\n")


async def scenario1():
    """
    Two FASTA files -> ParserFactory -> FASTAParser -> DNASample
    -> DoubleSampleComparator -> BandVisualizer
    """
    files = [str(DATA_DIR / "fa/y_chromosome_sample.fa"), str(DATA_DIR / "fa/y_chromosome_sample (kopya).fa")]
    loader = SampleLoader()
    log_path = LOG_DIR / "scenario1.logs"
    with log_path.open("w") as logfh, contextlib.redirect_stdout(logfh):
        t0 = time.time()
        batches = await loader.load_samples(files, batch_size=16384, aggregate=False, memmap_dir=str(MEMMAP_DIR))
        if any(len(v) == 0 for v in batches.values()):
            print("Scenario1: not enough samples")
            return
        s1_list = batches.get(files[0], [])
        s2_list = batches.get(files[1], [])
        if not s1_list or not s2_list:
            print("Scenario1: not enough batches")
            return

        comp = DoubleSampleComparator()
        total_len = min(sum(len(b.sequence) for b in s1_list), sum(len(b.sequence) for b in s2_list))
        sim = await comp.compare(s1_list, s2_list, batch_size=total_len)

        print(f"Scenario1 similarity: {sim}, time: {time.time() - t0:.12f}s")
        cleanup_memmaps(s1_list + s2_list)
        batches.clear()
        import gc
        gc.collect()
    print(f"Scenario1 completed. Detailed logs at {log_path}")





async def scenario1_b():
    """
    Two FASTA files -> ParserFactory -> FASTAParser -> DNASample
    -> DoubleSampleComparator -> BandVisualizer
    """
    files = [str(DATA_DIR / "fq/2_OHara_S1_psbA3_2019_minq7.fastq"), str(DATA_DIR / "fq/2_OHara_S1_psbA3_2019_minq7 (kopya).fastq")]
    loader = SampleLoader()
    log_path = LOG_DIR / "scenario1-b.logs"
    with log_path.open("w") as logfh, contextlib.redirect_stdout(logfh):
        t0 = time.time()
        batches = await loader.load_samples(files, batch_size=16384, aggregate=False, memmap_dir=str(MEMMAP_DIR))
        if any(len(v) == 0 for v in batches.values()):
            print("Scenario1: not enough samples")
            return
        s1_list = batches.get(files[0], [])
        s2_list = batches.get(files[1], [])
        if not s1_list or not s2_list:
            print("Scenario1: not enough batches")
            return

        comp = DoubleSampleComparator()
        total_len = min(sum(len(b.sequence) for b in s1_list), sum(len(b.sequence) for b in s2_list))
        sim = await comp.compare(s1_list, s2_list, batch_size=total_len)

        print(f"Scenario1 similarity: {sim}, time: {time.time() - t0:.12f}s")
        cleanup_memmaps(s1_list + s2_list)
        batches.clear()
        import gc
        gc.collect()
    print(f"Scenario1-B completed. Detailed logs at {log_path}")





async def scenario1_c():
    """
    Two FASTA files -> ParserFactory -> FASTAParser -> DNASample
    -> DoubleSampleComparator -> BandVisualizer
    """
    files = [str(DATA_DIR / "vcf/newALL.chr14.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf"), str(DATA_DIR / "vcf/newALL.chr18.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf")]
    loader = SampleLoader()
    log_path = LOG_DIR / "scenario1-c.logs"
    with log_path.open("w") as logfh, contextlib.redirect_stdout(logfh):
        t0 = time.time()
        batches = await loader.load_samples(files, batch_size=16384, aggregate=False, memmap_dir=str(MEMMAP_DIR))
        if any(len(v) == 0 for v in batches.values()):
            print("Scenario1: not enough samples")
            return
        s1_list = batches.get(files[0], [])
        s2_list = batches.get(files[1], [])
        if not s1_list or not s2_list:
            print("Scenario1: not enough batches")
            return

        comp = DoubleSampleComparator()
        total_len = min(sum(len(b.sequence) for b in s1_list), sum(len(b.sequence) for b in s2_list))
        sim = await comp.compare(s1_list, s2_list, batch_size=total_len)

        print(f"Scenario1 similarity: {sim}, time: {time.time() - t0:.12f}s")
        cleanup_memmaps(s1_list + s2_list)
        batches.clear()
        import gc
        gc.collect()
    print(f"Scenario1-C completed. Detailed logs at {log_path}")






async def scenario1_d():
    """
    Two FASTA files -> ParserFactory -> FASTAParser -> DNASample
    -> DoubleSampleComparator -> BandVisualizer
    """
    files = [str(DATA_DIR / "bigbed/ENCFF675AFH.bigBed"), str(DATA_DIR / "bigbed/ENCFF675AFH (kopya).bigBed")]
    loader = SampleLoader()
    log_path = LOG_DIR / "scenario1-d.logs"
    with log_path.open("w") as logfh, contextlib.redirect_stdout(logfh):
        t0 = time.time()
        batches = await loader.load_samples(files, batch_size=16384, aggregate=False, memmap_dir=str(MEMMAP_DIR))
        if any(len(v) == 0 for v in batches.values()):
            print("Scenario1: not enough samples")
            return
        s1_list = batches.get(files[0], [])
        s2_list = batches.get(files[1], [])
        if not s1_list or not s2_list:
            print("Scenario1: not enough batches")
            return

        comp = DoubleSampleComparator()
        total_len = min(sum(len(b.sequence) for b in s1_list), sum(len(b.sequence) for b in s2_list))
        sim = await comp.compare(s1_list, s2_list, batch_size=total_len)

        print(f"Scenario1 similarity: {sim}, time: {time.time() - t0:.12f}s")
        cleanup_memmaps(s1_list + s2_list)
        batches.clear()
        import gc
        gc.collect()
    print(f"Scenario1-D completed. Detailed logs at {log_path}")






async def scenario2():
    """
    Multiple FASTA files -> ParserFactory -> FASTAParser -> DNASample
    -> SequenceMatcher -> DoubleSampleComparator -> BandVisualizer
    """
    files = [
        str(DATA_DIR / "fa/md_dna/mt_dna.fa"),
        str(DATA_DIR / "fa/md_dna/mt_dna (kopya).fa"),
        str(DATA_DIR / "fa/md_dna/mt_dna (başka kopya).fa"),
        str(DATA_DIR / "fa/md_dna/mt_dna (3. kopya).fa"),
        str(DATA_DIR / "fa/md_dna/mt_dna (4. kopya).fa"),
        str(DATA_DIR / "fa/md_dna/mt_dna (5. kopya).fa"),
        str(DATA_DIR / "fa/md_dna/mt_dna (6. kopya).fa"),
        str(DATA_DIR / "fa/md_dna/mt_dna (7. kopya).fa"),
        str(DATA_DIR / "fa/md_dna/mt_dna (8. kopya).fa"),
        str(DATA_DIR / "fq/s_1_0010_end1.fastq"),
    ]
    loader = SampleLoader()
    scenario_name = "scenario2"
    log_path = LOG_DIR / "scenario2-ext-t.logs"
    with log_path.open("w") as logfh, contextlib.redirect_stdout(logfh):
        t0 = time.perf_counter()

        matcher = SequenceMatcher(target=None, pool=[])
        best_result = await matcher.match_streaming_paths(files[0], files[1:], batch_size=16384, loader=loader)
        if not best_result:
            print("Scenario2: no best sample found (stream)")
            return
        best_path = best_result["best_match"]
        log_msg(scenario_name, f"stream_best: {os.path.basename(best_path)} score={best_result['similarity']:.12f}")

        comp = DoubleSampleComparator()
        step = time.perf_counter()
        sim = await comp.compare_stream(files[0], best_path, batch_size=16384)
        log_msg(scenario_name, f"double_compare_stream: {time.perf_counter() - step:.12f}s")

        total = time.perf_counter() - t0
        log_msg(scenario_name, f"total: {total:.12f}s")
        print(f"Scenario2 best: {best_path}, similarity: {sim}, time: {total:.12f}s")
        cleanup_memmaps([])
        import gc
        gc.collect()
    print(f"Scenario2 completed. Detailed logs at {log_path}")





async def scenario2_b():
    """
    Multiple FASTA files -> ParserFactory -> FASTAParser -> DNASample
    -> SequenceMatcher -> DoubleSampleComparator -> BandVisualizer
    """
    files = [
        str(DATA_DIR / "fq/s_1_0010_end1.fastq"),
        str(DATA_DIR / "fq/s_1_0001_end1.fastq"),
        str(DATA_DIR / "fq/2_OHara_S1_psbA3_2019_minq7.fastq"),
        str(DATA_DIR / "fq/2_OHara_S1_psbA3_2019_minq7 (kopya).fastq"),
        str(DATA_DIR / "fq/4_OHara_S1B_18S_2019_minq7.fastq"),
        str(DATA_DIR / "fq/4_OHara_S1B_rbcLa_2019_minq7.fastq"),
        str(DATA_DIR / "fq/6_Swamp_S1_ITS2_2019_minq7.fastq"),
        #str(DATA_DIR / "fq/s_1_0010_end1 (kopya).fastq"),
    ]
    loader = SampleLoader()
    scenario_name = "scenario2"
    log_path = LOG_DIR / "scenario2-b.logs"
    with log_path.open("w") as logfh, contextlib.redirect_stdout(logfh):
        t0 = time.perf_counter()

        matcher = SequenceMatcher(target=None, pool=[])
        best_result = await matcher.match_streaming_paths(files[0], files[1:], batch_size=16384, loader=loader)
        if not best_result:
            print("Scenario2: no best sample found (stream)")
            return
        best_path = best_result["best_match"]
        log_msg(scenario_name, f"stream_best: {os.path.basename(best_path)} score={best_result['similarity']:.12f}")

        comp = DoubleSampleComparator()
        step = time.perf_counter()
        sim = await comp.compare_stream(files[0], best_path, batch_size=16384)
        log_msg(scenario_name, f"double_compare_stream: {time.perf_counter() - step:.12f}s")

        total = time.perf_counter() - t0
        log_msg(scenario_name, f"total: {total:.12f}s")
        print(f"Scenario2 best: {best_path}, similarity: {sim}, time: {total:.12f}s")
        cleanup_memmaps([])
        import gc
        gc.collect()
    print(f"Scenario2 completed. Detailed logs at {log_path}")




async def scenario2_c():
    """
    Multiple FASTA files -> ParserFactory -> FASTAParser -> DNASample
    -> SequenceMatcher -> DoubleSampleComparator -> BandVisualizer
    """
    files = [
        str(DATA_DIR / "vcf/newALL.chr14.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf"),
        str(DATA_DIR / "vcf/newALL.chr18.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf"),
        str(DATA_DIR / "vcf/newALL.chr19.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf"),
        str(DATA_DIR / "vcf/newALL.chr14.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes (kopya).vcf"),
        #str(DATA_DIR / "fq/s_1_0010_end1 (kopya).fastq"),
    ]
    loader = SampleLoader()
    scenario_name = "scenario2"
    log_path = LOG_DIR / "scenario2-c.logs"
    with log_path.open("w") as logfh, contextlib.redirect_stdout(logfh):
        t0 = time.perf_counter()

        matcher = SequenceMatcher(target=None, pool=[])
        best_result = await matcher.match_streaming_paths(files[0], files[1:], batch_size=16384, loader=loader)
        if not best_result:
            print("Scenario2: no best sample found (stream)")
            return
        best_path = best_result["best_match"]
        log_msg(scenario_name, f"stream_best: {os.path.basename(best_path)} score={best_result['similarity']:.12f}")

        comp = DoubleSampleComparator()
        step = time.perf_counter()
        sim = await comp.compare_stream(files[0], best_path, batch_size=16384)
        log_msg(scenario_name, f"double_compare_stream: {time.perf_counter() - step:.12f}s")

        total = time.perf_counter() - t0
        log_msg(scenario_name, f"total: {total:.12f}s")
        print(f"Scenario2 best: {best_path}, similarity: {sim}, time: {total:.12f}s")
        cleanup_memmaps([])
        import gc
        gc.collect()
    print(f"Scenario2 completed. Detailed logs at {log_path}")




async def scenario3():
    """
    Multiple FASTA + one FASTQ -> ParserFactory -> FASTAParser/FASTQParser -> DNASample
    -> SequenceMatcher -> DoubleSampleComparator -> BandVisualizer
    """
    files = [
        str(DATA_DIR / "fa/md_dna/mt_dna.fa"),
        str(DATA_DIR / "fa/md_dna/mt_dna (kopya).fa"),
        str(DATA_DIR / "fa/md_dna/mt_dna (başka kopya).fa"),
        str(DATA_DIR / "fa/md_dna/mt_dna (3. kopya).fa"),
        str(DATA_DIR / "fa/md_dna/mt_dna (4. kopya).fa"),
        str(DATA_DIR / "fa/md_dna/mt_dna (5. kopya).fa"),
        str(DATA_DIR / "fa/md_dna/mt_dna (6. kopya).fa"),
        str(DATA_DIR / "fa/md_dna/mt_dna (7. kopya).fa"),
        str(DATA_DIR / "fa/md_dna/mt_dna (8. kopya).fa"),
        str(DATA_DIR / "vcf/clinvar.vcf"),
    ]
    loader = SampleLoader()
    scenario_name = "scenario3"
    t0 = time.perf_counter()

    matcher = SequenceMatcher(target=None, pool=[])
    best_result = await matcher.match_streaming_paths(files[0], files[1:], batch_size=16384, loader=loader)
    if not best_result:
        print("Scenario3: no best sample found (stream)")
        return
    best_path = best_result["best_match"]
    log_msg(scenario_name, f"stream_best: {os.path.basename(best_path)} score={best_result['similarity']:.12f}")

    comp = DoubleSampleComparator()
    step = time.perf_counter()
    sim = await comp.compare_stream(files[0], best_path, batch_size=16384)
    log_msg(scenario_name, f"double_compare_stream: {time.perf_counter() - step:.12f}s")

    total = time.perf_counter() - t0
    log_msg(scenario_name, f"total: {total:.12f}s")
    print(f"Scenario3 best: {best_path}, similarity: {sim}, time: {total:.12f}s")
    cleanup_memmaps([])
    import gc
    gc.collect()




async def scenario4():
    """
    Multiple FASTA + one FASTQ -> ParserFactory -> FASTAParser/FASTQParser -> DNASample
    -> SequenceMatcher -> DoubleSampleComparator -> BandVisualizer
    """
    files = [
        str(DATA_DIR / "fa/md_dna/mt_dna.fa"),
        str(DATA_DIR / "fa/md_dna/mt_dna.fa"),
        str(DATA_DIR / "fa/md_dna/mt_dna.fa"),
        str(DATA_DIR / "fa/md_dna/mt_dna.fa"),
        str(DATA_DIR / "fa/md_dna/mt_dna.fa"),
        str(DATA_DIR / "bigwig/wgEncodeUwRepliSeqGm12812S3PctSignalRep1.bigWig"),
    ]
    loader = SampleLoader()
    scenario_name = "scenario4"
    t0 = time.perf_counter()

    matcher = SequenceMatcher(target=None, pool=[])
    best_result = await matcher.match_streaming_paths(files[0], files[1:], batch_size=16384, loader=loader)
    if not best_result:
        print("Scenario4: no best sample found (stream)")
        return
    best_path = best_result["best_match"]
    log_msg(scenario_name, f"stream_best: {os.path.basename(best_path)} score={best_result['similarity']:.12f}")

    comp = DoubleSampleComparator()
    step = time.perf_counter()
    sim = await comp.compare_stream(files[0], best_path, batch_size=16384)
    log_msg(scenario_name, f"double_compare_stream: {time.perf_counter() - step:.12f}s")

    total = time.perf_counter() - t0
    log_msg(scenario_name, f"total: {total:.12f}s")
    print(f"Scenario4 best: {best_path}, similarity: {sim}, time: {total:.12f}s")
    cleanup_memmaps([])
    import gc
    gc.collect()




async def scenario5():
    """
    Multiple FASTA + one FASTQ -> ParserFactory -> FASTAParser/FASTQParser -> DNASample
    -> SequenceMatcher -> DoubleSampleComparator -> BandVisualizer
    """
    files = [
        str(DATA_DIR / "fa/md_dna/mt_dna.fa"),
        str(DATA_DIR / "fa/md_dna/mt_dna.fa"),
        str(DATA_DIR / "fa/md_dna/mt_dna.fa"),
        str(DATA_DIR / "fa/md_dna/mt_dna.fa"),
        str(DATA_DIR / "fa/md_dna/mt_dna.fa"),
        str(DATA_DIR / "bigbed/ENCFF675AFH.bigBed"),
    ]
    loader = SampleLoader()
    scenario_name = "scenario5"
    t0 = time.perf_counter()

    matcher = SequenceMatcher(target=None, pool=[])
    best_result = await matcher.match_streaming_paths(files[0], files[1:], batch_size=16384, loader=loader)
    if not best_result:
        print("Scenario5: no best sample found (stream)")
        return
    best_path = best_result["best_match"]
    log_msg(scenario_name, f"stream_best: {os.path.basename(best_path)} score={best_result['similarity']:.12f}")

    comp = DoubleSampleComparator()
    step = time.perf_counter()
    sim = await comp.compare_stream(files[0], best_path, batch_size=16384)
    log_msg(scenario_name, f"double_compare_stream: {time.perf_counter() - step:.12f}s")

    total = time.perf_counter() - t0
    log_msg(scenario_name, f"total: {total:.12f}s")
    print(f"Scenario5 best: {best_path}, similarity: {sim}, time: {total:.12f}s")
    cleanup_memmaps([])
    import gc
    gc.collect()





async def scenario6():
    """
    Multiple FASTA + one FASTQ -> ParserFactory -> FASTAParser/FASTQParser -> DNASample
    -> SequenceMatcher -> DoubleSampleComparator -> BandVisualizer
    """
    files = [
        str(DATA_DIR / "bigwig/wgEncodeUwRepliSeqGm12812S3PctSignalRep1 (3. kopya).bigWig"),
        str(DATA_DIR / "vcf/clinvar.vcf"),
        str(DATA_DIR / "bigwig/wgEncodeUwRepliSeqGm12812S3PctSignalRep1.bigWig"),
    ]
    loader = SampleLoader()
    scenario_name = "scenario6"
    t0 = time.perf_counter()

    matcher = SequenceMatcher(target=None, pool=[])
    best_result = await matcher.match_streaming_paths(files[0], files[1:], batch_size=16384, loader=loader)
    if not best_result:
        print("Scenario6: no best sample found (stream)")
        return
    best_path = best_result["best_match"]
    log_msg(scenario_name, f"stream_best: {os.path.basename(best_path)} score={best_result['similarity']:.12f}")

    comp = DoubleSampleComparator()
    step = time.perf_counter()
    sim = await comp.compare_stream(files[0], best_path, batch_size=16384)
    log_msg(scenario_name, f"double_compare_stream: {time.perf_counter() - step:.12f}s")

    total = time.perf_counter() - t0
    log_msg(scenario_name, f"total: {total:.12f}s")
    print(f"Scenario6 best: {best_path}, similarity: {sim}, time: {total:.12f}s")
    cleanup_memmaps([])
    import gc
    gc.collect()




async def scenario7():
    """
    Two FASTA files -> ParserFactory -> FASTAParser -> DNASample
    -> DoubleSampleComparator -> BandVisualizer
    """
    files = [str(DATA_DIR / "bigwig/wgEncodeUwRepliSeqGm12812S3PctSignalRep1.bigWig"), str(DATA_DIR / "bigwig/wgEncodeUwRepliSeqGm12812S3PctSignalRep1 (3. kopya).bigWig")]
    loader = SampleLoader()
    log_path = LOG_DIR / "scenario7.logs"
    with log_path.open("w") as logfh, contextlib.redirect_stdout(logfh):
        t0 = time.time()
        batches = await loader.load_samples(files, batch_size=16384, aggregate=False, memmap_dir=str(MEMMAP_DIR))
        if any(len(v) == 0 for v in batches.values()):
            print("Scenario7: not enough samples")
            return
        s1_list = batches.get(files[0], [])
        s2_list = batches.get(files[1], [])
        if not s1_list or not s2_list:
            print("Scenario7: not enough batches")
            return

        comp = DoubleSampleComparator()
        total_len = min(sum(len(b.sequence) for b in s1_list), sum(len(b.sequence) for b in s2_list))
        sim = await comp.compare(s1_list, s2_list, batch_size=total_len)

        print(f"Scenario7 similarity: {sim}, time: {time.time() - t0:.12f}s")
        cleanup_memmaps(s1_list + s2_list)
        batches.clear()
        import gc
        gc.collect()
    print(f"Scenario7 completed. Detailed logs at {log_path}")







async def scenario_hid():
    """
    HID file -> HIDVisualizer -> CSV output
    """
    log_file = LOG_DIR / "scenario_hidxx.logs"
    with log_file.open("w") as logfh, contextlib.redirect_stdout(logfh):
        hid_rel = DATA_DIR / "mixture" / "Promega_PowerPlex_Fusion_6C" / "PPF6C_H4_3P-C_10D-30D-60D_1_H07_22.hid"
        if not hid_rel.exists():
            print(f"Scenario HID: target .hid not found at {hid_rel}; skipping")
            return

        out_csv = LOG_DIR / "hid_output_xx.csv"
        hv = HIDVisualizer()
        do_visualize = True  # set False to skip generating WEBP trace
        t0 = time.perf_counter()
        await hv.visualize(str(hid_rel), output_path=str(out_csv), visualize=do_visualize)
        print(f"Scenario HID: wrote {out_csv} in {time.perf_counter() - t0:.6f}s")
    print(f"Scenario HID completed. Detailed logs at {log_file}")





async def scenario_abi():
    """
    ABI/AB1 file -> ABIChromatogramVisualizer -> CSV output (all ABIF key/values)
    """
    log_file = LOG_DIR / "scenario_abi.logs"
    with log_file.open("w") as logfh, contextlib.redirect_stdout(logfh):
        abi_file = "data_samples/sample.abi"
        if not os.path.exists(abi_file):
            abi_file = "sample.abi"
        if not os.path.exists(abi_file):
            print(f"Scenario ABI: no ABI file found at {abi_file}; skipping")
            return

        out_csv = LOG_DIR / "abi_output.csv"
        viz = ABIChromatogramVisualizer()
        do_visualize = True  # set False to skip generating WEBP trace
        t0 = time.perf_counter()
        await viz.visualize(str(abi_file), output_path=str(out_csv), visualize=do_visualize)
        print(f"Scenario ABI: wrote {out_csv} in {time.perf_counter() - t0:.6f}s")
    print(f"Scenario ABI completed. Detailed logs at {log_file}")





async def scenario_fsa():
    """
    FSA file -> FSAElectropherogramVisualizer -> CSV outputs
    """
    log_file = LOG_DIR / "scenario_fsa.logs"
    with log_file.open("w") as logfh, contextlib.redirect_stdout(logfh):
        fsa_file = DATA_DIR / "fsa" / "sample.fsa"
        if not fsa_file.exists():
            fsa_file = Path("sample.fsa")
        if not fsa_file.exists():
            print(f"Scenario FSA: no FSA file found at {fsa_file}; skipping")
            return

        out_csv = LOG_DIR / "fsa_output.csv"
        viz = FSAElectropherogramVisualizer()
        do_visualize = True  # set False to skip generating WEBP trace
        t0 = time.perf_counter()
        await viz.visualize(str(fsa_file), output_path=str(out_csv), visualize=do_visualize)
        print(f"Scenario FSA: wrote {out_csv} in {time.perf_counter() - t0:.6f}s")
    print(f"Scenario FSA completed. Detailed logs at {log_file}")





async def scenario_data_band():
    """
    Synthetic DATA9-12 traces -> DataBandVisualizer -> WEBP output
    """
    log_file = LOG_DIR / "scenario_databand.logs"
    with log_file.open("w") as logfh, contextlib.redirect_stdout(logfh):
        def synth_trace(peaks, amps, sigma, length=2000, noise_mu=50, noise_sigma=8, seed=None):
            rng = np.random.default_rng(seed)
            x = np.arange(length)
            signal = rng.normal(noise_mu, noise_sigma, length)
            for p, a in zip(peaks, amps):
                signal += a * np.exp(-0.5 * ((x - p) / sigma) ** 2)
            signal = np.clip(signal, 0, None)
            return signal.astype(int)

        data9 = synth_trace(peaks=[180, 620, 1180, 1650], amps=[600, 900, 750, 500], sigma=25, seed=1)
        data10 = synth_trace(peaks=[240, 760, 1320, 1750], amps=[700, 850, 680, 520], sigma=35, seed=2)
        data11 = synth_trace(peaks=[150, 540, 980, 1540], amps=[500, 720, 900, 650], sigma=30, seed=3)
        data12 = synth_trace(peaks=[320, 880, 1420, 1880], amps=[820, 760, 700, 600], sigma=28, seed=4)

        to_row = lambda arr: ";".join(str(int(v)) for v in arr.tolist())
        trace_rows = [
            ["key", "value"],
            ["DATA9", to_row(data9)],
            ["DATA10", to_row(data10)],
            ["DATA11", to_row(data11)],
            ["DATA12", to_row(data12)],
        ]
        trace_csv = LOG_DIR / "databand_trace.csv"
        with trace_csv.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerows(trace_rows)

        viz = DataBandVisualizer()
        viz.render_from_trace_rows(trace_rows, trace_csv.with_suffix(".webp"))
        print(f"Scenario DataBand: wrote trace CSV {trace_csv} and WEBP {trace_csv.with_suffix('.webp')}")
    print(f"Scenario DataBand completed. Detailed logs at {log_file}")





async def scenario_str_searcher():
    """
    STRSearcher: search a pattern across multiple FASTA files, log best match.
    """
    log_file = LOG_DIR / "scenario_str_searcher.logs"
    with log_file.open("w") as logfh, contextlib.redirect_stdout(logfh):
        pattern = "AGCTAGTTCGGATATAAACNGTGTTNACATCGNCANGTAAGCNTGACNTCTAATATATTN"
        files = [
            str(DATA_DIR / "fa/md_dna/mt_dna.fa"),
            str(DATA_DIR / "fa/md_dna/mt_dna (kopya).fa"),
            str(DATA_DIR / "fa/md_dna/mt_dna (başka kopya).fa"),
            str(DATA_DIR / "fa/md_dna/mt_dna (3. kopya).fa"),
            str(DATA_DIR / "fa/md_dna/mt_dna (4. kopya).fa"),
            str(DATA_DIR / "fa/md_dna/mt_dna (5. kopya).fa"),
            str(DATA_DIR / "fa/md_dna/mt_dna (6. kopya).fa"),
            str(DATA_DIR / "fa/md_dna/mt_dna (7. kopya).fa"),
            str(DATA_DIR / "fa/md_dna/mt_dna (8. kopya).fa"),
        ]
        files = [f for f in files if Path(f).exists()]
        if len(files) < 2:
            print("Scenario STR: not enough FASTA files found; skipping")
            return

        searcher = STRSearcher()
        t0 = time.perf_counter()
        result = await searcher.search(pattern, files)
        if result:
            print(
                f"Scenario STR best: file={result['file']}, pos={result['position']}, "
                f"substring={result['substring']}, similarity={result['similarity']:.6f}, "
                f"time={time.perf_counter() - t0:.6f}s"
            )
        else:
            print("Scenario STR: no match found")
    print(f"Scenario STR completed. Detailed logs at {log_file}")





async def scenario_kernel():
    """
    KernelMatrix: two cases:
      1) build without saving vectors (conditional off)
      2) build with vector save + conditional batch check
    """
    log_file = LOG_DIR / "scenario_kernel.logs"
    import logging
    km_logger = logging.getLogger("KernelMatrix")
    handler = logging.FileHandler(log_file, mode="w")
    handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
    km_logger.addHandler(handler)
    km_logger.setLevel(logging.INFO)
    t0_global = time.perf_counter()

    with log_file.open("a") as logfh, contextlib.redirect_stdout(logfh):
        rng = np.random.default_rng(42)
        length = 16384
        n_samples = 16500
        batch_size = 16384
        base = rng.normal(0, 1, length).astype(np.float32)
        batches = []
        t_gen = time.perf_counter()
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch = []
            for i in range(start, end):
                noise = rng.normal(0, 0.01, length).astype(np.float32)
                vec = base + noise
                batch.append(DNASample(f"sample_{i}", vec, file_format="synthetic"))
            batches.append(batch)
        gen_time = time.perf_counter() - t_gen

        # Case 1: no vector save, conditional False
        t0 = time.perf_counter()
        km = KernelMatrix.build_from_batches(
            batches,
            block_size=0,
            memmap_dir=str(MEMMAP_DIR),
            conditional=False,
            save_vectors_path=None,
            logs_dir=str(LOG_DIR),
        )
        build_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        best = km.best_match("sample_0")
        query_time = time.perf_counter() - t1

        print(f"Kernel generation time for {n_samples} samples (batch {batch_size}): {gen_time:.6f}s")
        print(f"[Case1] Kernel build time: {build_time:.6f}s")
        print(f"[Case1] Kernel query time (best match to sample_0): {query_time:.6f}s")
        print(f"[Case1] Kernel best match to sample_0: {best} (expected close to sample_1)")

        batches.clear()
        km.cleanup()
        del km
        import gc
        gc.collect()

        # Case 2: save vectors, conditional batch check, separate log
        log_file2 = LOG_DIR / "scenario_kernel_case2.logs"
        handler2 = logging.FileHandler(log_file2, mode="w")
        handler2.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
        km_logger.addHandler(handler2)

        with log_file2.open("a") as logfh2, contextlib.redirect_stdout(logfh2):
            batches = []
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch = []
                for i in range(start, end):
                    noise = rng.normal(0, 0.01, length).astype(np.float32)
                    vec = base + noise
                    batch.append(DNASample(f"sample_{i}#batch{i - start}", vec, file_format="synthetic"))
                batches.append(batch)

            vec_save = LOG_DIR / "kernel_vectors.npy"
            t0 = time.perf_counter()
            km2 = KernelMatrix.build_from_batches(
                batches,
                block_size=0,
                memmap_dir=str(MEMMAP_DIR),
                conditional=True,
                save_vectors_path=str(vec_save),
                logs_dir=str(LOG_DIR),
            )
            build_time2 = time.perf_counter() - t0
            best2 = km2.best_match("sample_0#batch0")
            print(f"[Case2] Kernel build time (saved vectors): {build_time2:.6f}s")
            print(f"[Case2] Kernel best match to sample_0#batch0: {best2} (expected close to sample_1#batch1)")
            batches.clear()
            km2.cleanup()
            del km2
            gc.collect()
        km_logger.removeHandler(handler2)
        handler2.close()

    # Case 3: high-dimensional tensor data (SP kernel path) with known closest pair
    log_file3 = LOG_DIR / "scenario_kernel_case3.logs"
    handler3 = logging.FileHandler(log_file3, mode="w")
    handler3.setFormatter(logging.Formatter("[%(name)s] %(levelname)s: %(message)s"))
    km_logger.addHandler(handler3)
    with log_file3.open("a") as logfh3, contextlib.redirect_stdout(logfh3):
        seq_length = 16384
        meta_params = 1
        meta_keys = 8
        tensor_shape = (seq_length, meta_params, meta_keys)  # ndim=3 -> SP path
        base_tensor = rng.normal(0, 1, tensor_shape).astype(np.float32)
        samples = []
        closest_id = "tensor_1"
        params_pool = [f"param_{p}" for p in range(10)]
        keys_pool = [f"key_{k}" for k in range(10)]
        notes_pool = [f"note_variant_{i}" for i in range(10)]
        for i in range(5000):
            # tensor_1 is identical to tensor_0; others have larger noise to ensure separation
            noise_scale = 0.0 if i == 1 else 0.05
            # build metadata with potentially overlapping but not identical entries
            chosen_params = rng.choice(params_pool, size=meta_params, replace=True).tolist()
            chosen_keys = rng.choice(keys_pool, size=meta_keys + 3, replace=True).tolist()  # ensure >4 pairs
            kv_pairs = {k: float(rng.normal()) for k in chosen_keys}
            metadata = {
                "params": chosen_params,
                "keys": chosen_keys,
                "values": kv_pairs,
                "note": rng.choice(notes_pool),
            }
            samples.append(
                DNASample(
                    f"tensor_{i}",
                    base_tensor + rng.normal(0, noise_scale, tensor_shape).astype(np.float32),
                    "synthetic",
                    metadata=metadata,
                )
            )
        t0 = time.perf_counter()
        km3 = KernelMatrix(
            samples,
            memmap_dir=str(MEMMAP_DIR),
            logs_dir=str(LOG_DIR),
            conditional=False,
        )
        build_time3 = time.perf_counter() - t0
        best3 = km3.best_match("tensor_0")
        print(f"[Case3] SP kernel build time: {build_time3:.6f}s")
        print(f"[Case3] Expected closest: {closest_id}; got: {best3}")
        km3.cleanup()
        del km3
        gc.collect()
    km_logger.removeHandler(handler3)
    handler3.close()

    km_logger.removeHandler(handler)
    handler.close()
    total_time = time.perf_counter() - t0_global
    print(f"Scenario Kernel completed in {total_time:.6f}s. Detailed logs at {log_file}, {log_file2}, and {log_file3}")





async def main():
    #await scenario1()
    #await scenario1_b()
    #await scenario1_c()
    #await scenario1_d()
    #await scenario2()
    #await scenario2_b()
    #await scenario2_c()
    #await scenario3()
    #await scenario4()
    #await scenario5()
    #await scenario_hid()
    #await scenario_abi()
    #await scenario_fsa()
    #await scenario_data_band()
    await scenario_kernel()
    #await scenario_str_searcher()

if __name__ == "__main__":
    asyncio.run(main())
