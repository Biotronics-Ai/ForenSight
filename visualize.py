import asyncio
from typing import Optional
import numpy as np
import tempfile
import csv
import math
from PIL import Image, ImageDraw
import re
import os 
from base import BaseVisualizer
from Bio import SeqIO




class HIDVisualizer(BaseVisualizer):
    """
    Creates a CSV summary from HID (ABIF-based) mixture files.
    Extracts an approximate per-peak table with columns:
    locus, allele, peak_height_rfu, sequence_length, sequence (optional)
    using ABIF traces (DATA9-12) and peak locations (PLOC2).
    Expects these ABIF fields to be present; if traces/positions are missing,
    only the CSV headers will be emitted.
    """

    async def visualize(
        self,
        filepath: str,
        output_path: Optional[str] = None,
        max_positions: int = 10000,
        include_sequence: bool = True,
        visualize: bool = True,
    ):
        self.log_prepare(f"Preparing HID CSV for {filepath}")
        if output_path and not output_path.lower().endswith(".csv"):
            output_path = f"{output_path}.csv"

        def _safe_value(val):
            if isinstance(val, bytes):
                try:
                    return val.decode(errors="ignore")
                except Exception:
                    return repr(val)
            if isinstance(val, (list, tuple, np.ndarray)):
                return ";".join(str(x) for x in val[:50]) + ("..." if len(val) > 50 else "")
            return val

        def _parse():
            record = SeqIO.read(filepath, "abi")
            raw = record.annotations.get("abif_raw", {})
            meta_rows = [["key", "value"]]
            trace_rows = [["key", "value"]]
            excluded_rows = [["key", "value"]]

            include_keys = {
                "PLOC1", "PLOC2",
                "DATA1", "DATA2", "DATA3", "DATA4", "DATA5", "DATA6", "DATA7", "DATA8",
                "DATA9", "DATA10", "DATA11", "DATA12",
                "PBAS1", "PBAS2",
                "Dye#1", "DyeN1", "DyeN2", "DyeN3", "DyeN4", "DyeW1", "DyeW2", "DyeW3", "DyeW4", "DySN1",
                "RunN1", "LANE1", "LIMS1", "SMPL1",
                "RUND1", "RUND2", "RUND3", "RUND4", "RUNT1", "RUNT2", "RUNT3", "RUNT4",
                "TUBE1", "APrN1", "APrV1", "APrX1",
                "SVER1", "SVER2", "SVER3",
                "GTyp1", "InVt1", "InSc1", "EPVt1", "LsrP1", "RGNm1", "Rate1", "SCAN1", "Scan1", "RevC1",
                "RMdN1", "RMdV1", "RMdX1", "RMXV1",
                "S/N%1",
            }
            include_prefixes = ("AUDT", "BufT")
            trace_keys = {
                "PLOC1", "PLOC2",
                "DATA1", "DATA2", "DATA3", "DATA4", "DATA5", "DATA6", "DATA7", "DATA8",
                "DATA9", "DATA10", "DATA11", "DATA12",
                "PBAS1", "PBAS2",
            }

            for k, v in raw.items():
                kv = [k, _safe_value(v)]
                if (k in include_keys) or any(k.startswith(pref) for pref in include_prefixes):
                    meta_rows.append(kv)
                else:
                    excluded_rows.append(kv)
                if k in trace_keys:
                    trace_rows.append(kv)
            # parse APrX1 XML if present
            aprx_rows = []
            aprx_xml = None
            if "APrX1" in raw:
                aprx_xml = _safe_value(raw.get("APrX1"))
                try:
                    import xml.etree.ElementTree as ET
                    root = ET.fromstring(aprx_xml)
                    for param in root.findall(".//parameter"):
                        name = param.attrib.get("name", "parameter")
                        value = (param.text or "").strip()
                        # explode semicolon-separated key/value tuples if present
                        if "key=" in value and "value=" in value and ";" in value:
                            parts = [p for p in value.split(";") if p]
                            for p in parts:
                                if p.startswith("key="):
                                    try:
                                        k_part, v_part = p.split(",value=")
                                        subkey = k_part.replace("key=", "")
                                        subval = v_part
                                    except ValueError:
                                        subkey, subval = "", p
                                    aprx_rows.append([name, subkey, subval])
                                else:
                                    aprx_rows.append([name, "", p])
                        else:
                            aprx_rows.append([name, "", value])
                    # size standard info
                    ss = root.find(".//sizeStandard")
                    if ss is not None:
                        ss_name = ss.attrib.get("name", "")
                        ss_version = ss.attrib.get("version", "")
                        aprx_rows.append(["size_standard_name", "", ss_name])
                        aprx_rows.append(["size_standard_version", "", ss_version])
                        sizes = ss.findtext("sizes", default="")
                        if sizes:
                            aprx_rows.append(["size_standard_sizes", "", sizes])
                except Exception:
                    aprx_rows = []
            return meta_rows, trace_rows, excluded_rows, aprx_rows, aprx_xml

        loop = asyncio.get_event_loop()
        meta_rows, trace_rows, excluded_rows, aprx_rows, aprx_xml = await loop.run_in_executor(None, _parse)

        csv_path = output_path
        if csv_path is None:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="") as tmp:
                csv_path = tmp.name
        with open(csv_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerows(meta_rows)
        self.log_write(f"Saved HID metadata CSV: {csv_path} (fields: {len(meta_rows)-1})")

        # write trace/basecall CSV
        trace_path = csv_path.replace(".csv", ".trace.csv")
        with open(trace_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerows(trace_rows)
        self.log_write(f"Saved HID trace/basecall CSV: {trace_path}")

        # write excluded params CSV
        excl_path = csv_path.replace(".csv", ".excluded.csv")
        with open(excl_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerows(excluded_rows)
        self.log_write(f"Saved HID excluded metadata CSV: {excl_path}")

        # write parsed APrX1 parameters if available
        if aprx_rows:
            aprx_path = csv_path.replace(".csv", ".aprx.csv")
            with open(aprx_path, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["parameter", "subkey", "value"])
                writer.writerows(aprx_rows)
            self.log_write(f"Saved HID APrX1 parameters CSV: {aprx_path} (rows: {len(aprx_rows)})")
        if aprx_xml:
            xml_path = csv_path.replace(".csv", ".aprx.xml")
            with open(xml_path, "w", encoding="utf-8") as fh:
                fh.write(aprx_xml if isinstance(aprx_xml, str) else str(aprx_xml))
            self.log_write(f"Saved HID APrX1 raw XML: {xml_path}")

        # render DATA9-12 bands if requested (base on trace CSV path)
        if visualize:
            DataBandVisualizer().render_from_trace_rows(trace_rows, trace_path.replace(".csv", ".webp"))



class ABIChromatogramVisualizer(BaseVisualizer):
    """
    Extracts ABI/AB1 chromatogram files into a simple key/value CSV of all
    ABIF parameters (dynamic; no hardcoded fields).
    """

    async def visualize(self, filepath: str, output_path: Optional[str] = None, visualize: bool = True):
        self.log_prepare(f"Extracting ABI/AB1 metadata: {filepath}")

        def _extract():
            record = SeqIO.read(filepath, "abi")
            raw = record.annotations.get("abif_raw", {})
            rows = [["key", "value"]]
            trace_rows = [["key", "value"]]
            excluded_rows = [["key", "value"]]

            include_keys = {
                "PLOC1", "PLOC2",
                "DATA1", "DATA2", "DATA3", "DATA4", "DATA5", "DATA6", "DATA7", "DATA8",
                "DATA9", "DATA10", "DATA11", "DATA12",
                "PBAS1", "PBAS2",
                "Dye#1", "DyeN1", "DyeN2", "DyeN3", "DyeN4", "DyeW1", "DyeW2", "DyeW3", "DyeW4", "DySN1",
                "RunN1", "LANE1", "LIMS1", "SMPL1",
                "RUND1", "RUND2", "RUND3", "RUND4", "RUNT1", "RUNT2", "RUNT3", "RUNT4",
                "TUBE1", "APrN1", "APrV1", "APrX1",
                "SVER1", "SVER2", "SVER3",
                "GTyp1", "InVt1", "InSc1", "EPVt1", "LsrP1", "RGNm1", "Rate1", "SCAN1", "Scan1", "RevC1",
                "RMdN1", "RMdV1", "RMdX1", "RMXV1",
                "S/N%1",
            }
            include_prefixes = ("AUDT", "BufT")
            trace_keys = {
                "PLOC1", "PLOC2",
                "DATA1", "DATA2", "DATA3", "DATA4", "DATA5", "DATA6", "DATA7", "DATA8",
                "DATA9", "DATA10", "DATA11", "DATA12",
                "PBAS1", "PBAS2",
            }
            def _safe_value(val):
                if isinstance(val, bytes):
                    try:
                        return val.decode(errors="ignore")
                    except Exception:
                        return repr(val)
                if isinstance(val, (list, tuple, np.ndarray)):
                    return ";".join(str(x) for x in val[:50]) + ("..." if len(val) > 50 else "")
                return val
            for k, v in raw.items():
                kv = [k, _safe_value(v)]
                if (k in include_keys) or any(k.startswith(pref) for pref in include_prefixes):
                    rows.append(kv)
                else:
                    excluded_rows.append(kv)
                if k in trace_keys:
                    trace_rows.append(kv)
            # parse APrX1 if present
            aprx_rows = []
            aprx_xml = None
            if "APrX1" in raw:
                aprx_xml = _safe_value(raw.get("APrX1"))
                try:
                    import xml.etree.ElementTree as ET
                    root = ET.fromstring(aprx_xml)
                    for param in root.findall(".//parameter"):
                        name = param.attrib.get("name", "parameter")
                        value = (param.text or "").strip()
                        aprx_rows.append([name, value])
                    ss = root.find(".//sizeStandard")
                    if ss is not None:
                        ss_name = ss.attrib.get("name", "")
                        ss_version = ss.attrib.get("version", "")
                        aprx_rows.append(["size_standard_name", ss_name])
                        aprx_rows.append(["size_standard_version", ss_version])
                        sizes = ss.findtext("sizes", default="")
                        if sizes:
                            aprx_rows.append(["size_standard_sizes", sizes])
                except Exception:
                    aprx_rows = []
            return rows, excluded_rows, trace_rows, aprx_rows, aprx_xml

        if output_path and not output_path.lower().endswith(".csv"):
            output_path = f"{output_path}.csv"
        if output_path is None:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="") as tmp:
                output_path = tmp.name
        rows, excluded_rows, trace_rows, aprx_rows, aprx_xml = await asyncio.get_event_loop().run_in_executor(None, _extract)
        with open(output_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerows(rows)
        self.log_write(f"Saved ABI metadata CSV: {output_path}")
        # trace/basecall parameters to sidecar
        trace_path = output_path.replace(".csv", ".trace.csv")
        with open(trace_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerows(trace_rows)
        self.log_write(f"Saved ABI trace/basecall CSV: {trace_path}")
        # render DATA9-12 bands if requested
        if visualize:
            DataBandVisualizer().render_from_trace_rows(trace_rows, trace_path.replace(".csv", ".webp"))
        # excluded parameters to sidecar
        excl_path = output_path.replace(".csv", ".excluded.csv")
        with open(excl_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerows(excluded_rows)
        self.log_write(f"Saved ABI excluded metadata CSV: {excl_path}")
        if aprx_rows:
            aprx_path = output_path.replace(".csv", ".aprx.csv")
            with open(aprx_path, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["key", "value"])
                writer.writerows(aprx_rows)
            self.log_write(f"Saved ABI APrX1 CSV: {aprx_path}")
        if aprx_xml:
            xml_path = output_path.replace(".csv", ".aprx.xml")
            with open(xml_path, "w", encoding="utf-8") as fh:
                fh.write(aprx_xml if isinstance(aprx_xml, str) else str(aprx_xml))
            self.log_write(f"Saved ABI APrX1 raw XML: {xml_path}")




class FSAElectropherogramVisualizer(BaseVisualizer):
    """
    Extracts FSA capillary electrophoresis outputs into a CSV (similar to HID/ABI).
    Columns: locus, allele, peak_location, peak_height_rfu, sequence_length, sequence (optional).
    """

    async def visualize(
        self,
        filepath: str,
        output_path: Optional[str] = None,
        max_positions: int = 10000,
        include_sequence: bool = True,
        visualize: bool = True,
    ):
        self.log_prepare(f"Extracting FSA metadata: {filepath}")

        def _extract():
            record = SeqIO.read(filepath, "abi")
            raw = record.annotations.get("abif_raw", {})
            # collect metadata partitions
            def _safe_value(val):
                if isinstance(val, bytes):
                    try:
                        return val.decode(errors="ignore")
                    except Exception:
                        return repr(val)
                if isinstance(val, (list, tuple, np.ndarray)):
                    return ";".join(str(x) for x in val[:50]) + ("..." if len(val) > 50 else "")
                return val

            meta_rows = [["key", "value"]]
            trace_rows = [["key", "value"]]
            excluded_rows = [["key", "value"]]

            include_keys = {
                "PLOC1", "PLOC2",
                "DATA1", "DATA2", "DATA3", "DATA4", "DATA5", "DATA6", "DATA7", "DATA8",
                "DATA9", "DATA10", "DATA11", "DATA12",
                "PBAS1", "PBAS2",
                "Dye#1", "DyeN1", "DyeN2", "DyeN3", "DyeN4", "DyeW1", "DyeW2", "DyeW3", "DyeW4", "DySN1",
                "RunN1", "LANE1", "LIMS1", "SMPL1",
                "RUND1", "RUND2", "RUND3", "RUND4", "RUNT1", "RUNT2", "RUNT3", "RUNT4",
                "TUBE1", "APrN1", "APrV1", "APrX1",
                "SVER1", "SVER2", "SVER3",
                "GTyp1", "InVt1", "InSc1", "EPVt1", "LsrP1", "RGNm1", "Rate1", "SCAN1", "Scan1", "RevC1",
                "RMdN1", "RMdV1", "RMdX1", "RMXV1",
                "S/N%1",
            }
            include_prefixes = ("AUDT", "BufT")
            trace_keys = {
                "PLOC1", "PLOC2",
                "DATA1", "DATA2", "DATA3", "DATA4", "DATA5", "DATA6", "DATA7", "DATA8",
                "DATA9", "DATA10", "DATA11", "DATA12",
                "PBAS1", "PBAS2",
            }
            for k, v in raw.items():
                kv = [k, _safe_value(v)]
                if (k in include_keys) or any(k.startswith(pref) for pref in include_prefixes):
                    meta_rows.append(kv)
                else:
                    excluded_rows.append(kv)
                if k in trace_keys:
                    trace_rows.append(kv)
            # parse APrX1 if present
            aprx_rows = []
            aprx_xml = None
            if "APrX1" in raw:
                aprx_xml = _safe_value(raw.get("APrX1"))
                try:
                    import xml.etree.ElementTree as ET
                    root = ET.fromstring(aprx_xml)
                    for param in root.findall(".//parameter"):
                        name = param.attrib.get("name", "parameter")
                        value = (param.text or "").strip()
                        aprx_rows.append([name, value])
                    ss = root.find(".//sizeStandard")
                    if ss is not None:
                        ss_name = ss.attrib.get("name", "")
                        ss_version = ss.attrib.get("version", "")
                        aprx_rows.append(["size_standard_name", ss_name])
                        aprx_rows.append(["size_standard_version", ss_version])
                        sizes = ss.findtext("sizes", default="")
                        if sizes:
                            aprx_rows.append(["size_standard_sizes", sizes])
                except Exception:
                    aprx_rows = []
            return meta_rows, trace_rows, excluded_rows, aprx_rows, aprx_xml

        loop = asyncio.get_event_loop()
        meta_rows, trace_rows, excluded_rows, aprx_rows, aprx_xml = await loop.run_in_executor(None, _extract)

        if output_path and not output_path.lower().endswith(".csv"):
            output_path = f"{output_path}.csv"
        if output_path is None:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w", newline="") as tmp:
                output_path = tmp.name
        with open(output_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerows(meta_rows)
        self.log_write(f"Saved FSA metadata CSV: {output_path} (fields: {len(meta_rows)-1})")

        # write kept metadata key-value CSV alongside
        meta_path = output_path.replace(".csv", ".meta.csv")
        with open(meta_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerows(meta_rows)
        self.log_write(f"Saved FSA metadata CSV: {meta_path} (fields: {len(meta_rows)-1})")

        # write trace/basecall CSV
        trace_path = output_path.replace(".csv", ".trace.csv")
        with open(trace_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerows(trace_rows)
        self.log_write(f"Saved FSA trace/basecall CSV: {trace_path}")

        # write excluded params CSV
        excl_path = output_path.replace(".csv", ".excluded.csv")
        with open(excl_path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerows(excluded_rows)
        self.log_write(f"Saved FSA excluded metadata CSV: {excl_path}")

        # APrX sidecars
        if aprx_rows:
            aprx_path = output_path.replace(".csv", ".aprx.csv")
            with open(aprx_path, "w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["key", "value"])
                writer.writerows(aprx_rows)
            self.log_write(f"Saved FSA APrX1 CSV: {aprx_path}")
        if aprx_xml:
            xml_path = output_path.replace(".csv", ".aprx.xml")
            with open(xml_path, "w", encoding="utf-8") as fh:
                fh.write(aprx_xml if isinstance(aprx_xml, str) else str(aprx_xml))
            self.log_write(f"Saved FSA APrX1 raw XML: {xml_path}")

        # render DATA9-12 bands if requested
        if visualize:
            DataBandVisualizer().render_from_trace_rows(trace_rows, trace_path.replace(".csv", ".webp"))


class DataBandVisualizer(BaseVisualizer):
    """
    Renders DATA9-12 traces (if present) into a WEBP heatband image from trace_rows.
    Expects trace_rows like [["key","value"], ...] where value is comma/semicolon-separated series.
    """

    async def visualize(self, *args, **kwargs):
        """Not used directly; call render_from_trace_rows instead."""
        raise NotImplementedError("Use render_from_trace_rows for DataBandVisualizer")

    def _parse_series(self, val: str) -> np.ndarray:
        if not isinstance(val, str):
            return np.array([], dtype=float)
        parts = re.split(r"[;,]", val)
        arr = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            try:
                arr.append(float(p))
            except ValueError:
                continue
        return np.array(arr, dtype=float)

    def render_from_trace_rows(self, trace_rows, output_path: str, height: int = 400, max_width: int | None = None):
        if not trace_rows or len(trace_rows) <= 1:
            self.log_encode("No trace rows to visualize (DATA9-12 missing)")
            return
        # Collect possibly multiple rows per DATA key (some files repeat keys or split values across rows).
        data_map = {"DATA9": [], "DATA10": [], "DATA11": [], "DATA12": []}
        for k, v in trace_rows[1:]:
            if k in data_map:
                data_map[k].append(v)
        if not data_map:
            self.log_encode("No DATA9-12 entries found; skipping trace visualization")
            return

        series = []
        for key in ("DATA9", "DATA10", "DATA11", "DATA12"):
            chunks = data_map.get(key, [])
            if not chunks:
                continue
            combined = ";".join(str(c) for c in chunks)
            arr = self._parse_series(combined)
            if arr.size:
                series.append((key, arr))

        if not series:
            self.log_encode("Trace series empty after parsing; skipping visualization")
            return

        max_len = max((len(arr) for _, arr in series))
        if max_width:
            max_len = min(max_len, max_width)
        global_max = max((float(np.max(arr)) if arr.size else 0.0 for _, arr in series), default=0.0)
        width = max_len
        top = 50
        left = 60
        bottom = 40
        right = 20
        canvas = np.zeros((height + top + bottom, width + left + right, 3), dtype=np.uint8)

        legend = [
            ("A (DATA - 9)", (0, 180, 255)),
            ("C (DATA - 10)", (0, 200, 0)),
            ("G (DATA - 11)", (255, 200, 0)),
            ("T (DATA - 12)", (220, 0, 200)),
        ]
        colors = [np.array(c, dtype=np.uint8) for _, c in legend]

        for idx, (name, arr) in enumerate(series):
            if arr.size == 0:
                continue
            maxv = np.max(arr) if arr.size else 1.0
            norm = arr / maxv if maxv != 0 else arr
            color = colors[idx % len(colors)]
            for x in range(min(len(norm) - 1, width - 1)):
                y1 = top + int((1.0 - np.clip(norm[x], 0.0, 1.0)) * (height - 1))
                y2 = top + int((1.0 - np.clip(norm[x + 1], 0.0, 1.0)) * (height - 1))
                y_min, y_max = sorted((y1, y2))
                canvas[y_min:y_max + 1, left + x, :] = color

        # axes and labels
        canvas[top:, left, :] = 255  # y-axis
        canvas[top + height - 1, left:, :] = 255  # x-axis
        image = Image.fromarray(canvas, mode="RGB")
        draw = ImageDraw.Draw(image)
        title = "Traces (DATA9:A, DATA10:C, DATA11:G, DATA12:T)"
        tb = draw.textbbox((0, 0), title)
        title_w, title_h = tb[2] - tb[0], tb[3] - tb[1]
        draw.text(((left + width + right - title_w) // 2, 5), title, fill=(255, 255, 255))
        draw.text((left + 5, top + height + 5), "x: position (index)", fill=(255, 255, 255))
        y_label = f"y: RFU  -  (max  ≈  {int(global_max)})"
        draw.text((35, top - title_h - 30), y_label, fill=(255, 255, 255))

        # x-axis ticks (5 segments)
        if width <= 60:
            tick_positions = list(range(width))
        else:
            max_ticks = 20
            step = max(1, width // max_ticks)
            tick_positions = list(range(0, width, step))
            if tick_positions[-1] != width - 1:
                tick_positions.append(width - 1)
        for x_idx in tick_positions:
            x_pos = left + x_idx
            canvas[top + height - 5:top + height, x_pos, :] = 255
            draw.text((x_pos - 10, top + height + 2), str(x_idx), fill=(200, 200, 200))

        # y-axis ticks (0..global_max)
        y_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
        for frac in y_ticks:
            y_pos = top + int((1.0 - frac) * (height - 1))
            canvas[y_pos, left:left + 5, :] = 255
            draw.text((2, y_pos - 7), str(int(global_max * frac)), fill=(200, 200, 200))

        # legend at bottom with gap
        legend_gap_y = 25
        legend_y = top + height + legend_gap_y
        legend_x = left + 10
        gap_x = 140
        for idx, (label, color) in enumerate(legend):
            lx = legend_x + idx * gap_x
            ly = legend_y
            draw.rectangle([lx, ly, lx + 14, ly + 14], fill=color)
            draw.text((lx + 20, ly - 2), label, fill=(255, 255, 255))

        stem, ext = os.path.splitext(output_path)
        if not ext:
            ext = ".webp"
        out_path = f"{stem}{ext}"
        image.save(out_path, format="WEBP", quality=90)
        self.log_write(f"Saved DATA9-12 trace visualization: {out_path}")
