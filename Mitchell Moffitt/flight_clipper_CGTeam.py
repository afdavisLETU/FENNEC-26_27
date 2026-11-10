"""
FENNEC Flight Clipper
=====================
Turns an ArduPilot DataFlash .bin log into an .xlsx workbook with one sheet
per sensor / message type (IMU, ATT, RATE, GPS, BAT, ...).

Features
  * Fast built-in .bin parser (no pymavlink needed)
  * 3D flight-path simulation with play / pause / speed and attitude readout
  * Crop the flight to a time range (type it, drag on the timeline, use the
    playback position, or auto-detect arm -> disarm)
  * File-name builder for the  <Flight#><Heli>_<CG mm><A/F/M>_<Wind>.xlsx
    convention, e.g. 300R_02.2A_M.xlsx  (CG -18.7 .. +18.7 mm, datum 00.0M)
  * Choose the output folder; every sheet from the .bin is saved

Run:  python flight_clipper.py
Needs: numpy, pandas, matplotlib, xlsxwriter (or openpyxl as a slower fallback)
"""

import json
import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# =============================================================================
# 1. DataFlash .bin parser
# =============================================================================
HEAD1, HEAD2, FMT_ID, FMT_LEN = 0xA3, 0x95, 128, 89

# format char -> (numpy dtype, scale applied after reading)
FMT_CHARS = {
    "b": ("i1", None), "B": ("u1", None), "h": ("<i2", None), "H": ("<u2", None),
    "i": ("<i4", None), "I": ("<u4", None), "f": ("<f4", None), "d": ("<f8", None),
    "n": ("S4", None), "N": ("S16", None), "Z": ("S64", None),
    "c": ("<i2", 0.01), "C": ("<u2", 0.01), "e": ("<i4", 0.01), "E": ("<u4", 0.01),
    "L": ("<i4", 1e-7), "M": ("u1", None), "q": ("<i8", None), "Q": ("<u8", None),
    "a": (("<i2", (32,)), None),
}

_ILLEGAL_XML = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


class BinLog:
    """Parsed DataFlash log.  `messages` maps name -> DataFrame."""

    def __init__(self, path, progress=None):
        self.path = str(path)
        self.messages = {}
        self.formats = {}
        self.skipped = []
        self._parse(progress)

    # ---------------------------------------------------------------- parse
    def _parse(self, progress):
        buf = np.fromfile(self.path, dtype=np.uint8)
        data = buf.tobytes()
        n = len(data)
        lengths = {FMT_ID: FMT_LEN}
        fmts = {}
        offsets = {}
        i = 0
        last_report = 0
        while i + 3 <= n:
            if data[i] != HEAD1 or data[i + 1] != HEAD2:
                i += 1
                continue
            mid = data[i + 2]
            ln = lengths.get(mid)
            if ln is None or i + ln > n:
                i += 1
                continue
            if mid == FMT_ID:
                typ, tlen = data[i + 3], data[i + 4]
                name = data[i + 5:i + 9].rstrip(b"\0").decode("ascii", "replace")
                fmt = data[i + 9:i + 25].rstrip(b"\0").decode("ascii", "replace")
                cols = data[i + 25:i + 89].rstrip(b"\0").decode("ascii", "replace")
                lengths[typ] = tlen
                fmts[typ] = (name, fmt, cols.split(","), tlen)
            offsets.setdefault(mid, []).append(i)
            i += ln
            if progress and i - last_report > 2_000_000:
                last_report = i
                progress(0.7 * i / n, "Scanning log...")

        fmts[FMT_ID] = ("FMT", "BBnNZ", ["Type", "Length", "Name", "Format", "Columns"], FMT_LEN)
        total = len(offsets)
        for k, (mid, offs) in enumerate(offsets.items()):
            if mid not in fmts:
                continue
            name, fmt, cols, ln = fmts[mid]
            if progress:
                progress(0.7 + 0.3 * k / total, f"Decoding {name}...")
            try:
                df = self._decode(buf, np.asarray(offs, dtype=np.int64), fmt, cols, ln)
            except Exception as exc:  # malformed format: keep going
                self.skipped.append(f"{name}: {exc}")
                continue
            self.formats[name] = (fmt, cols)
            if name in self.messages:  # same name re-defined: append
                df = pd.concat([self.messages[name], df], ignore_index=True)
            self.messages[name] = df
        if progress:
            progress(1.0, "Done")

    @staticmethod
    def _decode(buf, offs, fmt, cols, ln):
        if len(fmt) != len(cols):
            raise ValueError("format/column count mismatch")
        fields, scales = [], {}
        seen = {}
        for ch, col in zip(fmt, cols):
            if ch not in FMT_CHARS:
                raise ValueError(f"unknown format char '{ch}'")
            dt, sc = FMT_CHARS[ch]
            if col in seen:  # duplicate column names -> make unique
                seen[col] += 1
                col = f"{col}_{seen[col]}"
            else:
                seen[col] = 0
            fields.append((col, dt))
            if sc:
                scales[col] = sc
        dtype = np.dtype(fields)
        if dtype.itemsize != ln - 3:
            raise ValueError(f"size {dtype.itemsize} != {ln - 3}")
        raw = buf[offs[:, None] + np.arange(3, ln)]
        rec = np.ascontiguousarray(raw).view(dtype).ravel()

        out = {}
        for col, _ in fields:
            v = rec[col]
            if v.dtype.kind == "S":
                s = pd.Series(v).str.decode("latin-1").str.rstrip("\x00")
                out[col] = s.str.replace(_ILLEGAL_XML, "", regex=True)
            elif v.ndim == 2:  # 'a' arrays -> one column per element
                for j in range(v.shape[1]):
                    out[f"{col}_{j}"] = v[:, j]
            elif col in scales:
                out[col] = v.astype(np.float64) * scales[col]
            else:
                out[col] = v
        return pd.DataFrame(out)

    # -------------------------------------------------------------- helpers
    def time_range(self):
        lo, hi = np.inf, -np.inf
        for df in self.messages.values():
            if "TimeUS" in df and len(df):
                t = df["TimeUS"].to_numpy()
                t = t[t > 0]
                if len(t):
                    lo, hi = min(lo, t.min()), max(hi, t.max())
        if not np.isfinite(lo):
            return 0.0, 0.0
        return lo / 1e6, hi / 1e6

    def flight_path(self):
        """Return (t_s, east_m, north_m, up_m, source) for the 3D plot."""
        def local(lat, lng):
            lat0, lng0 = lat[0], lng[0]
            north = np.radians(lat - lat0) * 6378137.0
            east = np.radians(lng - lng0) * 6378137.0 * np.cos(np.radians(lat0))
            return east, north

        pos = self.messages.get("POS")
        if pos is not None and len(pos) > 1:
            p = pos[(pos["Lat"] != 0) | (pos["Lng"] != 0)]
            if len(p) > 1:
                e, nn = local(p["Lat"].to_numpy(), p["Lng"].to_numpy())
                up = p["RelHomeAlt"].to_numpy() if "RelHomeAlt" in p else p["Alt"].to_numpy() - p["Alt"].iloc[0]
                return p["TimeUS"].to_numpy() / 1e6, e, nn, up, "POS (GPS / EKF position)"
        xkf = self.messages.get("XKF1")
        if xkf is not None and len(xkf) > 1:
            x = xkf[xkf["C"] == xkf["C"].min()] if "C" in xkf else xkf
            return (x["TimeUS"].to_numpy() / 1e6, x["PE"].to_numpy(), x["PN"].to_numpy(),
                    -x["PD"].to_numpy(), "XKF1 (EKF local position)")
        gps = self.messages.get("GPS")
        if gps is not None and len(gps) > 1:
            g = gps[(gps["Lat"] != 0)]
            if len(g) > 1:
                e, nn = local(g["Lat"].to_numpy(), g["Lng"].to_numpy())
                return (g["TimeUS"].to_numpy() / 1e6, e, nn,
                        g["Alt"].to_numpy() - g["Alt"].iloc[0], "GPS")
        return None

    def arm_events(self):
        """List of (time_s, armed_bool) from ARM messages (or EV ids 10/11)."""
        ev = []
        arm = self.messages.get("ARM")
        if arm is not None and len(arm):
            for t, s in zip(arm["TimeUS"], arm["ArmState"]):
                ev.append((t / 1e6, bool(s)))
        elif "EV" in self.messages:
            e = self.messages["EV"]
            for t, i in zip(e["TimeUS"], e["Id"]):
                if i in (10, 11):
                    ev.append((t / 1e6, i == 10))
        return sorted(ev)

    def detect_flight(self):
        """Guess the flight window: first arm -> following disarm."""
        lo, hi = self.time_range()
        ev = self.arm_events()
        start = next((t for t, a in ev if a), None)
        if start is None:
            return lo, hi
        end = next((t for t, a in ev if not a and t > start), hi)
        return start, end


# =============================================================================
# 2. File-name convention   <Flight#><Heli>_<NN.N><A/F/M>_<L/M/H>.xlsx
# =============================================================================
CG_LIMIT = 18.7
NAME_RE = re.compile(r"^(\d+)([A-Za-z])_(\d{1,2}\.\d{1,2})([AFM])_([LMH])(?:\.xlsx)?$", re.I)


def build_filename(flight, heli, cg_mm, direction, wind):
    """build_filename(300, 'R', 2.2, 'A', 'M') -> '300R_02.2A_M.xlsx'"""
    flight = int(flight)
    heli = str(heli).strip().upper()
    direction = str(direction).strip().upper()
    wind = str(wind).strip().upper()
    cg = round(abs(float(cg_mm)), 1)
    if flight < 0:
        raise ValueError("Flight number must be a positive whole number")
    if len(heli) != 1 or not heli.isalpha():
        raise ValueError("Helicopter type must be a single letter (e.g. R or B)")
    if direction not in ("A", "F", "M"):
        raise ValueError("Direction must be A (aft), F (forward) or M (middle)")
    if cg > CG_LIMIT:
        raise ValueError(f"CG must be within {CG_LIMIT} mm of the datum")
    if direction == "M" and cg != 0:
        raise ValueError("Middle (M) is the datum, so the CG must be 00.0")
    if direction != "M" and cg == 0:
        raise ValueError("A CG of 00.0 is the datum: use M (middle)")
    if wind not in ("L", "M", "H"):
        raise ValueError("Wind level must be L, M or H")
    return f"{flight}{heli}_{cg:04.1f}{direction}_{wind}.xlsx"


def parse_filename(name):
    """'300R_02.2A_M.xlsx' -> (300, 'R', 2.2, 'A', 'M').  Raises ValueError."""
    m = NAME_RE.match(os.path.basename(name.strip()))
    if not m:
        raise ValueError("Expected a name like 300R_02.2A_M.xlsx")
    flight, heli, cg, d, wind = m.groups()
    build_filename(flight, heli, cg, d, wind)  # validates ranges
    return int(flight), heli.upper(), float(cg), d.upper(), wind.upper()


def signed_cg(cg_mm, direction):
    """Forward = +, Aft = -, Middle = 0 (same convention as the preprocessing code)."""
    return {"F": 1, "A": -1, "M": 0}[direction] * round(abs(float(cg_mm)), 1)


# =============================================================================
# 3. Excel export
# =============================================================================
EXCEL_MAX_ROWS = 1_048_575            # data rows per sheet (1 row for header)
UNCROPPED_SHEETS = {"FMT", "FMTU", "UNIT", "MULT", "PARM", "VER", "FILE"}


def export_xlsx(log, out_path, t_start=None, t_end=None, sheets=None,
                keep_meta_uncropped=True, info=None, progress=None, cancel=None):
    """Write the selected message types of `log` to `out_path`.

    t_start / t_end are in seconds since boot (TimeUS / 1e6).  Messages with no
    TimeUS (e.g. FMT) are always written whole.  Returns {sheet: rows}.
    """
    names = sorted(sheets if sheets is not None else log.messages.keys())
    frames = {}
    for name in names:
        df = log.messages[name]
        if "TimeUS" in df and t_start is not None and t_end is not None \
                and not (keep_meta_uncropped and name in UNCROPPED_SHEETS):
            t = df["TimeUS"].to_numpy() / 1e6
            df = df[(t >= t_start) & (t <= t_end)]
        if "TimeUS" in df:
            df = df.copy()
            df.insert(1, "Time_s", df["TimeUS"].to_numpy() / 1e6)
        frames[name] = df.reset_index(drop=True)

    total_cells = sum(max(df.size, 1) for df in frames.values()) or 1
    done = 0
    written = {}

    try:
        import xlsxwriter  # noqa: F401
        engine = "xlsxwriter"
    except ImportError:
        engine = "openpyxl"

    tmp_path = str(out_path) + ".partial"
    if engine == "xlsxwriter":
        import xlsxwriter
        wb = xlsxwriter.Workbook(tmp_path, {"constant_memory": True,
                                            "nan_inf_to_errors": True,
                                            "strings_to_numbers": False,
                                            "strings_to_formulas": False,
                                            "strings_to_urls": False})
        bold = wb.add_format({"bold": True, "bg_color": "#DDE7F3", "border": 1})
    else:
        from openpyxl import Workbook
        wb = Workbook(write_only=True)

    def add_sheet(title, df):
        nonlocal done
        cols = list(df.columns)
        values = [df[c].to_numpy() for c in cols]
        values = [v.astype(object) if v.dtype.kind in "SUO" else v for v in values]
        nrows = len(df)
        if engine == "xlsxwriter":
            ws = wb.add_worksheet(title)
            ws.freeze_panes(1, 0)
            ws.write_row(0, 0, cols, bold)
            for j, c in enumerate(cols):
                ws.set_column(j, j, max(9, min(22, len(str(c)) + 3)))
        else:
            ws = wb.create_sheet(title)
            ws.append(cols)
        chunk = 20000
        for r0 in range(0, nrows, chunk):
            if cancel and cancel.is_set():
                raise InterruptedError("Export cancelled")
            block = list(zip(*(v[r0:r0 + chunk].tolist() for v in values)))
            if engine == "xlsxwriter":
                for k, row in enumerate(block):
                    ws.write_row(r0 + k + 1, 0, row)
            else:
                for row in block:
                    ws.append([None if isinstance(x, float) and not np.isfinite(x) else x
                               for x in row])
            done += len(block) * len(cols)
            if progress:
                progress(done / total_cells, f"Writing {title}  ({min(r0 + chunk, nrows):,}/{nrows:,} rows)")

    try:
        for name, df in frames.items():
            if len(df) <= EXCEL_MAX_ROWS:
                add_sheet(name[:31], df)
                written[name[:31]] = len(df)
            else:  # too big for one sheet -> NAME, NAME_2, ...
                for p, r0 in enumerate(range(0, len(df), EXCEL_MAX_ROWS)):
                    title = name if p == 0 else f"{name}_{p + 1}"
                    part = df.iloc[r0:r0 + EXCEL_MAX_ROWS]
                    add_sheet(title[:31], part)
                    written[title[:31]] = len(part)
            done += 1

        # Info sheet goes LAST so pd.read_excel(file) still opens the first sensor sheet.
        rows = [("Field", "Value"),
                ("Source .bin", log.path),
                ("Exported", time.strftime("%Y-%m-%d %H:%M:%S")),
                ("Crop start (s)", t_start if t_start is not None else "full log"),
                ("Crop end (s)", t_end if t_end is not None else "full log"),
                ("Crop duration (s)", (t_end - t_start) if t_start is not None else "")]
        rows += list((info or {}).items())
        rows += [("Sheet: " + k, f"{v:,} rows") for k, v in written.items()]
        if engine == "xlsxwriter":
            ws = wb.add_worksheet("Info")
            ws.set_column(0, 0, 22)
            ws.set_column(1, 1, 60)
            for r, (k, v) in enumerate(rows):
                ws.write(r, 0, k, bold)
                ws.write(r, 1, v)
            wb.close()
        else:
            ws = wb.create_sheet("Info")
            for k, v in rows:
                ws.append([k, v])
            wb.save(tmp_path)
        os.replace(tmp_path, out_path)
    except BaseException:
        try:
            if engine == "xlsxwriter":
                wb.close()
        except Exception:
            pass
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
    if progress:
        progress(1.0, "Saved")
    return written


# =============================================================================
# 4. GUI
# =============================================================================
CONFIG_PATH = Path.home() / ".fennec_flight_clipper.json"


def load_config():
    try:
        return json.loads(CONFIG_PATH.read_text())
    except Exception:
        return {}


def save_config(cfg):
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    except Exception:
        pass


def run_gui():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.widgets import SpanSelector

    cfg = load_config()

    # ---- dark palette ---------------------------------------------------
    BG, PANEL, FIELD, BORDER = "#16181d", "#1f2229", "#2a2e37", "#3a3f4b"
    FG, MUTED, ACCENT = "#e6e8eb", "#9aa3b2", "#4f9dff"
    OK_FG, ERR_FG = "#7ee787", "#ff7b72"
    matplotlib.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
        "axes.edgecolor": BORDER, "axes.labelcolor": FG, "text.color": FG,
        "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": BORDER,
        "legend.facecolor": PANEL, "legend.edgecolor": BORDER, "legend.labelcolor": FG,
    })

    class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title("FENNEC Flight Clipper  -  .bin  ->  .xlsx")
            self.geometry("1400x880")
            self.minsize(1100, 700)
            self.log = None
            self.path = None
            self.att = None
            self.playing = False
            self.cancel_evt = threading.Event()
            self._syncing = False

            self._apply_dark_theme()

            pw = ttk.PanedWindow(self, orient="horizontal")
            pw.pack(fill="both", expand=True)
            left = self._scrollable(pw)
            right = ttk.Frame(pw)
            pw.add(self._left_outer, weight=0)
            pw.add(right, weight=1)
            self._build_controls(left)
            self._build_plots(right)
            self._set_enabled(False)
            self.protocol("WM_DELETE_WINDOW", self._on_close)

        # ------------------------------------------------------ dark theme
        def _apply_dark_theme(self):
            self.configure(bg=BG)
            st = ttk.Style(self)
            st.theme_use("clam")
            font = ("Segoe UI", 9)
            st.configure(".", background=BG, foreground=FG, fieldbackground=FIELD, bordercolor=BORDER,
                         lightcolor=BORDER, darkcolor=BORDER, troughcolor=FIELD, focuscolor=ACCENT,
                         selectbackground=ACCENT, selectforeground="#ffffff", insertcolor=FG,
                         arrowcolor=FG, font=font)
            st.configure("TFrame", background=BG)
            st.configure("TLabel", background=BG, foreground=FG)
            st.configure("TLabelframe", background=BG, bordercolor=BORDER)
            st.configure("Head.TLabelframe", background=BG, bordercolor=BORDER)
            st.configure("Head.TLabelframe.Label", background=BG, foreground=ACCENT,
                         font=("Segoe UI", 10, "bold"))
            st.configure("TButton", background=FIELD, foreground=FG, padding=4, borderwidth=1)
            st.map("TButton", background=[("disabled", PANEL), ("pressed", BORDER), ("active", "#353a45")],
                   foreground=[("disabled", "#5c6370")])
            st.configure("Big.TButton", font=("Segoe UI", 11, "bold"), padding=6,
                         background=ACCENT, foreground="#ffffff")
            st.map("Big.TButton", background=[("disabled", PANEL), ("pressed", "#2f6fd1"),
                                              ("active", "#6aaeff")])
            st.configure("Play.TButton", font=("Segoe UI", 11, "bold"), padding=(10, 4),
                         background="#238636", foreground="#ffffff")
            st.map("Play.TButton", background=[("disabled", PANEL), ("active", "#2ea043")])
            for w in ("TCheckbutton", "TRadiobutton"):
                st.configure(w, background=BG, foreground=FG, indicatorbackground=FIELD,
                             indicatorforeground=FG)
                st.map(w, background=[("active", BG)], indicatorbackground=[("selected", ACCENT)])
            for w in ("TEntry", "TSpinbox", "TCombobox"):
                st.configure(w, fieldbackground=FIELD, foreground=FG, background=FIELD,
                             bordercolor=BORDER, arrowcolor=FG)
                st.map(w, fieldbackground=[("readonly", FIELD), ("disabled", PANEL)],
                       foreground=[("disabled", MUTED)])
            st.configure("Horizontal.TProgressbar", background=ACCENT, troughcolor=FIELD)
            st.configure("Vertical.TScrollbar", background=FIELD, troughcolor=BG, arrowcolor=MUTED)
            st.configure("Horizontal.TScale", background=ACCENT, troughcolor=FIELD)
            st.configure("TPanedwindow", background=BORDER)
            self.option_add("*TCombobox*Listbox.background", FIELD)
            self.option_add("*TCombobox*Listbox.foreground", FG)
            self.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
            # dark Windows title bar
            if sys.platform == "win32":
                try:
                    import ctypes
                    self.update_idletasks()
                    hwnd = ctypes.windll.user32.GetParent(self.winfo_id())
                    val = ctypes.c_int(1)
                    for attr in (20, 19):
                        if ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(val), 4) == 0:
                            break
                except Exception:
                    pass

        # ------------------------------------------------------ layout bits
        def _scrollable(self, parent):
            outer = ttk.Frame(parent, width=430)
            self._left_outer = outer
            canvas = tk.Canvas(outer, width=430, highlightthickness=0, bg=BG)
            sb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
            inner = ttk.Frame(canvas, padding=8)
            inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
            canvas.create_window((0, 0), window=inner, anchor="nw")
            canvas.configure(yscrollcommand=sb.set)
            canvas.pack(side="left", fill="both", expand=True)
            sb.pack(side="right", fill="y")
            canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-e.delta // 120, "units")
                            if str(e.widget).startswith(str(outer)) else None)
            return inner

        def _section(self, parent, title):
            f = ttk.LabelFrame(parent, text=title, padding=8, style="Head.TLabelframe")
            f.pack(fill="x", pady=(0, 8))
            return f

        def _build_controls(self, p):
            # ---- 1. bin file
            s = self._section(p, "1.  Flight log (.bin)")
            self.bin_var = tk.StringVar(value=cfg.get("last_bin", ""))
            ttk.Entry(s, textvariable=self.bin_var, width=46).grid(row=0, column=0, columnspan=2, sticky="we")
            ttk.Button(s, text="Browse...", command=self.browse_bin).grid(row=1, column=0, sticky="we", pady=4)
            ttk.Button(s, text="Load", command=self.load_bin).grid(row=1, column=1, sticky="we", pady=4, padx=(4, 0))
            self.summary = ttk.Label(s, text="No log loaded.", foreground=MUTED, wraplength=390, justify="left")
            self.summary.grid(row=2, column=0, columnspan=2, sticky="w")
            s.columnconfigure(0, weight=1)
            s.columnconfigure(1, weight=1)

            # ---- 2. crop
            s = self._section(p, "2.  Crop flight (seconds since boot)")
            self.t0_var, self.t1_var = tk.DoubleVar(), tk.DoubleVar()
            ttk.Label(s, text="Start").grid(row=0, column=0, sticky="w")
            self.t0_spin = ttk.Spinbox(s, textvariable=self.t0_var, width=11, increment=0.5,
                                       from_=0, to=1e7, command=self.crop_changed)
            self.t0_spin.grid(row=0, column=1, sticky="w")
            ttk.Label(s, text="End").grid(row=0, column=2, sticky="w", padx=(10, 0))
            self.t1_spin = ttk.Spinbox(s, textvariable=self.t1_var, width=11, increment=0.5,
                                       from_=0, to=1e7, command=self.crop_changed)
            self.t1_spin.grid(row=0, column=3, sticky="w")
            for w in (self.t0_spin, self.t1_spin):
                w.bind("<Return>", lambda e: self.crop_changed())
                w.bind("<FocusOut>", lambda e: self.crop_changed())
            self.dur_lbl = ttk.Label(s, text="Duration: -")
            self.dur_lbl.grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 4))
            b = ttk.Frame(s)
            b.grid(row=2, column=0, columnspan=4, sticky="we")
            self.crop_btns = [
                ttk.Button(b, text="Full log", command=self.crop_full),
                ttk.Button(b, text="Auto (arm -> disarm)", command=self.crop_auto),
                ttk.Button(b, text="Start = playhead", command=lambda: self.crop_from_playhead(0)),
                ttk.Button(b, text="End = playhead", command=lambda: self.crop_from_playhead(1)),
            ]
            for k, w in enumerate(self.crop_btns):
                w.grid(row=k // 2, column=k % 2, sticky="we", padx=2, pady=2)
            b.columnconfigure(0, weight=1)
            b.columnconfigure(1, weight=1)
            ttk.Label(s, text="Tip: drag across the timeline plot to select a range.",
                      foreground=MUTED).grid(row=3, column=0, columnspan=4, sticky="w")
            self.keep_meta = tk.BooleanVar(value=True)
            ttk.Checkbutton(s, text="Keep PARM / FMT / VER / ... sheets uncropped",
                            variable=self.keep_meta).grid(row=4, column=0, columnspan=4, sticky="w")

            # ---- 3. file name
            s = self._section(p, "3.  File name   (e.g. 300R_02.2A_M.xlsx)")
            self.flight_var = tk.StringVar(value=str(cfg.get("flight", 300)))
            self.heli_var = tk.StringVar(value=cfg.get("heli", "R"))
            self.cg_var = tk.StringVar(value="00.0")
            self.dir_var = tk.StringVar(value="M")
            self.wind_var = tk.StringVar(value=cfg.get("wind", "M"))
            self.name_var = tk.StringVar()

            ttk.Label(s, text="Flight #").grid(row=0, column=0, sticky="w")
            ttk.Spinbox(s, textvariable=self.flight_var, from_=0, to=99999, width=8,
                        command=self.fields_changed).grid(row=0, column=1, sticky="w")
            ttk.Label(s, text="Heli type").grid(row=0, column=2, sticky="w", padx=(10, 0))
            ttk.Combobox(s, textvariable=self.heli_var, values=["R", "B"], width=4).grid(row=0, column=3, sticky="w")

            ttk.Label(s, text="CG (mm)").grid(row=1, column=0, sticky="w", pady=(6, 0))
            self.cg_spin = ttk.Spinbox(s, textvariable=self.cg_var, from_=0.0, to=CG_LIMIT,
                                       increment=0.1, format="%04.1f", width=8, command=self.fields_changed)
            self.cg_spin.grid(row=1, column=1, sticky="w", pady=(6, 0))
            df = ttk.Frame(s)
            df.grid(row=1, column=2, columnspan=2, sticky="w", padx=(10, 0), pady=(6, 0))
            for val, txt in (("A", "Aft"), ("M", "Middle"), ("F", "Fwd")):
                ttk.Radiobutton(df, text=f"{txt} ({val})", value=val, variable=self.dir_var,
                                command=self.dir_changed).pack(side="left")

            ttk.Label(s, text="Wind").grid(row=2, column=0, sticky="w", pady=(6, 0))
            wf = ttk.Frame(s)
            wf.grid(row=2, column=1, columnspan=3, sticky="w", pady=(6, 0))
            for val, txt in (("L", "Low"), ("M", "Medium"), ("H", "High")):
                ttk.Radiobutton(wf, text=f"{txt} ({val})", value=val, variable=self.wind_var,
                                command=self.fields_changed).pack(side="left")

            ttk.Label(s, text="File name").grid(row=3, column=0, sticky="w", pady=(8, 0))
            ne = ttk.Entry(s, textvariable=self.name_var, width=28, font=("Consolas", 11))
            ne.grid(row=3, column=1, columnspan=3, sticky="we", pady=(8, 0))
            ne.bind("<Return>", lambda e: self.name_typed())
            ne.bind("<FocusOut>", lambda e: self.name_typed())
            self.name_msg = ttk.Label(s, text="", wraplength=390, justify="left")
            self.name_msg.grid(row=4, column=0, columnspan=4, sticky="w")
            ttk.Label(s, text=f"CG range: {CG_LIMIT} mm Aft ... 00.0 (datum) ... {CG_LIMIT} mm Fwd.  "
                              "You can also type a full name and press Enter.",
                      foreground=MUTED, wraplength=390, justify="left").grid(row=5, column=0, columnspan=4, sticky="w")
            for v in (self.flight_var, self.heli_var, self.cg_var):
                v.trace_add("write", lambda *a: self.fields_changed())

            # ---- 4. sheets
            s = self._section(p, "4.  Sheets to save (one per sensor)")
            lf = ttk.Frame(s)
            lf.pack(fill="x")
            self.sheet_list = tk.Listbox(lf, selectmode="multiple", height=8, exportselection=False, bg=FIELD, fg=FG, selectbackground=ACCENT, selectforeground="#ffffff", highlightthickness=1, highlightbackground=BORDER, highlightcolor=ACCENT, relief="flat", borderwidth=0,
                                         font=("Consolas", 9))
            sb = ttk.Scrollbar(lf, orient="vertical", command=self.sheet_list.yview)
            self.sheet_list.configure(yscrollcommand=sb.set)
            self.sheet_list.pack(side="left", fill="x", expand=True)
            sb.pack(side="right", fill="y")
            bf = ttk.Frame(s)
            bf.pack(fill="x", pady=(4, 0))
            ttk.Button(bf, text="Select all", command=lambda: self.sheet_list.select_set(0, "end")).pack(side="left")
            ttk.Button(bf, text="Select none", command=lambda: self.sheet_list.select_clear(0, "end")).pack(side="left", padx=4)

            # ---- 5. output folder
            s = self._section(p, "5.  Save to folder")
            self.out_var = tk.StringVar(value=cfg.get("out_dir", str(Path.home() / "Documents")))
            ttk.Entry(s, textvariable=self.out_var, width=46).pack(fill="x")
            bf = ttk.Frame(s)
            bf.pack(fill="x", pady=(4, 0))
            ttk.Button(bf, text="Browse...", command=self.browse_out).pack(side="left")
            ttk.Button(bf, text="Open folder", command=self.open_out).pack(side="left", padx=4)
            self.auto_inc = tk.BooleanVar(value=True)
            ttk.Checkbutton(s, text="Add 1 to flight # after each save", variable=self.auto_inc).pack(anchor="w", pady=(4, 0))

            # ---- 6. export
            s = self._section(p, "6.  Export")
            self.export_btn = ttk.Button(s, text="Save .xlsx", style="Big.TButton", command=self.export)
            self.export_btn.pack(fill="x")
            self.cancel_btn = ttk.Button(s, text="Cancel", command=self.cancel_evt.set, state="disabled")
            self.cancel_btn.pack(fill="x", pady=(4, 0))
            self.prog = ttk.Progressbar(s, maximum=1.0)
            self.prog.pack(fill="x", pady=(6, 0))
            self.status = ttk.Label(s, text="", wraplength=390, justify="left")
            self.status.pack(anchor="w")

            self.fields_changed()

        def _build_plots(self, p):
            self.fig = Figure(figsize=(9, 8), dpi=100)
            gs = self.fig.add_gridspec(2, 1, height_ratios=[3.4, 1], hspace=0.12)
            self.ax3 = self.fig.add_subplot(gs[0], projection="3d")
            self.axt = self.fig.add_subplot(gs[1])
            self.fig.subplots_adjust(left=0.06, right=0.93, top=0.96, bottom=0.07)
            self.canvas = FigureCanvasTkAgg(self.fig, master=p)
            self.canvas.get_tk_widget().configure(bg=BG, highlightthickness=0)

            # ---- playback controls (under the plots)
            ctrl = ttk.Frame(p, padding=(8, 6))
            ctrl.pack(side="bottom", fill="x")
            row1 = ttk.Frame(ctrl)
            row1.pack(fill="x")
            self.restart_btn = ttk.Button(row1, text="⏮", width=3, command=self.restart_play)
            self.restart_btn.pack(side="left")
            self.play_btn = ttk.Button(row1, text="▶  Play", width=10, style="Play.TButton",
                                       command=self.toggle_play)
            self.play_btn.pack(side="left", padx=(4, 0))
            ttk.Label(row1, text="Speed").pack(side="left", padx=(12, 2))
            self.speed_var = tk.StringVar(value="2x")
            ttk.Combobox(row1, textvariable=self.speed_var, width=5, state="readonly",
                         values=["0.5x", "1x", "2x", "5x", "10x", "20x"]).pack(side="left")
            self.only_crop = tk.BooleanVar(value=False)
            ttk.Checkbutton(row1, text="Play cropped range only", variable=self.only_crop).pack(side="left", padx=12)
            self.play_var = tk.DoubleVar()
            self.scrub = ttk.Scale(row1, orient="horizontal", variable=self.play_var,
                                   command=lambda v: self.update_playhead())
            self.scrub.pack(side="left", fill="x", expand=True, padx=6)
            self.readout = ttk.Label(ctrl, text="", font=("Consolas", 10), foreground=MUTED)
            self.readout.pack(anchor="w", pady=(4, 0))

            self.canvas.get_tk_widget().pack(fill="both", expand=True)
            self.span = SpanSelector(self.axt, self.on_span, "horizontal", useblit=True, interactive=False,
                                     props=dict(alpha=0.25, facecolor="tab:orange"))
            self.rotor_angle = 0.0
            self._draw_empty()

        def _style_3d(self, ax):
            """Dark panes and a fixed camera: the 3D view can't be rotated, panned or zoomed."""
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.set_pane_color((0.12, 0.13, 0.16, 1.0))
                axis.line.set_color(BORDER)
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.set_facecolor(BG)
            ax.computed_zorder = False   # so the helicopter is always drawn on top of the path
            ax.view_init(elev=24, azim=-60)
            ax.disable_mouse_rotation()

        def _draw_empty(self):
            self.ax3.clear()
            self.axt.clear()
            self._style_3d(self.ax3)
            self.ax3.set_title("Load a .bin file to see the flight path")
            self.axt.set_xlabel("Time since boot (s)")
            self.canvas.draw_idle()

        def _set_enabled(self, on):
            st = "normal" if on else "disabled"
            for w in self.crop_btns + [self.export_btn, self.play_btn, self.restart_btn]:
                w.configure(state=st)

        # ------------------------------------------------------ loading
        def browse_bin(self):
            init = os.path.dirname(self.bin_var.get()) or str(Path.home() / "Downloads")
            f = filedialog.askopenfilename(title="Choose a flight log", initialdir=init,
                                           filetypes=[("ArduPilot logs", "*.bin *.BIN"), ("All files", "*.*")])
            if f:
                self.bin_var.set(f)
                self.load_bin()

        def load_bin(self):
            path = self.bin_var.get().strip().strip('"')
            if not os.path.isfile(path):
                messagebox.showerror("Flight Clipper", f"File not found:\n{path}")
                return
            self._set_enabled(False)
            self.stop_play()
            self.summary.configure(text="Loading...")

            def work():
                try:
                    log = BinLog(path, progress=lambda f, m: self.after(0, self._progress, f, m))
                    self.after(0, self._loaded, log)
                except Exception as exc:
                    traceback.print_exc()
                    self.after(0, lambda: (messagebox.showerror("Flight Clipper", f"Could not read log:\n{exc}"),
                                           self.summary.configure(text="Load failed.")))
            threading.Thread(target=work, daemon=True).start()

        def _progress(self, frac, msg):
            self.prog["value"] = frac
            self.status.configure(text=msg)

        def _loaded(self, log):
            self.log = log
            self.path = log.flight_path()
            cfg["last_bin"] = log.path
            save_config(cfg)
            lo, hi = log.time_range()
            self.t_lo, self.t_hi = lo, hi
            att = log.messages.get("ATT")
            self.att = (att["TimeUS"].to_numpy() / 1e6, att["Roll"].to_numpy(), att["Pitch"].to_numpy(),
                        att["Yaw"].to_numpy()) if att is not None and len(att) else None
            self.arm_ev = log.arm_events()

            rows = sum(len(d) for d in log.messages.values())
            msg = (f"{os.path.basename(log.path)}\n{len(log.messages)} message types, {rows:,} rows, "
                   f"{hi - lo:.1f} s of log ({lo:.1f} - {hi:.1f} s)")
            if self.arm_ev:
                msg += "\nArm events: " + ", ".join(f"{'ARM' if a else 'DISARM'} @ {t:.1f}s" for t, a in self.arm_ev)
            if log.skipped:
                msg += f"\nSkipped (bad format): {', '.join(log.skipped)}"
            self.summary.configure(text=msg)

            self.sheet_list.delete(0, "end")
            for name in sorted(log.messages):
                self.sheet_list.insert("end", f"{name:<6}{len(log.messages[name]):>10,} rows")
            self.sheet_list.select_set(0, "end")
            self._sheet_names = sorted(log.messages)

            for w in (self.t0_spin, self.t1_spin):
                w.configure(from_=lo, to=hi)
            self.scrub.configure(from_=lo, to=hi)
            self._draw_log()
            self.crop_auto()
            self.play_var.set(self._play_bounds()[0])
            self.update_playhead()
            self._set_enabled(True)
            self.prog["value"] = 0
            self.status.configure(text="Log loaded.  Press Play to watch the flight, set crop + file name, then Save.")

        # ------------------------------------------------------ helicopter model
        # Body frame: x forward, y right, z down (same as ArduPilot).  Length ~2 units.
        HELI_PARTS = {
            "body":  ([[0.55, 0, -0.02], [-0.30, 0, -0.06]], 9, "#ffb347"),
            "nose":  ([[0.55, 0, -0.02], [0.62, 0, 0.0]], 6, "#7ec8ff"),
            "boom":  ([[-0.30, 0, -0.08], [-1.15, 0, -0.12]], 3, "#ffb347"),
            "fin":   ([[-1.10, 0, -0.12], [-1.22, 0, -0.38]], 3, "#ffb347"),
            "stab":  ([[-0.95, -0.18, -0.12], [-0.95, 0.18, -0.12]], 2.5, "#ffb347"),
            "mast":  ([[0.0, 0, -0.08], [0.0, 0, -0.32]], 2.5, "#c9d1d9"),
            "skidL": ([[-0.38, -0.24, 0.22], [0.42, -0.24, 0.22], [0.52, -0.24, 0.14]], 2, "#c9d1d9"),
            "skidR": ([[-0.38, 0.24, 0.22], [0.42, 0.24, 0.22], [0.52, 0.24, 0.14]], 2, "#c9d1d9"),
            "strut1": ([[0.25, -0.24, 0.22], [0.20, -0.08, 0.02], [0.20, 0.08, 0.02], [0.25, 0.24, 0.22]], 1.8, "#c9d1d9"),
            "strut2": ([[-0.20, -0.24, 0.22], [-0.15, -0.08, 0.02], [-0.15, 0.08, 0.02], [-0.20, 0.24, 0.22]], 1.8, "#c9d1d9"),
        }
        ROTOR_R, ROTOR_Z = 0.95, -0.32

        def _create_heli(self, ax):
            self.heli_lines = {}
            for name, (_, lw, col) in self.HELI_PARTS.items():
                ln, = ax.plot([], [], [], color=col, lw=lw, solid_capstyle="round", zorder=10)
                self.heli_lines[name] = ln
            self.heli_lines["blades"], = ax.plot([], [], [], color="#e6edf3", lw=2.2, zorder=11)
            self.heli_lines["disc"], = ax.plot([], [], [], color="#e6edf3", lw=0.8, alpha=0.35, zorder=9)
            self.heli_lines["tail_rotor"], = ax.plot([], [], [], color="#e6edf3", lw=1.8, zorder=11)
            self.heli_lines["drop"], = ax.plot([], [], [], color=MUTED, lw=0.8, ls=":", zorder=5)
            self.heli_lines["shadow"], = ax.plot([], [], [], "o", color="#000000", alpha=0.45, ms=9, zorder=4)
            ax.plot([], [], [], color="#ffb347", lw=5, label="Helicopter")

        def _heli_body_points(self):
            """All model parts in body coordinates, including the spinning rotors."""
            pts = {k: np.array(v[0], float) for k, v in self.HELI_PARTS.items()}
            a, R, z = self.rotor_angle, self.ROTOR_R, self.ROTOR_Z
            c, s = np.cos(a), np.sin(a)
            pts["blades"] = np.array([[R * c, R * s, z], [-R * c, -R * s, z]])
            th = np.linspace(0, 2 * np.pi, 40)
            pts["disc"] = np.column_stack([R * np.cos(th), R * np.sin(th), np.full_like(th, z)])
            tr = 0.2
            ta = a * 2.3
            pts["tail_rotor"] = np.array([[-1.15 + tr * np.cos(ta), 0.07, -0.16 + tr * np.sin(ta)],
                                          [-1.15 - tr * np.cos(ta), 0.07, -0.16 - tr * np.sin(ta)]])
            return pts

        def _place_heli(self, e0, n0, u0, roll, pitch, yaw):
            r, p_, y = np.radians([roll, pitch, yaw])
            cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p_), np.sin(p_), np.cos(y), np.sin(y)
            # body (FRD) -> NED rotation, ZYX Euler (yaw, pitch, roll)
            Rm = np.array([[cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
                           [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
                           [-sp, sr * cp, cr * cp]])
            S, kz = self.heli_scale, self.heli_kz
            for name, pts in self._heli_body_points().items():
                ned = pts @ Rm.T * S
                self.heli_lines[name].set_data_3d(e0 + ned[:, 1], n0 + ned[:, 0], u0 - ned[:, 2] * kz)
            self.heli_lines["drop"].set_data_3d([e0, e0], [n0, n0], [u0, self.z_floor])
            self.heli_lines["shadow"].set_data_3d([e0], [n0], [self.z_floor])

        # ------------------------------------------------------ plotting
        def _draw_log(self):
            ax, axt = self.ax3, self.axt
            ax.clear()
            axt.clear()
            self._style_3d(ax)
            if self.path is None:
                ax.set_title("No position data in this log")
                self._path_ds = None
            else:
                t, e, n, u, src = self.path
                step = max(1, len(t) // 4000)
                self._path_ds = (t[::step], e[::step], n[::step], u[::step])
                ax.plot(e[::step], n[::step], u[::step], color="#5c6370", lw=0.8, label="Full log")
                self.crop_line, = ax.plot([], [], [], color=ACCENT, lw=2, label="Cropped flight")
                self.trail, = ax.plot([], [], [], color="#ffb347", lw=1.2, alpha=0.8, label="Trail")
                self.start_pt, = ax.plot([], [], [], "o", color="#3fb950", ms=7, label="Crop start")
                self.end_pt, = ax.plot([], [], [], "s", color="#f85149", ms=7, label="Crop end")
                ax.set_xlabel("East (m)", fontsize=9)
                ax.set_ylabel("North (m)", fontsize=9)
                ax.set_zlabel("Up (m)", fontsize=9)
                ax.set_title(f"Flight path  -  {src}", fontsize=10)

                span = max(np.ptp(e), np.ptp(n), 1.0) * 1.1
                ec, nc = (e.min() + e.max()) / 2, (n.min() + n.max()) / 2
                z0, z1 = min(u.min(), 0) - 0.5, max(u.max(), 1) + span * 0.06
                ax.set_xlim(ec - span / 2, ec + span / 2)
                ax.set_ylim(nc - span / 2, nc + span / 2)
                ax.set_zlim(z0, z1)
                bz = float(np.clip((z1 - z0) / span, 0.35, 1.0))
                ax.set_box_aspect((1, 1, bz), zoom=1.3)
                # Scale so the model looks the same shape whatever the axis ranges are.
                self.heli_scale = span * 0.1
                self.heli_kz = (z1 - z0) / (bz * span)
                self.z_floor = z0
                self._create_heli(ax)
                ax.legend(loc="upper left", fontsize=8)

                axt.plot(t, u, color=ACCENT, lw=1, label="Altitude (m)")
                axt.set_ylabel("Altitude (m)")
            ctun = self.log.messages.get("CTUN")
            if ctun is not None and "ThO" in ctun:
                a2 = axt.twinx()
                a2.plot(ctun["TimeUS"] / 1e6, ctun["ThO"], color="#d2a8ff", lw=0.7, alpha=0.6)
                a2.set_ylabel("Throttle out", color="#d2a8ff", fontsize=8)
                a2.tick_params(labelsize=7)
                self._twin = a2
            for tt, armed in self.arm_ev:
                col = "#3fb950" if armed else "#f85149"
                axt.axvline(tt, color=col, ls="--", lw=1)
                axt.text(tt, axt.get_ylim()[1], " ARM" if armed else " DISARM", fontsize=7, color=col, va="top")
            axt.set_xlim(self.t_lo, self.t_hi)
            axt.grid(True, alpha=0.3)
            axt.set_xlabel("Time since boot (s)   -   drag to set the crop range")
            self.crop_patch = axt.axvspan(self.t_lo, self.t_hi, color="tab:orange", alpha=0.15)
            self.play_line = axt.axvline(self.t_lo, color="#ffb347", lw=1.5)
            self.canvas.draw_idle()

        def _redraw_crop(self):
            if self.log is None:
                return
            t0, t1 = self.t0_var.get(), self.t1_var.get()
            self.crop_patch.remove()
            self.crop_patch = self.axt.axvspan(t0, t1, color="tab:orange", alpha=0.15)
            if self._path_ds is not None:
                t, e, n, u = self._path_ds
                m = (t >= t0) & (t <= t1)
                if m.any():
                    self.crop_line.set_data_3d(e[m], n[m], u[m])
                    i0, i1 = np.flatnonzero(m)[[0, -1]]
                    self.start_pt.set_data_3d([e[i0]], [n[i0]], [u[i0]])
                    self.end_pt.set_data_3d([e[i1]], [n[i1]], [u[i1]])
                else:
                    for ln in (self.crop_line, self.start_pt, self.end_pt):
                        ln.set_data_3d([], [], [])
            self.canvas.draw_idle()

        def _armed_at(self, tp):
            if not self.arm_ev:
                return True
            state = False
            for t, a in self.arm_ev:
                if t <= tp:
                    state = a
            return state

        def update_playhead(self):
            if self.log is None:
                return
            tp = self.play_var.get()
            self.play_line.set_xdata([tp, tp])
            armed = self._armed_at(tp)
            txt = f"t = {tp:8.2f} s   {'ARMED   ' if armed else 'disarmed'}"
            if self.path is not None:
                t, e, n, u, _ = self.path
                k = int(np.clip(np.searchsorted(t, tp), 0, len(t) - 1))
                roll = pitch = 0.0
                yaw = 0.0
                if self.att is not None:
                    ta, r, pch, y = self.att
                    j = int(np.clip(np.searchsorted(ta, tp), 0, len(ta) - 1))
                    roll, pitch, yaw = r[j], pch[j], y[j]
                self._place_heli(e[k], n[k], u[k], roll, pitch, yaw)
                k0 = max(0, np.searchsorted(t, tp - 10.0))  # last 10 s of flight as a trail
                self.trail.set_data_3d(e[k0:k + 1], n[k0:k + 1], u[k0:k + 1])
                txt += (f"   alt {u[k]:6.1f} m   roll {roll:6.1f}°   pitch {pitch:6.1f}°"
                        f"   yaw {yaw:6.1f}°")
                if tp > t[-1]:
                    txt += "   (no position data after this time)"
            self.readout.configure(text=txt)
            self.canvas.draw_idle()

        # ------------------------------------------------------ playback
        def toggle_play(self):
            if self.playing:
                self.stop_play()
            else:
                self.playing = True
                self.play_btn.configure(text="⏸  Pause")
                lo, hi = self._play_bounds()
                if not (lo <= self.play_var.get() < hi - 1e-3):
                    self.play_var.set(lo)
                self._last_tick = time.perf_counter()
                self._tick()

        def stop_play(self):
            self.playing = False
            self.play_btn.configure(text="▶  Play")

        def restart_play(self):
            self.play_var.set(self._play_bounds()[0])
            self.update_playhead()

        def _play_bounds(self):
            if self.only_crop.get():
                return self.t0_var.get(), self.t1_var.get()
            if self.path is not None:  # whole file, limited to where there is position data
                t = self.path[0]
                return max(self.t_lo, t[0]), min(self.t_hi, t[-1])
            return self.t_lo, self.t_hi

        def _tick(self):
            if not self.playing:
                return
            now = time.perf_counter()
            dt = (now - self._last_tick) * float(self.speed_var.get().rstrip("x"))
            self._last_tick = now
            lo, hi = self._play_bounds()
            tp = self.play_var.get() + dt
            if self._armed_at(tp):
                self.rotor_angle += 0.9  # visible spin (a real rotor strobes at screen frame rates)
            if tp >= hi:
                self.play_var.set(hi)
                self.update_playhead()
                self.stop_play()
                return
            self.play_var.set(max(tp, lo))
            self.update_playhead()
            self.after(40, self._tick)

        # ------------------------------------------------------ crop
        def _set_crop(self, t0, t1):
            t0, t1 = sorted((float(t0), float(t1)))
            t0, t1 = max(t0, self.t_lo), min(t1, self.t_hi)
            self.t0_var.set(round(t0, 2))
            self.t1_var.set(round(t1, 2))
            self.dur_lbl.configure(text=f"Duration: {t1 - t0:.2f} s   ({t0:.2f} -> {t1:.2f})")
            self._redraw_crop()

        def crop_changed(self):
            if self.log is None:
                return
            try:
                self._set_crop(self.t0_var.get(), self.t1_var.get())
            except (tk.TclError, ValueError):
                pass

        def crop_full(self):
            self._set_crop(self.t_lo, self.t_hi)

        def crop_auto(self):
            self._set_crop(*self.log.detect_flight())

        def crop_from_playhead(self, which):
            tp = self.play_var.get()
            if which == 0:
                self._set_crop(tp, max(tp, self.t1_var.get()))
            else:
                self._set_crop(min(tp, self.t0_var.get()), tp)

        def on_span(self, xmin, xmax):
            if self.log is not None and xmax - xmin > 0.05:
                self._set_crop(xmin, xmax)

        # ------------------------------------------------------ naming
        def dir_changed(self):
            if self.dir_var.get() == "M":
                self.cg_var.set("00.0")
            self.fields_changed()

        def fields_changed(self):
            if self._syncing:
                return
            try:
                cg = float(self.cg_var.get() or 0)
                d = self.dir_var.get()
                if cg == 0 and d != "M":
                    d = "M"
                    self.dir_var.set("M")
                elif cg != 0 and d == "M":
                    raise ValueError("Pick Aft (A) or Forward (F) for a non-zero CG")
                name = build_filename(self.flight_var.get(), self.heli_var.get(), cg, d, self.wind_var.get())
                self._syncing = True
                self.name_var.set(name)
                self._syncing = False
                self._name_ok(name)
            except ValueError as exc:
                self.name_msg.configure(text=f"Error: {exc}", foreground=ERR_FG)
                self._valid_name = None

        def name_typed(self):
            if self._syncing:
                return
            txt = self.name_var.get().strip()
            if not txt.lower().endswith(".xlsx"):
                txt += ".xlsx"
            try:
                f, h, cg, d, w = parse_filename(txt)
            except ValueError as exc:
                self.name_msg.configure(text=f"Error: {exc}", foreground=ERR_FG)
                self._valid_name = None
                return
            self._syncing = True
            self.flight_var.set(str(f))
            self.heli_var.set(h)
            self.cg_var.set(f"{cg:04.1f}")
            self.dir_var.set(d)
            self.wind_var.set(w)
            self._syncing = False
            self.fields_changed()

        def _name_ok(self, name):
            f, h, cg, d, w = parse_filename(name)
            desc = {"A": "aft of", "F": "forward of", "M": "at"}[d]
            wind = {"L": "low", "M": "medium", "H": "high"}[w]
            self.name_msg.configure(
                text=f"OK: flight {f}, heli {h}, CG {cg:.1f} mm {desc} datum "
                     f"(signed {signed_cg(cg, d):+.1f} mm), {wind} wind", foreground=OK_FG)
            self._valid_name = name

        # ------------------------------------------------------ output
        def browse_out(self):
            d = filedialog.askdirectory(title="Choose the save folder", initialdir=self.out_var.get() or None)
            if d:
                self.out_var.set(d)

        def open_out(self):
            d = self.out_var.get()
            if os.path.isdir(d):
                os.startfile(d) if sys.platform == "win32" else None

        def export(self):
            if self.log is None:
                return
            self.fields_changed()
            name = getattr(self, "_valid_name", None)
            if not name:
                messagebox.showerror("Flight Clipper", "Fix the file name first.")
                return
            out_dir = self.out_var.get().strip().strip('"')
            if not out_dir:
                messagebox.showerror("Flight Clipper", "Choose a folder to save to.")
                return
            os.makedirs(out_dir, exist_ok=True)
            out = os.path.join(out_dir, name)
            if os.path.exists(out) and not messagebox.askyesno("Flight Clipper", f"{name} already exists.\nOverwrite it?"):
                return
            sel = [self._sheet_names[i] for i in self.sheet_list.curselection()]
            if not sel:
                messagebox.showerror("Flight Clipper", "Select at least one sheet.")
                return
            t0, t1 = self.t0_var.get(), self.t1_var.get()
            if t1 - t0 <= 0:
                messagebox.showerror("Flight Clipper", "The crop range is empty.")
                return
            f, h, cg, d, w = parse_filename(name)
            info = {"Flight number": f, "Helicopter type": h, "CG (mm)": cg, "CG direction": d,
                    "CG signed (mm, +F / -A)": signed_cg(cg, d), "Wind level": w}

            cfg.update(out_dir=out_dir, heli=h, wind=w, flight=f)
            save_config(cfg)
            self.cancel_evt.clear()
            self.export_btn.configure(state="disabled")
            self.cancel_btn.configure(state="normal")
            t_start = time.perf_counter()

            def work():
                try:
                    written = export_xlsx(self.log, out, t0, t1, sel, self.keep_meta.get(), info,
                                          progress=lambda fr, m: self.after(0, self._progress, fr, m),
                                          cancel=self.cancel_evt)
                    self.after(0, self._exported, out, written, time.perf_counter() - t_start)
                except InterruptedError:
                    self.after(0, self._export_done, "Export cancelled.")
                except PermissionError:
                    self.after(0, self._export_done, f"Can't write {name} - is it open in Excel?")
                except Exception as exc:
                    traceback.print_exc()
                    self.after(0, self._export_done, f"Export failed: {exc}")
            threading.Thread(target=work, daemon=True).start()

        def _export_done(self, msg):
            self.export_btn.configure(state="normal")
            self.cancel_btn.configure(state="disabled")
            self.status.configure(text=msg)

        def _exported(self, out, written, secs):
            self._export_done(f"Saved {os.path.basename(out)}: {len(written)} sheets, "
                              f"{sum(written.values()):,} rows in {secs:.0f} s\n{out}")
            if self.auto_inc.get():
                try:
                    self.flight_var.set(str(int(self.flight_var.get()) + 1))
                except ValueError:
                    pass

        def _on_close(self):
            self.playing = False
            self.cancel_evt.set()
            self.destroy()

    App().mainloop()


# =============================================================================
# 5. Command-line use (no GUI):
#    python flight_clipper.py log.bin 300R_02.2A_M.xlsx --start 120 --end 300 --out C:\data
# =============================================================================
def main(argv):
    import argparse
    ap = argparse.ArgumentParser(description="Convert an ArduPilot .bin log to .xlsx")
    ap.add_argument("bin", nargs="?")
    ap.add_argument("name", nargs="?", help="e.g. 300R_02.2A_M.xlsx")
    ap.add_argument("--start", type=float, help="crop start, seconds since boot")
    ap.add_argument("--end", type=float, help="crop end, seconds since boot")
    ap.add_argument("--auto", action="store_true", help="crop arm -> disarm")
    ap.add_argument("--out", default=".", help="output folder")
    a = ap.parse_args(argv)
    if not a.bin:
        run_gui()
        return
    parse_filename(a.name)
    log = BinLog(a.bin)
    t0, t1 = log.time_range()
    if a.auto:
        t0, t1 = log.detect_flight()
    t0 = a.start if a.start is not None else t0
    t1 = a.end if a.end is not None else t1
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, a.name)
    f, h, cg, d, w = parse_filename(a.name)
    last = [0]

    def prog(fr, m):
        if fr - last[0] > 0.05 or fr >= 1:
            last[0] = fr
            print(f"{fr * 100:5.1f}%  {m}", flush=True)
    written = export_xlsx(log, out, t0, t1, info={"Flight number": f, "Helicopter type": h, "CG (mm)": cg,
                                                  "CG direction": d, "Wind level": w}, progress=prog)
    print(f"Saved {out}  ({len(written)} sheets)")


if __name__ == "__main__":
    main(sys.argv[1:])
