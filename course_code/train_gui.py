"""YOLO-Master PPE training console (Tkinter).

Start / continue training with one click. Training state lives in results/ppe/<model>/:
last.pt is written after every epoch, so closing the window never loses more than the
current (unfinished) epoch, and the next "开始 / 继续训练" resumes from there.

Launch: .venv\\Scripts\\pythonw.exe course_code\\train_gui.py   (or the desktop shortcut)
"""

import csv
import datetime
import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
TRAIN = ROOT / "course_code" / "train_ppe.py"
RUNS = ROOT / "results" / "ppe"
MODELS = ["YOLO-Master-v0.1-N", "YOLO-Master-EsMoE-N"]
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\[K")
FONT = ("Microsoft YaHei UI", 10)


def read_state(model):
    """Return (done_epochs, total_epochs, rows, resumable) for a run folder."""
    run = RUNS / model
    rows = []
    if (run / "results.csv").exists():
        with open(run / "results.csv", encoding="utf-8") as f:
            rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
    total = None
    if (run / "args.yaml").exists():
        m = re.search(r"^epochs:\s*(\d+)", (run / "args.yaml").read_text(encoding="utf-8"), re.M)
        total = int(m.group(1)) if m else None
    resumable = (run / "weights" / "last.pt").exists() and total is not None and len(rows) < total
    return len(rows), total, rows, resumable


def external_pid(model):
    """PID of a train_ppe.py run for this model started outside this window (via its RUNNING file)."""
    flag = RUNS / model / "RUNNING"
    try:
        import psutil

        pid = int(flag.read_text())
        if psutil.pid_exists(pid) and "train_ppe.py" in " ".join(psutil.Process(pid).cmdline()):
            return pid
    except Exception:
        pass
    return None


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("YOLO-Master 施工现场 PPE 检测 · 训练控制台")
        k = self.winfo_fpixels("1i") / 96  # Windows display scaling (100% = 96 dpi)
        self.geometry(f"{int(900 * k)}x{int(680 * k)}")
        self.minsize(int(760 * k), int(560 * k))
        self.proc = None
        self.lines = queue.Queue()
        self.closing = False
        self._build()
        self.refresh()
        self.after(300, self._pump)
        self.after(2000, self._tick)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------------------------------------------------------------- UI
    def _build(self):
        style = ttk.Style(self)
        style.configure(".", font=FONT)
        style.configure("Big.TButton", font=("Microsoft YaHei UI", 11, "bold"), padding=8)

        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")
        ttk.Label(top, text="模型").grid(row=0, column=0, sticky="w")
        self.model = tk.StringVar(value=MODELS[0])
        cb = ttk.Combobox(top, textvariable=self.model, values=MODELS, state="readonly", width=24)
        cb.grid(row=0, column=1, padx=(4, 16))
        cb.bind("<<ComboboxSelected>>", lambda e: self.refresh())
        ttk.Label(top, text="总轮数").grid(row=0, column=2, sticky="w")
        self.epochs = tk.IntVar(value=100)
        self.epochs_box = ttk.Spinbox(top, from_=1, to=1000, textvariable=self.epochs, width=6)
        self.epochs_box.grid(row=0, column=3, padx=(4, 16))
        ttk.Label(top, text="Batch").grid(row=0, column=4, sticky="w")
        self.batch = tk.IntVar(value=8)  # 16 overflows 8 GB VRAM and becomes ~20x slower
        self.batch_box = ttk.Spinbox(top, values=(2, 4, 8, 12, 16), textvariable=self.batch, width=5)
        self.batch_box.grid(row=0, column=5, padx=(4, 0))

        btns = ttk.Frame(self, padding=(10, 0))
        btns.pack(fill="x")
        self.start_btn = ttk.Button(btns, text="▶ 开始训练", style="Big.TButton", command=self.start)
        self.start_btn.pack(side="left")
        self.pause_btn = ttk.Button(btns, text="⏸ 本轮结束后暂停", command=self.pause, state="disabled")
        self.pause_btn.pack(side="left", padx=8)
        self.kill_btn = ttk.Button(btns, text="■ 立即停止", command=self.kill, state="disabled")
        self.kill_btn.pack(side="left")
        ttk.Button(btns, text="打开结果文件夹", command=self.open_folder).pack(side="right")
        self.restart_btn = ttk.Button(btns, text="重新训练", command=self.restart)
        self.restart_btn.pack(side="right", padx=8)

        info = ttk.Frame(self, padding=10)
        info.pack(fill="x")
        self.status = tk.StringVar()
        ttk.Label(info, textvariable=self.status, font=("Microsoft YaHei UI", 11, "bold")).pack(anchor="w")
        self.bar = ttk.Progressbar(info, maximum=100)
        self.bar.pack(fill="x", pady=6)
        self.metrics = tk.StringVar()
        ttk.Label(info, textvariable=self.metrics).pack(anchor="w")

        mid = ttk.Frame(self, padding=(10, 0))
        mid.pack(fill="both", expand=True)
        self.chart = tk.Canvas(mid, height=170, bg="white", highlightthickness=1, highlightbackground="#ccc")
        self.chart.pack(fill="x")
        self.chart.bind("<Configure>", lambda e: self.draw_chart())
        self.log = tk.Text(mid, height=12, font=("Consolas", 9), bg="#111", fg="#ddd", wrap="none")
        self.log.pack(fill="both", expand=True, pady=(8, 10))
        self.rows = []

    # ---------------------------------------------------------------- state
    def refresh(self):
        done, total, self.rows, resumable = read_state(self.model.get())
        own = self.proc is not None and self.proc.poll() is None
        self.ext = None if own else external_pid(self.model.get())
        running = own or self.ext is not None
        if total:
            self.bar["value"] = 100 * done / total
        else:
            self.bar["value"] = 0
        if running:
            where = f"（后台进程 PID {self.ext}，日志见 logs 文件夹）" if self.ext else ""
            self.status.set(f"训练中…  第 {min(done + 1, total or done + 1)} / {total or self.epochs.get()} 轮{where}")
        elif total and done >= total:
            self.status.set(f"已完成：{done} / {total} 轮")
        elif resumable:
            self.status.set(f"已暂停：完成 {done} / {total} 轮，点击“继续训练”从第 {done + 1} 轮接着训练")
        else:
            self.status.set("未开始")
        if self.rows:
            r = self.rows[-1]
            best = max(float(x.get("metrics/mAP50-95(B)", 0) or 0) for x in self.rows)
            self.metrics.set(
                f"最近一轮  mAP50={float(r.get('metrics/mAP50(B)', 0)):.3f}   "
                f"mAP50-95={float(r.get('metrics/mAP50-95(B)', 0)):.3f}   "
                f"box_loss={float(r.get('train/box_loss', 0)):.3f}   最佳 mAP50-95={best:.3f}"
            )
        else:
            self.metrics.set("暂无训练记录")
        fixed = resumable or running
        self.epochs_box.configure(state="disabled" if fixed else "normal")
        self.batch_box.configure(state="disabled" if fixed else "normal")
        self.start_btn.configure(
            text="▶ 继续训练" if resumable else "▶ 开始训练",
            state="disabled" if running or (total and done >= total) else "normal",
        )
        self.pause_btn.configure(state="normal" if running else "disabled")
        self.kill_btn.configure(state="normal" if running else "disabled")
        self.restart_btn.configure(state="disabled" if running else "normal")
        self.draw_chart()

    def draw_chart(self):
        c = self.chart
        c.delete("all")
        w, h, pad = c.winfo_width(), c.winfo_height(), 30
        c.create_text(pad, 10, text="验证集 mAP50 (蓝) / mAP50-95 (橙)", anchor="w", font=("Microsoft YaHei UI", 9))
        if len(self.rows) < 1 or w < 50:
            return
        n = max(len(self.rows), 2)
        for key, color in (("metrics/mAP50(B)", "#2a6fdb"), ("metrics/mAP50-95(B)", "#e8833a")):
            pts = []
            for i, r in enumerate(self.rows):
                y = float(r.get(key, 0) or 0)
                pts += [pad + (w - 2 * pad) * i / (n - 1), h - pad - (h - 2 * pad - 10) * y]
            if len(pts) >= 4:
                c.create_line(*pts, fill=color, width=2)
            else:
                c.create_oval(pts[0] - 3, pts[1] - 3, pts[0] + 3, pts[1] + 3, fill=color, outline="")
        c.create_line(pad, h - pad, w - pad, h - pad, fill="#999")
        c.create_text(pad, h - pad + 12, text="1", font=FONT)
        c.create_text(w - pad, h - pad + 12, text=str(len(self.rows)), font=FONT)

    # ---------------------------------------------------------------- actions
    def start(self):
        model = self.model.get()
        _, _, _, resumable = read_state(model)
        cmd = [str(PYTHON), "-u", str(TRAIN), "--model", model]
        if resumable:
            cmd.append("--resume")
        else:
            cmd += ["--epochs", str(self.epochs.get()), "--batch", str(self.batch.get())]
        self.log_line(f"$ {' '.join(cmd[2:])}")
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        self.proc = subprocess.Popen(
            cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()
        self.refresh()

    def pause(self):
        (RUNS / self.model.get()).mkdir(parents=True, exist_ok=True)
        (RUNS / self.model.get() / "STOP").touch()
        self.pause_btn.configure(state="disabled")
        self.log_line("[界面] 已请求暂停：当前这一轮训练和验证结束、保存检查点后自动停止。")

    def kill(self):
        pid = self.proc.pid if self.proc and self.proc.poll() is None else getattr(self, "ext", None)
        if pid:
            # /T also stops the dataloader worker processes
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True,
                           creationflags=subprocess.CREATE_NO_WINDOW)
            (RUNS / self.model.get() / "RUNNING").unlink(missing_ok=True)
            self.log_line("[界面] 已立即停止，进度保留到上一轮结束时的检查点。")
        self.refresh()

    def restart(self):
        run = RUNS / self.model.get()
        if not run.exists():
            return self.start()
        if not messagebox.askyesno("重新训练", f"把现有训练记录改名备份，然后从头开始训练？\n{run}"):
            return
        run.rename(run.with_name(f"{run.name}_backup_{datetime.datetime.now():%Y%m%d_%H%M%S}"))
        self.refresh()
        self.start()

    def open_folder(self):
        run = RUNS / self.model.get()
        os.startfile(run if run.exists() else RUNS if RUNS.exists() else ROOT / "results")

    def on_close(self):
        if not (self.proc and self.proc.poll() is None):
            return self.destroy()
        ans = messagebox.askyesnocancel(
            "训练进行中",
            "是：本轮结束后保存进度再退出（推荐，窗口会等到本轮结束）\n"
            "否：立即退出（保留到上一轮结束的进度）\n取消：继续训练",
        )
        if ans is None:
            return
        if ans:
            self.closing = True
            self.pause()
            self.status.set("正在等待本轮结束并保存进度，完成后窗口自动关闭…")
        else:
            self.kill()
            self.destroy()

    # ---------------------------------------------------------------- plumbing
    def _reader(self, proc):
        buf = b""
        while chunk := proc.stdout.read1(4096) if hasattr(proc.stdout, "read1") else proc.stdout.read(1):
            buf += chunk
            *parts, buf = re.split(rb"[\r\n]", buf)
            for p in parts:
                if p.strip():
                    self.lines.put(ANSI.sub("", p.decode("utf-8", "replace")))
        self.lines.put(None)

    def _pump(self):
        last_progress = None
        try:
            while True:
                line = self.lines.get_nowait()
                if line is None:
                    code = self.proc.returncode if self.proc else None
                    self.log_line(f"[界面] 训练进程已结束（返回码 {code}）。")
                    self.refresh()
                    if self.closing:
                        self.destroy()
                        return
                    continue
                if re.search(r"\d+%.*\d+/\d+", line) and "Class" not in line:
                    last_progress = line  # tqdm-style bar: only show the latest state
                    continue
                self.log_line(line)
        except queue.Empty:
            pass
        if last_progress:
            self.log_line(last_progress, replace_progress=True)
        self.after(300, self._pump)

    def _tick(self):
        self.refresh()
        self.after(2000, self._tick)

    def log_line(self, text, replace_progress=False):
        self.log.configure(state="normal")
        last = self.log.index("end-2c linestart")
        if replace_progress and self.log.get(last, "end-1c").startswith("» "):
            self.log.delete(last, "end-1c")
        self.log.insert("end", ("» " if replace_progress else "") + text + "\n")
        if int(self.log.index("end-1c").split(".")[0]) > 2000:
            self.log.delete("1.0", "500.0")
        self.log.see("end")
        self.log.configure(state="disabled")


if __name__ == "__main__":
    try:  # crisp text on scaled (125%/150%) laptop displays
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass
    if not PYTHON.exists():
        messagebox.showerror("缺少环境", f"找不到 {PYTHON}")
        sys.exit(1)
    App().mainloop()
