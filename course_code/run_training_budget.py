"""Run three transfer experiments with a shared hard four-hour wall-clock limit."""
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if __name__ == '__main__':
    start = time.monotonic()
    runs = [
        ['--model','YOLO-Master-v0.1-N','--epochs','100','--hours','0.95'],
        ['--model','YOLO-Master-v0.1-N','--epochs','100','--hours','0.95','--cls-pw','0.5','--name','YOLO-Master-v0.1-N-clspw0.5-retrain'],
        ['--model','YOLO-Master-EsMoE-N','--epochs','50','--hours','0.65'],
    ]
    for args in runs:
        remaining = 4*3600-(time.monotonic()-start)
        if remaining <= 0:
            raise SystemExit('Four-hour budget exhausted')
        proc = subprocess.Popen([sys.executable,str(ROOT/'course_code/train_ppe.py'),*args],cwd=ROOT)
        try:
            code = proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            if sys.platform == 'win32':
                subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],check=False)
            else:
                proc.kill()
            raise SystemExit('Stopped at shared four-hour limit; keep saved checkpoints')
        if code:
            raise SystemExit(code)
    print(f'Total training wall clock: {time.monotonic()-start:.1f} seconds')
