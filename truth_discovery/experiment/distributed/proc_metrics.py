
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import psutil

@dataclass
class ProcSample:
    pid: int
    label: str
    baseline_rss_mb: float = float("nan")
    peak_rss_mb: float = float("nan")
    cpu_user_sec: float = float("nan")
    cpu_system_sec: float = float("nan")
    samples: int = 0
    errors: int = 0
    history: List[float] = field(default_factory=list)

    def peak_delta_rss_mb(self) -> float:
        if self.baseline_rss_mb != self.baseline_rss_mb:
            return float("nan")
        if self.peak_rss_mb != self.peak_rss_mb:
            return float("nan")
        return max(self.peak_rss_mb - self.baseline_rss_mb, 0.0)

class MultiProcessSampler:
    def __init__(self, interval_sec: float = 0.05, keep_history: bool = False):
        self.interval_sec = max(float(interval_sec), 0.005)
        self.keep_history = bool(keep_history)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._samples: Dict[int, ProcSample] = {}

    def track(self, pid: int, label: str) -> None:
        if pid in self._samples:
            return
        sample = ProcSample(pid=int(pid), label=str(label))
        try:
            proc = psutil.Process(pid)
            sample.baseline_rss_mb = float(proc.memory_info().rss) / (1024.0 * 1024.0)
            sample.peak_rss_mb = sample.baseline_rss_mb
        except Exception:
            sample.errors += 1
        self._samples[pid] = sample

    def untrack(self, pid: int) -> None:
        self._samples.pop(int(pid), None)

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[int, ProcSample]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.interval_sec * 4.0, 0.2))

        self._sample_all(final=True)
        return dict(self._samples)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._sample_all(final=False)
            time.sleep(self.interval_sec)

    def _sample_all(self, *, final: bool) -> None:
        for pid, sample in list(self._samples.items()):
            try:
                proc = psutil.Process(pid)
                rss_mb = float(proc.memory_info().rss) / (1024.0 * 1024.0)
                if sample.baseline_rss_mb != sample.baseline_rss_mb:
                    sample.baseline_rss_mb = rss_mb
                if sample.peak_rss_mb != sample.peak_rss_mb:
                    sample.peak_rss_mb = rss_mb
                else:
                    sample.peak_rss_mb = max(sample.peak_rss_mb, rss_mb)
                sample.samples += 1
                if self.keep_history:
                    sample.history.append(rss_mb)
                if final:
                    times = proc.cpu_times()
                    sample.cpu_user_sec = float(times.user)
                    sample.cpu_system_sec = float(times.system)
            except Exception:
                sample.errors += 1

def aggregate_fleet(samples: Dict[int, ProcSample]) -> Dict[str, float]:

    if not samples:
        return {
            "fleet_peak_rss_mb_max": float("nan"),
            "fleet_peak_rss_mb_sum": float("nan"),
            "fleet_peak_delta_rss_mb_max": float("nan"),
            "fleet_peak_delta_rss_mb_sum": float("nan"),
            "fleet_cpu_user_sec_sum": float("nan"),
            "fleet_cpu_system_sec_sum": float("nan"),
            "fleet_proc_count": 0,
        }
    peaks = [s.peak_rss_mb for s in samples.values() if s.peak_rss_mb == s.peak_rss_mb]
    deltas = [s.peak_delta_rss_mb() for s in samples.values()]
    deltas = [d for d in deltas if d == d]
    cpu_user = [s.cpu_user_sec for s in samples.values() if s.cpu_user_sec == s.cpu_user_sec]
    cpu_sys = [s.cpu_system_sec for s in samples.values() if s.cpu_system_sec == s.cpu_system_sec]
    return {
        "fleet_peak_rss_mb_max": float(max(peaks)) if peaks else float("nan"),
        "fleet_peak_rss_mb_sum": float(sum(peaks)) if peaks else float("nan"),
        "fleet_peak_delta_rss_mb_max": float(max(deltas)) if deltas else float("nan"),
        "fleet_peak_delta_rss_mb_sum": float(sum(deltas)) if deltas else float("nan"),
        "fleet_cpu_user_sec_sum": float(sum(cpu_user)) if cpu_user else float("nan"),
        "fleet_cpu_system_sec_sum": float(sum(cpu_sys)) if cpu_sys else float("nan"),
        "fleet_proc_count": len(samples),
    }
