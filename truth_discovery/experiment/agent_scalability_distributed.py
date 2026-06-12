
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from truth_discovery.experiment.agent_scalability_benchmark import (
    HYBRID_METHOD_NAME,
    METHOD_NAMES,
    TEXT_CAPABLE_METHODS,
    WorkloadSpec,
    _PeakRssSampler,
    _batches_to_numeric_df,
    _batches_to_text_df,
    _build_workload_specs,
    _bounded_float,
    _collect_workload_batches,
    _compute_group_scores,
    _evaluate_hybrid_requirement,
    _extract_agents_from_batches,
    _format_float,
    _load_batches,
    _mean_percentage_score,
    _non_negative_int,
    _normalize_weights,
    _parse_agent_counts,
    _positive_int,
    _resolve_num_workloads,
    _run_numeric_method_by_name,
    _run_text_method_by_name,
    _sanitize_path_stem,
    _select_agent_subsets,
    _summarize_by_method_agent,
    _summarize_overall,
    _write_json,
)
from truth_discovery.experiment.distributed.method_adapters import (
    METHOD_PROTOCOLS,
    DistributedRunner,
)
from truth_discovery.experiment.distributed.proc_metrics import (
    MultiProcessSampler,
    aggregate_fleet,
)
from truth_discovery.experiment.distributed.transport_zmq import (
    ZmqCoordinatorTransport,
)

WORKER_ENTRY = Path(__file__).resolve().parent / "agent_worker_entry.py"
DEFAULT_DATA_DIR = ROOT / "truth_discovery" / "data" / "data_agent-50_news-300"
DEFAULT_INPUT_JSON = str(DEFAULT_DATA_DIR / "num_demo.json")
DEFAULT_OUTPUT_ROOT = str(Path(__file__).resolve().parent / "outputs")

class WorkerPool:

    def __init__(
        self,
        agent_ids: Sequence[str],
        coord_addr: str,
        serialization: str,
        python_executable: str,
        preload_sbert: bool = False,
        spawn_stagger_sec: float = 0.0,
    ):
        self.agent_ids = list(agent_ids)
        self.coord_addr = coord_addr
        self.serialization = serialization
        self.python_executable = python_executable
        self.preload_sbert = bool(preload_sbert)
        self.spawn_stagger_sec = max(0.0, float(spawn_stagger_sec))
        self.processes: Dict[str, subprocess.Popen] = {}

    def start(self) -> None:
        for idx, agent_id in enumerate(self.agent_ids):
            if idx > 0 and self.spawn_stagger_sec > 0:

                time.sleep(self.spawn_stagger_sec)
            cmd = [
                self.python_executable,
                str(WORKER_ENTRY),
                "--agent-id",
                agent_id,
                "--coord-addr",
                self.coord_addr,
                "--serialization",
                self.serialization,
            ]
            if self.preload_sbert:
                cmd.append("--preload-sbert")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                close_fds=True,
            )
            self.processes[agent_id] = proc

    def pids(self) -> Dict[str, int]:
        return {agent_id: proc.pid for agent_id, proc in self.processes.items()}

    def shutdown(self, coord_transport, *, grace_sec: float = 5.0) -> List[str]:
        errors: List[str] = []

        try:
            coord_transport.broadcast(self.agent_ids, {"type": "shutdown"})
            try:
                coord_transport.gather(self.agent_ids, timeout_sec=grace_sec)
            except TimeoutError:
                errors.append("shutdown gather timed out")
        except Exception as exc:
            errors.append(f"shutdown broadcast failed: {exc!r}")

        deadline = time.time() + grace_sec
        for agent_id, proc in self.processes.items():
            remaining = max(0.0, deadline - time.time())
            try:
                proc.wait(timeout=remaining if remaining > 0 else 0.1)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    errors.append(f"agent {agent_id} (pid {proc.pid}) had to be killed")
        return errors

@contextmanager
def _coord_session(
    *,
    agent_ids: Sequence[str],
    coord_addr: str,
    serialization: str,
):
    coord = ZmqCoordinatorTransport(endpoint=coord_addr, serialization=serialization)
    coord.start(expected_agents=agent_ids)
    try:
        yield coord
    finally:
        coord.close()

def _run_inproc(
    *,
    method_name: str,
    numeric_df: pd.DataFrame,
    text_df: Optional[pd.DataFrame],
    websites: Sequence[str],
    memory_sample_interval: float,
    use_sbert: bool,
) -> Dict[str, Any]:
    sampler = _PeakRssSampler(memory_sample_interval)
    wall_start = time.perf_counter()
    sampler.start()
    use_text_path = (
        method_name in TEXT_CAPABLE_METHODS
        and text_df is not None
        and not text_df.empty
    )
    try:
        if use_text_path:
            source_scores, method_compute_sec = _run_text_method_by_name(
                method_name, text_df, use_sbert=use_sbert
            )
        else:
            source_scores, method_compute_sec = _run_numeric_method_by_name(
                method_name, numeric_df
            )
        success = True
        error_message = ""
        mean_score, scored_count = _mean_percentage_score(source_scores, websites, method_name)
    except Exception:
        success = False
        error_message = traceback.format_exc(limit=20)
        method_compute_sec = float("nan")
        mean_score = float("nan")
        scored_count = 0
    baseline_rss_mb, peak_rss_mb, peak_delta_rss_mb = sampler.stop()
    wall_elapsed_sec = time.perf_counter() - wall_start

    return {
        "method": method_name,
        "success": success,
        "error": error_message,
        "elapsed_sec": float(method_compute_sec),
        "method_compute_sec": float(method_compute_sec),
        "wall_elapsed_sec": float(wall_elapsed_sec),
        "baseline_rss_mb": float(baseline_rss_mb) if not pd.isna(baseline_rss_mb) else float("nan"),
        "peak_rss_mb": float(peak_rss_mb) if not pd.isna(peak_rss_mb) else float("nan"),
        "peak_delta_rss_mb": float(peak_delta_rss_mb) if not pd.isna(peak_delta_rss_mb) else float("nan"),
        "mean_source_score_pct": float(mean_score),
        "scored_website_count": int(scored_count),
        "coord_aggregate_sec": float("nan"),
        "network_sec": float("nan"),
        "rpc_round_trips": 0,
        "bytes_sent_total": float("nan"),
        "bytes_recv_total": float("nan"),
        "rpc_count": 0,
        "rpc_latency_p50_ms": float("nan"),
        "rpc_latency_p95_ms": float("nan"),
        "fleet_peak_rss_mb_max": float("nan"),
        "fleet_peak_rss_mb_sum": float("nan"),
        "fleet_peak_delta_rss_mb_max": float("nan"),
        "fleet_peak_delta_rss_mb_sum": float("nan"),
        "fleet_cpu_user_sec_sum": float("nan"),
        "fleet_cpu_system_sec_sum": float("nan"),
        "fleet_proc_count": 0,
    }

def _drain_hellos(coord, agent_ids: Sequence[str], timeout_sec: float) -> Dict[str, int]:

    pids: Dict[str, int] = {}
    deadline = time.time() + timeout_sec
    outstanding = set(agent_ids)
    while outstanding and time.time() < deadline:
        try:
            replies = coord.gather(list(outstanding), timeout_sec=max(0.05, deadline - time.time()))
        except TimeoutError:
            break
        for agent_id, msg in replies.items():
            if isinstance(msg, dict) and msg.get("type") == "hello":
                pids[agent_id] = int(msg.get("pid", 0))
            outstanding.discard(agent_id)
    if outstanding:
        raise RuntimeError(f"workers did not send hello within {timeout_sec}s: {sorted(outstanding)}")
    return pids

def _run_distributed(
    *,
    method_name: str,
    numeric_df: pd.DataFrame,
    text_df: Optional[pd.DataFrame],
    websites: Sequence[str],
    selected_agents: Sequence[str],
    coord,
    pool: WorkerPool,
    proc_sampler: MultiProcessSampler,
    memory_sample_interval: float,
    use_sbert: bool,
) -> Dict[str, Any]:
    coord.reset_stats()
    proc_sampler = MultiProcessSampler(interval_sec=memory_sample_interval)
    for agent_id, pid in pool.pids().items():
        if agent_id in set(selected_agents):
            proc_sampler.track(pid, label=agent_id)

    coord_baseline_sampler = _PeakRssSampler(memory_sample_interval)
    coord_baseline_sampler.start()
    proc_sampler.start()
    wall_start = time.perf_counter()

    try:
        protocol = METHOD_PROTOCOLS[method_name]
        runner = DistributedRunner(coord_transport=coord, protocol=protocol)

        use_text_path = (
            method_name in TEXT_CAPABLE_METHODS
            and text_df is not None
            and not text_df.empty
        )
        result = runner.run(
            agent_ids=selected_agents,
            numeric_df=numeric_df,
            text_df=text_df if use_text_path else None,
            use_sbert=use_sbert,
        )
        success = result.success
        error_message = result.error
        method_compute_sec = result.method_compute_sec
        coord_aggregate_sec = result.coord_aggregate_sec
        network_sec = result.network_sec
        rpc_round_trips = result.rpc_round_trips
        if success:
            mean_score, scored_count = _mean_percentage_score(
                result.source_scores, websites, method_name
            )
        else:
            mean_score = float("nan")
            scored_count = 0
    except Exception:
        success = False
        error_message = traceback.format_exc(limit=20)
        method_compute_sec = float("nan")
        coord_aggregate_sec = float("nan")
        network_sec = float("nan")
        rpc_round_trips = 0
        mean_score = float("nan")
        scored_count = 0

    wall_elapsed_sec = time.perf_counter() - wall_start
    baseline_rss_mb, peak_rss_mb, peak_delta_rss_mb = coord_baseline_sampler.stop()
    samples = proc_sampler.stop()
    fleet = aggregate_fleet(samples)
    transport_snapshot = coord.stats.snapshot()

    if success and not pd.isna(method_compute_sec) and not pd.isna(network_sec):
        elapsed_with_network = float(method_compute_sec) + float(network_sec)
    else:
        elapsed_with_network = float("nan")

    return {
        "method": method_name,
        "success": success,
        "error": error_message,
        "elapsed_sec": elapsed_with_network,
        "method_compute_sec": float(method_compute_sec),
        "wall_elapsed_sec": float(wall_elapsed_sec),
        "baseline_rss_mb": float(baseline_rss_mb) if not pd.isna(baseline_rss_mb) else float("nan"),
        "peak_rss_mb": float(peak_rss_mb) if not pd.isna(peak_rss_mb) else float("nan"),
        "peak_delta_rss_mb": float(peak_delta_rss_mb) if not pd.isna(peak_delta_rss_mb) else float("nan"),
        "mean_source_score_pct": float(mean_score),
        "scored_website_count": int(scored_count),
        "coord_aggregate_sec": float(coord_aggregate_sec),
        "network_sec": float(network_sec),
        "rpc_round_trips": int(rpc_round_trips),
        "bytes_sent_total": float(transport_snapshot["bytes_sent"]),
        "bytes_recv_total": float(transport_snapshot["bytes_recv"]),
        "rpc_count": int(transport_snapshot["rpc_count"]),
        "rpc_latency_p50_ms": float(transport_snapshot["rpc_latency_p50_ms"]),
        "rpc_latency_p95_ms": float(transport_snapshot["rpc_latency_p95_ms"]),
        "fleet_peak_rss_mb_max": float(fleet["fleet_peak_rss_mb_max"]),
        "fleet_peak_rss_mb_sum": float(fleet["fleet_peak_rss_mb_sum"]),
        "fleet_peak_delta_rss_mb_max": float(fleet["fleet_peak_delta_rss_mb_max"]),
        "fleet_peak_delta_rss_mb_sum": float(fleet["fleet_peak_delta_rss_mb_sum"]),
        "fleet_cpu_user_sec_sum": float(fleet["fleet_cpu_user_sec_sum"]),
        "fleet_cpu_system_sec_sum": float(fleet["fleet_cpu_system_sec_sum"]),
        "fleet_proc_count": int(fleet["fleet_proc_count"]),
    }

def _print_progress(
    *,
    mode: str,
    workload_index: int,
    agent_count: int,
    subset_index: int,
    total_subsets: int,
    method_name: str,
    result_row: Dict[str, Any],
):
    status = "OK" if result_row.get("success") else "FAIL"
    extras = ""
    if mode != "inproc":
        extras = (
            f" net={_format_float(result_row.get('network_sec'))}s"
            f" bytes={int(result_row.get('bytes_sent_total') or 0) + int(result_row.get('bytes_recv_total') or 0)}"
            f" p95={_format_float(result_row.get('rpc_latency_p95_ms'))}ms"
            f" rpc={int(result_row.get('rpc_count') or 0)}"
            f" fleetMem={_format_float(result_row.get('fleet_peak_delta_rss_mb_sum'))}MB"
        )
    print(
        f"[{mode}|workload {workload_index}] agents={agent_count} subset={subset_index}/{total_subsets} "
        f"method={method_name} status={status} time={_format_float(result_row.get('elapsed_sec'))}s "
        f"wall={_format_float(result_row.get('wall_elapsed_sec'))}s "
        f"memΔ={_format_float(result_row.get('peak_delta_rss_mb'))}MB"
        f"{extras}"
    )

def _build_unique_endpoint(base: str) -> str:
    if "ipc://" in base or base.endswith("*"):
        return base
    if base.endswith(":*") or "://*" in base:
        return base
    if base.count(":") >= 2 and not base.endswith(":0"):

        return base
    return base + (f"-{uuid.uuid4().hex[:8]}" if base.startswith("ipc://") else "")

def _summarize_time_memory(raw_results_df: pd.DataFrame) -> pd.DataFrame:

    df = raw_results_df.copy()
    numeric_columns = [
        "agent_count",
        "success",
        "elapsed_sec",
        "wall_elapsed_sec",
        "network_sec",
        "peak_rss_mb",
        "peak_delta_rss_mb",
        "fleet_peak_rss_mb_sum",
        "fleet_peak_delta_rss_mb_sum",
        "fleet_peak_delta_rss_mb_max",
    ]
    for column in numeric_columns:
        if column in df.columns and column != "success":
            df[column] = pd.to_numeric(df[column], errors="coerce")

    success_mask = df["success"].fillna(False).astype(bool)
    fleet_peak = (
        df["fleet_peak_rss_mb_sum"]
        if "fleet_peak_rss_mb_sum" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    fleet_delta = (
        df["fleet_peak_delta_rss_mb_sum"]
        if "fleet_peak_delta_rss_mb_sum" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    coord_peak = (
        df["peak_rss_mb"]
        if "peak_rss_mb" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    coord_delta = (
        df["peak_delta_rss_mb"]
        if "peak_delta_rss_mb" in df.columns
        else pd.Series(np.nan, index=df.index)
    )
    df["total_peak_rss_mb"] = coord_peak + fleet_peak.fillna(0.0)
    df.loc[coord_peak.isna(), "total_peak_rss_mb"] = np.nan
    df["total_peak_delta_rss_mb"] = coord_delta + fleet_delta.fillna(0.0)
    df.loc[coord_delta.isna(), "total_peak_delta_rss_mb"] = np.nan

    metric_columns = [
        "elapsed_sec",
        "wall_elapsed_sec",
        "network_sec",
        "peak_rss_mb",
        "peak_delta_rss_mb",
        "fleet_peak_rss_mb_sum",
        "fleet_peak_delta_rss_mb_sum",
        "fleet_peak_delta_rss_mb_max",
        "total_peak_rss_mb",
        "total_peak_delta_rss_mb",
    ]
    for column in metric_columns:
        if column in df.columns:
            df.loc[~success_mask, column] = np.nan

    summary_df = (
        df.groupby(["method", "agent_count"], sort=False)
        .agg(
            run_count=("method", "size"),
            success_count=("success", lambda series: int(pd.Series(series).fillna(False).sum())),
            mean_wall_time_sec=("wall_elapsed_sec", "mean"),
            std_wall_time_sec=("wall_elapsed_sec", "std"),
            mean_elapsed_with_comm_sec=("elapsed_sec", "mean"),
            mean_network_sec=("network_sec", "mean"),
            mean_coord_peak_rss_mb=("peak_rss_mb", "mean"),
            mean_coord_peak_delta_rss_mb=("peak_delta_rss_mb", "mean"),
            mean_fleet_peak_rss_mb_sum=("fleet_peak_rss_mb_sum", "mean"),
            mean_fleet_peak_delta_rss_mb_sum=("fleet_peak_delta_rss_mb_sum", "mean"),
            mean_fleet_peak_delta_rss_mb_max=("fleet_peak_delta_rss_mb_max", "mean"),
            mean_total_peak_rss_mb=("total_peak_rss_mb", "mean"),
            mean_total_peak_delta_rss_mb=("total_peak_delta_rss_mb", "mean"),
        )
        .reset_index()
    )
    summary_df["success_rate"] = summary_df["success_count"] / summary_df["run_count"]
    ordered_columns = [
        "method",
        "agent_count",
        "run_count",
        "success_count",
        "success_rate",
        "mean_wall_time_sec",
        "std_wall_time_sec",
        "mean_elapsed_with_comm_sec",
        "mean_network_sec",
        "mean_total_peak_rss_mb",
        "mean_total_peak_delta_rss_mb",
        "mean_coord_peak_rss_mb",
        "mean_coord_peak_delta_rss_mb",
        "mean_fleet_peak_rss_mb_sum",
        "mean_fleet_peak_delta_rss_mb_sum",
        "mean_fleet_peak_delta_rss_mb_max",
    ]
    return summary_df[ordered_columns]

def _run_benchmark(args: argparse.Namespace) -> int:
    input_json_path = os.path.abspath(args.input_json)
    all_batches = _load_batches(input_json_path)
    total_batches = len(all_batches)

    text_batches: Optional[List[Dict[str, Any]]] = None
    text_input_path = ""
    if args.text_mode:
        if not args.input_text_json:
            raise ValueError("--text-mode requires --input-text-json")
        text_input_path = os.path.abspath(args.input_text_json)
        text_batches = _load_batches(text_input_path)

    num_workloads = _resolve_num_workloads(
        args.num_workloads, total_batches, args.start_batch, args.batch_group_size
    )
    workload_specs = _build_workload_specs(
        total_batches,
        start_batch=args.start_batch,
        batch_group_size=args.batch_group_size,
        num_workloads=num_workloads,
    )
    first_workload_batches = _collect_workload_batches(all_batches, workload_specs[0])
    max_agents = len(_extract_agents_from_batches(first_workload_batches))
    agent_counts = _parse_agent_counts(args.agent_counts, max_agents)

    methods = (
        METHOD_NAMES
        if args.methods.strip().lower() == "all"
        else [token.strip() for token in args.methods.split(",") if token.strip()]
    )
    unsupported = [m for m in methods if m not in METHOD_NAMES]
    if unsupported:
        raise ValueError(f"Unsupported methods: {unsupported}")

    time_w, mem_w, qual_w = _normalize_weights(
        args.time_weight, args.memory_weight, args.quality_weight
    )

    experiment_id = (
        args.experiment_id
        or f"agent_scalability_dist_{args.mode}_{_sanitize_path_stem(input_json_path)}_{int(time.time())}"
    )
    output_dir = os.path.abspath(os.path.join(args.output_root, experiment_id))
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 108)
    print(f"Distributed agent scalability benchmark — mode={args.mode}")
    print("=" * 108)
    print(f"Input JSON      : {input_json_path}")
    print(f"Experiment ID   : {experiment_id}")
    print(f"Output dir      : {output_dir}")
    print(f"Workloads       : {len(workload_specs)} | batch_group_size={args.batch_group_size}")
    print(f"Agent counts    : {agent_counts}")
    print(f"Methods         : {methods}")
    print(f"Coord endpoint  : {args.coord_addr if args.mode != 'inproc' else 'n/a'}")
    print(f"Serialization   : {args.serialization}")
    print(f"netem delay (ms): {args.netem_delay_ms}")
    print(f"Text mode       : {bool(args.text_mode)} (use_sbert={bool(args.use_sbert)})")
    if args.text_mode:
        print(f"Input text JSON : {text_input_path}")
    print(
        f"Composite weights: time={time_w:.2f} memory={mem_w:.2f} quality={qual_w:.2f}"
    )

    raw_rows: List[Dict[str, Any]] = []

    for workload in workload_specs:
        workload_batches = _collect_workload_batches(all_batches, workload)
        workload_agents = _extract_agents_from_batches(workload_batches)
        text_workload_batches = (
            _collect_workload_batches(text_batches, workload)
            if text_batches is not None
            else None
        )
        for agent_count in agent_counts:
            if agent_count > len(workload_agents):
                continue
            subsets = _select_agent_subsets(
                workload_agents,
                agent_count=agent_count,
                subset_samples=args.subset_samples,
                base_seed=args.subset_seed,
                workload_index=workload.workload_index,
                strategy=args.selection_strategy,
            )
            for subset_offset, selected_agents in enumerate(subsets, start=1):
                filtered_df = _batches_to_numeric_df(
                    workload_batches, agent_subset=selected_agents
                )
                text_filtered_df: Optional[pd.DataFrame] = None
                if text_workload_batches is not None:
                    text_filtered_df = _batches_to_text_df(
                        text_workload_batches, agent_subset=selected_agents
                    )
                    if args.text_mode and text_filtered_df.empty:
                        raise ValueError(
                            "Text mode did not extract any string news-summary records "
                            f"from {text_input_path} for workload={workload.workload_index}, "
                            f"agent_count={agent_count}."
                        )
                if args.text_mode and text_filtered_df is not None and not text_filtered_df.empty:
                    websites = (
                        text_filtered_df["website"]
                        .astype(str)
                        .drop_duplicates()
                        .tolist()
                    )
                else:
                    websites = (
                        filtered_df["website"].astype(str).drop_duplicates().tolist()
                        if not filtered_df.empty
                        else list(selected_agents)
                    )
                rows_for_subset = _execute_subset(
                    args=args,
                    methods=methods,
                    workload=workload,
                    agent_count=agent_count,
                    subset_offset=subset_offset,
                    total_subsets=len(subsets),
                    selected_agents=selected_agents,
                    filtered_df=filtered_df,
                    text_df=text_filtered_df if args.text_mode else None,
                    websites=websites,
                )
                for row in rows_for_subset:
                    row["workload_index"] = workload.workload_index
                    row["batch_start"] = workload.batch_start
                    row["batch_end"] = workload.batch_end
                    row["batch_group_size"] = workload.batch_group_size
                    row["agent_count"] = agent_count
                    row["available_agent_count"] = len(workload_agents)
                    row["selected_agents"] = "|".join(selected_agents)
                    row["selected_agent_count"] = len(selected_agents)
                    row["subset_index"] = subset_offset
                    row["total_subsets_for_agent_count"] = len(subsets)
                    row["selection_strategy"] = args.selection_strategy
                    row["subset_seed"] = args.subset_seed
                    row["paired_numeric_input_item_count"] = int(filtered_df.shape[0])
                    row["paired_numeric_input_object_count"] = (
                        int(filtered_df["object"].nunique()) if not filtered_df.empty else 0
                    )
                    if args.text_mode and text_filtered_df is not None:
                        row["input_item_count"] = int(text_filtered_df.shape[0])
                        row["input_object_count"] = (
                            int(text_filtered_df["object"].nunique())
                            if not text_filtered_df.empty
                            else 0
                        )
                    else:
                        row["input_item_count"] = int(filtered_df.shape[0])
                        row["input_object_count"] = (
                            int(filtered_df["object"].nunique()) if not filtered_df.empty else 0
                        )
                    row["experiment_id"] = experiment_id
                    row["input_json"] = input_json_path
                    row["input_text_json"] = text_input_path if args.text_mode else ""
                    row["mode"] = args.mode
                    row["transport"] = "zmq-tcp" if args.mode == "dist-tcp" else "inproc"
                    row["serialization"] = args.serialization if args.mode != "inproc" else ""
                    row["coord_addr"] = args.coord_addr if args.mode != "inproc" else ""
                    row["netem_delay_ms"] = args.netem_delay_ms
                    row["text_mode"] = bool(args.text_mode)
                    row["use_sbert"] = bool(args.use_sbert)
                    raw_rows.append(row)

    if not raw_rows:
        print("No runs were produced; check workload/agent_count configuration.")
        return 1

    raw_results_df = pd.DataFrame(raw_rows)
    scored_results_df = _compute_group_scores(
        raw_results_df,
        time_weight=time_w,
        memory_weight=mem_w,
        quality_weight=qual_w,
    )
    method_agent_summary_df = _summarize_by_method_agent(scored_results_df)
    overall_summary_df = _summarize_overall(scored_results_df, method_agent_summary_df)
    time_memory_summary_df = _summarize_time_memory(raw_results_df)
    effective_top_k = args.hybrid_top_k if HYBRID_METHOD_NAME in methods else 0
    hybrid_check = _evaluate_hybrid_requirement(
        overall_summary_df,
        hybrid_method_name=HYBRID_METHOD_NAME,
        top_k=effective_top_k,
    )

    raw_results_path = os.path.join(output_dir, "raw_runs.csv")
    scored_results_path = os.path.join(output_dir, "scored_runs.csv")
    method_agent_summary_path = os.path.join(output_dir, "summary_by_method_agent_count.csv")
    overall_summary_path = os.path.join(output_dir, "summary_overall.csv")
    time_memory_summary_path = os.path.join(output_dir, "summary_time_memory.csv")
    benchmark_summary_path = os.path.join(output_dir, "benchmark_summary.json")

    raw_results_df.to_csv(raw_results_path, index=False)
    scored_results_df.to_csv(scored_results_path, index=False)
    method_agent_summary_df.to_csv(method_agent_summary_path, index=False)
    overall_summary_df.to_csv(overall_summary_path, index=False)
    time_memory_summary_df.to_csv(time_memory_summary_path, index=False)
    _write_json(
        benchmark_summary_path,
        {
            "experiment_id": experiment_id,
            "mode": args.mode,
            "config": {
                "input_json": input_json_path,
                "start_batch": args.start_batch,
                "batch_group_size": args.batch_group_size,
                "num_workloads": len(workload_specs),
                "agent_counts": agent_counts,
                "subset_samples": args.subset_samples,
                "subset_seed": args.subset_seed,
                "selection_strategy": args.selection_strategy,
                "methods": methods,
                "weights": {"time": time_w, "memory": mem_w, "quality": qual_w},
                "hybrid_top_k": effective_top_k,
                "transport": "zmq-tcp" if args.mode == "dist-tcp" else "inproc",
                "serialization": args.serialization,
                "coord_addr": args.coord_addr,
                "netem_delay_ms": args.netem_delay_ms,
                "text_mode": bool(args.text_mode),
                "input_text_json": text_input_path,
                "use_sbert": bool(args.use_sbert),
            },
            "workloads": [asdict(w) for w in workload_specs],
            "hybrid_check": hybrid_check,
            "artifacts": {
                "raw_runs_csv": raw_results_path,
                "scored_runs_csv": scored_results_path,
                "summary_by_method_agent_count_csv": method_agent_summary_path,
                "summary_overall_csv": overall_summary_path,
                "summary_time_memory_csv": time_memory_summary_path,
            },
        },
    )

    print("\nTime/memory summary:")
    summary_columns = [
        "method",
        "agent_count",
        "success_count",
        "run_count",
        "mean_wall_time_sec",
        "mean_elapsed_with_comm_sec",
        "mean_network_sec",
        "mean_total_peak_delta_rss_mb",
        "mean_coord_peak_delta_rss_mb",
        "mean_fleet_peak_delta_rss_mb_sum",
    ]
    print(time_memory_summary_df[summary_columns].round(4).to_string(index=False))
    print(f"\nHybrid check: {hybrid_check['message']}")
    print(f"raw runs              : {raw_results_path}")
    print(f"scored runs           : {scored_results_path}")
    print(f"method-agent summary  : {method_agent_summary_path}")
    print(f"overall summary       : {overall_summary_path}")
    print(f"time/memory summary   : {time_memory_summary_path}")
    print(f"benchmark summary     : {benchmark_summary_path}")

    return 0 if hybrid_check.get("passed", False) else 2

def _execute_subset(
    *,
    args: argparse.Namespace,
    methods: Sequence[str],
    workload: WorkloadSpec,
    agent_count: int,
    subset_offset: int,
    total_subsets: int,
    selected_agents: Sequence[str],
    filtered_df: pd.DataFrame,
    text_df: Optional[pd.DataFrame],
    websites: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    text_payload = text_df if args.text_mode and text_df is not None else None
    if args.mode == "inproc":
        for method_name in methods:
            row = _run_inproc(
                method_name=method_name,
                numeric_df=filtered_df.copy(),
                text_df=text_payload.copy() if text_payload is not None else None,
                websites=websites,
                memory_sample_interval=args.memory_sample_interval,
                use_sbert=bool(args.use_sbert),
            )
            _print_progress(
                mode=args.mode,
                workload_index=workload.workload_index,
                agent_count=agent_count,
                subset_index=subset_offset,
                total_subsets=total_subsets,
                method_name=method_name,
                result_row=row,
            )
            rows.append(row)
        return rows

    if args.mode != "dist-tcp":
        raise ValueError(f"Unknown mode: {args.mode}")

    coord_addr = args.coord_addr

    def _needs_worker_sbert(method_name: str) -> bool:
        return bool(
            args.text_mode
            and args.use_sbert
            and method_name in {"LASOTruth", "SenteTruth"}
        )

    method_groups: List[Tuple[bool, List[str]]] = []
    for method_name in methods:
        preload_sbert = _needs_worker_sbert(method_name)
        if method_groups and method_groups[-1][0] == preload_sbert:
            method_groups[-1][1].append(method_name)
        else:
            method_groups.append((preload_sbert, [method_name]))

    for preload_sbert, grouped_methods in method_groups:
        pool = WorkerPool(
            agent_ids=selected_agents,
            coord_addr=coord_addr,
            serialization=args.serialization,
            python_executable=sys.executable,
            preload_sbert=preload_sbert,
            spawn_stagger_sec=args.worker_spawn_stagger_sec,
        )
        with _coord_session(
            agent_ids=selected_agents,
            coord_addr=coord_addr,
            serialization=args.serialization,
        ) as coord:
            pool.start()
            try:
                _drain_hellos(coord, selected_agents, timeout_sec=args.worker_startup_timeout_sec)
                sampler_seed = MultiProcessSampler(interval_sec=args.memory_sample_interval)
                for method_name in grouped_methods:
                    row = _run_distributed(
                        method_name=method_name,
                        numeric_df=filtered_df.copy(),
                        text_df=text_payload.copy() if text_payload is not None else None,
                        websites=websites,
                        selected_agents=selected_agents,
                        coord=coord,
                        pool=pool,
                        proc_sampler=sampler_seed,
                        memory_sample_interval=args.memory_sample_interval,
                        use_sbert=bool(args.use_sbert),
                    )
                    _print_progress(
                        mode=args.mode,
                        workload_index=workload.workload_index,
                        agent_count=agent_count,
                        subset_index=subset_offset,
                        total_subsets=total_subsets,
                        method_name=method_name,
                        result_row=row,
                    )
                    rows.append(row)
            finally:
                errors = pool.shutdown(coord)
                for err in errors:
                    print(f"[shutdown warning] {err}", file=sys.stderr)

        time.sleep(0.05)
    return rows

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-process / cross-machine scalability benchmark for the five "
            "truth-discovery methods. Companion to exp5 with a real socket transport."
        ),
    )
    parser.add_argument("--input-json", default=DEFAULT_INPUT_JSON)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--experiment-id", default="")
    parser.add_argument("--mode", choices=["inproc", "dist-tcp"], default="dist-tcp")
    parser.add_argument(
        "--coord-addr",
        default="tcp://127.0.0.1:5555",
        help="ZeroMQ endpoint coordinator binds (and workers connect to).",
    )
    parser.add_argument(
        "--serialization",
        default="msgpack",
        choices=["msgpack", "pickle", "json"],
    )
    parser.add_argument("--start-batch", type=_positive_int, default=1)
    parser.add_argument("--batch-group-size", type=_positive_int, default=10)
    parser.add_argument("--num-workloads", type=_non_negative_int, default=1)
    parser.add_argument("--agent-counts", default="auto")
    parser.add_argument("--subset-samples", type=_positive_int, default=2)
    parser.add_argument("--subset-seed", type=int, default=20260407)
    parser.add_argument("--selection-strategy", choices=["random", "prefix"], default="random")
    parser.add_argument("--methods", default="all")
    parser.add_argument("--time-weight", type=_bounded_float, default=0.45)
    parser.add_argument("--memory-weight", type=_bounded_float, default=0.35)
    parser.add_argument("--quality-weight", type=_bounded_float, default=0.20)
    parser.add_argument("--hybrid-top-k", type=_non_negative_int, default=3)
    parser.add_argument("--memory-sample-interval", type=float, default=0.05)
    parser.add_argument(
        "--worker-startup-timeout-sec",
        type=float,
        default=20.0,
        help="Maximum wait for all workers to send HELLO after spawn.",
    )
    parser.add_argument(
        "--netem-delay-ms",
        type=float,
        default=0.0,
        help=(
            "Annotation only — record the loopback delay you injected via "
            "`sudo tc qdisc add dev lo root netem delay Xms` so the CSV labels are accurate."
        ),
    )
    parser.add_argument(
        "--text-mode",
        action="store_true",
        help=(
            "Run text-mode workloads (BasicTruth/LASOTruth/SenteTruth) "
            "instead of the numeric pipeline. Requires --input-text-json."
        ),
    )
    parser.add_argument(
        "--input-text-json",
        default="",
        help=(
            "Path to a text-style demo JSON (each item's `response` is a string). "
            "Required when --text-mode is set."
        ),
    )
    parser.add_argument(
        "--use-sbert",
        action="store_true",
        help=(
            "Use the real SBERT encoder in text-mode (default falls back to "
            "TF-IDF). When set in dist-tcp mode, every worker preloads the "
            "model so per-process RSS reflects the duplication cost."
        ),
    )
    parser.add_argument(
        "--worker-spawn-stagger-sec",
        type=float,
        default=0.0,
        help=(
            "Sleep this many seconds between consecutive worker spawns. Useful "
            "for large agent counts in text-mode where many simultaneous SBERT "
            "cold loads would thunder-herd the CPU and miss the HELLO timeout."
        ),
    )
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    return _run_benchmark(args)

if __name__ == "__main__":
    raise SystemExit(main())
