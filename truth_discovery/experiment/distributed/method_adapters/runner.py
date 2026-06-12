
from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import pandas as pd

def _get_inproc_method_runner():
    mod = importlib.import_module(
        "truth_discovery.experiment.agent_scalability_benchmark"
    )
    return getattr(mod, "_run_numeric_method_by_name")

def _get_text_method_runner():
    mod = importlib.import_module(
        "truth_discovery.experiment.agent_scalability_benchmark"
    )
    return getattr(mod, "_run_text_method_by_name")

@dataclass(frozen=True)
class MethodProtocol:

    method_name: str
    broadcast_reliability: bool = True

    extra_round_trips: int = 0

    per_batch_gather: bool = True

METHOD_PROTOCOLS: Dict[str, MethodProtocol] = {
    "BasicTruth": MethodProtocol(
        method_name="BasicTruth",
        broadcast_reliability=True,
        extra_round_trips=0,
        per_batch_gather=True,
    ),
    "DecentTruth": MethodProtocol(
        method_name="DecentTruth",
        broadcast_reliability=True,
        extra_round_trips=0,
        per_batch_gather=True,
    ),
    "SenFeedTruth": MethodProtocol(
        method_name="SenFeedTruth",
        broadcast_reliability=True,
        extra_round_trips=0,
        per_batch_gather=True,
    ),
    "SenteTruth": MethodProtocol(
        method_name="SenteTruth",
        broadcast_reliability=True,
        extra_round_trips=0,
        per_batch_gather=True,
    ),
    "LASOTruth": MethodProtocol(
        method_name="LASOTruth",
        broadcast_reliability=True,
        extra_round_trips=0,
        per_batch_gather=True,
    ),
}

@dataclass
class RunnerResult:
    success: bool
    error: str
    method_compute_sec: float
    wall_elapsed_sec: float
    coord_aggregate_sec: float
    network_sec: float
    source_scores: Dict[str, float] = field(default_factory=dict)
    agent_compute_seconds: Dict[str, float] = field(default_factory=dict)
    rpc_round_trips: int = 0

class DistributedRunner:

    def __init__(self, coord_transport, protocol: MethodProtocol):
        self.coord = coord_transport
        self.protocol = protocol

    def run(
        self,
        *,
        agent_ids: Sequence[str],
        numeric_df: pd.DataFrame,
        text_df: pd.DataFrame | None = None,
        use_sbert: bool = False,
    ) -> RunnerResult:

        wall_start = time.perf_counter()
        rpc_rounds = 0
        agent_compute: Dict[str, float] = {agent: 0.0 for agent in agent_ids}
        text_mode = text_df is not None and not text_df.empty
        try:
            if text_mode:
                self._gather_text_workload(
                    agent_ids=agent_ids,
                    text_df=text_df,
                    agent_compute=agent_compute,
                )
            else:
                self._gather_workload(
                    agent_ids=agent_ids,
                    numeric_df=numeric_df,
                    agent_compute=agent_compute,
                )
            rpc_rounds += 1

            for _ in range(self.protocol.extra_round_trips):
                self._extra_round_trip(agent_ids)
                rpc_rounds += 1

            aggregate_start = time.perf_counter()
            if text_mode:
                text_runner = _get_text_method_runner()
                source_scores, method_compute_sec = text_runner(
                    self.protocol.method_name, text_df, use_sbert=use_sbert
                )
            else:
                inproc_runner = _get_inproc_method_runner()
                source_scores, method_compute_sec = inproc_runner(
                    self.protocol.method_name, numeric_df
                )
            coord_aggregate_sec = time.perf_counter() - aggregate_start

            if self.protocol.broadcast_reliability:
                self._broadcast_reliability(agent_ids, source_scores)
                rpc_rounds += 1

            wall_elapsed_sec = time.perf_counter() - wall_start
            network_sec = max(wall_elapsed_sec - coord_aggregate_sec, 0.0)
            return RunnerResult(
                success=True,
                error="",
                method_compute_sec=float(method_compute_sec),
                wall_elapsed_sec=float(wall_elapsed_sec),
                coord_aggregate_sec=float(coord_aggregate_sec),
                network_sec=float(network_sec),
                source_scores=source_scores,
                agent_compute_seconds=agent_compute,
                rpc_round_trips=rpc_rounds,
            )
        except Exception as exc:
            wall_elapsed_sec = time.perf_counter() - wall_start
            return RunnerResult(
                success=False,
                error=f"{type(exc).__name__}: {exc}",
                method_compute_sec=float("nan"),
                wall_elapsed_sec=float(wall_elapsed_sec),
                coord_aggregate_sec=float("nan"),
                network_sec=float("nan"),
                rpc_round_trips=rpc_rounds,
            )

    def _gather_workload(
        self,
        *,
        agent_ids: Sequence[str],
        numeric_df: pd.DataFrame,
        agent_compute: Dict[str, float],
    ) -> None:
        slices = self._slice_by_agent(numeric_df, agent_ids)
        for agent_id in agent_ids:
            payload = {
                "type": "process_batch",
                "method": self.protocol.method_name,
                "items": slices.get(agent_id, []),
            }
            self.coord.send(agent_id, payload)
        replies = self.coord.gather(agent_ids)
        for agent_id, reply in replies.items():
            if isinstance(reply, dict):
                local_compute = reply.get("local_compute_sec", 0.0)
                try:
                    agent_compute[agent_id] = float(local_compute)
                except (TypeError, ValueError):
                    agent_compute[agent_id] = 0.0

    def _gather_text_workload(
        self,
        *,
        agent_ids: Sequence[str],
        text_df: pd.DataFrame,
        agent_compute: Dict[str, float],
    ) -> None:
        slices = self._slice_text_by_agent(text_df, agent_ids)
        for agent_id in agent_ids:
            payload = {
                "type": "process_text_batch",
                "method": self.protocol.method_name,
                "items": slices.get(agent_id, []),
            }
            self.coord.send(agent_id, payload)
        replies = self.coord.gather(agent_ids)
        for agent_id, reply in replies.items():
            if isinstance(reply, dict):
                local_compute = reply.get("local_compute_sec", 0.0)
                try:
                    agent_compute[agent_id] = float(local_compute)
                except (TypeError, ValueError):
                    agent_compute[agent_id] = 0.0

    def _extra_round_trip(self, agent_ids: Sequence[str]) -> None:
        payload = {"type": "iterate", "method": self.protocol.method_name}
        for agent_id in agent_ids:
            self.coord.send(agent_id, payload)
        self.coord.gather(agent_ids)

    def _broadcast_reliability(
        self,
        agent_ids: Sequence[str],
        source_scores: Dict[str, float],
    ) -> None:
        payload = {
            "type": "reliability_update",
            "method": self.protocol.method_name,
            "reliability": {str(k): float(v) for k, v in source_scores.items()},
        }
        self.coord.broadcast(agent_ids, payload)

        self.coord.gather(agent_ids)

    @staticmethod
    def _slice_by_agent(
        numeric_df: pd.DataFrame,
        agent_ids: Sequence[str],
    ) -> Dict[str, List[List[Any]]]:
        slices: Dict[str, List[List[Any]]] = {agent_id: [] for agent_id in agent_ids}
        if numeric_df is None or numeric_df.empty:
            return slices

        for row in numeric_df.itertuples(index=False):
            agent = str(getattr(row, "website"))
            if agent not in slices:
                continue
            slices[agent].append(
                [
                    str(getattr(row, "object")),
                    float(getattr(row, "fact")),
                    int(getattr(row, "batch_index")),
                ]
            )
        return slices

    @staticmethod
    def _slice_text_by_agent(
        text_df: pd.DataFrame,
        agent_ids: Sequence[str],
    ) -> Dict[str, List[List[Any]]]:
        slices: Dict[str, List[List[Any]]] = {agent_id: [] for agent_id in agent_ids}
        if text_df is None or text_df.empty:
            return slices
        for row in text_df.itertuples(index=False):
            agent = str(getattr(row, "website"))
            if agent not in slices:
                continue
            slices[agent].append(
                [
                    str(getattr(row, "object")),
                    str(getattr(row, "fact")),
                    int(getattr(row, "batch_index")),
                ]
            )
        return slices
