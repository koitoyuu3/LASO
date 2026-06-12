
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import pandas as pd

from .zk_proof import ZKProofEngine, attach_grouped_proofs

class SenFeedTruthDiscovery:

    def __init__(
        self,
        gamma: float = 1.0,
        map_shape_a: float = 2.0,
        map_scale_beta: float = 1.0,
        history_truth_weight: float = 1.0,
        initial_weight: float = 1.0,
        weight_floor: float = 1e-12,
        composite_mode: bool = True,
        distance_mode: str = "relative",
        weight_evolution_epsilon: float = 1.0,
        iterative_max_iters: int = 50,
        iterative_convergence_tol: float = 1e-6,
        use_history_in_iterative: bool = True,
        enable_zk_proof: bool = True,
        zk_proof_secret: Optional[str] = None,
    ):
        assert gamma > 0, "gamma must be > 0"
        assert map_shape_a > 1.0, "map_shape_a should be > 1 to keep numerator positive"
        assert map_scale_beta > 0, "map_scale_beta must be > 0"
        assert history_truth_weight >= 0, "history_truth_weight must be >= 0"
        assert initial_weight > 0, "initial_weight must be > 0"
        assert weight_floor > 0, "weight_floor must be > 0"
        assert distance_mode in {"relative", "absolute"}, "distance_mode must be 'relative' or 'absolute'"
        assert weight_evolution_epsilon > 0, "weight_evolution_epsilon must be > 0"
        assert iterative_max_iters >= 1, "iterative_max_iters must be >= 1"
        assert iterative_convergence_tol > 0, "iterative_convergence_tol must be > 0"

        self.gamma = gamma
        self.map_shape_a = map_shape_a
        self.map_scale_beta = map_scale_beta
        self.history_truth_weight = history_truth_weight
        self.initial_weight = initial_weight
        self.weight_floor = weight_floor
        self.composite_mode = composite_mode
        self.distance_mode = distance_mode
        self.weight_evolution_epsilon = weight_evolution_epsilon
        self.iterative_max_iters = iterative_max_iters
        self.iterative_convergence_tol = iterative_convergence_tol
        self.use_history_in_iterative = use_history_in_iterative
        self.enable_zk_proof = enable_zk_proof

        self.source_weights: Dict[str, float] = {}

        self.previous_source_weights: Dict[str, float] = {}

        self.previous_truth: Dict[str, float] = {}

        self.cumulative_squared_error: Dict[str, float] = {}

        self.cumulative_obs_count: Dict[str, int] = {}

        self.last_iteration_count: int = 0
        self.last_weight_evolution: Dict[str, float] = {}
        self.last_td_mode: str = "incremental"
        self.zk_proof_engine = ZKProofEngine(
            prover_id="SenFeedTruthDiscovery",
            secret_seed=zk_proof_secret,
        )

    def reset_history(self):

        self.source_weights = {}
        self.previous_source_weights = {}
        self.previous_truth = {}
        self.cumulative_squared_error = {}
        self.cumulative_obs_count = {}
        self.last_iteration_count = 0
        self.last_weight_evolution = {}
        self.last_td_mode = "incremental"

    def get_source_reliability(self, website: Optional[str] = None) -> Dict:

        if website is not None:
            return {"weight": self.source_weights.get(website, self.initial_weight)}
        return {k: {"weight": v} for k, v in self.source_weights.items()}

    def train(self, dataframe: pd.DataFrame) -> pd.DataFrame:

        df = self._prepare_dataframe(dataframe)
        self._initialize_sources(df)

        result_chunks = []
        unique_timestamps = sorted(df["timestamp"].unique())

        for timestamp in unique_timestamps:
            current_df = df[df["timestamp"] == timestamp].copy()
            truth_map, _, _, final_weights = self._run_composite_timestamp(current_df)

            self.previous_truth = truth_map.copy()
            self.previous_source_weights = {
                str(source): float(weight)
                for source, weight in final_weights.items()
            }

            current_df["global_truth"] = current_df["object"].map(truth_map)
            current_df["source_weight"] = current_df["website"].map(
                lambda s: self.source_weights.get(s, self.initial_weight)
            )
            result_chunks.append(current_df)

        result_df = pd.concat(result_chunks, ignore_index=True)

        if self.enable_zk_proof:
            result_df = attach_grouped_proofs(
                result_df,
                method_name="SenFeedTruthDiscovery",
                source_scores=self.source_weights,
                group_columns=("timestamp", "object"),
                truth_column="global_truth",
                iteration_count=self.last_iteration_count,
                proof_engine=self.zk_proof_engine,
            )

        return result_df

    def _prepare_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        required = {"website", "fact", "object"}
        missing = required - set(dataframe.columns)
        if missing:
            raise ValueError(f"missing required columns: {sorted(missing)}")

        df = dataframe.copy()
        if "timestamp" not in df.columns:
            df["timestamp"] = 0

        df["fact"] = pd.to_numeric(df["fact"], errors="coerce")
        if df["fact"].isna().any():
            bad_rows = df[df["fact"].isna()].head(5)
            raise ValueError(
                "fact column must be numeric for SenFEED truth discovery.\n"
                f"sample invalid rows:\n{bad_rows}"
            )

        df = df.sort_values(["timestamp", "object", "website"]).reset_index(drop=True)
        return df

    def _initialize_sources(self, df: pd.DataFrame):
        for source in df["website"].unique():
            if source not in self.source_weights:
                self.source_weights[source] = self.initial_weight
            if source not in self.previous_source_weights:
                self.previous_source_weights[source] = self.initial_weight
            if source not in self.cumulative_squared_error:
                self.cumulative_squared_error[source] = 0.0
            if source not in self.cumulative_obs_count:
                self.cumulative_obs_count[source] = 0

    def _distance(self, value: float, truth: float) -> float:
        residual = float(value) - float(truth)
        if self.distance_mode == "absolute":
            return residual * residual

        scale = max(abs(float(truth)), 1e-8)
        normalized = residual / scale
        return normalized * normalized

    def _compute_weighted_truth(
        self,
        object_df: pd.DataFrame,
        include_history_truth: bool,
        weight_map: Optional[Dict[str, float]] = None,
    ) -> float:

        object_id = str(object_df["object"].iloc[0])
        numerator = 0.0
        denominator = 0.0

        for _, row in object_df.iterrows():
            source = str(row["website"])
            value = float(row["fact"])
            source_weights = self.source_weights if weight_map is None else weight_map
            w_ts = max(self.weight_floor, source_weights.get(source, self.initial_weight))

            numerator += w_ts * value
            denominator += w_ts

        if include_history_truth and object_id in self.previous_truth:
            numerator += self.history_truth_weight * float(self.previous_truth[object_id])
            denominator += self.history_truth_weight

        if denominator <= self.weight_floor:
            return float(object_df["fact"].mean())
        return float(numerator / denominator)

    def _estimate_truth_map(
        self,
        df_t: pd.DataFrame,
        include_history_truth: bool,
        weight_map: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        truth_map: Dict[str, float] = {}
        for object_id in df_t["object"].unique():
            obj_df = df_t[df_t["object"] == object_id]
            truth_map[str(object_id)] = self._compute_weighted_truth(
                obj_df,
                include_history_truth,
                weight_map=weight_map,
            )
        return truth_map

    def _collect_error_statistics(
        self, df_t: pd.DataFrame, truth_map: Dict[str, float]
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, int]]:
        abs_error_sum: Dict[str, float] = {}
        squared_error_sum: Dict[str, float] = {}
        observation_count: Dict[str, int] = {}

        for _, row in df_t.iterrows():
            source = str(row["website"])
            object_id = str(row["object"])
            value = float(row["fact"])
            truth = float(truth_map[object_id])
            dist = self._distance(value, truth)

            abs_error_sum[source] = abs_error_sum.get(source, 0.0) + math.sqrt(dist)
            squared_error_sum[source] = squared_error_sum.get(source, 0.0) + dist
            observation_count[source] = observation_count.get(source, 0) + 1

        return abs_error_sum, squared_error_sum, observation_count

    def _compute_batch_weight_update(
        self,
        squared_err: Dict[str, float],
        obs_count: Dict[str, int],
    ) -> Dict[str, float]:
        updated: Dict[str, float] = {}
        for source, count in obs_count.items():
            numerator = (2.0 * self.map_shape_a - 2.0) + float(count)
            denominator = (2.0 * self.map_scale_beta) + self.gamma * float(squared_err.get(source, 0.0))
            if denominator <= self.weight_floor:
                updated[source] = self.weight_floor
            else:
                updated[source] = max(self.weight_floor, float(numerator / denominator))
        return updated

    def _compute_incremental_candidate(
        self,
        df_t: pd.DataFrame,
        include_history_truth: bool,
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, int]]:
        truth_map = self._estimate_truth_map(df_t, include_history_truth=include_history_truth)
        _, squared_err, obs_count = self._collect_error_statistics(df_t, truth_map)

        candidate_weights: Dict[str, float] = {}
        for source in [str(s) for s in df_t["website"].unique()]:
            cumulative_squared_error = (
                self.cumulative_squared_error.get(source, 0.0) + squared_err.get(source, 0.0)
            )
            cumulative_obs_count = self.cumulative_obs_count.get(source, 0) + obs_count.get(source, 0)

            numerator = (2.0 * self.map_shape_a - 2.0) + float(cumulative_obs_count)
            denominator = (2.0 * self.map_scale_beta) + self.gamma * cumulative_squared_error
            if denominator <= self.weight_floor:
                candidate_weights[source] = self.weight_floor
            else:
                candidate_weights[source] = max(self.weight_floor, float(numerator / denominator))

        return truth_map, candidate_weights, squared_err, obs_count

    def _commit_statistics(
        self,
        squared_err: Dict[str, float],
        obs_count: Dict[str, int],
    ) -> None:
        for source, count in obs_count.items():
            self.cumulative_squared_error[source] = (
                self.cumulative_squared_error.get(source, 0.0) + squared_err.get(source, 0.0)
            )
            self.cumulative_obs_count[source] = self.cumulative_obs_count.get(source, 0) + int(count)

    def _compute_weight_evolution(
        self,
        df_t: pd.DataFrame,
        candidate_weights: Dict[str, float],
    ) -> Dict[str, float]:
        evolution: Dict[str, float] = {}
        for source in [str(s) for s in df_t["website"].unique()]:
            previous = float(self.previous_source_weights.get(source, self.initial_weight))
            current = float(candidate_weights.get(source, previous))
            evolution[source] = abs(current - previous)
        return evolution

    def _weight_evolution_threshold(self, n_sources: int) -> float:
        return float(self.weight_evolution_epsilon) / max(int(n_sources), 1)

    def _should_run_iterative(
        self,
        df_t: pd.DataFrame,
        weight_evolution: Dict[str, float],
    ) -> bool:
        if not self.composite_mode:
            return False
        if not self.previous_truth:
            return True

        threshold = self._weight_evolution_threshold(df_t["website"].nunique())
        return any(delta > threshold for delta in weight_evolution.values())

    def _run_iterative_truth_discovery(
        self,
        df_t: pd.DataFrame,
        include_history_truth: bool,
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, int], Dict[str, float], int]:
        active_sources = [str(source) for source in df_t["website"].unique()]
        working_weights: Dict[str, float] = {
            source: max(self.weight_floor, float(self.source_weights.get(source, self.initial_weight)))
            for source in active_sources
        }

        truth_map: Dict[str, float] = {}
        squared_err: Dict[str, float] = {}
        obs_count: Dict[str, int] = {}
        iteration_count = 0

        for iteration in range(1, self.iterative_max_iters + 1):
            truth_map = self._estimate_truth_map(
                df_t,
                include_history_truth=include_history_truth,
                weight_map=working_weights,
            )
            _, squared_err, obs_count = self._collect_error_statistics(df_t, truth_map)
            updated_weights = self._compute_batch_weight_update(squared_err, obs_count)

            max_delta = max(
                abs(updated_weights.get(source, working_weights[source]) - working_weights[source])
                for source in active_sources
            )
            working_weights = {
                source: max(self.weight_floor, float(updated_weights.get(source, working_weights[source])))
                for source in active_sources
            }
            iteration_count = iteration
            if max_delta <= self.iterative_convergence_tol:
                break

        return truth_map, squared_err, obs_count, working_weights, iteration_count

    def _run_incremental_map_update(
        self, df_t: pd.DataFrame
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, int]]:

        truth_map, candidate_weights, squared_err, obs_count = self._compute_incremental_candidate(
            df_t,
            include_history_truth=True,
        )
        self._commit_statistics(squared_err, obs_count)
        for source, weight in candidate_weights.items():
            self.source_weights[source] = max(self.weight_floor, float(weight))
        self.last_iteration_count = 1
        return truth_map, squared_err, obs_count

    def _run_composite_timestamp(
        self,
        df_t: pd.DataFrame,
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, int], Dict[str, float]]:
        if not self.composite_mode:
            truth_map, squared_err, obs_count = self._run_incremental_map_update(df_t)
            final_weights = {
                str(source): float(self.source_weights.get(str(source), self.initial_weight))
                for source in df_t["website"].unique()
            }
            self.last_weight_evolution = {
                source: abs(final_weights[source] - self.previous_source_weights.get(source, self.initial_weight))
                for source in final_weights
            }
            self.last_td_mode = "incremental"
            return truth_map, squared_err, obs_count, final_weights

        truth_map, candidate_weights, squared_err, obs_count = self._compute_incremental_candidate(
            df_t,
            include_history_truth=bool(self.previous_truth),
        )
        weight_evolution = self._compute_weight_evolution(df_t, candidate_weights)
        self.last_weight_evolution = weight_evolution

        if self._should_run_iterative(df_t, weight_evolution):
            truth_map, squared_err, obs_count, final_weights, iteration_count = self._run_iterative_truth_discovery(
                df_t,
                include_history_truth=bool(self.previous_truth) and self.use_history_in_iterative,
            )
            for source, weight in final_weights.items():
                self.source_weights[source] = max(self.weight_floor, float(weight))
            self.last_iteration_count = iteration_count
            self.last_td_mode = "iterative"
        else:
            for source, weight in candidate_weights.items():
                self.source_weights[source] = max(self.weight_floor, float(weight))
            final_weights = {
                str(source): float(candidate_weights.get(str(source), self.initial_weight))
                for source in df_t["website"].unique()
            }
            self.last_iteration_count = 1
            self.last_td_mode = "incremental"

        self._commit_statistics(squared_err, obs_count)
        return truth_map, squared_err, obs_count, final_weights
