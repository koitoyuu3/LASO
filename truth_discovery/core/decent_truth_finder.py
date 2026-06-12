
import numpy as np
from scipy.special import erfc
import pandas as pd
from typing import Dict, List, Optional, Tuple

from .zk_proof import ZKProofEngine, attach_grouped_proofs

class EnhancedTruthFinder:

    def __init__(self,
                 alpha: float = 0.9,
                 beta: float = 100.0,
                 initial_reliability: float = 0.5,
                 enable_zk_proof: bool = True,
                 zk_proof_secret: Optional[str] = None):

        self.alpha = alpha
        self.beta = beta
        self.initial_reliability = initial_reliability
        self.enable_zk_proof = enable_zk_proof
        self.zk_proof_engine = ZKProofEngine(
            prover_id="EnhancedTruthFinder",
            secret_seed=zk_proof_secret,
        )

        self.source_reliability_history: Dict[str, Dict[str, float]] = {}

        self.global_truth_history: Dict[int, Dict[str, float]] = {}
        self.last_iteration_count: int = 0

    def _initialize_source_reliability(self, dataframe: pd.DataFrame) -> None:

        for website in dataframe['website'].unique():
            if website not in self.source_reliability_history:
                self.source_reliability_history[website] = {
                    'r': self.initial_reliability,
                    'e': 0.0,
                    'k': 1.0,
                }

    def _build_local_estimates(self, dataframe: pd.DataFrame) -> Dict[str, Dict[str, float]]:

        local_estimates: Dict[str, Dict[str, float]] = {}
        for _, row in dataframe.iterrows():
            if pd.isna(row['fact']):
                continue
            local_estimates.setdefault(str(row['website']), {})[str(row['object'])] = float(row['fact'])
        return local_estimates

    def _initialize_global_truth(
        self, local_estimates: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:

        epsilon = 1e-10
        weighted: Dict[str, List[Tuple[float, float]]] = {}
        for website, object_values in local_estimates.items():
            r = max(epsilon, float(
                self.source_reliability_history.get(website, {}).get('r', self.initial_reliability)
            ))
            for object_id, value in object_values.items():
                weighted.setdefault(object_id, []).append((value, r))

        x_hat: Dict[str, float] = {}
        for object_id, value_weights in weighted.items():
            total_w = sum(w for _, w in value_weights)
            x_hat[object_id] = (
                sum(v * w for v, w in value_weights) / total_w
                if total_w > 0
                else float(np.mean([v for v, _ in value_weights]))
            )
        return x_hat

    def _run_global_td(self, dataframe: pd.DataFrame) -> Dict[str, float]:

        local_estimates = self._build_local_estimates(dataframe)
        if not local_estimates:
            return {}
        return self._initialize_global_truth(local_estimates)

    def _update_source_reliability(
        self,
        dataframe: pd.DataFrame,
        global_truth: Dict[str, float],
    ) -> None:

        epsilon = 1e-10
        for website in dataframe['website'].unique():
            if website not in self.source_reliability_history:
                self.source_reliability_history[website] = {
                    'r': self.initial_reliability, 'e': 0.0, 'k': 1.0,
                }

            source_data = dataframe[dataframe['website'] == website]
            errors = [
                (float(row['fact']) - float(global_truth[str(row['object'])])) ** 2
                for _, row in source_data.iterrows()
                if str(row['object']) in global_truth
            ]
            if not errors:
                continue

            prev_e = self.source_reliability_history[website]['e']
            prev_k = self.source_reliability_history[website]['k']

            new_e = float(np.mean(errors))
            new_k = erfc(self.beta * abs(new_e - prev_e))
            k_updated = self.alpha * prev_k + (1 - self.alpha) * new_k
            new_r = k_updated / (new_e + epsilon)

            self.source_reliability_history[website]['e'] = new_e
            self.source_reliability_history[website]['k'] = k_updated
            self.source_reliability_history[website]['r'] = new_r

    def _update_dataframe(
        self,
        dataframe: pd.DataFrame,
        global_truth: Dict[str, float],
    ) -> None:

        dataframe['source_reliability'] = dataframe['website'].map(
            lambda w: self.source_reliability_history.get(w, {}).get('r', self.initial_reliability)
        )
        dataframe['error_metric'] = dataframe['website'].map(
            lambda w: self.source_reliability_history.get(w, {}).get('e', 0.0)
        )
        dataframe['consistency_metric'] = dataframe['website'].map(
            lambda w: self.source_reliability_history.get(w, {}).get('k', 1.0)
        )
        dataframe['global_truth'] = dataframe['object'].map(global_truth)

    def iteration(self, dataframe: pd.DataFrame) -> pd.DataFrame:

        global_truth = self._run_global_td(dataframe)
        self._update_source_reliability(dataframe, global_truth)
        self._update_dataframe(dataframe, global_truth)
        return dataframe

    def process_batch(
        self,
        dataframe: pd.DataFrame,
        epoch: Optional[int] = None,
    ) -> pd.DataFrame:

        dataframe = dataframe.copy()
        dataframe['fact'] = pd.to_numeric(dataframe['fact'], errors='raise')

        self._initialize_source_reliability(dataframe)
        dataframe = self.iteration(dataframe)
        self.last_iteration_count = 1

        global_truth = (
            dataframe[["object", "global_truth"]]
            .drop_duplicates(subset=["object"])
            .set_index("object")["global_truth"]
            .astype(float)
            .to_dict()
        )
        if epoch is not None:
            self.global_truth_history[epoch] = global_truth

        if self.enable_zk_proof:
            source_scores = {w: float(v['r']) for w, v in self.source_reliability_history.items()}
            dataframe = attach_grouped_proofs(
                dataframe,
                method_name="EnhancedTruthFinder",
                source_scores=source_scores,
                group_columns=("object",),
                truth_column="global_truth",
                iteration_count=self.last_iteration_count,
                proof_engine=self.zk_proof_engine,
            )

        return dataframe

    def get_source_reliability(self, website: Optional[str] = None) -> Dict:

        if website is not None:
            return self.source_reliability_history.get(website, {}).copy()
        return {w: v.copy() for w, v in self.source_reliability_history.items()}

    def reset_history(self) -> None:

        self.source_reliability_history = {}
        self.global_truth_history = {}
        self.last_iteration_count = 0
