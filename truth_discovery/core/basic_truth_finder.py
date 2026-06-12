
import numpy as np
from numpy.linalg import norm
import math
import pandas as pd
from typing import Dict, Optional, Callable, Union, List
from sklearn.feature_extraction.text import TfidfVectorizer

from .zk_proof import ZKProofEngine, attach_grouped_proofs
from .text_vectorization import build_tfidf_vectorizer

def sigmoid(x):

    return 1 / (1 + math.exp(-x))

class BasicTruthFinder:

    def __init__(self,
                 implication: Callable[[str, str], float],
                 dampening_factor: float = 0.3,
                 influence_related: float = 0.5,
                 enable_zk_proof: bool = True,
                 zk_proof_secret: Optional[str] = None):

        assert 0 < dampening_factor < 1, "dampening_factor must be in (0, 1)"
        assert 0 <= influence_related <= 1, "influence_related must be in [0, 1]"

        self.implication = implication
        self.dampening_factor = dampening_factor
        self.influence_related = influence_related
        self.enable_zk_proof = enable_zk_proof
        self.last_iteration_count = 0
        self.website_trustworthiness: Dict[str, float] = {}
        self.zk_proof_engine = ZKProofEngine(
            prover_id="BasicTruthFinder",
            secret_seed=zk_proof_secret,
        )

    def calculate_confidence(self, df: pd.DataFrame) -> pd.DataFrame:

        def trustworthiness_score(x):

            x = min(0.999, max(0.001, x))
            return -math.log(1 - x)

        for i, row in df.iterrows():

            ts = df.loc[df["fact"] == row["fact"], "trustworthiness"]

            v = sum(trustworthiness_score(t) for t in ts)
            df.at[i, "fact_confidence"] = v

        return df

    def adjust_confidence(self, df: pd.DataFrame) -> pd.DataFrame:

        update = {}

        for i, row1 in df.iterrows():
            f1 = row1["fact"]
            s = 0

            for j, row2 in df.drop_duplicates("fact").iterrows():
                f2 = row2["fact"]
                if f1 == f2:
                    continue

                s += row2["fact_confidence"] * self.implication(f2, f1)

            update[i] = self.influence_related * s + row1["fact_confidence"]

        for i, row1 in df.iterrows():
            df.at[i, "fact_confidence"] = update[i]

        return df

    def compute_fact_confidence(self, df: pd.DataFrame) -> pd.DataFrame:

        f = lambda x: sigmoid(self.dampening_factor * x)
        for i, row in df.iterrows():
            df.at[i, "fact_confidence"] = f(row["fact_confidence"])
        return df

    def update_fact_confidence(self, df: pd.DataFrame) -> pd.DataFrame:

        for object_ in df["object"].unique():
            indices = df["object"] == object_
            d = df.loc[indices].copy()
            d = self.calculate_confidence(d)
            d = self.adjust_confidence(d)
            df.loc[indices] = self.compute_fact_confidence(d)
        return df

    def update_website_trustworthiness(self, df: pd.DataFrame) -> pd.DataFrame:

        for website in df["website"].unique():
            indices = df["website"] == website
            cs = df.loc[indices, "fact_confidence"]

            trustworthiness = sum(cs) / len(cs)
            trustworthiness = max(0.001, min(0.999, trustworthiness))
            df.loc[indices, "trustworthiness"] = trustworthiness
        return df

    def iteration(self, df: pd.DataFrame) -> pd.DataFrame:

        df = self.update_fact_confidence(df)
        df = self.update_website_trustworthiness(df)
        return df

    def _estimate_global_truth_text(self, dataframe: pd.DataFrame) -> Dict[str, str]:

        truth_map: Dict[str, str] = {}
        for object_id in dataframe["object"].unique():
            obj_df = dataframe[dataframe["object"] == object_id]
            if obj_df.empty:
                continue
            best_idx = obj_df["fact_confidence"].astype(float).idxmax()
            truth_map[str(object_id)] = str(dataframe.loc[best_idx, "fact"])
        return truth_map

    def _initialize_trustworthiness(
        self,
        dataframe: pd.DataFrame,
        initial_trustworthiness: Union[float, Dict[str, float]],
        default_trustworthiness: float,
    ) -> pd.Series:
        if isinstance(initial_trustworthiness, dict):
            values = dataframe["website"].astype(str).map(
                lambda website: float(initial_trustworthiness.get(str(website), default_trustworthiness))
            )
        else:
            values = pd.Series(
                np.ones(len(dataframe.index)) * float(initial_trustworthiness),
                index=dataframe.index,
                dtype=float,
            )
        return values.clip(lower=0.001, upper=0.999)

    def process_batch(
        self,
        dataframe: pd.DataFrame,
        implication_texts: Optional[List[str]] = None,
        initial_trustworthiness: float = 0.5,
    ) -> pd.DataFrame:

        if dataframe is None or dataframe.empty:
            columns = ["website", "fact", "object", "trustworthiness", "fact_confidence", "global_truth"]
            if dataframe is not None:
                columns = list(dict.fromkeys(list(dataframe.columns) + columns))
            return pd.DataFrame(columns=columns)

        dataframe = dataframe.copy()
        dataframe["website"] = dataframe["website"].astype(str)
        dataframe["fact"] = dataframe["fact"].astype(str)
        dataframe["object"] = dataframe["object"].astype(str)

        visible_texts = implication_texts
        if visible_texts is None:
            visible_texts = dataframe["fact"].tolist()
        visible_texts = [str(text) for text in visible_texts]
        if visible_texts:
            self.implication = create_implication_function_from_texts(visible_texts)

        init = self.website_trustworthiness if self.website_trustworthiness else initial_trustworthiness
        dataframe["trustworthiness"] = self._initialize_trustworthiness(
            dataframe,
            init,
            initial_trustworthiness,
        )
        dataframe["fact_confidence"] = np.zeros(len(dataframe.index))

        out = self.iteration(dataframe)
        self.last_iteration_count = 1

        truth_map = self._estimate_global_truth_text(out)
        out["global_truth"] = out["object"].map(truth_map)

        latest_scores = (
            out[["website", "trustworthiness"]]
            .drop_duplicates(subset=["website"], keep="last")
            .set_index("website")["trustworthiness"]
            .astype(float)
            .to_dict()
        )
        self.website_trustworthiness.update(latest_scores)
        out["source_reliability"] = out["website"].map(
            lambda website: float(self.website_trustworthiness[str(website)])
        )

        if self.enable_zk_proof:
            out = attach_grouped_proofs(
                out,
                method_name="BasicTruthFinder",
                source_scores=latest_scores,
                group_columns=("object",),
                truth_column="global_truth",
                iteration_count=self.last_iteration_count,
                proof_engine=self.zk_proof_engine,
            )
        return out

    def get_source_reliability(self, website: Optional[str] = None) -> Dict:

        if website is not None:
            if str(website) not in self.website_trustworthiness:
                return {}
            return {"trustworthiness": float(self.website_trustworthiness[str(website)])}
        return {
            str(source): {"trustworthiness": float(score)}
            for source, score in self.website_trustworthiness.items()
        }

    def reset_history(self) -> None:
        self.website_trustworthiness = {}

def create_implication_function_from_texts(texts: list,
                                          vectorizer: Optional[TfidfVectorizer] = None) -> Callable[[str, str], float]:

    if vectorizer is None:
        vectorizer = build_tfidf_vectorizer(texts, stop_words=None)
        vectorizer.fit(texts)

    def similarity(w1: str, w2: str) -> float:

        V = vectorizer.transform([w1, w2])
        v1, v2 = np.asarray(V.todense())
        dot_product = np.dot(v1, v2)
        norm1 = norm(v1)
        norm2 = norm(v2)

        if norm1 > 0 and norm2 > 0:
            return dot_product / (norm1 * norm2)
        else:
            return 0.0

    def implication(f1: str, f2: str) -> float:

        return similarity(f1.lower(), f2.lower())

    return implication
