
import numpy as np
from numpy.linalg import norm
import os
import pandas as pd
from typing import Dict, Optional, Union, List
import warnings
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from .zk_proof import ZKProofEngine, attach_grouped_proofs
from .text_vectorization import build_tfidf_vectorizer

HAS_SBERT = True

class SenteTruthFinder:

    def __init__(self,
                 initial_credibility: float = 0.5,
                 sbert_model: Optional[str] = None,
                 use_sbert: bool = True,
                 strict_sbert: bool = False,
                 local_files_only: Optional[bool] = None,
                 enable_zk_proof: bool = True,
                 zk_proof_secret: Optional[str] = None):

        self.initial_credibility = initial_credibility
        self.use_sbert = use_sbert and HAS_SBERT
        self.strict_sbert = strict_sbert
        if local_files_only is None:
            local_files_only = (
                os.environ.get("HF_HUB_OFFLINE", "0") == "1"
                or os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
            )
        self.local_files_only = local_files_only
        self.enable_zk_proof = enable_zk_proof
        self.last_iteration_count = 0
        self.zk_proof_engine = ZKProofEngine(
            prover_id="SenteTruthFinder",
            secret_seed=zk_proof_secret,
        )

        self.node_credibility: Dict[str, float] = {}

        self.global_truth_history: Dict[str, str] = {}

        if self.use_sbert:
            model_name = (
                sbert_model
                or os.environ.get("TRUTHFINDER_SBERT_MODEL")
                or 'paraphrase-multilingual-MiniLM-L12-v2'
            )
            try:
                model_source = model_name
                if self.local_files_only and not os.path.isdir(model_name):
                    repo_id = model_name if "/" in model_name else f"sentence-transformers/{model_name}"
                    model_source = snapshot_download(repo_id, local_files_only=True)
                old_hf_offline = os.environ.get("HF_HUB_OFFLINE")
                old_tf_offline = os.environ.get("TRANSFORMERS_OFFLINE")
                if self.local_files_only:
                    os.environ["HF_HUB_OFFLINE"] = "1"
                    os.environ["TRANSFORMERS_OFFLINE"] = "1"
                try:
                    self.encoder = SentenceTransformer(
                        model_source,
                        local_files_only=self.local_files_only,
                    )
                finally:
                    if self.local_files_only:
                        if old_hf_offline is None:
                            os.environ.pop("HF_HUB_OFFLINE", None)
                        else:
                            os.environ["HF_HUB_OFFLINE"] = old_hf_offline
                        if old_tf_offline is None:
                            os.environ.pop("TRANSFORMERS_OFFLINE", None)
                        else:
                            os.environ["TRANSFORMERS_OFFLINE"] = old_tf_offline
                print(f"Using SBERT model: {model_name}")
            except Exception as e:
                if self.strict_sbert:
                    raise RuntimeError(
                        f"Failed to load SBERT model {model_name} while strict_sbert=True: {e}"
                    ) from e
                warnings.warn(f"Failed to load SBERT model {model_name}: {e}. Falling back to TF-IDF.")
                self.use_sbert = False
                self.encoder = None
        else:
            self.encoder = None

        if not self.use_sbert:
            self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
            print("Using TF-IDF as fallback for LASO encoding")

    def _encode_texts(self, texts: List[str]) -> np.ndarray:

        if self.use_sbert and self.encoder is not None:

            embeddings = self.encoder.encode(texts, convert_to_numpy=True)
            return embeddings
        else:

            self.tfidf_vectorizer = build_tfidf_vectorizer(texts, max_features=1000)
            vectors = self.tfidf_vectorizer.fit_transform(texts)
            return vectors.toarray()

    def _calculate_LASO_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:

        n = len(embeddings)
        if n == 0:
            return np.zeros((0, 0), dtype=float)

        embeddings = np.asarray(embeddings, dtype=float)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = np.divide(
            embeddings,
            norms,
            out=np.zeros_like(embeddings, dtype=float),
            where=norms > 0,
        )
        similarity_matrix = normalized @ normalized.T
        similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)
        np.fill_diagonal(similarity_matrix, 1.0)
        return similarity_matrix

    def _calculate_trustworthiness(self, similarity_matrix: np.ndarray, node_idx: int) -> float:

        n = len(similarity_matrix)
        if n <= 1:
            return 1.0

        similarities = []
        for j in range(n):
            if j != node_idx:
                similarities.append(similarity_matrix[node_idx, j])

        if len(similarities) > 0:
            phi = np.mean(similarities)
        else:
            phi = 1.0

        return float(phi)

    def _calculate_trustworthiness_scores(self, similarity_matrix: np.ndarray) -> List[float]:
        return [
            self._calculate_trustworthiness(similarity_matrix, idx)
            for idx in range(len(similarity_matrix))
        ]

    def _aggregate_truth(self,
                        df: pd.DataFrame,
                        embeddings: np.ndarray,
                        similarity_matrix: np.ndarray) -> Dict[str, str]:

        if df.empty:
            return {}

        object_id = df['object'].iloc[0]
        n = len(df)

        phi_values = self._calculate_trustworthiness_scores(similarity_matrix)

        scores = []
        texts = []

        for idx, (_, row) in enumerate(df.iterrows()):
            website = row['website']
            text = str(row['fact'])

            C_i = self.node_credibility.get(website, self.initial_credibility)

            phi_i = phi_values[idx]

            score = C_i * phi_i

            scores.append(score)
            texts.append(text)

        best_idx = np.argmax(scores)
        truth_text = texts[best_idx]

        return {object_id: truth_text}

    def _update_credibility(self,
                           df: pd.DataFrame,
                           embeddings: np.ndarray,
                           similarity_matrix: np.ndarray,
                           global_truth: Dict[str, str]):

        if df.empty:
            return

        phi_values = self._calculate_trustworthiness_scores(similarity_matrix)
        credibilities = []
        trustworthiness_scores = []

        for idx, (_, row) in enumerate(df.iterrows()):
            website = row['website']
            C_i = self.node_credibility.get(website, self.initial_credibility)
            phi_i = phi_values[idx]

            credibilities.append(C_i)
            trustworthiness_scores.append(phi_i)

        sum_C = sum(credibilities)
        sum_C_phi = sum(C_i * phi_i for C_i, phi_i in zip(credibilities, trustworthiness_scores))

        for idx, (_, row) in enumerate(df.iterrows()):
            website = row['website']
            C_i = self.node_credibility.get(website, self.initial_credibility)
            phi_i = phi_values[idx]

            if abs(sum_C_phi) > 1e-12:
                new_C_i = (sum_C / sum_C_phi) * C_i * phi_i
            else:

                new_C_i = C_i

            self.node_credibility[website] = new_C_i

    def _initialize_credibility(self, dataframe: pd.DataFrame):

        websites = dataframe['website'].unique()
        for website in websites:
            if website not in self.node_credibility:
                self.node_credibility[website] = self.initial_credibility

    def iteration(self, dataframe: pd.DataFrame) -> pd.DataFrame:

        self._initialize_credibility(dataframe)

        all_truth = {}

        for object_id in dataframe['object'].unique():
            obj_data = dataframe[dataframe['object'] == object_id].copy()

            texts = [str(row['fact']) for _, row in obj_data.iterrows()]

            embeddings = self._encode_texts(texts)

            similarity_matrix = self._calculate_LASO_similarity_matrix(embeddings)

            truth = self._aggregate_truth(obj_data, embeddings, similarity_matrix)
            all_truth.update(truth)

            self._update_credibility(obj_data, embeddings, similarity_matrix, all_truth)

        self.global_truth_history.update({str(k): str(v) for k, v in all_truth.items()})

        dataframe['source_reliability'] = dataframe['website'].map(
            lambda w: self.node_credibility.get(w, self.initial_credibility)
        )
        dataframe['global_truth'] = dataframe['object'].map(all_truth)

        return dataframe

    def process_batch(self,
                      dataframe: pd.DataFrame,
                      epoch: Optional[int] = None) -> pd.DataFrame:

        if dataframe is None or dataframe.empty:
            columns = ["website", "fact", "object", "source_reliability", "global_truth"]
            if dataframe is not None:
                columns = list(dict.fromkeys(list(dataframe.columns) + columns))
            return pd.DataFrame(columns=columns)

        dataframe = dataframe.copy()
        dataframe['fact'] = dataframe['fact'].astype(str)
        if "batch_index" not in dataframe.columns and epoch is not None:
            dataframe["batch_index"] = int(epoch)

        dataframe = self.iteration(dataframe)
        self.last_iteration_count = 1

        if self.enable_zk_proof:
            source_scores = {website: float(score) for website, score in self.node_credibility.items()}
            dataframe = attach_grouped_proofs(
                dataframe,
                method_name="SenteTruthFinder",
                source_scores=source_scores,
                group_columns=("object",),
                truth_column="global_truth",
                iteration_count=self.last_iteration_count,
                proof_engine=self.zk_proof_engine,
                extra_public_data={"use_sbert": bool(self.use_sbert)},
            )

        return dataframe

    def get_source_reliability(self, website: Optional[str] = None) -> Dict:

        if website is not None:
            return {'credibility': self.node_credibility.get(website, self.initial_credibility)}
        else:
            return {w: {'credibility': c} for w, c in self.node_credibility.items()}

    def reset_history(self):

        self.node_credibility = {}
        self.global_truth_history = {}
