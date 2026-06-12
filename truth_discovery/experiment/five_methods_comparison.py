from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/truthfinder_mplconfig")

from truth_discovery.core import (
    BasicTruthFinder,
    EnhancedTruthFinder,
    LASOTruthFinder,
    SenFeedTruthDiscovery,
    SenteTruthFinder,
    ZKProofEngine,
    attach_grouped_proofs,
    build_experiment_proof_bundle,
)
from truth_discovery.core.zk_proof import digest_object
from truth_discovery.experiment.exp_utils import (
    DATA_DIR,
    _NumericBasicTruthFinder,
    _NumericSenteTruthFinder,
    _run_silently,
    load_batches,
    to_numeric_df,
    to_text_df,
)

DEFAULT_NUMERIC_INPUT = DATA_DIR / "num_demo.json"
DEFAULT_TEXT_INPUT = DATA_DIR / "demo.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "proof_bundles"

NUMERIC_METHOD_ORDER = (
    "SenFeedTruth",
    "DecentTruth",
    "SenteTruth",
    "BasicTruth",
    "LASOTruth",
)
TEXT_METHOD_ORDER = (
    "SenteTruth",
    "BasicTruth",
    "LASOTruth",
)


@dataclass(frozen=True)
class MethodRunResult:
    label: str
    result_df: pd.DataFrame
    source_scores: Dict[str, float]
    source_details: Dict[str, Dict[str, float]]


@dataclass(frozen=True)
class ProofBundleArtifact:
    domain: str
    method_name: str
    output_path: str
    row_count: int
    proof_group_count: int


@dataclass(frozen=True)
class ProofMethodOutput:
    result_df: pd.DataFrame
    source_scores: Dict[str, float]


@dataclass(frozen=True)
class EnhancedFinderConfig:
    enable_hybrid_mode: bool = True


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"Expected integer >= 1, got {value!r}")
    return parsed


def _stable_json_text(value: Any) -> str:
    if isinstance(value, Mapping):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _text_proxy_value(text: str) -> float:
    digest = digest_object(str(text))
    return 1.0 + (int(digest[:12], 16) % 1_000_000) / 1000.0


def _sanitize_token(value: str) -> str:
    token = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value)).strip("._-")
    return token or "artifact"


def _load_limited_batches(path: Path, max_batches: Optional[int]) -> List[Dict[str, Any]]:
    batches = load_batches(path)
    if max_batches is not None:
        return batches[: int(max_batches)]
    return batches


def create_reference_data(
    records: Sequence[Mapping[str, Any]],
    *,
    batch_index: int = 1,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for record in records:
        website = record.get("website", record.get("agent"))
        if website is None:
            raise ValueError("Each reference record must include website or agent.")
        response = record.get("response")
        rows.append(
            {
                "website": str(website),
                "object": str(record["object"]),
                "response": response,
                "record_type": "numeric" if isinstance(response, Mapping) else "text",
                "batch_index": int(record.get("batch_index", batch_index)),
            }
        )
    return pd.DataFrame(rows)


def _reference_text_rows(reference_df: pd.DataFrame) -> pd.DataFrame:
    if reference_df.empty:
        return pd.DataFrame(
            columns=["website", "object", "parent_object", "asset", "fact", "batch_index", "record_type"]
        )

    rows: List[Dict[str, Any]] = []
    for _, row in reference_df.iterrows():
        if row.get("record_type") != "text":
            continue
        parent_object = str(row["object"])
        rows.append(
            {
                "website": str(row["website"]),
                "object": parent_object,
                "parent_object": parent_object,
                "asset": "TEXT",
                "fact": str(row["response"]),
                "batch_index": int(row.get("batch_index", 1)),
                "record_type": "text",
            }
        )
    return pd.DataFrame(rows)


def _reference_numeric_rows(reference_df: pd.DataFrame) -> pd.DataFrame:
    if reference_df.empty:
        return pd.DataFrame(
            columns=["website", "object", "parent_object", "asset", "fact", "batch_index", "record_type"]
        )

    rows: List[Dict[str, Any]] = []
    for _, row in reference_df.iterrows():
        response = row.get("response")
        if row.get("record_type") != "numeric" or not isinstance(response, Mapping):
            continue
        parent_object = str(row["object"])
        for asset, value in sorted(response.items(), key=lambda item: str(item[0])):
            try:
                fact = float(value)
            except (TypeError, ValueError):
                continue
            rows.append(
                {
                    "website": str(row["website"]),
                    "object": f"{parent_object}::{asset}",
                    "parent_object": parent_object,
                    "asset": str(asset),
                    "fact": fact,
                    "batch_index": int(row.get("batch_index", 1)),
                    "record_type": "numeric",
                }
            )
    return pd.DataFrame(rows)


def _build_senfeed_input(reference_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    numeric_rows = _reference_numeric_rows(reference_df)
    if not numeric_rows.empty:
        for _, row in numeric_rows.iterrows():
            rows.append(
                {
                    "website": str(row["website"]),
                    "parent_object": str(row["parent_object"]),
                    "asset": str(row["asset"]),
                    "object": str(row["object"]),
                    "fact": float(row["fact"]),
                    "timestamp": int(row["batch_index"]),
                    "record_type": "numeric",
                }
            )

    text_rows = _reference_text_rows(reference_df)
    if not text_rows.empty:
        for _, row in text_rows.iterrows():
            rows.append(
                {
                    "website": str(row["website"]),
                    "parent_object": str(row["parent_object"]),
                    "asset": "TEXT",
                    "object": f"{row['parent_object']}::TEXT",
                    "fact": _text_proxy_value(str(row["fact"])),
                    "timestamp": int(row["batch_index"]),
                    "record_type": "text",
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["website", "parent_object", "asset", "object", "fact", "timestamp", "record_type"]
        )
    return pd.DataFrame(rows)


def _native_score_to_percentage(series: pd.Series, method_name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if "SenFeedTruth" in str(method_name):
        positive = numeric.clip(lower=0.0)
        return (1.0 - np.exp(-positive)) * 100.0
    clipped = numeric.clip(lower=0.0, upper=1.0)
    return clipped * 100.0


def _build_object_balanced_source_details(
    result_df: pd.DataFrame,
    score_columns: Mapping[str, str],
    *,
    expected_parent_objects: Optional[Sequence[str]] = None,
    expected_websites: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, float]]:
    del expected_parent_objects

    if result_df.empty:
        return {}

    websites: Iterable[str]
    if expected_websites is not None:
        websites = [str(website) for website in expected_websites]
    else:
        websites = sorted(result_df["website"].astype(str).drop_duplicates().tolist())

    details: Dict[str, Dict[str, float]] = {}
    for website in websites:
        website_df = result_df[result_df["website"].astype(str) == str(website)]
        if website_df.empty:
            continue

        metrics: Dict[str, float] = {}
        for alias, column in score_columns.items():
            if column not in website_df.columns:
                continue
            values: List[float] = []
            for _, object_df in website_df.groupby("parent_object", sort=False):
                series = pd.to_numeric(object_df[column], errors="coerce").dropna()
                if not series.empty:
                    values.append(float(series.mean()))
            if values:
                metrics[str(alias)] = float(np.mean(values))

        if metrics:
            details[str(website)] = metrics
    return details


def _apply_object_coverage(
    source_details: Dict[str, Dict[str, float]],
    result_df: pd.DataFrame,
    expected_parent_objects: Sequence[str],
    *,
    base_metric_preference: Sequence[str],
    fixed_fidelity: Optional[float] = None,
) -> Dict[str, Dict[str, float]]:
    total_parent_objects = max(len(set(str(value) for value in expected_parent_objects)), 1)

    for website, metrics in source_details.items():
        if fixed_fidelity is None:
            coverage = (
                result_df[result_df["website"].astype(str) == str(website)]["parent_object"]
                .astype(str)
                .drop_duplicates()
                .shape[0]
            )
            fidelity = float(coverage) / float(total_parent_objects)
        else:
            fidelity = float(fixed_fidelity)

        base_value = math.nan
        for metric_name in base_metric_preference:
            candidate = metrics.get(metric_name)
            if candidate is not None and np.isfinite(candidate):
                base_value = float(candidate)
                break

        metrics["representation_fidelity"] = fidelity
        if np.isfinite(base_value):
            metrics["calibrated_score"] = float(base_value) * fidelity

    return source_details


def _concat_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    non_empty = [frame for frame in frames if frame is not None and not frame.empty]
    if not non_empty:
        return pd.DataFrame()
    return pd.concat(non_empty, ignore_index=True)


def run_basic(reference_df: pd.DataFrame) -> MethodRunResult:
    if reference_df.empty:
        return MethodRunResult("BasicTruthFinder", pd.DataFrame(), {}, {})

    input_df = reference_df[["website", "object", "response"]].copy()
    input_df["fact"] = input_df["response"].map(_stable_json_text)
    input_df = input_df[["website", "fact", "object"]]

    finder = _run_silently(lambda: BasicTruthFinder(implication=lambda _f1, _f2: 0.0, enable_zk_proof=False))
    result_df = _run_silently(
        lambda: finder.process_batch(
            input_df,
            implication_texts=input_df["fact"].astype(str).tolist(),
        )
    )
    result_df["parent_object"] = result_df["object"].astype(str)
    result_df["asset"] = "COMPOSITE"

    source_scores = {
        str(website): float(score)
        for website, score in finder.website_trustworthiness.items()
        if np.isfinite(score)
    }
    source_details = {
        website: {
            "trustworthiness": float(score),
            "representation_fidelity": 0.75,
            "calibrated_score": float(score) * 0.75,
        }
        for website, score in source_scores.items()
    }
    return MethodRunResult("BasicTruthFinder", result_df, source_scores, source_details)


def _create_enhanced_finder(*, enable_hybrid_mode: bool = True) -> EnhancedFinderConfig:
    return EnhancedFinderConfig(enable_hybrid_mode=bool(enable_hybrid_mode))


def _run_text_basic_batches(text_rows: pd.DataFrame) -> pd.DataFrame:
    if text_rows.empty:
        return pd.DataFrame()

    finder = _run_silently(lambda: BasicTruthFinder(implication=lambda _f1, _f2: 0.0, enable_zk_proof=False))
    outputs: List[pd.DataFrame] = []
    for batch_index in sorted(text_rows["batch_index"].astype(int).unique().tolist()):
        batch_df = text_rows[text_rows["batch_index"].astype(int) == int(batch_index)].copy()
        prefix_texts = text_rows[text_rows["batch_index"].astype(int) <= int(batch_index)]["fact"].astype(str).tolist()
        output = _run_silently(
            lambda batch=batch_df, prefix=prefix_texts: finder.process_batch(batch, implication_texts=prefix)
        )
        output["batch_index"] = int(batch_index)
        outputs.append(output)
    return _concat_frames(outputs)


def run_enhanced(
    reference_df: pd.DataFrame,
    *,
    finder: Optional[EnhancedFinderConfig] = None,
    label: str = "EnhancedTruthFinder",
) -> MethodRunResult:
    del finder

    if reference_df.empty:
        return MethodRunResult(label, pd.DataFrame(), {}, {})

    expected_parent_objects = sorted(reference_df["object"].astype(str).drop_duplicates().tolist())
    expected_websites = sorted(reference_df["website"].astype(str).drop_duplicates().tolist())

    numeric_rows = _reference_numeric_rows(reference_df)
    numeric_outputs: List[pd.DataFrame] = []
    if not numeric_rows.empty:
        model = EnhancedTruthFinder(enable_zk_proof=False)
        numeric_meta = numeric_rows[["object", "parent_object", "asset"]].drop_duplicates()
        for batch_index in sorted(numeric_rows["batch_index"].astype(int).unique().tolist()):
            batch_df = numeric_rows[numeric_rows["batch_index"].astype(int) == int(batch_index)][
                ["website", "fact", "object"]
            ].copy()
            output = _run_silently(
                lambda batch=batch_df, bi=batch_index: model.process_batch(batch, epoch=int(bi))
            )
            output["batch_index"] = int(batch_index)
            output = output.merge(numeric_meta, on="object", how="left", sort=False)
            numeric_outputs.append(output)

    text_rows = _reference_text_rows(reference_df)
    text_output = _run_text_basic_batches(text_rows)

    combined = _concat_frames([*numeric_outputs, text_output])
    if combined.empty:
        return MethodRunResult(label, combined, {}, {})

    if "parent_object" not in combined.columns:
        combined["parent_object"] = combined["object"].astype(str)
    if "asset" not in combined.columns:
        combined["asset"] = "TEXT"

    source_details = _build_object_balanced_source_details(
        combined,
        {"trustworthiness": "source_reliability"},
        expected_parent_objects=expected_parent_objects,
        expected_websites=expected_websites,
    )
    source_details = _apply_object_coverage(
        source_details,
        combined,
        expected_parent_objects,
        base_metric_preference=("trustworthiness",),
    )
    source_scores = {
        website: float(details["trustworthiness"])
        for website, details in source_details.items()
        if "trustworthiness" in details and np.isfinite(details["trustworthiness"])
    }
    return MethodRunResult(label, combined, source_scores, source_details)


def run_senfeed(reference_df: pd.DataFrame) -> MethodRunResult:
    senfeed_input = _build_senfeed_input(reference_df)
    if senfeed_input.empty:
        return MethodRunResult("SenFeedTruthFinder-CompositeTD", pd.DataFrame(), {}, {})

    model = SenFeedTruthDiscovery(enable_zk_proof=False)
    result_df = _run_silently(
        lambda: model.train(senfeed_input[["website", "fact", "object", "timestamp"]].copy())
    )
    meta = senfeed_input[
        ["website", "object", "timestamp", "parent_object", "asset", "record_type"]
    ].drop_duplicates()
    result_df = result_df.merge(meta, on=["website", "object", "timestamp"], how="left", sort=False)
    result_df["batch_index"] = result_df["timestamp"].astype(int)

    expected_parent_objects = sorted(reference_df["object"].astype(str).drop_duplicates().tolist())
    expected_websites = sorted(reference_df["website"].astype(str).drop_duplicates().tolist())
    source_details = _build_object_balanced_source_details(
        result_df,
        {"weight": "source_weight"},
        expected_parent_objects=expected_parent_objects,
        expected_websites=expected_websites,
    )
    source_details = _apply_object_coverage(
        source_details,
        result_df,
        expected_parent_objects,
        base_metric_preference=("weight",),
    )
    source_scores = {
        website: float(details["weight"])
        for website, details in source_details.items()
        if "weight" in details and np.isfinite(details["weight"])
    }
    return MethodRunResult("SenFeedTruthFinder-CompositeTD", result_df, source_scores, source_details)


def _numeric_senfeed_result_df(numeric_df: pd.DataFrame) -> ProofMethodOutput:
    input_df = numeric_df.rename(columns={"batch_index": "timestamp"})[
        ["website", "fact", "object", "timestamp"]
    ].copy()
    model = SenFeedTruthDiscovery(enable_zk_proof=False)
    result_df = _run_silently(lambda: model.train(input_df))
    source_scores = {
        str(website): float(metrics.get("weight", np.nan))
        for website, metrics in model.get_source_reliability().items()
        if np.isfinite(metrics.get("weight", np.nan))
    }
    return ProofMethodOutput(result_df=result_df, source_scores=source_scores)


def _numeric_decent_result_df(numeric_df: pd.DataFrame) -> ProofMethodOutput:
    model = EnhancedTruthFinder(enable_zk_proof=False)
    outputs: List[pd.DataFrame] = []
    for batch_index in sorted(numeric_df["batch_index"].astype(int).unique().tolist()):
        batch_df = numeric_df[numeric_df["batch_index"].astype(int) == int(batch_index)][
            ["website", "fact", "object"]
        ].copy()
        output = _run_silently(
            lambda batch=batch_df, bi=batch_index: model.process_batch(batch, epoch=int(bi))
        )
        output["batch_index"] = int(batch_index)
        outputs.append(output)
    result_df = _concat_frames(outputs)
    source_scores = {
        str(website): float(metrics.get("r", np.nan))
        for website, metrics in model.get_source_reliability().items()
        if np.isfinite(metrics.get("r", np.nan))
    }
    return ProofMethodOutput(result_df=result_df, source_scores=source_scores)


def _numeric_sente_result_df(numeric_df: pd.DataFrame) -> ProofMethodOutput:
    model = _run_silently(lambda: _NumericSenteTruthFinder(enable_zk_proof=False))
    outputs: List[pd.DataFrame] = []
    for batch_index in sorted(numeric_df["batch_index"].astype(int).unique().tolist()):
        batch_df = numeric_df[numeric_df["batch_index"].astype(int) == int(batch_index)][
            ["website", "fact", "object", "batch_index"]
        ].copy()
        output = _run_silently(lambda batch=batch_df: model.process_batch(batch))
        outputs.append(output)
    result_df = _concat_frames(outputs)
    source_scores = {
        str(website): float(score)
        for website, score in model.node_credibility.items()
        if np.isfinite(score)
    }
    return ProofMethodOutput(result_df=result_df, source_scores=source_scores)


def _numeric_basic_result_df(numeric_df: pd.DataFrame) -> ProofMethodOutput:
    model = _run_silently(lambda: _NumericBasicTruthFinder(enable_zk_proof=False))
    outputs: List[pd.DataFrame] = []
    for batch_index in sorted(numeric_df["batch_index"].astype(int).unique().tolist()):
        batch_df = numeric_df[numeric_df["batch_index"].astype(int) == int(batch_index)][
            ["website", "fact", "object", "batch_index"]
        ].copy()
        output = _run_silently(lambda batch=batch_df: model.process_batch(batch))
        outputs.append(output)
    result_df = _concat_frames(outputs)
    source_scores = {
        str(website): float(score)
        for website, score in model.website_trustworthiness.items()
        if np.isfinite(score)
    }
    return ProofMethodOutput(result_df=result_df, source_scores=source_scores)


def _numeric_LASO_result_df(numeric_df: pd.DataFrame) -> ProofMethodOutput:
    model = LASOTruthFinder(enable_zk_proof=False, use_sbert=False)
    empty_text_df = pd.DataFrame(columns=["website", "object", "fact", "batch_index"])
    outputs: List[pd.DataFrame] = []
    for batch_index in sorted(numeric_df["batch_index"].astype(int).unique().tolist()):
        batch_df = numeric_df[numeric_df["batch_index"].astype(int) == int(batch_index)][
            ["website", "fact", "object", "batch_index"]
        ].copy()
        numeric_out, _ = _run_silently(lambda batch=batch_df: model.process_batch(batch, empty_text_df.copy()))
        outputs.append(numeric_out)
    result_df = _concat_frames(outputs)
    source_scores = {
        str(website): float(metrics.get("reliability", np.nan))
        for website, metrics in model.get_reliability().items()
        if np.isfinite(metrics.get("reliability", np.nan))
    }
    return ProofMethodOutput(result_df=result_df, source_scores=source_scores)


def _text_sente_result_df(text_df: pd.DataFrame) -> ProofMethodOutput:
    model = _run_silently(
        lambda: SenteTruthFinder(enable_zk_proof=False, use_sbert=True, local_files_only=True)
    )
    outputs: List[pd.DataFrame] = []
    for batch_index in sorted(text_df["batch_index"].astype(int).unique().tolist()):
        batch_df = text_df[text_df["batch_index"].astype(int) == int(batch_index)][
            ["website", "fact", "object", "batch_index"]
        ].copy()
        output = _run_silently(lambda batch=batch_df: model.process_batch(batch))
        outputs.append(output)
    result_df = _concat_frames(outputs)
    source_scores = {
        str(website): float(score)
        for website, score in model.node_credibility.items()
        if np.isfinite(score)
    }
    return ProofMethodOutput(result_df=result_df, source_scores=source_scores)


def _text_basic_result_df(text_df: pd.DataFrame) -> ProofMethodOutput:
    model = _run_silently(lambda: BasicTruthFinder(implication=lambda _f1, _f2: 0.0, enable_zk_proof=False))
    outputs: List[pd.DataFrame] = []
    for batch_index in sorted(text_df["batch_index"].astype(int).unique().tolist()):
        batch_df = text_df[text_df["batch_index"].astype(int) == int(batch_index)][
            ["website", "fact", "object", "batch_index"]
        ].copy()
        prefix_texts = text_df[text_df["batch_index"].astype(int) <= int(batch_index)]["fact"].astype(str).tolist()
        output = _run_silently(
            lambda batch=batch_df, prefix=prefix_texts: model.process_batch(batch, implication_texts=prefix)
        )
        outputs.append(output)
    result_df = _concat_frames(outputs)
    source_scores = {
        str(website): float(score)
        for website, score in model.website_trustworthiness.items()
        if np.isfinite(score)
    }
    return ProofMethodOutput(result_df=result_df, source_scores=source_scores)


def _text_LASO_result_df(text_df: pd.DataFrame) -> ProofMethodOutput:
    model = LASOTruthFinder(enable_zk_proof=False, use_sbert=True, local_files_only=True)
    empty_numeric_df = pd.DataFrame(columns=["website", "object", "fact", "batch_index"])
    outputs: List[pd.DataFrame] = []
    for batch_index in sorted(text_df["batch_index"].astype(int).unique().tolist()):
        batch_df = text_df[text_df["batch_index"].astype(int) == int(batch_index)][
            ["website", "fact", "object", "batch_index"]
        ].copy()
        _, text_out = _run_silently(lambda batch=batch_df: model.process_batch(empty_numeric_df.copy(), batch))
        outputs.append(text_out)
    result_df = _concat_frames(outputs)
    source_scores = {
        str(website): float(metrics.get("reliability", np.nan))
        for website, metrics in model.get_reliability().items()
        if np.isfinite(metrics.get("reliability", np.nan))
    }
    return ProofMethodOutput(result_df=result_df, source_scores=source_scores)


NUMERIC_PROOF_RUNNERS: Dict[str, Callable[[pd.DataFrame], ProofMethodOutput]] = {
    "SenFeedTruth": _numeric_senfeed_result_df,
    "DecentTruth": _numeric_decent_result_df,
    "SenteTruth": _numeric_sente_result_df,
    "BasicTruth": _numeric_basic_result_df,
    "LASOTruth": _numeric_LASO_result_df,
}

TEXT_PROOF_RUNNERS: Dict[str, Callable[[pd.DataFrame], ProofMethodOutput]] = {
    "SenteTruth": _text_sente_result_df,
    "BasicTruth": _text_basic_result_df,
    "LASOTruth": _text_LASO_result_df,
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2)


def _attach_single_method_proof(
    *,
    result_df: pd.DataFrame,
    method_name: str,
    domain: str,
    source_scores: Mapping[str, float],
) -> pd.DataFrame:
    if result_df.empty:
        return result_df

    proof_df = result_df.copy()
    proof_df["proof_scope"] = f"{domain}:{method_name}:full_run"

    claim_columns = [
        column
        for column in proof_df.columns
        if column
        not in {
            "proof",
            "proof_id",
            "proof_scheme",
            "proof_verified",
            "proof_verification_mode",
        }
    ]

    truth_column = "global_truth" if "global_truth" in proof_df.columns else None
    proof_df = attach_grouped_proofs(
        proof_df,
        method_name=method_name,
        source_scores=dict(source_scores),
        group_columns=("proof_scope",),
        truth_column=truth_column,
        claim_columns=tuple(claim_columns),
        proof_engine=ZKProofEngine(prover_id=method_name),
        extra_public_data={"domain": domain, "row_count": int(len(result_df))},
        proof_backend="schnorr",
    )
    return proof_df.drop(columns=["proof_scope"])


def _write_proof_bundle(
    *,
    result_df: pd.DataFrame,
    domain: str,
    method_name: str,
    output_dir: Path,
    experiment_prefix: str,
) -> ProofBundleArtifact:
    if result_df.empty:
        raise ValueError(f"{method_name} produced an empty result dataframe for {domain}.")

    experiment_id = f"{experiment_prefix}_{domain}_{_sanitize_token(method_name)}"
    bundle = build_experiment_proof_bundle(
        result_df,
        experiment_id=experiment_id,
        method=method_name,
    )
    output_path = output_dir / f"{method_name}.{domain}.proof_bundle.json"
    _write_json(output_path, bundle.to_dict())
    return ProofBundleArtifact(
        domain=domain,
        method_name=method_name,
        output_path=str(output_path.resolve()),
        row_count=int(bundle.row_count),
        proof_group_count=int(len(bundle.proof_groups)),
    )


def generate_numeric_proof_bundles(
    *,
    input_path: Path = DEFAULT_NUMERIC_INPUT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_batches: Optional[int] = None,
    methods: Sequence[str] = NUMERIC_METHOD_ORDER,
) -> List[ProofBundleArtifact]:
    batches = _load_limited_batches(Path(input_path), max_batches)
    numeric_df = to_numeric_df(batches)
    artifacts: List[ProofBundleArtifact] = []
    for method_name in methods:
        runner = NUMERIC_PROOF_RUNNERS[method_name]
        method_output = runner(numeric_df.copy())
        result_df = _attach_single_method_proof(
            result_df=method_output.result_df,
            method_name=method_name,
            domain="numeric",
            source_scores=method_output.source_scores,
        )
        artifacts.append(
            _write_proof_bundle(
                result_df=result_df,
                domain="numeric",
                method_name=method_name,
                output_dir=Path(output_dir),
                experiment_prefix=_sanitize_token(Path(input_path).stem),
            )
        )
    return artifacts


def generate_text_proof_bundles(
    *,
    input_path: Path = DEFAULT_TEXT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_batches: Optional[int] = None,
    methods: Sequence[str] = TEXT_METHOD_ORDER,
) -> List[ProofBundleArtifact]:
    batches = _load_limited_batches(Path(input_path), max_batches)
    text_df = to_text_df(batches)
    artifacts: List[ProofBundleArtifact] = []
    for method_name in methods:
        runner = TEXT_PROOF_RUNNERS[method_name]
        method_output = runner(text_df.copy())
        result_df = _attach_single_method_proof(
            result_df=method_output.result_df,
            method_name=method_name,
            domain="text",
            source_scores=method_output.source_scores,
        )
        artifacts.append(
            _write_proof_bundle(
                result_df=result_df,
                domain="text",
                method_name=method_name,
                output_dir=Path(output_dir),
                experiment_prefix=_sanitize_token(Path(input_path).stem),
            )
        )
    return artifacts


def generate_requested_proof_bundles(
    *,
    numeric_input: Path = DEFAULT_NUMERIC_INPUT,
    text_input: Path = DEFAULT_TEXT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    max_batches: Optional[int] = None,
) -> List[ProofBundleArtifact]:
    artifacts = generate_numeric_proof_bundles(
        input_path=Path(numeric_input),
        output_dir=Path(output_dir),
        max_batches=max_batches,
    )
    artifacts.extend(
        generate_text_proof_bundles(
            input_path=Path(text_input),
            output_dir=Path(output_dir),
            max_batches=max_batches,
        )
    )
    return artifacts


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate only *.proof_bundle.json artifacts, with exactly one proof group per method."
    )
    parser.add_argument(
        "--numeric-input",
        default=str(DEFAULT_NUMERIC_INPUT),
        help="Numeric JSON input path. Default: truth_discovery/data_agent-10/num_demo.json",
    )
    parser.add_argument(
        "--text-input",
        default=str(DEFAULT_TEXT_INPUT),
        help="Text JSON input path. Default: truth_discovery/data_agent-10/demo.json",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for output *.proof_bundle.json files.",
    )
    parser.add_argument(
        "--max-batches",
        type=_positive_int,
        default=None,
        help="Optional cap on the number of batches consumed from each input.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_cli()
    args = parser.parse_args(argv)

    artifacts = generate_requested_proof_bundles(
        numeric_input=Path(args.numeric_input),
        text_input=Path(args.text_input),
        output_dir=Path(args.output_dir),
        max_batches=args.max_batches,
    )

    print("Generated proof bundles:")
    for artifact in artifacts:
        print(
            f"- {artifact.method_name} [{artifact.domain}] "
            f"rows={artifact.row_count} groups={artifact.proof_group_count} "
            f"path={artifact.output_path}"
        )
    return 0


__all__ = [
    "DEFAULT_NUMERIC_INPUT",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_TEXT_INPUT",
    "EnhancedFinderConfig",
    "MethodRunResult",
    "NUMERIC_METHOD_ORDER",
    "ProofBundleArtifact",
    "TEXT_METHOD_ORDER",
    "_build_object_balanced_source_details",
    "_build_senfeed_input",
    "_create_enhanced_finder",
    "_native_score_to_percentage",
    "create_reference_data",
    "generate_numeric_proof_bundles",
    "generate_requested_proof_bundles",
    "generate_text_proof_bundles",
    "main",
    "run_basic",
    "run_enhanced",
    "run_senfeed",
]


if __name__ == "__main__":
    raise SystemExit(main())
