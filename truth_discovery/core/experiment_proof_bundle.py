from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional

import pandas as pd

from .zk_proof import digest_object


def _strip_proof_payload_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    removable = [column for column in ["proof"] if column in dataframe.columns]
    if not removable:
        return dataframe.copy()
    return dataframe.drop(columns=removable).copy()


def _records_from_dataframe(dataframe: pd.DataFrame) -> List[Dict[str, Any]]:
    working = dataframe.copy()
    if working.index.name is not None or not isinstance(working.index, pd.RangeIndex):
        working = working.reset_index()
    return working.to_dict("records")


def _resolve_method(result_df: pd.DataFrame, explicit_method: Optional[str]) -> str:
    if explicit_method:
        return str(explicit_method)
    if "proof" in result_df.columns and not result_df.empty:
        proof_payload = result_df.iloc[0].get("proof")
        if isinstance(proof_payload, Mapping) and proof_payload.get("method"):
            return str(proof_payload["method"])
    if "method" in result_df.columns and not result_df.empty:
        method_value = result_df.iloc[0].get("method")
        if pd.notna(method_value):
            return str(method_value)
    return "unknown"


def _ensure_bytes32_hex(value: Any) -> str:
    text = str(value).lower().removeprefix("0x")
    if len(text) != 64:
        text = digest_object(value)
    return f"0x{text}"


def groth16_proof_to_solidity_call(proof: Mapping[str, Any]) -> Dict[str, Any]:
    proof_json = proof.get("proof_json")
    public_json = proof.get("public_json")
    if not isinstance(proof_json, Mapping):
        raise ValueError("proof_json is required for Groth16 calldata export")
    if not isinstance(public_json, list):
        raise ValueError("public_json is required for Groth16 calldata export")
    if len(public_json) < 4:
        raise ValueError("unexpected Groth16 public_json length")

    def _to_uint(value: Any) -> int:
        return int(str(value))

    pi_a = proof_json.get("pi_a")
    pi_b = proof_json.get("pi_b")
    pi_c = proof_json.get("pi_c")
    if not isinstance(pi_a, list) or len(pi_a) < 2:
        raise ValueError("unexpected Groth16 pi_a structure")
    if not isinstance(pi_b, list) or len(pi_b) < 2:
        raise ValueError("unexpected Groth16 pi_b structure")
    if not isinstance(pi_c, list) or len(pi_c) < 2:
        raise ValueError("unexpected Groth16 pi_c structure")

    a = [_to_uint(pi_a[0]), _to_uint(pi_a[1])]
    b = [
        [_to_uint(pi_b[0][1]), _to_uint(pi_b[0][0])],
        [_to_uint(pi_b[1][1]), _to_uint(pi_b[1][0])],
    ]
    c = [_to_uint(pi_c[0]), _to_uint(pi_c[1])]
    pub_signals = [_to_uint(value) for value in public_json]

    payload = {
        "a": a,
        "b": b,
        "c": c,
        "pubSignals": pub_signals,
        "publicSignalCount": len(pub_signals),
        "outputStatementHashField": pub_signals[0],
        "outputTruth": pub_signals[1],
        "outputTotalWeight": pub_signals[2],
        "outputIsValid": pub_signals[3],
    }
    if len(pub_signals) >= 3:
        payload.update(
            {
                "claimedTruth": pub_signals[-3],
                "claimedTotalWeight": pub_signals[-2],
                "inputStatementHashField": pub_signals[-1],
            }
        )
    return payload


@dataclass
class ExperimentProofGroup:
    group_id: Dict[str, Any]
    group_digest: str
    result_digest: str
    row_count: int
    proof_id: str
    proof_scheme: str
    proof_backend: Optional[str]
    statement_hash: Optional[str]
    statement_digest_sha256: Optional[str]
    verification_key_path: Optional[str]
    artifact_dir: Optional[str]
    public_json: Any
    proof_json: Any
    proof_payload: Any
    contract_call: Optional[Dict[str, Any]]
    contract_submission: Dict[str, Any]
    rows: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentProofBundle:
    experiment_id: str
    experiment_digest: str
    method: str
    row_count: int
    result_digest: str
    rows: Optional[List[Dict[str, Any]]]
    proof_groups: List[ExperimentProofGroup]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["proof_groups"] = [group.to_dict() for group in self.proof_groups]
        return payload


def build_experiment_proof_bundle(
    result_df: pd.DataFrame,
    *,
    experiment_id: str,
    method: Optional[str] = None,
    include_rows: bool = True,
    include_group_rows: bool = False,
) -> ExperimentProofBundle:
    if result_df.empty:
        raise ValueError("result_df must not be empty")

    required_columns = {"proof_id", "proof_scheme", "proof"}
    missing_columns = required_columns - set(result_df.columns)
    if missing_columns:
        raise ValueError(f"missing required proof columns: {sorted(missing_columns)}")

    resolved_method = _resolve_method(result_df, method)

    rows_df = _strip_proof_payload_columns(result_df)
    result_rows_payload = _records_from_dataframe(rows_df)
    result_rows = result_rows_payload if include_rows else None
    result_digest = digest_object(result_rows_payload)

    proof_groups: List[ExperimentProofGroup] = []
    group_digest_entries: List[Dict[str, Any]] = []

    for proof_id, group_df in result_df.groupby("proof_id", sort=False, dropna=False):
        if pd.isna(proof_id):
            continue

        proof_payload = group_df.iloc[0].get("proof")
        if not isinstance(proof_payload, Mapping):
            raise ValueError(f"invalid proof payload for proof_id={proof_id}")

        statement = proof_payload.get("statement") if isinstance(proof_payload.get("statement"), Mapping) else {}
        group_id = statement.get("group_id") if isinstance(statement.get("group_id"), Mapping) else {}
        group_digest = digest_object(group_id)

        group_rows_df = _strip_proof_payload_columns(group_df)
        group_rows_payload = _records_from_dataframe(group_rows_df)
        group_result_digest = digest_object(group_rows_payload)

        scheme = str(proof_payload.get("scheme") or group_df.iloc[0]["proof_scheme"])
        contract_call = None
        if scheme == "groth16_truth_aggregation_v1":
            contract_call = groth16_proof_to_solidity_call(proof_payload)

        statement_digest_sha256 = str(
            proof_payload.get("statement_digest_sha256") or digest_object(statement)
        )
        normalized_proof_id = str(proof_payload.get("proof_id") or proof_id)

        contract_submission = {
            "experimentId": _ensure_bytes32_hex(experiment_id),
            "groupId": _ensure_bytes32_hex(group_digest),
            "resultDigest": _ensure_bytes32_hex(group_result_digest),
            "proofId": _ensure_bytes32_hex(normalized_proof_id),
            "statementDigestSha256": _ensure_bytes32_hex(statement_digest_sha256),
        }
        if contract_call is not None:
            contract_submission.update(contract_call)

        proof_groups.append(
            ExperimentProofGroup(
                group_id=dict(group_id),
                group_digest=group_digest,
                result_digest=group_result_digest,
                row_count=int(len(group_df)),
                proof_id=normalized_proof_id,
                proof_scheme=scheme,
                proof_backend=proof_payload.get("backend"),
                statement_hash=(
                    str(proof_payload.get("statement_hash"))
                    if proof_payload.get("statement_hash") is not None
                    else None
                ),
                statement_digest_sha256=statement_digest_sha256,
                verification_key_path=proof_payload.get("verification_key_path"),
                artifact_dir=proof_payload.get("artifact_dir"),
                public_json=proof_payload.get("public_json"),
                proof_json=proof_payload.get("proof_json"),
                proof_payload=dict(proof_payload),
                contract_call=contract_call,
                contract_submission=contract_submission,
                rows=group_rows_payload if include_group_rows else None,
            )
        )
        group_digest_entries.append(
            {
                "proof_id": normalized_proof_id,
                "group_digest": group_digest,
                "result_digest": group_result_digest,
            }
        )

    experiment_digest = digest_object(
        {
            "experiment_id": str(experiment_id),
            "method": resolved_method,
            "result_digest": result_digest,
            "proof_groups": group_digest_entries,
        }
    )

    return ExperimentProofBundle(
        experiment_id=str(experiment_id),
        experiment_digest=experiment_digest,
        method=resolved_method,
        row_count=int(len(result_df)),
        result_digest=result_digest,
        rows=result_rows,
        proof_groups=proof_groups,
    )
