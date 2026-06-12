
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import pandas as pd

_BN128_FR = 21888242871839275222246405745257275088548364400416034343698204186575808495617

_MODP_2048_P = int(
    "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22"
    "514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6"
    "F44C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3D"
    "C2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB"
    "9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3BE39E772C180E8603"
    "9B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA0510"
    "15728E5A8AACAA68FFFFFFFFFFFFFFFF",
    16,
)
_MODP_2048_Q = (_MODP_2048_P - 1) // 2
_MODP_2048_G = 4

def _normalize_json_value(value: Any) -> Any:

    try:
        import numpy as np
    except Exception:
        np = None

    if value is None:
        return None
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if value != value:
            return "NaN"
        if value == float("inf"):
            return "Infinity"
        if value == float("-inf"):
            return "-Infinity"
        return value
    if np is not None and isinstance(value, np.generic):
        return _normalize_json_value(value.item())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            str(k): _normalize_json_value(v)
            for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(v) for v in value]
    if isinstance(value, set):
        return sorted(_normalize_json_value(v) for v in value)
    return str(value)

def _canonical_json(data: Any) -> str:
    return json.dumps(
        _normalize_json_value(data),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )

def _sha256_hex(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

def digest_object(data: Any) -> str:

    return _sha256_hex(_canonical_json(data))

def _to_int_hex(value: int) -> str:
    return hex(int(value))

def _from_int_hex(value: str) -> int:
    return int(str(value), 16)

def _run_cmd(cmd: Sequence[str], cwd: Optional[str] = None) -> str:

    result = subprocess.run(
        list(cmd),
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    output = (result.stdout or "") + (result.stderr or "")
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(cmd)}\n{output}"
        )
    return output

def _write_json(path: str, data: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, separators=(",", ":"))

def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _extract_numeric_claim(value: Any) -> float:

    try:
        import numpy as np
    except Exception:
        np = None

    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        if value != value:
            return 0.0
        return float(value)
    if np is not None and isinstance(value, np.generic):
        return _extract_numeric_claim(value.item())

    text = str(value)
    currency = re.findall(r"\$(-?\d+(?:\.\d+)?)", text)
    if currency:
        return float(currency[0])

    generic = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", text)
    if generic:
        return float(generic[0])
    return 0.0

def _int_json_list(values: Sequence[int]) -> Sequence[str]:

    return [str(int(v)) for v in values]

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

@dataclass
class Groth16TruthAggregationEngine:

    circuits_dir: Optional[str] = None
    ptau_power: int = 12
    protocol: Optional[str] = None

    def __post_init__(self):
        self.circuits_dir = self.circuits_dir or os.path.join(os.path.dirname(__file__), "circuits")
        self.protocol = (self.protocol or os.environ.get("TRUTHFINDER_SNARK_PROTOCOL", "groth16")).strip().lower()
        if self.protocol not in {"groth16", "plonk"}:
            raise ValueError(f"unsupported snark protocol: {self.protocol}")
        self.circom_bin = os.path.join(self.circuits_dir, "node_modules", ".bin", "circom")
        self.snarkjs_bin = os.path.join(self.circuits_dir, "node_modules", ".bin", "snarkjs")
        self.artifacts_root = os.path.join(self.circuits_dir, "snarkjs_artifacts")
        _ensure_dir(self.artifacts_root)

        if not os.path.exists(self.circom_bin):
            raise FileNotFoundError(f"circom binary not found: {self.circom_bin}")
        if not os.path.exists(self.snarkjs_bin):
            raise FileNotFoundError(f"snarkjs binary not found: {self.snarkjs_bin}")

    def _artifact_dir(self, n_sources: int) -> str:
        return os.path.join(self.artifacts_root, f"truth_aggregation_{self.protocol}_v2_n{int(n_sources)}")

    def _circuit_text(self, n_sources: int) -> str:
        n = int(n_sources)
        return f"""
// ---- Inline helpers (self-contained, no circomlib dependency) ----

template Num2Bits(n) {{
    signal input in;
    signal output out[n];
    var lc = 0;
    for (var i = 0; i < n; i++) {{
        out[i] <-- (in >> i) & 1;
        out[i] * (out[i] - 1) === 0;
        lc += out[i] * (1 << i);
    }}
    lc === in;
}}

template IsZero() {{
    signal input in;
    signal output out;
    signal inv;
    inv <-- in != 0 ? 1/in : 0;
    out <== -in * inv + 1;
    in * out === 0;
}}

template IsEqual() {{
    signal input in[2];
    signal output out;
    component isz = IsZero();
    isz.in <== in[0] - in[1];
    out <== isz.out;
}}

template LessThan(n) {{
    signal input in[2];
    signal output out;
    component n2b = Num2Bits(n + 1);
    n2b.in <== in[0] + (1 << n) - in[1];
    out <== 1 - n2b.out[n];
}}

template LessEqThan(n) {{
    signal input in[2];
    signal output out;
    component lt = LessThan(n);
    lt.in[0] <== in[0];
    lt.in[1] <== in[1] + 1;
    out <== lt.out;
}}

template GreaterThan(n) {{
    signal input in[2];
    signal output out;
    component lt = LessThan(n);
    lt.in[0] <== in[1];
    lt.in[1] <== in[0];
    out <== lt.out;
}}

template GreaterEqThan(n) {{
    signal input in[2];
    signal output out;
    component lt = LessThan(n);
    lt.in[0] <== in[0];
    lt.in[1] <== in[1];
    out <== 1 - lt.out;
}}

template Sum(n) {{
    signal input in[n];
    signal output out;
    signal partial[n+1];
    partial[0] <== 0;
    for (var i = 0; i < n; i++) {{
        partial[i+1] <== partial[i] + in[i];
    }}
    out <== partial[n];
}}

// ---- LASOTruth Weighted Median Proof ----
//
// Prove that claimed_truth is the weighted median of (values, weights):
//   (a) claimed_truth appears in values
//   (b) weight(v_i ≤ claimed_truth) × 2 ≥ total_weight
//   (c) weight(v_i < claimed_truth) × 2 < total_weight

template LASOTruthProof(n) {{
    signal input values[n];
    signal input weights[n];
    signal input claimed_truth;
    signal input claimed_total_weight;
    signal input statement_hash;

    signal output public_statement_hash;
    signal output public_truth;
    signal output public_total_weight;
    signal output is_valid;

    // (a) claimed_truth must exist in values
    component eq[n];
    signal match_flag[n];
    for (var i = 0; i < n; i++) {{
        eq[i] = IsEqual();
        eq[i].in[0] <== values[i];
        eq[i].in[1] <== claimed_truth;
        match_flag[i] <== eq[i].out;
    }}
    component sum_match = Sum(n);
    for (var i = 0; i < n; i++) {{
        sum_match.in[i] <== match_flag[i];
    }}
    component has_match = GreaterThan(32);
    has_match.in[0] <== sum_match.out;
    has_match.in[1] <== 0;
    has_match.out === 1;

    // (b) weight_leq = Σ w_i where v_i ≤ claimed_truth
    component leq[n];
    signal w_leq[n];
    for (var i = 0; i < n; i++) {{
        leq[i] = LessEqThan(64);
        leq[i].in[0] <== values[i];
        leq[i].in[1] <== claimed_truth;
        w_leq[i] <== leq[i].out * weights[i];
    }}
    component sum_leq = Sum(n);
    for (var i = 0; i < n; i++) {{
        sum_leq.in[i] <== w_leq[i];
    }}

    // (c) weight_lt = Σ w_i where v_i < claimed_truth
    component lt_cmp[n];
    signal w_lt[n];
    for (var i = 0; i < n; i++) {{
        lt_cmp[i] = LessThan(64);
        lt_cmp[i].in[0] <== values[i];
        lt_cmp[i].in[1] <== claimed_truth;
        w_lt[i] <== lt_cmp[i].out * weights[i];
    }}
    component sum_lt = Sum(n);
    for (var i = 0; i < n; i++) {{
        sum_lt.in[i] <== w_lt[i];
    }}

    // total weight
    component sum_w = Sum(n);
    for (var i = 0; i < n; i++) {{
        sum_w.in[i] <== weights[i];
    }}
    claimed_total_weight === sum_w.out;

    // median condition: 2 × weight_leq ≥ total_weight
    component median_geq = GreaterEqThan(64);
    median_geq.in[0] <== 2 * sum_leq.out;
    median_geq.in[1] <== claimed_total_weight;
    median_geq.out === 1;

    // minimality: 2 × weight_lt < total_weight
    component median_lt = LessThan(64);
    median_lt.in[0] <== 2 * sum_lt.out;
    median_lt.in[1] <== claimed_total_weight;
    median_lt.out === 1;

    // binding hash
    signal hash_terms[{2*n + 2}];
    component sum_hash = Sum({2*n + 2});
    for (var k = 0; k < n; k++) {{
        hash_terms[2*k] <== values[k] * (k + 1);
        hash_terms[2*k + 1] <== weights[k] * (k + 1001);
    }}
    hash_terms[{2*n}] <== claimed_truth;
    hash_terms[{2*n + 1}] <== claimed_total_weight;
    for (var t = 0; t < {2*n + 2}; t++) {{
        sum_hash.in[t] <== hash_terms[t];
    }}
    statement_hash === sum_hash.out;

    public_statement_hash <== statement_hash;
    public_truth <== claimed_truth;
    public_total_weight <== claimed_total_weight;
    is_valid <== 1;
}}

component main = LASOTruthProof({n});
""".strip()

    def ensure_setup(self, n_sources: int):
        n = int(n_sources)
        if n <= 0:
            raise ValueError("n_sources must be > 0 for Groth16 proof generation")

        artifact_dir = self._artifact_dir(n)
        _ensure_dir(artifact_dir)

        circuit_path = os.path.join(artifact_dir, "truth_aggregation.circom")
        r1cs_path = os.path.join(artifact_dir, "truth_aggregation.r1cs")
        wasm_path = os.path.join(artifact_dir, "truth_aggregation.wasm")
        sym_path = os.path.join(artifact_dir, "truth_aggregation.sym")
        ptau_0_path = os.path.join(artifact_dir, f"pot{self.ptau_power}_0000.ptau")
        ptau_final_path = os.path.join(artifact_dir, f"pot{self.ptau_power}_final.ptau")
        zkey_path = os.path.join(artifact_dir, "truth_aggregation_final.zkey")
        vkey_path = os.path.join(artifact_dir, "verification_key.json")

        if all(os.path.exists(p) for p in [r1cs_path, wasm_path, zkey_path, vkey_path]):
            return

        with open(circuit_path, "w", encoding="utf-8") as f:
            f.write(self._circuit_text(n))

        _run_cmd([self.circom_bin, circuit_path, "-r", r1cs_path, "-w", wasm_path, "-s", sym_path])
        _run_cmd([self.snarkjs_bin, "ptn", "bn128", str(self.ptau_power), ptau_0_path])
        _run_cmd([self.snarkjs_bin, "pt2", ptau_0_path, ptau_final_path])
        if self.protocol == "groth16":
            _run_cmd([self.snarkjs_bin, "g16s", r1cs_path, ptau_final_path, zkey_path])
        else:
            _run_cmd([self.snarkjs_bin, "pks", r1cs_path, ptau_final_path, zkey_path])
        _run_cmd([self.snarkjs_bin, "zkev", zkey_path, vkey_path])

    def prove(
        self,
        *,
        n_sources: int,
        values: Sequence[int],
        weights: Sequence[int],
    ) -> Dict[str, Any]:
        n = int(n_sources)
        if n <= 0:
            raise ValueError("n_sources must be > 0")
        if len(values) != n or len(weights) != n:
            raise ValueError("values/weights length must equal n_sources")

        self.ensure_setup(n)

        artifact_dir = self._artifact_dir(n)
        wasm_path = os.path.join(artifact_dir, "truth_aggregation.wasm")
        zkey_path = os.path.join(artifact_dir, "truth_aggregation_final.zkey")
        vkey_path = os.path.join(artifact_dir, "verification_key.json")

        total_weight = int(sum(int(w) for w in weights))
        pairs = sorted(zip(values, weights), key=lambda x: int(x[0]))
        cumul = 0
        claimed_truth = int(pairs[0][0])
        for v, w in pairs:
            cumul += int(w)
            if 2 * cumul >= total_weight:
                claimed_truth = int(v)
                break

        statement_hash_int = self._binding_hash_int(values, weights, claimed_truth, total_weight)

        input_payload = {
            "values": _int_json_list(values),
            "weights": _int_json_list(weights),
            "claimed_truth": str(claimed_truth),
            "claimed_total_weight": str(total_weight),
            "statement_hash": str(statement_hash_int),
        }

        with tempfile.TemporaryDirectory(prefix="truthfinder_g16_") as tmpdir:
            input_path = os.path.join(tmpdir, "input.json")
            proof_path = os.path.join(tmpdir, "proof.json")
            public_path = os.path.join(tmpdir, "public.json")
            _write_json(input_path, input_payload)

            if self.protocol == "groth16":
                prove_log = _run_cmd([self.snarkjs_bin, "g16f", input_path, wasm_path, zkey_path, proof_path, public_path])
                scheme = "groth16_truth_aggregation_v1"
                backend = "snarkjs_groth16"
            else:
                prove_log = _run_cmd([self.snarkjs_bin, "pkf", input_path, wasm_path, zkey_path, proof_path, public_path])
                scheme = "plonk_truth_aggregation_v1"
                backend = "snarkjs_plonk"
            proof_json = _read_json(proof_path)
            public_json = _read_json(public_path)

        return {
            "scheme": scheme,
            "backend": backend,
            "artifact_dir": artifact_dir,
            "verification_key_path": vkey_path,
            "truth": claimed_truth,
            "total_weight": total_weight,
            "proof_json": proof_json,
            "public_json": public_json,
            "prove_log": prove_log,
            "verify_log": None,
            "verified": None,
            "verification_performed_in_python": False,
        }

    @staticmethod
    def _binding_hash_int(
        values: Sequence[int],
        weights: Sequence[int],
        claimed_truth: int,
        total_weight: int,
    ) -> int:
        acc = 0
        for i, v in enumerate(values):
            acc += int(v) * (i + 1)
        for i, w in enumerate(weights):
            acc += int(w) * (i + 1001)
        acc += int(claimed_truth)
        acc += int(total_weight)
        return int(acc) % _BN128_FR

@dataclass
class ZKProofEngine:

    prover_id: str = "TruthFinderProver"
    secret_seed: Optional[str] = None

    def __post_init__(self):
        seed = self.secret_seed or os.environ.get("TRUTHFINDER_ZK_SECRET")
        if seed is None:
            self._secret = secrets.randbelow(_MODP_2048_Q - 1) + 1
        else:
            material = f"{self.prover_id}:{seed}".encode("utf-8")
            self._secret = (int.from_bytes(hashlib.sha256(material).digest(), "big") % (_MODP_2048_Q - 1)) + 1
        self._public_key = pow(_MODP_2048_G, self._secret, _MODP_2048_P)

    @property
    def public_key(self) -> int:
        return self._public_key

    def generate(self, statement: Mapping[str, Any]) -> Dict[str, Any]:
        statement_obj = _normalize_json_value(statement)
        statement_hash = digest_object(statement_obj)

        nonce = secrets.randbelow(_MODP_2048_Q - 1) + 1
        commitment = pow(_MODP_2048_G, nonce, _MODP_2048_P)

        challenge_material = {
            "scheme": "schnorr_nizk_v1",
            "group": "modp2048_subgroup",
            "prover_id": self.prover_id,
            "public_key": _to_int_hex(self._public_key),
            "commitment": _to_int_hex(commitment),
            "statement_hash": statement_hash,
        }
        challenge = int(digest_object(challenge_material), 16) % _MODP_2048_Q
        if challenge == 0:
            challenge = 1

        response = (nonce + challenge * self._secret) % _MODP_2048_Q

        proof_id = digest_object(
            {
                "statement_hash": statement_hash,
                "public_key": _to_int_hex(self._public_key),
                "commitment": _to_int_hex(commitment),
                "response": _to_int_hex(response),
            }
        )

        proof = {
            "proof_id": proof_id,
            "scheme": "schnorr_nizk_v1",
            "backend": "schnorr",
            "group": "modp2048_subgroup",
            "generated_at": int(time.time()),
            "prover_id": self.prover_id,
            "statement": statement_obj,
            "statement_hash": statement_hash,
            "public_key": _to_int_hex(self._public_key),
            "commitment": _to_int_hex(commitment),
            "challenge": _to_int_hex(challenge),
            "response": _to_int_hex(response),
        }
        return proof

def _verify_schnorr(proof: Mapping[str, Any]) -> bool:
    if not isinstance(proof, Mapping):
        return False
    if proof.get("scheme") != "schnorr_nizk_v1":
        return False

    statement = proof.get("statement")
    statement_hash = digest_object(statement)
    if statement_hash != str(proof.get("statement_hash")):
        return False

    public_key = _from_int_hex(str(proof.get("public_key")))
    commitment = _from_int_hex(str(proof.get("commitment")))
    challenge = _from_int_hex(str(proof.get("challenge")))
    response = _from_int_hex(str(proof.get("response")))

    challenge_material = {
        "scheme": "schnorr_nizk_v1",
        "group": "modp2048_subgroup",
        "prover_id": proof.get("prover_id"),
        "public_key": str(proof.get("public_key")),
        "commitment": str(proof.get("commitment")),
        "statement_hash": statement_hash,
    }
    expected_challenge = int(digest_object(challenge_material), 16) % _MODP_2048_Q
    if expected_challenge == 0:
        expected_challenge = 1
    if challenge != expected_challenge:
        return False

    left = pow(_MODP_2048_G, response, _MODP_2048_P)
    right = (commitment * pow(public_key, challenge, _MODP_2048_P)) % _MODP_2048_P
    return left == right

def verify_proof_package(proof: Mapping[str, Any]) -> bool:

    try:
        if not isinstance(proof, Mapping):
            return False
        scheme = str(proof.get("scheme", ""))
        if scheme == "schnorr_nizk_v1":
            return _verify_schnorr(proof)
        if scheme in {"groth16_truth_aggregation_v1", "plonk_truth_aggregation_v1"}:
            return bool(proof.get("verified", False))
        return False
    except Exception:
        return False

def attach_grouped_proofs(
    dataframe: pd.DataFrame,
    *,
    method_name: str,
    source_scores: Optional[Mapping[str, float]] = None,
    group_columns: Sequence[str] = ("object",),
    truth_column: Optional[str] = "global_truth",
    website_column: str = "website",
    claim_columns: Sequence[str] = ("website", "fact"),
    iteration_count: Optional[int] = None,
    proof_engine: Optional[ZKProofEngine] = None,
    extra_public_data: Optional[Mapping[str, Any]] = None,
    proof_backend: Optional[str] = None,
    snark_engine: Optional[Groth16TruthAggregationEngine] = None,
    value_scale: int = 1000,
    weight_scale: int = 1000000,
) -> pd.DataFrame:

    df = dataframe.copy()
    if df.empty:
        return df

    group_columns = list(group_columns)
    missing = [col for col in group_columns if col not in df.columns]
    if missing:
        raise ValueError(f"missing group columns for proof attachment: {missing}")

    backend = (proof_backend or os.environ.get("TRUTHFINDER_PROOF_BACKEND", "snarkjs_groth16")).strip().lower()
    normalized_scores = {str(k): float(v) for k, v in (source_scores or {}).items()}

    schnorr_engine = proof_engine or ZKProofEngine(prover_id=method_name)
    if backend in {"snarkjs_groth16", "snarkjs_plonk"}:
        protocol = "plonk" if backend == "snarkjs_plonk" else "groth16"
        groth16_engine = snark_engine or Groth16TruthAggregationEngine(protocol=protocol)
    else:
        groth16_engine = None

    proof_rows = []
    for group_key, group_df in df.groupby(group_columns, dropna=False, sort=False):
        key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
        group_id = {
            col: _normalize_json_value(value)
            for col, value in zip(group_columns, key_tuple)
        }

        used_claim_columns = [c for c in claim_columns if c in group_df.columns]
        if used_claim_columns:
            claim_df = group_df[used_claim_columns].copy()
            for col in used_claim_columns:
                claim_df[col] = claim_df[col].map(_normalize_json_value)
            claim_df = claim_df.sort_values(used_claim_columns, kind="mergesort")
            claim_records = claim_df.to_dict("records")
        else:
            claim_records = []

        if truth_column and truth_column in group_df.columns:
            truth_values = [_normalize_json_value(v) for v in group_df[truth_column].drop_duplicates().tolist()]
            truth_payload = truth_values[0] if len(truth_values) == 1 else truth_values
        else:
            truth_payload = None

        related_scores = {}
        if website_column in group_df.columns and normalized_scores:
            websites = sorted(set(group_df[website_column].astype(str).tolist()))
            related_scores = {w: normalized_scores[w] for w in websites if w in normalized_scores}

        statement = {
            "method": method_name,
            "group_id": group_id,
            "row_count": int(len(group_df)),
            "input_digest": digest_object(claim_records),
            "truth_digest": digest_object(truth_payload),
            "source_score_digest": digest_object(related_scores),
            "iteration_count": int(iteration_count) if iteration_count is not None else None,
            "extra_public_data": _normalize_json_value(extra_public_data) if extra_public_data else None,
        }
        statement_hash_hex = digest_object(statement)

        if backend in {"snarkjs_groth16", "snarkjs_plonk"}:
            n_sources = int(len(group_df))
            values_float = [_extract_numeric_claim(v) for v in group_df.get("fact", pd.Series([0.0] * n_sources))]
            if website_column in group_df.columns:
                websites = group_df[website_column].astype(str).tolist()
                weights_float = [float(normalized_scores.get(w, 1.0)) for w in websites]
            else:
                weights_float = [1.0] * n_sources

            values_int = [int(round(max(0.0, v) * value_scale)) for v in values_float]
            weights_int = [int(round(max(0.0, w) * weight_scale)) for w in weights_float]
            if all(w == 0 for w in weights_int) and len(weights_int) > 0:
                weights_int[0] = weight_scale

            if groth16_engine is None:
                raise ValueError("SNARK proof backend selected but snark engine is unavailable")

            snark_pack = groth16_engine.prove(
                n_sources=n_sources,
                values=values_int,
                weights=weights_int,
            )

            proof = {
                "proof_id": digest_object(
                    {
                        "statement_hash": statement_hash_hex,
                        "public_json": snark_pack.get("public_json"),
                        "proof_json": snark_pack.get("proof_json"),
                    }
                ),
                "scheme": snark_pack["scheme"],
                "backend": snark_pack["backend"],
                "generated_at": int(time.time()),
                "method": method_name,
                "statement": statement,
                "statement_hash": str(snark_pack["public_json"][0]) if snark_pack.get("public_json") else None,
                "statement_digest_sha256": statement_hash_hex,
                "proof_json": snark_pack["proof_json"],
                "public_json": snark_pack["public_json"],
                "verification_key_path": snark_pack["verification_key_path"],
                "artifact_dir": snark_pack["artifact_dir"],
                "truth": snark_pack["truth"],
                "total_weight": snark_pack["total_weight"],
                "verified": None,
                "verification_performed_in_python": False,
                "verification_hint": "verify externally (e.g. on-chain verifier contract)",
            }
            verified = None
        else:
            proof = schnorr_engine.generate(statement)
            proof["verified"] = None
            proof["verification_performed_in_python"] = False
            proof["verification_hint"] = "verify externally"
            verified = None

        row = {col: val for col, val in zip(group_columns, key_tuple)}
        row.update(
            {
                "proof": proof,
                "proof_id": proof["proof_id"],
                "proof_scheme": proof["scheme"],
                "proof_verified": verified,
                "proof_verification_mode": "external",
            }
        )
        proof_rows.append(row)

    proof_df = pd.DataFrame(proof_rows)
    return df.merge(proof_df, on=group_columns, how="left", sort=False)
