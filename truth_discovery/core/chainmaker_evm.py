from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping, MutableMapping


def _to_uint_string(value: Any) -> str:
    return str(int(str(value)))


def _validate_submission(submission: Mapping[str, Any]) -> None:
    required_keys = {
        "experimentId",
        "groupId",
        "resultDigest",
        "proofId",
        "statementDigestSha256",
        "a",
        "b",
        "c",
        "pubSignals",
    }
    missing = required_keys - set(submission.keys())
    if missing:
        raise ValueError(f"missing ChainMaker submission keys: {sorted(missing)}")

    a = submission["a"]
    b = submission["b"]
    c = submission["c"]
    pub = submission["pubSignals"]
    if not isinstance(a, list) or len(a) != 2:
        raise ValueError("submission.a must be a 2-element list")
    if not isinstance(b, list) or len(b) != 2 or any(not isinstance(row, list) or len(row) != 2 for row in b):
        raise ValueError("submission.b must be a 2x2 list")
    if not isinstance(c, list) or len(c) != 2:
        raise ValueError("submission.c must be a 2-element list")
    if not isinstance(pub, list) or len(pub) != 15:
        raise ValueError("submission.pubSignals must be a 15-element list for current n=4 Groth16 proof")


def build_chainmaker_verify_params(contract_submission: Mapping[str, Any]) -> List[Dict[str, str]]:
    _validate_submission(contract_submission)
    a = contract_submission["a"]
    b = contract_submission["b"]
    c = contract_submission["c"]
    pub = contract_submission["pubSignals"]

    flat_values = [
        a[0],
        a[1],
        b[0][0],
        b[0][1],
        b[1][0],
        b[1][1],
        c[0],
        c[1],
        *pub,
    ]
    return [{"uint256": _to_uint_string(value)} for value in flat_values]


def build_chainmaker_submit_params(
    contract_submission: Mapping[str, Any],
    *,
    result_uri: str = "",
    proof_uri: str = "",
) -> List[Dict[str, str]]:
    _validate_submission(contract_submission)
    params: List[Dict[str, str]] = [
        {"bytes32": str(contract_submission["experimentId"])},
        {"bytes32": str(contract_submission["groupId"])},
        {"bytes32": str(contract_submission["resultDigest"])},
        {"bytes32": str(contract_submission["proofId"])},
        {"bytes32": str(contract_submission["statementDigestSha256"])},
        {"string": str(result_uri)},
        {"string": str(proof_uri)},
    ]
    params.extend(build_chainmaker_verify_params(contract_submission))
    return params


def export_chainmaker_group_payloads(
    bundle_payload: Mapping[str, Any],
    *,
    output_dir: str,
    bundle_name: str,
    result_uri: str = "",
    proof_uri: str = "",
) -> Dict[str, Any]:
    proof_groups = bundle_payload.get("proof_groups")
    if not isinstance(proof_groups, list) or not proof_groups:
        raise ValueError("bundle_payload.proof_groups must be a non-empty list")

    os.makedirs(output_dir, exist_ok=True)
    manifest: MutableMapping[str, Any] = {
        "bundle_name": bundle_name,
        "contract_file": "truth_discovery/core/circuits/TruthSingleProofRegistryN4.sol",
        "verify_method": "verifySelectedProofFlat",
        "submit_method": "submitSelectedProofFlat",
        "groups": [],
    }

    for index, group in enumerate(proof_groups):
        contract_submission = group.get("contract_submission")
        if not isinstance(contract_submission, Mapping):
            raise ValueError(f"proof_groups[{index}].contract_submission is required")

        verify_params = build_chainmaker_verify_params(contract_submission)
        submit_params = build_chainmaker_submit_params(
            contract_submission,
            result_uri=result_uri,
            proof_uri=proof_uri,
        )

        base_name = f"{bundle_name}.group_{index}"
        verify_path = os.path.abspath(os.path.join(output_dir, f"{base_name}.verifySelectedProofFlat.params.json"))
        submit_path = os.path.abspath(os.path.join(output_dir, f"{base_name}.submitSelectedProofFlat.params.json"))

        with open(verify_path, "w", encoding="utf-8") as file_obj:
            json.dump(verify_params, file_obj, ensure_ascii=False, indent=2)
        with open(submit_path, "w", encoding="utf-8") as file_obj:
            json.dump(submit_params, file_obj, ensure_ascii=False, indent=2)

        manifest["groups"].append(
            {
                "group_index": index,
                "proof_id": group.get("proof_id"),
                "group_digest": group.get("group_digest"),
                "verify_params_path": verify_path,
                "submit_params_path": submit_path,
            }
        )

    manifest_path = os.path.abspath(os.path.join(output_dir, f"{bundle_name}.chainmaker_manifest.json"))
    with open(manifest_path, "w", encoding="utf-8") as file_obj:
        json.dump(manifest, file_obj, ensure_ascii=False, indent=2)
    return dict(manifest, manifest_path=manifest_path)
