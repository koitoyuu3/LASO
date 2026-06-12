
from __future__ import annotations

import pickle
from typing import Any, Tuple

import msgpack

SUPPORTED_BACKENDS: Tuple[str, ...] = ("msgpack", "pickle", "json")

def encode(payload: Any, backend: str = "msgpack") -> bytes:
    if backend == "msgpack":
        return msgpack.packb(payload, use_bin_type=True)
    if backend == "pickle":
        return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    if backend == "json":
        import json

        return json.dumps(payload, ensure_ascii=False).encode("utf-8")
    raise ValueError(f"Unsupported serialization backend: {backend!r}")

def decode(blob: bytes, backend: str = "msgpack") -> Any:
    if backend == "msgpack":
        return msgpack.unpackb(blob, raw=False)
    if backend == "pickle":
        return pickle.loads(blob)
    if backend == "json":
        import json

        return json.loads(blob.decode("utf-8"))
    raise ValueError(f"Unsupported serialization backend: {backend!r}")
