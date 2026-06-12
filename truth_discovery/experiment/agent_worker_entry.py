
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from truth_discovery.experiment.distributed.transport_zmq import (
    ZmqAgentTransport,
)

def _now_ms() -> float:
    return time.perf_counter() * 1000.0

def _maybe_load_sbert(state: Dict[str, Any]) -> None:

    if state.get("_sbert_loaded"):
        return
    state["_sbert_loaded"] = True
    try:
        from truth_discovery.core import LASOTruthFinder

        holder = LASOTruthFinder(enable_zk_proof=False, use_sbert=True)

        try:
            holder._encode_texts(["warmup"])
        except Exception:
            pass
        state["_sbert_holder"] = holder
    except Exception as exc:
        state["_sbert_load_error"] = f"{type(exc).__name__}: {exc}"

def _encode_text_slice(state: Dict[str, Any], items):

    holder = state.get("_sbert_holder")
    if holder is None:
        return 0, 0
    texts = []
    for record in items:
        try:
            text = str(record[1])
        except (IndexError, TypeError):
            continue
        if text:
            texts.append(text)
    if not texts:
        return 0, 0
    try:

        embeddings = holder._encode_texts(texts)
    except Exception:
        return len(texts), 0
    try:
        dim = int(getattr(embeddings, "shape", [0, 0])[1])
    except Exception:
        dim = 0
    return len(texts), dim

def _handle(message: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    msg_type = message.get("type") if isinstance(message, dict) else None
    if msg_type == "process_text_batch":
        start = time.perf_counter()
        items = message.get("items") or []

        if message.get("method") == "BasicTruth":
            encoded_count, embed_dim = 0, 0
        else:
            encoded_count, embed_dim = _encode_text_slice(state, items)
        local_compute_sec = time.perf_counter() - start
        state["last_item_count"] = len(items)
        return {
            "type": "ack",
            "subtype": "process_text_batch",
            "agent_id": state["agent_id"],
            "item_count": len(items),
            "encoded_count": encoded_count,
            "embedding_dim": embed_dim,
            "local_compute_sec": float(local_compute_sec),
            "sbert_loaded": bool(state.get("_sbert_holder") is not None),
        }
    if msg_type == "process_batch":
        start = time.perf_counter()
        items = message.get("items") or []

        item_count = 0
        object_set = set()
        running_sum = 0.0
        for record in items:
            try:
                object_id = record[0]
                fact = float(record[1])
            except (IndexError, TypeError, ValueError):
                continue
            item_count += 1
            object_set.add(object_id)
            running_sum += fact
        local_compute_sec = time.perf_counter() - start
        state["last_item_count"] = item_count
        return {
            "type": "ack",
            "subtype": "process_batch",
            "agent_id": state["agent_id"],
            "item_count": item_count,
            "object_count": len(object_set),
            "running_sum": float(running_sum),
            "local_compute_sec": float(local_compute_sec),
        }
    if msg_type == "iterate":

        start = time.perf_counter()

        _ = sum(range(state.get("last_item_count", 0) % 1024))
        return {
            "type": "ack",
            "subtype": "iterate",
            "agent_id": state["agent_id"],
            "local_compute_sec": float(time.perf_counter() - start),
        }
    if msg_type == "reliability_update":
        snapshot = message.get("reliability") or {}
        if isinstance(snapshot, dict):
            state["reliability"] = dict(snapshot)
        return {
            "type": "ack",
            "subtype": "reliability_update",
            "agent_id": state["agent_id"],
            "snapshot_size": len(state.get("reliability", {})),
        }
    if msg_type == "shutdown":
        return {
            "type": "ack",
            "subtype": "shutdown",
            "agent_id": state["agent_id"],
        }
    return {
        "type": "error",
        "agent_id": state["agent_id"],
        "error": f"unknown message type: {msg_type!r}",
    }

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Distributed scalability benchmark — single-agent worker process."
    )
    parser.add_argument("--agent-id", required=True, help="Logical agent identifier.")
    parser.add_argument(
        "--coord-addr",
        required=True,
        help="ZeroMQ endpoint (e.g. tcp://127.0.0.1:5555) the coordinator is bound to.",
    )
    parser.add_argument(
        "--serialization",
        default="msgpack",
        choices=["msgpack", "pickle", "json"],
        help="Serialization backend; must match the coordinator.",
    )
    parser.add_argument(
        "--idle-timeout-sec",
        type=float,
        default=600.0,
        help="Maximum seconds to wait between coordinator messages before exiting.",
    )
    parser.add_argument(
        "--preload-sbert",
        action="store_true",
        help="Eagerly load the SBERT encoder at worker startup so per-process "
             "RSS sampling captures the model footprint (text-mode runs).",
    )
    args = parser.parse_args()

    transport = ZmqAgentTransport(
        endpoint=args.coord_addr,
        agent_id=args.agent_id,
        serialization=args.serialization,
    )
    transport.connect()

    state: Dict[str, Any] = {
        "agent_id": args.agent_id,
        "pid": os.getpid(),
        "started_at": time.time(),
        "last_item_count": 0,
        "reliability": {},
    }

    if args.preload_sbert:
        _maybe_load_sbert(state)

    transport.send(
        {
            "type": "hello",
            "agent_id": args.agent_id,
            "pid": os.getpid(),
            "started_at_ms": _now_ms(),
        }
    )

    try:
        while True:
            try:
                message = transport.recv(timeout_sec=args.idle_timeout_sec)
            except TimeoutError:
                break
            reply = _handle(message, state)
            transport.send(reply)
            if message.get("type") == "shutdown":
                break
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 2
    finally:
        transport.close()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
