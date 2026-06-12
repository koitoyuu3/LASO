
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Sequence

import zmq

from .serialization import decode, encode
from .transport_base import AgentTransport, CoordinatorTransport

_DEFAULT_LINGER_MS = 0
_DEFAULT_RECV_TIMEOUT_MS = 60_000

class ZmqCoordinatorTransport(CoordinatorTransport):
    def __init__(self, endpoint: str, serialization: str = "msgpack"):
        super().__init__(endpoint=endpoint, serialization=serialization)
        self._context: Optional[zmq.Context] = None
        self._router: Optional[zmq.Socket] = None
        self._pending_send_ts: Dict[str, float] = {}

    def start(self, expected_agents: Sequence[str]) -> None:
        self._context = zmq.Context.instance()
        self._router = self._context.socket(zmq.ROUTER)
        self._router.setsockopt(zmq.LINGER, _DEFAULT_LINGER_MS)
        self._router.bind(self.endpoint)

    def close(self) -> None:
        if self._router is not None:
            self._router.close(linger=_DEFAULT_LINGER_MS)
            self._router = None

    def send(self, agent_id: str, payload: Any) -> None:
        assert self._router is not None, "Coordinator transport not started."
        blob = encode(payload, backend=self.serialization)
        self._router.send_multipart([agent_id.encode("utf-8"), blob])
        self.stats.bytes_sent += len(agent_id) + len(blob)
        self.stats.messages_sent += 1
        self._pending_send_ts[agent_id] = time.perf_counter()

    def broadcast(self, agent_ids: Sequence[str], payload: Any) -> None:

        blob = encode(payload, backend=self.serialization)
        assert self._router is not None
        now = time.perf_counter()
        for agent_id in agent_ids:
            self._router.send_multipart([agent_id.encode("utf-8"), blob])
            self.stats.bytes_sent += len(agent_id) + len(blob)
            self.stats.messages_sent += 1
            self._pending_send_ts[agent_id] = now

    def gather(
        self,
        agent_ids: Sequence[str],
        timeout_sec: Optional[float] = None,
    ) -> Dict[str, Any]:
        assert self._router is not None
        outstanding = set(agent_ids)
        replies: Dict[str, Any] = {}
        deadline = None if timeout_sec is None else time.perf_counter() + timeout_sec
        poller = zmq.Poller()
        poller.register(self._router, zmq.POLLIN)
        while outstanding:
            wait_ms = _DEFAULT_RECV_TIMEOUT_MS
            if deadline is not None:
                remaining = max(0.0, deadline - time.perf_counter())
                wait_ms = int(remaining * 1000)
                if wait_ms <= 0:
                    raise TimeoutError(f"gather timed out, missing={sorted(outstanding)}")
            socks = dict(poller.poll(wait_ms))
            if self._router not in socks:
                raise TimeoutError(f"gather poll timed out, missing={sorted(outstanding)}")
            frames = self._router.recv_multipart()

            sender = frames[0].decode("utf-8")
            blob = frames[-1]
            self.stats.bytes_recv += len(frames[0]) + len(blob)
            self.stats.messages_recv += 1
            send_ts = self._pending_send_ts.pop(sender, None)
            if send_ts is not None:
                self.stats.rpc_latencies_ms.append((time.perf_counter() - send_ts) * 1000.0)
            replies[sender] = decode(blob, backend=self.serialization)
            outstanding.discard(sender)
        return replies

class ZmqAgentTransport(AgentTransport):
    def __init__(self, endpoint: str, agent_id: str, serialization: str = "msgpack"):
        super().__init__(endpoint=endpoint, agent_id=agent_id, serialization=serialization)
        self._context: Optional[zmq.Context] = None
        self._dealer: Optional[zmq.Socket] = None

    def connect(self) -> None:
        self._context = zmq.Context.instance()
        self._dealer = self._context.socket(zmq.DEALER)
        self._dealer.setsockopt(zmq.IDENTITY, self.agent_id.encode("utf-8"))
        self._dealer.setsockopt(zmq.LINGER, _DEFAULT_LINGER_MS)
        self._dealer.connect(self.endpoint)

    def close(self) -> None:
        if self._dealer is not None:
            self._dealer.close(linger=_DEFAULT_LINGER_MS)
            self._dealer = None

    def recv(self, timeout_sec: Optional[float] = None) -> Any:
        assert self._dealer is not None, "Agent transport not connected."
        if timeout_sec is not None:
            poller = zmq.Poller()
            poller.register(self._dealer, zmq.POLLIN)
            socks = dict(poller.poll(int(timeout_sec * 1000)))
            if self._dealer not in socks:
                raise TimeoutError("agent recv timed out")
        frames = self._dealer.recv_multipart()
        blob = frames[-1]
        self.stats.bytes_recv += sum(len(f) for f in frames)
        self.stats.messages_recv += 1
        return decode(blob, backend=self.serialization)

    def send(self, payload: Any) -> None:
        assert self._dealer is not None
        blob = encode(payload, backend=self.serialization)
        self._dealer.send_multipart([blob])
        self.stats.bytes_sent += len(blob)
        self.stats.messages_sent += 1
