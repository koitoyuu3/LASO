from __future__ import annotations

import base64
import copy
import json
import logging
import os
import re
import shlex
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

DEFAULT_ALLOWED_EXTENSIONS = [
    ".csv",
    ".json",
    ".jsonl",
    ".txt",
    ".md",
    ".pk",
    ".pickle",
    ".parquet",
]

DEFAULT_TOOL_TIMEOUT_SECONDS = 120
DEFAULT_MODEL_TIMEOUT_SECONDS = DEFAULT_TOOL_TIMEOUT_SECONDS
DEFAULT_CONNECT_TIMEOUT_SECONDS = 10
DEFAULT_MODEL_NUM_PREDICT = 8192
DEFAULT_MAX_CONCURRENT_LOCAL_NODES = 1
DEFAULT_LOGICAL_NODE_COUNT = 50
DEFAULT_BACKEND_PORT_START = 11431
DEFAULT_BACKEND_PORT_COUNT = 4
DEFAULT_MAX_CONCURRENT_REMOTE_NODES = DEFAULT_BACKEND_PORT_COUNT
DEFAULT_MAX_CONCURRENT_PER_BACKEND = 1
PREFERRED_PRICE_SOURCES = ("kucoin", "binance", "kraken", "bitfinex", "cryptocompare", "coingecko")
PRICE_CANDIDATE_PREVIEW_LIMIT = 180
DEFAULT_BATCH_PRICE_INTERVAL_MINUTES = 30
DEFAULT_REMOTE_MODEL = "qwen2.5:7b"
DEFAULT_GPU_COUNT_HINT = 4


def _build_default_backend_configs(
    backend_count: int = DEFAULT_BACKEND_PORT_COUNT,
    port_start: int = DEFAULT_BACKEND_PORT_START,
) -> List[Dict[str, Any]]:
    configs: List[Dict[str, Any]] = []
    for index in range(backend_count):
        configs.append(
            {
                "backendName": f"backend-{index + 1}",
                "url": f"http://127.0.0.1:{port_start + index}",
                "model": DEFAULT_REMOTE_MODEL,
                "gpuHint": index % DEFAULT_GPU_COUNT_HINT,
            }
        )
    return configs


def _build_default_node_configs(
    node_count: int = DEFAULT_LOGICAL_NODE_COUNT,
    backend_configs: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    backends = backend_configs or _build_default_backend_configs()
    if not backends:
        return []
    configs: List[Dict[str, Any]] = []
    for index in range(node_count):
        backend = backends[index % len(backends)]
        configs.append(
            {
                "nodeName": f"agent-{index + 1}",
                "url": backend["url"],
                "model": backend["model"],
                "backendName": backend["backendName"],
                "backendIndex": index % len(backends),
                "gpuHint": backend.get("gpuHint"),
            }
        )
    return configs


DEFAULT_BACKEND_CONFIGS = _build_default_backend_configs()
DEFAULT_NODE_CONFIGS = _build_default_node_configs(backend_configs=DEFAULT_BACKEND_CONFIGS)


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _coerce_optional_positive_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else None


def _coerce_bool(value: Any, default: bool) -> bool:
    if value in (None, ""):
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _is_local_url(url: Any) -> bool:
    value = str(url or "").strip()
    if not value:
        return True
    parsed = urlparse(value if "://" in value else f"http://{value}")
    host = (parsed.hostname or "").strip().lower()
    return host in {"", "127.0.0.1", "localhost", "::1"}

REMOTE_TOOL_SCRIPT = r'''
from __future__ import annotations

import base64
import csv
import json
import os
import pickle
import re
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional


SUPPORTED_PRICE_FILE_EXTENSIONS = (".pk", ".pickle", ".json", ".jsonl", ".csv")
PREFERRED_PRICE_SOURCES = ("kucoin", "binance", "kraken", "bitfinex", "cryptocompare", "coingecko")


def _split_tokens(text: str) -> List[str]:
    tokens = re.split(r"[^\w\u4e00-\u9fff]+", text.lower())
    return [token for token in tokens if token]


def _normalize_records(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        for key in ("records", "data", "items", "rows", "news", "articles"):
            value = data.get(key)
            if isinstance(value, list):
                records = value
                break
        else:
            records = [data]
    else:
        records = [data]

    normalized = []
    for item in records:
        if isinstance(item, dict):
            normalized.append(item)
        else:
            normalized.append({"value": str(item)})
    return normalized


def _truncate_value(value: Any, max_len: int = 240) -> Any:
    if isinstance(value, str):
        return value if len(value) <= max_len else value[:max_len]
    return value


def _validate_path(path: str, root: str) -> str:
    real_root = os.path.realpath(root)
    real_path = os.path.realpath(path)
    if real_path != real_root and not real_path.startswith(real_root + os.sep):
        raise ValueError(f"path out of allowed root: {path}")
    return real_path


def _load_records(path: str) -> List[Dict[str, Any]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        with open(path, "r", encoding="utf-8", newline="", errors="ignore") as fh:
            return [dict(row) for row in csv.DictReader(fh)]
    if ext == ".json":
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return _normalize_records(json.load(fh))
    if ext == ".jsonl":
        records = []
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return _normalize_records(records)
    if ext in (".txt", ".md"):
        rows = []
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for index, line in enumerate(fh, start=1):
                rows.append({"line_number": index, "text": line.rstrip("\n")})
        return rows
    if ext in (".pk", ".pickle"):
        with open(path, "rb") as fh:
            return _normalize_records(pickle.load(fh))
    if ext == ".parquet":
        try:
            import pandas as pd
        except Exception as exc:
            raise RuntimeError(f"parquet support unavailable: {exc}") from exc
        frame = pd.read_parquet(path)
        return _normalize_records(frame.to_dict(orient="records"))
    raise ValueError(f"unsupported file type: {ext}")


def _discover_data_files(payload: Dict[str, Any]) -> Dict[str, Any]:
    root = payload["data_root"]
    allowed_extensions = {ext.lower() for ext in payload.get("allowed_extensions") or []}
    limit = int(payload.get("limit") or 20)
    keywords = [str(item).lower() for item in (payload.get("keywords") or []) if str(item).strip()]
    prompt = str(payload.get("prompt") or "")
    if not keywords and prompt:
        keywords = _split_tokens(prompt)

    real_root = _validate_path(root, root)
    matches = []
    for dirpath, _, filenames in os.walk(real_root):
        for filename in filenames:
            path = os.path.join(dirpath, filename)
            ext = os.path.splitext(filename)[1].lower()
            if allowed_extensions and ext not in allowed_extensions:
                continue
            lower_path = path.lower()
            score = 0
            for keyword in keywords:
                if not keyword:
                    continue
                if keyword in lower_path:
                    score += 10 if keyword in filename.lower() else 3
            stat = os.stat(path)
            matches.append({
                "path": path,
                "filename": filename,
                "extension": ext,
                "size": stat.st_size,
                "modified_at": int(stat.st_mtime),
                "score": score,
            })
    matches.sort(key=lambda item: (-item["score"], item["path"]))
    return {
        "files": matches[:limit],
        "total": len(matches),
    }


def _peek_schema(payload: Dict[str, Any]) -> Dict[str, Any]:
    root = payload["data_root"]
    path = _validate_path(payload["path"], root)
    sample_size = int(payload.get("sample_size") or 3)
    records = _load_records(path)
    sample = []
    for item in records[:sample_size]:
        if isinstance(item, dict):
            sample.append({key: _truncate_value(value) for key, value in item.items()})
        else:
            sample.append(_truncate_value(item))
    fields = []
    if sample and isinstance(sample[0], dict):
        fields = list(sample[0].keys())
    return {
        "path": path,
        "record_count": len(records),
        "fields": fields,
        "sample": sample,
    }


def _read_records(payload: Dict[str, Any]) -> Dict[str, Any]:
    root = payload["data_root"]
    path = _validate_path(payload["path"], root)
    offset = int(payload.get("offset") or 0)
    limit = int(payload.get("limit") or 20)
    records = _load_records(path)
    sliced = records[offset: offset + limit]
    output = []
    for item in sliced:
        if isinstance(item, dict):
            output.append({key: _truncate_value(value, 400) for key, value in item.items()})
        else:
            output.append(_truncate_value(item, 400))
    return {
        "path": path,
        "offset": offset,
        "limit": limit,
        "returned": len(output),
        "has_more": offset + len(output) < len(records),
        "records": output,
    }


def main() -> None:
    action = sys.argv[1]
    payload = json.loads(base64.urlsafe_b64decode(sys.argv[2].encode("ascii")).decode("utf-8"))
    actions = {
        "discover_data_files": _discover_data_files,
        "peek_schema": _peek_schema,
        "read_records": _read_records,
    }
    if action not in actions:
        raise ValueError(f"unsupported action: {action}")
    result = actions[action](payload)
    print(json.dumps({"success": True, "data": result}, ensure_ascii=False, default=str))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False))
        raise
'''



class AgentJobStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create_job(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        job_id = uuid.uuid4().hex
        now = self._now()
        job = {
            "jobId": job_id,
            "requestId": str(request_payload.get("requestId") or ""),
            "prompt": request_payload.get("prompt") or "",
            "dataRoot": request_payload.get("dataRoot") or "/home/***/data",
            "nodeCount": 0,
            "status": "queued",
            "createdAt": now,
            "updatedAt": now,
            "workflowMode": str(request_payload.get("workflowMode") or ""),
            "results": [],
            "batches": [],
            "output": None,
            "outputFormat": None,
            "recordCount": 0,
            "progress": {
                "phase": "queued",
                "newsTotal": 0,
                "newsCompleted": 0,
                "priceTotal": 0,
                "priceCompleted": 0,
                "totalOutputRecords": 0,
            },
            "error": None,
        }
        with self._lock:
            self._jobs[job_id] = job
        return copy.deepcopy(job)

    def mark_running(self, job_id: str, node_count: Optional[int] = None) -> None:
        updates: Dict[str, Any] = {"status": "running", "updatedAt": self._now()}
        if node_count is not None:
            updates["nodeCount"] = node_count
        self._update(job_id, updates)

    def set_workflow_mode(self, job_id: str, workflow_mode: str) -> None:
        self._update(job_id, {"workflowMode": workflow_mode, "updatedAt": self._now()})

    def set_node_result(self, job_id: str, result: Dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs[job_id]
            results = [item for item in job.get("results", []) if item.get("nodeIndex") != result.get("nodeIndex")]
            results.append(result)
            results.sort(key=lambda item: item.get("nodeIndex", 0))
            job["results"] = results
            job["updatedAt"] = self._now()

    def append_batch(self, job_id: str, batch: Dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs[job_id]
            batches = list(job.get("batches", []))
            batches.append(batch)
            batches.sort(key=lambda item: item.get("batchIndex", 0))
            job["batches"] = batches
            job["updatedAt"] = self._now()

    def set_batches(
        self,
        job_id: str,
        batches: List[Dict[str, Any]],
        record_count: Optional[int] = None,
    ) -> None:
        updates: Dict[str, Any] = {
            "batches": list(batches),
            "updatedAt": self._now(),
        }
        if record_count is not None:
            updates["recordCount"] = record_count
        self._update(job_id, updates)

    def update_progress(self, job_id: str, progress_updates: Dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs[job_id]
            progress = dict(job.get("progress") or {})
            progress.update(progress_updates)
            job["progress"] = progress
            job["updatedAt"] = self._now()

    def set_output(
        self,
        job_id: str,
        output: Any,
        output_format: str,
        record_count: Optional[int] = None,
    ) -> None:
        updates: Dict[str, Any] = {
            "output": output,
            "outputFormat": output_format,
            "updatedAt": self._now(),
        }
        if record_count is not None:
            updates["recordCount"] = record_count
        self._update(job_id, updates)

    def mark_completed(self, job_id: str) -> None:
        self._update(job_id, {"status": "completed", "updatedAt": self._now()})

    def mark_failed(self, job_id: str, error: str) -> None:
        self._update(job_id, {"status": "failed", "error": error, "updatedAt": self._now()})

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            return copy.deepcopy(job) if job else None

    def _update(self, job_id: str, updates: Dict[str, Any]) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.update(updates)

    @staticmethod
    def _now() -> str:
        return datetime.utcnow().isoformat() + "Z"


class RemoteDataToolExecutor:
    def __init__(
        self,
        ssh_host: str,
        ssh_user: str,
        connect_timeout_seconds: int,
        timeout_seconds: int,
        data_root: str,
        allowed_extensions: Optional[List[str]] = None,
    ) -> None:
        self.ssh_host = ssh_host
        self.ssh_user = ssh_user
        self.connect_timeout_seconds = connect_timeout_seconds
        self.timeout_seconds = timeout_seconds
        self.data_root = data_root
        self.allowed_extensions = allowed_extensions or list(DEFAULT_ALLOWED_EXTENSIONS)

    def execute(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(args or {})
        payload.setdefault("data_root", self.data_root)
        payload.setdefault("allowed_extensions", self.allowed_extensions)
        return self._run_remote_action(tool_name, payload)

    def _run_remote_action(self, action: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        encoded_payload = base64.urlsafe_b64encode(
            json.dumps(payload, ensure_ascii=False).encode("utf-8")
        ).decode("ascii")
        if self.is_local_target():
            command = ["python3", "-", action, encoded_payload]
        else:
            command = [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                f"ConnectTimeout={self.connect_timeout_seconds}",
                self._build_target(),
                "python3",
                "-",
                action,
                encoded_payload,
            ]
        process = subprocess.run(
            command,
            input=REMOTE_TOOL_SCRIPT.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout_seconds,
            check=False,
        )
        stdout = process.stdout.decode("utf-8", errors="ignore").strip()
        stderr = process.stderr.decode("utf-8", errors="ignore").strip()
        if process.returncode != 0:
            raise RuntimeError(stderr or stdout or f"remote tool {action} failed")
        if not stdout:
            raise RuntimeError(f"remote tool {action} returned empty stdout")
        response = json.loads(stdout)
        if not response.get("success"):
            raise RuntimeError(response.get("error") or f"remote tool {action} failed")
        return response.get("data") or {}

    def is_local_target(self) -> bool:
        host = (self.ssh_host or "").strip().lower()
        return host in {"", "127.0.0.1", "localhost", "::1"}

    def _build_target(self) -> str:
        if not self.ssh_user:
            return self.ssh_host
        return f"{self.ssh_user}@{self.ssh_host}"


class StructuredToolAgentRunner:
    def __init__(
        self,
        prompt: str,
        node_config: Dict[str, Any],
        tool_executor: RemoteDataToolExecutor,
        timeout_seconds: Optional[int],
        connect_timeout_seconds: int,
        model_num_predict: int,
        max_steps: int,
        read_limit: int,
        task_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.prompt = prompt
        self.node_config = node_config
        self.tool_executor = tool_executor
        self.timeout_seconds = timeout_seconds
        self.connect_timeout_seconds = connect_timeout_seconds
        self.model_num_predict = model_num_predict
        self.max_steps = max_steps
        self.read_limit = read_limit
        self.task_payload = task_payload or {}
        self.tool_trace: List[Dict[str, Any]] = []

    def run(self) -> Dict[str, Any]:
        started_at = time.time()
        if self.task_payload.get("mode") == "paired_batch_item":
            return self._run_paired_batch_item(started_at)
        if self.task_payload.get("mode") == "news_summary_item":
            return self._run_news_summary_item(started_at)
        if self.task_payload.get("mode") == "weather_article_summary_item":
            return self._run_weather_article_summary_item(started_at)
        self.tool_trace.append(
            {
                "step": 0,
                "tool": "raw_prompt_only",
                "args": {
                    "promptLength": len(self.prompt),
                },
                "result": {
                    "mode": "exact_user_prompt",
                },
            }
        )
        raw_text = self._invoke_model(self.prompt, json_mode=False, num_predict=self.model_num_predict)
        response_text = raw_text.strip()
        if not response_text:
            return self._build_result(started_at, False, None, "model response empty")
        output_format = "json-array" if response_text.startswith("[") else "text"
        return self._build_result(started_at, True, response_text, None, output_format=output_format)

    def _run_paired_batch_item(self, started_at: float) -> Dict[str, Any]:
        batch_index = int(self.task_payload.get("batchIndex") or 0)
        news_index = int(self.task_payload.get("newsIndex") or 0)
        self.tool_trace.append(
            {
                "step": 0,
                "tool": "paired_batch_item",
                "args": {
                    "batchIndex": batch_index,
                    "newsIndex": news_index,
                },
                "result": {
                    "mode": "json-object",
                },
            }
        )
        prompt = self._build_paired_batch_prompt()
        raw_text = self._invoke_model(
            prompt,
            json_mode=True,
            num_predict=min(self.model_num_predict, 900),
            temperature=0.0,
        )
        items = self._normalize_paired_batch_response(raw_text)
        response_text = json.dumps(items, ensure_ascii=False)
        return self._build_result(
            started_at,
            True,
            response_text,
            None,
            output_format="json-array",
            record_count=len(items),
        )

    def _run_news_summary_item(self, started_at: float) -> Dict[str, Any]:
        news_index = int(self.task_payload.get("newsIndex") or 0)
        self.tool_trace.append(
            {
                "step": 0,
                "tool": "news_summary_item",
                "args": {
                    "newsIndex": news_index,
                    "timestampMissing": bool(self.task_payload.get("timestampMissing")),
                },
                "result": {
                    "mode": "json-object",
                },
            }
        )
        prompt = self._build_news_summary_prompt()
        raw_text = self._invoke_model(
            prompt,
            json_mode=True,
            num_predict=min(self.model_num_predict, 320),
            temperature=0.0,
        )
        summary_text = self._normalize_news_summary_response(
            raw_text,
            bool(self.task_payload.get("timestampMissing")),
        )
        return self._build_result(
            started_at,
            True,
            summary_text,
            None,
            output_format="text",
            record_count=1,
        )

    def _run_weather_article_summary_item(self, started_at: float) -> Dict[str, Any]:
        article_index = int(self.task_payload.get("articleIndex") or 0)
        self.tool_trace.append(
            {
                "step": 0,
                "tool": "weather_article_summary_item",
                "args": {
                    "articleIndex": article_index,
                    "timestampMissing": bool(self.task_payload.get("timestampMissing")),
                },
                "result": {
                    "mode": "json-object",
                },
            }
        )
        prompt = self._build_weather_article_summary_prompt()
        raw_text = self._invoke_model(
            prompt,
            json_mode=True,
            num_predict=min(self.model_num_predict, 320),
            temperature=0.0,
        )
        summary_text = self._normalize_news_summary_response(
            raw_text,
            bool(self.task_payload.get("timestampMissing")),
        )
        return self._build_result(
            started_at,
            True,
            summary_text,
            None,
            output_format="text",
            record_count=1,
        )

    def _build_news_summary_prompt(self) -> str:
        news_record = self.task_payload.get("newsRecord") or {}
        lines = [
            "你是一名精通多模态金融数据处理的数据工程助手。",
            "系统已经从远端数据源中读取出1条真实财经新闻记录，下面是原始数据。",
            "你必须直接阅读这条记录，并只输出合法 JSON 对象，不要输出解释，不要输出 Markdown，不要输出代码块。",
            'JSON 结构固定为：{"summary":"..."}',
            "规则：",
            "1. summary 必须是一句中文简短摘要，且同时包含核心事件、涉及实体、关键财务数字。",
            "2. 如果原文没有明确财务数字，必须写“未披露明确财务数字”。",
            "3. 不得编造新闻内容，不得输出任何 schema 之外的字段。",
            "4. 可参考时间字段优先级：published_at > timestamp > date > created_at。",
            f"当前 nodeName: {self.node_config.get('nodeName')}",
            f"当前 newsIndex: {self.task_payload.get('newsIndex')}",
            "[新闻原始记录]",
            json.dumps(news_record, ensure_ascii=False),
        ]
        return "\n".join(lines)

    def _build_weather_article_summary_prompt(self) -> str:
        article_record = self.task_payload.get("articleRecord") or {}
        lines = [
            "你是一名精通天气文本处理与数据清洗的数据工程助手。",
            "系统已经从远端数据源中读取出1条真实天气文章记录，下面是原始数据。",
            "你必须直接阅读这条记录，并只输出合法 JSON 对象，不要输出解释，不要输出 Markdown，不要输出代码块。",
            'JSON 结构固定为：{"summary":"..."}',
            "规则：",
            "1. summary 必须是一句中文简短摘要，且同时包含核心天气事件、涉及地点或实体、关键气象数值。",
            "2. 关键气象数值包括温度、降水概率、风速、风向、湿度、露点、告警数量等正文中明确出现的数值。",
            "3. 如果原文没有明确气象数值，必须写“未披露明确数值”。",
            "4. 必须以 `Article` 正文作为主要依据，不得直接抄写或改写 `Lsa_summary`、`Luhn_summary`、`Textrank_summary`、`Lexrank_summary`。",
            "5. 如果正文明确写明无活动告警，应如实体现；如果有未来天气变化，也可简要体现，但不得超出原文。",
            "6. 不得编造天气情况、地点、数值或趋势，不得输出任何 schema 之外的字段。",
            "7. 时间字段优先级：Date > published_at > timestamp > date > created_at。",
            f"当前 nodeName: {self.node_config.get('nodeName')}",
            f"当前 articleIndex: {self.task_payload.get('articleIndex')}",
            "[天气文章原始记录]",
            json.dumps(article_record, ensure_ascii=False),
        ]
        return "\n".join(lines)

    def _normalize_news_summary_response(self, raw_text: str, timestamp_missing: bool) -> str:
        summary = ""
        parsed_text = raw_text.strip()
        if not parsed_text:
            raise RuntimeError("news summary response empty")
        try:
            parsed = json.loads(parsed_text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            for key in ("summary", "newsSummary", "response"):
                value = parsed.get(key)
                if value:
                    summary = str(value).strip()
                    break
        elif isinstance(parsed, str):
            summary = parsed.strip()
        if not summary:
            summary = parsed_text
        summary = summary.strip()
        if timestamp_missing and "时间戳缺失" not in summary:
            summary = summary.rstrip("。")
            summary = f"{summary}，时间戳缺失。"
        return summary

    def _build_paired_batch_prompt(self) -> str:
        news_record = self.task_payload.get("newsRecord") or {}
        price_record = self.task_payload.get("priceRecord") or {}
        lines = [
            "你是一名精通多模态金融数据处理的数据工程助手。",
            "系统已经从远端数据源中读取出1条真实新闻记录和1条真实价格记录，下面两段原始数据需要由你亲自阅读。",
            "你必须只输出合法 JSON 对象，不要输出解释，不要输出 Markdown，不要输出代码块。",
            'JSON 结构固定为：{"newsSummary":"...","priceResponse":{"BTC":0,"ETH":0,"DOGE":0}}',
            "规则：",
            "1. `newsSummary` 必须是一句中文摘要，且同时包含核心事件、涉及实体、关键财务数字。",
            "2. 如果新闻原文没有明确财务数字，必须写‘未披露明确财务数字’。",
            "3. `priceResponse` 必须严格抄录给定价格记录中的 BTC、ETH、DOGE 数值，不得改写，不得补充。",
            "4. 不要输出 `agent` 和 `object`，系统会自行填写。",
            f"当前 nodeName: {self.node_config.get('nodeName')}",
            f"当前 batchIndex: {self.task_payload.get('batchIndex')}",
            f"当前 newsIndex: {self.task_payload.get('newsIndex')}",
            "[新闻原始记录]",
            json.dumps(news_record, ensure_ascii=False),
            "[价格原始记录]",
            json.dumps(price_record, ensure_ascii=False),
        ]
        return "\n".join(lines)

    def _normalize_paired_batch_response(self, raw_text: str) -> List[Dict[str, Any]]:
        parsed = json.loads(raw_text)
        news_summary = ""
        price_response: Optional[Dict[str, Any]] = None
        if isinstance(parsed, dict):
            news_value = parsed.get("newsSummary") or parsed.get("news_summary")
            if not news_value:
                news_block = parsed.get("news")
                if isinstance(news_block, dict):
                    news_value = news_block.get("response") or news_block.get("summary")
                elif isinstance(news_block, str):
                    news_value = news_block
            if news_value:
                news_summary = str(news_value).strip()
            price_value = parsed.get("priceResponse") or parsed.get("price_response")
            if price_value is None:
                price_block = parsed.get("price")
                if isinstance(price_block, dict):
                    price_value = price_block.get("response") if "response" in price_block else price_block
            if isinstance(price_value, dict):
                price_response = price_value
        if not news_summary:
            raise RuntimeError("paired batch response missing newsSummary")

        node_name = str(self.node_config.get("nodeName") or "")
        news_index = int(self.task_payload.get("newsIndex") or 0)
        price_record = self.task_payload.get("priceRecord") or {}
        timestamp_text = str(price_record.get("timestamp") or "")
        validated_price = {
            key: self._coerce_number((price_response or {}).get(key))
            for key in ("BTC", "ETH", "DOGE")
        }
        for key in ("BTC", "ETH", "DOGE"):
            source_value = self._coerce_number(price_record.get(key))
            if source_value is not None:
                validated_price[key] = source_value
        items = [
            {
                "agent": node_name,
                "object": f"第{news_index}条新闻简短摘要",
                "response": news_summary,
            },
            {
                "agent": node_name,
                "object": f"{timestamp_text} 的 BTC/ETH/DOGE 价格数据",
                "response": validated_price,
            },
        ]
        return items

    def _coerce_number(self, value: Any) -> Optional[Any]:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            number = float(value)
        else:
            match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
            if not match:
                return None
            number = float(match.group(0))
        return int(number) if number.is_integer() else number

    def _invoke_model(
        self,
        prompt: str,
        json_mode: bool,
        num_predict: int = 160,
        temperature: float = 0.1,
    ) -> str:
        request_body: Dict[str, Any] = {
            "model": self.node_config.get("model"),
            "prompt": prompt,
            "stream": False,
        }
        request_body["options"] = {"temperature": temperature, "num_predict": num_predict}
        if json_mode:
            request_body["format"] = "json"

        generate_url = self._normalize_base_url(self.node_config.get('url')) + '/api/generate'
        if self.tool_executor.is_local_target():
            command = ["curl", "-sS", "--connect-timeout", str(self.connect_timeout_seconds)]
            if self.timeout_seconds is not None:
                command.extend(["--max-time", str(self.timeout_seconds)])
            command.extend([
                "-H",
                "Content-Type: application/json",
                "--data-binary",
                "@-",
                generate_url,
            ])
            timeout = self.timeout_seconds + 5 if self.timeout_seconds is not None else None
        else:
            remote_command_parts = [
                "curl",
                "-sS",
                "--connect-timeout",
                str(self.connect_timeout_seconds),
            ]
            if self.timeout_seconds is not None:
                remote_command_parts.extend(["--max-time", str(self.timeout_seconds)])
            remote_command_parts.extend([
                "-H",
                "'Content-Type: application/json'",
                "--data-binary",
                "@-",
                shlex.quote(generate_url),
            ])
            remote_command = " ".join(remote_command_parts)
            command = [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                f"ConnectTimeout={self.connect_timeout_seconds}",
                self.tool_executor._build_target(),
                remote_command,
            ]
            timeout = (
                self.timeout_seconds + self.connect_timeout_seconds + 5
                if self.timeout_seconds is not None
                else None
            )
        try:
            process = subprocess.run(
                command,
                input=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            node_name = self.node_config.get("nodeName") or "unknown-node"
            raise RuntimeError(
                f"model request exceeded subprocess timeout after {timeout}s: "
                f"node={node_name}, url={generate_url}, num_predict={num_predict}"
            ) from exc
        stdout = process.stdout.decode("utf-8", errors="ignore").strip()
        stderr = process.stderr.decode("utf-8", errors="ignore").strip()
        if process.returncode != 0:
            if process.returncode == 28 and "Operation timed out" in (stderr or stdout):
                node_name = self.node_config.get("nodeName") or "unknown-node"
                model_name = self.node_config.get("model") or "unknown-model"
                timeout_hint = f"max_time={self.timeout_seconds}s, " if self.timeout_seconds is not None else ""
                guidance = (
                    "Increase modelTimeoutSeconds or lower modelNumPredict."
                    if self.timeout_seconds is not None
                    else "Check node availability or increase connectTimeoutSeconds."
                )
                raise RuntimeError(
                    f"curl request timed out: node={node_name}, model={model_name}, url={generate_url}, "
                    f"{timeout_hint}connect_timeout={self.connect_timeout_seconds}s, num_predict={num_predict}. "
                    f"{guidance}"
                )
            raise RuntimeError(stderr or stdout or f"invoke model failed: {self.node_config.get('nodeName')}")
        response = json.loads(stdout)
        if response.get("error"):
            error_text = str(response.get("error"))
            normalized_error = error_text.lower()
            if (
                "out of memory" in normalized_error
                or "cudamalloc failed" in normalized_error
                or "unable to allocate cuda" in normalized_error
            ):
                node_name = self.node_config.get("nodeName") or "unknown-node"
                model_name = self.node_config.get("model") or "unknown-model"
                raise RuntimeError(
                    f"{error_text}. node={node_name}, model={model_name}. "
                    "GPU memory is insufficient; reduce concurrent local nodes, use a smaller model, or shorten context."
                )
            raise RuntimeError(error_text)
        return str(response.get("response") or "")

    @staticmethod
    def _normalize_base_url(url: Optional[str]) -> str:
        value = (url or "").strip()
        return value[:-1] if value.endswith("/") else value

    def _build_result(
        self,
        started_at: float,
        success: bool,
        summary: Optional[str],
        error: Optional[str],
        output_format: str = "text",
        record_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        result = {
            "nodeIndex": self.node_config.get("nodeIndex"),
            "nodeName": self.node_config.get("nodeName"),
            "model": self.node_config.get("model"),
            "ollamaUrl": self.node_config.get("url"),
            "success": success,
            "summary": summary,
            "error": error,
            "outputFormat": output_format,
            "durationMs": int((time.time() - started_at) * 1000),
            "toolTrace": self.tool_trace,
        }
        if record_count is not None:
            result["recordCount"] = record_count
        return result


class OllamaAgentService:
    def __init__(self) -> None:
        self.job_store = AgentJobStore()

    def submit_job(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        job = self.job_store.create_job(payload)
        worker = threading.Thread(target=self._run_job, args=(job["jobId"], payload), daemon=True)
        worker.start()
        return job

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        return self.job_store.get_job(job_id)

    def _run_job(self, job_id: str, payload: Dict[str, Any]) -> None:
        try:
            prompt = str(payload.get("prompt") or "")
            explicit_workflow_mode = str(payload.get("workflowMode") or "").strip()
            workflow_mode = self._resolve_workflow_mode(payload, prompt)
            if self._looks_like_truthfinder_weather_prompt(prompt) and workflow_mode != "truthfinder_weather_full":
                logger.warning(
                    "forcing workflow mode to truthfinder_weather_full for TruthFinder weather prompt; "
                    "explicit=%s, resolved=%s",
                    explicit_workflow_mode or "<empty>",
                    workflow_mode or "<empty>",
                )
                workflow_mode = "truthfinder_weather_full"
            if self._looks_like_truthfinder_finance_prompt(prompt) and workflow_mode != "truthfinder_finance_full":
                logger.warning(
                    "forcing workflow mode to truthfinder_finance_full for TruthFinder finance prompt; "
                    "explicit=%s, resolved=%s",
                    explicit_workflow_mode or "<empty>",
                    workflow_mode or "<empty>",
                )
                workflow_mode = "truthfinder_finance_full"
            if workflow_mode:
                self.job_store.set_workflow_mode(job_id, workflow_mode)
            custom_nodes = payload.get("nodes") if isinstance(payload.get("nodes"), list) else None
            allow_custom_nodes = _coerce_bool(payload.get("allowCustomNodes"), False)
            requested_node_count: Any = payload.get("nodeCount")
            if workflow_mode == "truthfinder_weather_full":
                requested_node_count = self._extract_truthfinder_agent_count(prompt) or requested_node_count or 50
            if workflow_mode == "truthfinder_finance_full":
                requested_node_count = self._extract_truthfinder_agent_count(prompt) or requested_node_count or 50
            if workflow_mode in {"finance_timeline_full", "paired_batch_150", "truthfinder_finance_full", "truthfinder_weather_full"} and custom_nodes:
                logger.info(
                    "workflow %s ignores %s custom nodes and uses the default backend pool (allowCustomNodes=%s)",
                    workflow_mode,
                    len(custom_nodes),
                    allow_custom_nodes,
                )
                nodes = self._normalize_nodes(None, requested_node_count)
            else:
                nodes = self._normalize_nodes(custom_nodes, requested_node_count)
            self.job_store.mark_running(job_id, len(nodes))
            tool_timeout_seconds = _coerce_positive_int(
                payload.get("timeoutSeconds"),
                DEFAULT_TOOL_TIMEOUT_SECONDS,
            )
            model_timeout_seconds = _coerce_optional_positive_int(
                payload.get("modelTimeoutSeconds"),
                DEFAULT_MODEL_TIMEOUT_SECONDS,
            )
            connect_timeout_seconds = _coerce_positive_int(
                payload.get("connectTimeoutSeconds"),
                DEFAULT_CONNECT_TIMEOUT_SECONDS,
            )
            model_num_predict = _coerce_positive_int(
                payload.get("modelNumPredict"),
                DEFAULT_MODEL_NUM_PREDICT,
            )
            requested_max_workers = _coerce_positive_int(payload.get("maxConcurrentNodes"), 0)
            all_local_nodes = all(_is_local_url(node.get("url")) for node in nodes)
            default_max_workers = (
                DEFAULT_MAX_CONCURRENT_LOCAL_NODES
                if all_local_nodes
                else DEFAULT_MAX_CONCURRENT_REMOTE_NODES
            )
            max_workers = min(len(nodes), requested_max_workers or default_max_workers)
            unique_backend_count = len({str(node.get("url") or "") for node in nodes}) or 1
            max_concurrent_per_backend = _coerce_positive_int(
                payload.get("maxConcurrentPerBackend"),
                DEFAULT_MAX_CONCURRENT_PER_BACKEND,
            )
            if unique_backend_count < len(nodes):
                max_workers = min(max_workers, unique_backend_count * max_concurrent_per_backend)
            tool_executor = RemoteDataToolExecutor(
                ssh_host=str(payload.get("sshHost") or "***.***.***.***"),
                ssh_user=str(payload.get("sshUser") or "***"),
                connect_timeout_seconds=connect_timeout_seconds,
                timeout_seconds=tool_timeout_seconds,
                data_root=str(payload.get("dataRoot") or "/home/***/data"),
            )
            max_steps = _coerce_positive_int(payload.get("maxSteps"), 6)
            read_limit = _coerce_positive_int(payload.get("readLimit"), 25)
            self.job_store.update_progress(
                job_id,
                {
                    "phase": "dispatching",
                    "nodeCount": len(nodes),
                    "backendCount": unique_backend_count,
                    "maxWorkers": max_workers,
                },
            )
            if workflow_mode == "paired_batch_150":
                self.job_store.set_workflow_mode(job_id, workflow_mode)
                self._run_paired_batch_job(
                    job_id=job_id,
                    payload=payload,
                    prompt=prompt,
                    nodes=nodes,
                    tool_executor=tool_executor,
                    timeout_seconds=model_timeout_seconds,
                    connect_timeout_seconds=connect_timeout_seconds,
                    model_num_predict=model_num_predict,
                    max_steps=max_steps,
                    read_limit=read_limit,
                    max_workers=max_workers,
                )
                self.job_store.mark_completed(job_id)
                return
            if workflow_mode == "finance_timeline_full":
                self.job_store.set_workflow_mode(job_id, workflow_mode)
                self._run_finance_timeline_job(
                    job_id=job_id,
                    payload=payload,
                    prompt=prompt,
                    nodes=nodes,
                    tool_executor=tool_executor,
                    timeout_seconds=model_timeout_seconds,
                    connect_timeout_seconds=connect_timeout_seconds,
                    model_num_predict=model_num_predict,
                    max_steps=max_steps,
                    read_limit=read_limit,
                    max_workers=max_workers,
                )
                self.job_store.mark_completed(job_id)
                return
            if workflow_mode == "truthfinder_finance_full":
                self.job_store.set_workflow_mode(job_id, workflow_mode)
                self._run_truthfinder_finance_job(
                    job_id=job_id,
                    payload=payload,
                    prompt=prompt,
                    nodes=nodes,
                    tool_executor=tool_executor,
                    timeout_seconds=model_timeout_seconds,
                    connect_timeout_seconds=connect_timeout_seconds,
                    model_num_predict=model_num_predict,
                    max_steps=max_steps,
                    read_limit=read_limit,
                    max_workers=max_workers,
                )
                self.job_store.mark_completed(job_id)
                return
            if workflow_mode == "truthfinder_weather_full":
                self.job_store.set_workflow_mode(job_id, workflow_mode)
                self._run_truthfinder_weather_job(
                    job_id=job_id,
                    payload=payload,
                    prompt=prompt,
                    nodes=nodes,
                    tool_executor=tool_executor,
                    timeout_seconds=model_timeout_seconds,
                    connect_timeout_seconds=connect_timeout_seconds,
                    model_num_predict=model_num_predict,
                    max_steps=max_steps,
                    read_limit=read_limit,
                    max_workers=max_workers,
                )
                self.job_store.mark_completed(job_id)
                return

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_node = {
                    executor.submit(
                        self._run_node,
                        prompt,
                        node,
                        tool_executor,
                        model_timeout_seconds,
                        connect_timeout_seconds,
                        model_num_predict,
                        max_steps,
                        read_limit,
                        None,
                    ): node
                    for node in nodes
                }
                for future in as_completed(future_to_node):
                    node = future_to_node[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        logger.exception("node agent execution failed, node=%s", node.get("nodeName"))
                        result = {
                            "nodeIndex": node.get("nodeIndex"),
                            "nodeName": node.get("nodeName"),
                            "model": node.get("model"),
                            "ollamaUrl": node.get("url"),
                            "success": False,
                            "summary": None,
                            "error": str(exc),
                            "outputFormat": "text",
                            "durationMs": 0,
                            "toolTrace": [],
                        }
                    self.job_store.set_node_result(job_id, result)
            self.job_store.mark_completed(job_id)
        except Exception as exc:
            logger.exception("agent job failed, jobId=%s", job_id)
            self.job_store.mark_failed(job_id, str(exc))

    @staticmethod
    def _resolve_workflow_mode(payload: Dict[str, Any], prompt: str) -> str:
        explicit = str(payload.get("workflowMode") or "").strip()
        if explicit:
            return explicit
        normalized_prompt = prompt.lower()
        if "[执行模式] truthfinder_weather_full" in prompt or "[executionmode] truthfinder_weather_full" in normalized_prompt:
            return "truthfinder_weather_full"
        if "[执行模式] truthfinder_finance_full" in prompt or "[executionmode] truthfinder_finance_full" in normalized_prompt:
            return "truthfinder_finance_full"
        if "[执行模式] finance_timeline_full" in prompt or "[executionmode] finance_timeline_full" in normalized_prompt:
            return "finance_timeline_full"
        if "[执行模式] paired_batch_150" in prompt or "[executionmode] paired_batch_150" in normalized_prompt:
            return "paired_batch_150"
        if OllamaAgentService._looks_like_truthfinder_weather_prompt(prompt):
            return "truthfinder_weather_full"
        if OllamaAgentService._looks_like_truthfinder_finance_prompt(prompt):
            return "truthfinder_finance_full"
        if "每次调用大模型，获取一条价格数据，和一条新闻摘要数据" in prompt and "150" in prompt and "batch" in normalized_prompt:
            return "paired_batch_150"
        if (
            ("nasdaq_external_news_300.csv" in prompt or "300 条新闻" in prompt)
            and "btc" in normalized_prompt
            and "doge" in normalized_prompt
            and "统一列表" in prompt
        ):
            return "finance_timeline_full"
        return ""

    @staticmethod
    def _looks_like_truthfinder_finance_prompt(prompt: str) -> bool:
        normalized_prompt = prompt.lower()
        has_finance_markers = (
            ("nasdaq_external_news_300.csv" in prompt or "300 条新闻" in prompt)
            and "btc" in normalized_prompt
            and "doge" in normalized_prompt
        )
        has_truthfinder_markers = (
            "truthfinder" in normalized_prompt
            or "job.batches" in normalized_prompt
            or "多 agent 输入格式" in prompt
            or "每个 batch" in prompt
            or '"itemCount"' in prompt
        )
        return has_finance_markers and has_truthfinder_markers

    @staticmethod
    def _looks_like_truthfinder_weather_prompt(prompt: str) -> bool:
        normalized_prompt = prompt.lower()
        has_weather_markers = (
            "weather_external_news_300.csv" in normalized_prompt
            or "天气文章" in prompt
            or "天气新闻式" in prompt
        )
        has_truthfinder_markers = (
            "truthfinder" in normalized_prompt
            or "job.batches" in normalized_prompt
            or "每个 batch" in prompt
            or '"articleTimestamp"' in prompt
        )
        return has_weather_markers and has_truthfinder_markers

    @staticmethod
    def _extract_truthfinder_agent_count(prompt: str) -> Optional[int]:
        patterns = (
            r"agent-1\s*到\s*agent-(\d+)",
            r"(\d+)\s*个 agent",
            r"itemcount\s*.*?(\d+)",
            r"items\s*长度必须恒等于\s*(\d+)",
        )
        for pattern in patterns:
            match = re.search(pattern, prompt, flags=re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    @staticmethod
    def _build_node_states(nodes: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        return {
            int(node.get("nodeIndex", 0)): {
                "nodeIndex": int(node.get("nodeIndex", 0)),
                "nodeName": node.get("nodeName"),
                "model": node.get("model"),
                "ollamaUrl": node.get("url"),
                "generatedNews": 0,
                "generatedPrices": 0,
                "errorCount": 0,
                "lastError": None,
            }
            for node in nodes
        }

    def _load_finance_source_data(
        self,
        prompt: str,
        tool_executor: RemoteDataToolExecutor,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        price_roots, news_path = self._extract_finance_paths(prompt)
        if not price_roots or not news_path:
            raise RuntimeError("failed to parse finance dataset paths from prompt")
        start_date, end_date = self._extract_date_range(prompt)
        requested_news_count = self._extract_requested_news_count(prompt)

        news_records = self._read_all_remote_records(tool_executor, news_path, page_size=200)
        if requested_news_count is not None and len(news_records) < requested_news_count:
            raise RuntimeError(
                f"expected at least {requested_news_count} news records, got {len(news_records)} from {news_path}"
            )
        if requested_news_count is not None:
            news_records = news_records[:requested_news_count]
        resolved_price_paths = {
            asset: self._resolve_price_data_path(tool_executor, base_path, start_date, end_date)
            for asset, base_path in price_roots.items()
        }
        aligned_price_rows = self._build_aligned_price_rows(tool_executor, resolved_price_paths, start_date, end_date)
        if not news_records:
            raise RuntimeError("no readable news records found")
        return news_records, aligned_price_rows

    def _load_weather_article_source_data(
        self,
        prompt: str,
        tool_executor: RemoteDataToolExecutor,
    ) -> List[Dict[str, Any]]:
        article_path = self._extract_truthfinder_weather_path(prompt)
        if not article_path:
            raise RuntimeError("failed to parse weather article dataset path from prompt")
        requested_count = self._extract_requested_article_count(prompt)
        records = self._read_all_remote_records(tool_executor, article_path, page_size=200)
        if requested_count is not None and len(records) < requested_count:
            raise RuntimeError(
                f"expected at least {requested_count} article records, got {len(records)} from {article_path}"
            )
        if requested_count is not None:
            records = records[:requested_count]
        if not records:
            raise RuntimeError("no readable weather article records found")
        return records

    def _summarize_finance_news(
        self,
        job_id: str,
        prompt: str,
        news_records: List[Dict[str, Any]],
        price_total: int,
        nodes: List[Dict[str, Any]],
        tool_executor: RemoteDataToolExecutor,
        timeout_seconds: Optional[int],
        connect_timeout_seconds: int,
        model_num_predict: int,
        max_steps: int,
        read_limit: int,
        max_workers: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
        node_states = self._build_node_states(nodes)
        self.job_store.update_progress(
            job_id,
            {
                "phase": "summarizing_news",
                "newsTotal": len(news_records),
                "newsCompleted": 0,
                "priceTotal": price_total,
                "priceCompleted": 0,
                "totalOutputRecords": 0,
            },
        )

        summarized_news: List[Dict[str, Any]] = []
        errors: List[str] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task: Dict[Any, Dict[str, Any]] = {}
            for index, news_record in enumerate(news_records):
                node = nodes[index % len(nodes)]
                timestamp = self._extract_record_timestamp(
                    news_record,
                    ["published_at", "timestamp", "date", "created_at"],
                )
                task_payload = {
                    "mode": "news_summary_item",
                    "newsIndex": index + 1,
                    "newsRecord": news_record,
                    "timestampText": self._format_timestamp(timestamp),
                    "timestampMissing": timestamp is None,
                }
                future = executor.submit(
                    self._run_node,
                    prompt,
                    node,
                    tool_executor,
                    timeout_seconds,
                    connect_timeout_seconds,
                    model_num_predict,
                    max_steps,
                    read_limit,
                    task_payload,
                )
                future_to_task[future] = {
                    "node": node,
                    "newsIndex": index + 1,
                    "timestamp": timestamp,
                    "timestampText": self._format_timestamp(timestamp),
                    "timestampMissing": timestamp is None,
                }

            completed_news = 0
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                node = task["node"]
                state = node_states[int(node.get("nodeIndex", 0))]
                try:
                    result = future.result()
                    summary_text = str(result.get("summary") or "").strip()
                    if not summary_text:
                        raise RuntimeError("news summary empty")
                    summarized_news.append(
                        {
                            "newsIndex": task["newsIndex"],
                            "timestamp": task["timestamp"],
                            "timestampText": task["timestampText"],
                            "timestampMissing": task["timestampMissing"],
                            "nodeName": node.get("nodeName"),
                            "summary": summary_text,
                        }
                    )
                    state["generatedNews"] += 1
                except Exception as exc:
                    state["errorCount"] += 1
                    state["lastError"] = str(exc)
                    errors.append(f"newsIndex={task['newsIndex']}, node={node.get('nodeName')}, error={exc}")
                completed_news += 1
                if completed_news == len(news_records) or completed_news % 10 == 0:
                    self.job_store.update_progress(
                        job_id,
                        {
                            "phase": "summarizing_news",
                            "newsCompleted": completed_news,
                        },
                    )

        if errors:
            sample_errors = "; ".join(errors[:5])
            raise RuntimeError(f"finance timeline generation failed for {len(errors)} news items: {sample_errors}")

        summarized_news.sort(key=lambda item: int(item.get("newsIndex") or 0))
        return summarized_news, node_states

    def _summarize_weather_articles(
        self,
        job_id: str,
        prompt: str,
        article_records: List[Dict[str, Any]],
        nodes: List[Dict[str, Any]],
        tool_executor: RemoteDataToolExecutor,
        timeout_seconds: Optional[int],
        connect_timeout_seconds: int,
        model_num_predict: int,
        max_steps: int,
        read_limit: int,
        max_workers: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
        node_states = self._build_node_states(nodes)
        self.job_store.update_progress(
            job_id,
            {
                "phase": "summarizing_articles",
                "newsTotal": len(article_records),
                "newsCompleted": 0,
                "priceTotal": 0,
                "priceCompleted": 0,
                "totalOutputRecords": 0,
            },
        )

        summarized_articles: List[Dict[str, Any]] = []
        errors: List[str] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task: Dict[Any, Dict[str, Any]] = {}
            for index, article_record in enumerate(article_records):
                node = nodes[index % len(nodes)]
                timestamp = self._extract_record_timestamp(
                    article_record,
                    ["Date", "published_at", "timestamp", "date", "created_at"],
                )
                task_payload = {
                    "mode": "weather_article_summary_item",
                    "articleIndex": index + 1,
                    "articleRecord": article_record,
                    "timestampText": self._format_timestamp(timestamp),
                    "timestampMissing": timestamp is None,
                }
                future = executor.submit(
                    self._run_node,
                    prompt,
                    node,
                    tool_executor,
                    timeout_seconds,
                    connect_timeout_seconds,
                    model_num_predict,
                    max_steps,
                    read_limit,
                    task_payload,
                )
                future_to_task[future] = {
                    "node": node,
                    "articleIndex": index + 1,
                    "timestamp": timestamp,
                    "timestampText": self._format_timestamp(timestamp),
                    "timestampMissing": timestamp is None,
                }

            completed_articles = 0
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                node = task["node"]
                state = node_states[int(node.get("nodeIndex", 0))]
                try:
                    result = future.result()
                    summary_text = str(result.get("summary") or "").strip()
                    if not summary_text:
                        raise RuntimeError("weather article summary empty")
                    summarized_articles.append(
                        {
                            "articleIndex": task["articleIndex"],
                            "timestamp": task["timestamp"],
                            "timestampText": task["timestampText"],
                            "timestampMissing": task["timestampMissing"],
                            "nodeName": node.get("nodeName"),
                            "summary": summary_text,
                        }
                    )
                    state["generatedNews"] += 1
                except Exception as exc:
                    state["errorCount"] += 1
                    state["lastError"] = str(exc)
                    errors.append(f"articleIndex={task['articleIndex']}, node={node.get('nodeName')}, error={exc}")
                completed_articles += 1
                if completed_articles == len(article_records) or completed_articles % 10 == 0:
                    self.job_store.update_progress(
                        job_id,
                        {
                            "phase": "summarizing_articles",
                            "newsCompleted": completed_articles,
                        },
                    )

        if errors:
            sample_errors = "; ".join(errors[:5])
            raise RuntimeError(f"weather truthfinder generation failed for {len(errors)} articles: {sample_errors}")

        summarized_articles.sort(key=lambda item: int(item.get("articleIndex") or 0))
        return summarized_articles, node_states

    def _run_finance_timeline_job(
        self,
        job_id: str,
        payload: Dict[str, Any],
        prompt: str,
        nodes: List[Dict[str, Any]],
        tool_executor: RemoteDataToolExecutor,
        timeout_seconds: Optional[int],
        connect_timeout_seconds: int,
        model_num_predict: int,
        max_steps: int,
        read_limit: int,
        max_workers: int,
    ) -> None:
        self.job_store.update_progress(job_id, {"phase": "loading_data"})
        news_records, aligned_price_rows = self._load_finance_source_data(prompt, tool_executor)
        summarized_news, node_states = self._summarize_finance_news(
            job_id=job_id,
            prompt=prompt,
            news_records=news_records,
            price_total=len(aligned_price_rows),
            nodes=nodes,
            tool_executor=tool_executor,
            timeout_seconds=timeout_seconds,
            connect_timeout_seconds=connect_timeout_seconds,
            model_num_predict=model_num_predict,
            max_steps=max_steps,
            read_limit=read_limit,
            max_workers=max_workers,
        )

        self.job_store.update_progress(job_id, {"phase": "merging_output", "newsCompleted": len(news_records)})
        merged_entries: List[Dict[str, Any]] = []
        for item in summarized_news:
            merged_entries.append(
                {
                    "sortTimestamp": item["timestamp"],
                    "sortType": 0,
                    "sortIndex": item["newsIndex"],
                    "item": {
                        "agent": item["nodeName"],
                        "object": f"第{item['newsIndex']}条新闻简短摘要",
                        "response": item["summary"],
                    },
                }
            )
        for index, price_row in enumerate(aligned_price_rows):
            node = nodes[index % len(nodes)]
            state = node_states[int(node.get("nodeIndex", 0))]
            state["generatedPrices"] += 1
            merged_entries.append(
                {
                    "sortTimestamp": self._parse_timestamp(price_row.get("timestamp")),
                    "sortType": 1,
                    "sortIndex": index,
                    "item": {
                        "agent": node.get("nodeName"),
                        "object": f"{price_row.get('timestamp')} 的 BTC/ETH/DOGE 价格数据",
                        "response": {
                            "BTC": self._coerce_number(price_row.get("BTC")),
                            "ETH": self._coerce_number(price_row.get("ETH")),
                            "DOGE": self._coerce_number(price_row.get("DOGE")),
                        },
                    },
                }
            )

        sortable_entries = [entry for entry in merged_entries if entry.get("sortTimestamp") is not None]
        unsortable_entries = [entry for entry in merged_entries if entry.get("sortTimestamp") is None]
        sortable_entries.sort(
            key=lambda entry: (
                entry.get("sortTimestamp"),
                entry.get("sortType", 0),
                entry.get("sortIndex", 0),
            )
        )
        unsortable_entries.sort(key=lambda entry: (entry.get("sortType", 0), entry.get("sortIndex", 0)))
        final_output = [entry["item"] for entry in sortable_entries] + [entry["item"] for entry in unsortable_entries]

        self.job_store.set_output(job_id, final_output, "json-array", len(final_output))
        self.job_store.update_progress(
            job_id,
            {
                "phase": "completed",
                "newsCompleted": len(news_records),
                "priceCompleted": len(aligned_price_rows),
                "totalOutputRecords": len(final_output),
            },
        )

        for node in nodes:
            node_index = int(node.get("nodeIndex", 0))
            state = node_states[node_index]
            self.job_store.set_node_result(
                job_id,
                {
                    "nodeIndex": node_index,
                    "nodeName": node.get("nodeName"),
                    "model": node.get("model"),
                    "ollamaUrl": node.get("url"),
                    "success": state["errorCount"] == 0,
                    "summary": (
                        f"summarized {state['generatedNews']} news items and emitted "
                        f"{state['generatedPrices']} price rows"
                    ),
                    "error": state["lastError"],
                    "outputFormat": "text",
                    "durationMs": 0,
                    "toolTrace": [],
                    "recordCount": state["generatedNews"] + state["generatedPrices"],
                },
            )

    def _run_truthfinder_finance_job(
        self,
        job_id: str,
        payload: Dict[str, Any],
        prompt: str,
        nodes: List[Dict[str, Any]],
        tool_executor: RemoteDataToolExecutor,
        timeout_seconds: Optional[int],
        connect_timeout_seconds: int,
        model_num_predict: int,
        max_steps: int,
        read_limit: int,
        max_workers: int,
    ) -> None:
        self.job_store.update_progress(job_id, {"phase": "loading_data"})
        news_records, aligned_price_rows = self._load_finance_source_data(prompt, tool_executor)
        summarized_news, node_states = self._summarize_finance_news(
            job_id=job_id,
            prompt=prompt,
            news_records=news_records,
            price_total=len(aligned_price_rows),
            nodes=nodes,
            tool_executor=tool_executor,
            timeout_seconds=timeout_seconds,
            connect_timeout_seconds=connect_timeout_seconds,
            model_num_predict=model_num_predict,
            max_steps=max_steps,
            read_limit=read_limit,
            max_workers=max_workers,
        )

        self.job_store.update_progress(job_id, {"phase": "merging_output", "newsCompleted": len(news_records)})
        agent_count = self._extract_truthfinder_agent_count(prompt) or len(nodes)
        agent_count = max(1, min(agent_count, len(nodes)))
        agent_names = [f"agent-{index + 1}" for index in range(agent_count)]

        sortable_entries: List[Dict[str, Any]] = []
        unsortable_entries: List[Dict[str, Any]] = []
        for item in summarized_news:
            target = sortable_entries if item["timestamp"] is not None else unsortable_entries
            target.append(
                {
                    "sortTimestamp": item["timestamp"],
                    "sortType": 0,
                    "sortIndex": item["newsIndex"],
                    "kind": "news",
                    "newsIndex": item["newsIndex"],
                    "summary": item["summary"],
                }
            )
        for index, price_row in enumerate(aligned_price_rows):
            sortable_entries.append(
                {
                    "sortTimestamp": self._parse_timestamp(price_row.get("timestamp")),
                    "sortType": 1,
                    "sortIndex": index,
                    "kind": "price",
                    "priceRow": price_row,
                }
            )

        sortable_entries.sort(
            key=lambda entry: (
                entry.get("sortTimestamp"),
                entry.get("sortType", 0),
                entry.get("sortIndex", 0),
            )
        )
        unsortable_entries.sort(key=lambda entry: (entry.get("sortType", 0), entry.get("sortIndex", 0)))

        ordered_entries = sortable_entries + unsortable_entries
        batches: List[Dict[str, Any]] = []
        for batch_index, entry in enumerate(ordered_entries, start=1):
            if entry["kind"] == "news":
                news_index = int(entry["newsIndex"])
                object_text = f"第{news_index}条新闻简短摘要"
                items = [
                    {
                        "agent": agent_name,
                        "object": object_text,
                        "response": entry["summary"],
                    }
                    for agent_name in agent_names
                ]
                batches.append(
                    {
                        "batchIndex": batch_index,
                        "itemCount": len(items),
                        "items": items,
                        "priceTimestamp": None,
                    }
                )
                continue

            price_row = entry["priceRow"]
            object_text = f"{price_row.get('timestamp')} 的 BTC/ETH/DOGE 价格数据"
            response_payload = {
                "BTC": self._coerce_number(price_row.get("BTC")),
                "ETH": self._coerce_number(price_row.get("ETH")),
                "DOGE": self._coerce_number(price_row.get("DOGE")),
            }
            items = [
                {
                    "agent": agent_name,
                    "object": object_text,
                    "response": dict(response_payload),
                }
                for agent_name in agent_names
            ]
            batches.append(
                {
                    "batchIndex": batch_index,
                    "itemCount": len(items),
                    "items": items,
                    "priceTimestamp": price_row.get("timestamp"),
                }
            )

        total_output_records = len(batches) * len(agent_names)
        self.job_store.set_batches(job_id, batches, record_count=total_output_records)
        self.job_store.update_progress(
            job_id,
            {
                "phase": "completed",
                "newsCompleted": len(news_records),
                "priceCompleted": len(aligned_price_rows),
                "totalOutputRecords": total_output_records,
            },
        )

        for node in nodes:
            node_index = int(node.get("nodeIndex", 0))
            state = node_states[node_index]
            self.job_store.set_node_result(
                job_id,
                {
                    "nodeIndex": node_index,
                    "nodeName": node.get("nodeName"),
                    "model": node.get("model"),
                    "ollamaUrl": node.get("url"),
                    "success": state["errorCount"] == 0,
                    "summary": (
                        f"summarized {state['generatedNews']} news items and materialized "
                        f"{len(batches)} truthfinder objects"
                    ),
                    "error": state["lastError"],
                    "outputFormat": "text",
                    "durationMs": 0,
                    "toolTrace": [],
                    "recordCount": len(batches),
                },
            )

    def _run_truthfinder_weather_job(
        self,
        job_id: str,
        payload: Dict[str, Any],
        prompt: str,
        nodes: List[Dict[str, Any]],
        tool_executor: RemoteDataToolExecutor,
        timeout_seconds: Optional[int],
        connect_timeout_seconds: int,
        model_num_predict: int,
        max_steps: int,
        read_limit: int,
        max_workers: int,
    ) -> None:
        self.job_store.update_progress(job_id, {"phase": "loading_data"})
        article_records = self._load_weather_article_source_data(prompt, tool_executor)
        summarized_articles, node_states = self._summarize_weather_articles(
            job_id=job_id,
            prompt=prompt,
            article_records=article_records,
            nodes=nodes,
            tool_executor=tool_executor,
            timeout_seconds=timeout_seconds,
            connect_timeout_seconds=connect_timeout_seconds,
            model_num_predict=model_num_predict,
            max_steps=max_steps,
            read_limit=read_limit,
            max_workers=max_workers,
        )

        self.job_store.update_progress(job_id, {"phase": "building_batches", "newsCompleted": len(article_records)})
        agent_count = self._extract_truthfinder_agent_count(prompt) or len(nodes)
        agent_count = max(1, min(agent_count, len(nodes)))
        agent_names = [f"agent-{index + 1}" for index in range(agent_count)]

        batches: List[Dict[str, Any]] = []
        for batch_index, entry in enumerate(summarized_articles, start=1):
            object_text = f"第{batch_index}条天气文章简短摘要"
            items = [
                {
                    "agent": agent_name,
                    "object": object_text,
                    "response": entry["summary"],
                }
                for agent_name in agent_names
            ]
            batches.append(
                {
                    "batchIndex": batch_index,
                    "itemCount": len(items),
                    "items": items,
                    "articleTimestamp": entry["timestampText"] or None,
                }
            )

        total_output_records = len(batches) * len(agent_names)
        self.job_store.set_batches(job_id, batches, record_count=total_output_records)
        self.job_store.update_progress(
            job_id,
            {
                "phase": "completed",
                "newsCompleted": len(article_records),
                "priceCompleted": 0,
                "totalOutputRecords": total_outputRecords if False else total_output_records,
            },
        )

        for node in nodes:
            node_index = int(node.get("nodeIndex", 0))
            state = node_states[node_index]
            self.job_store.set_node_result(
                job_id,
                {
                    "nodeIndex": node_index,
                    "nodeName": node.get("nodeName"),
                    "model": node.get("model"),
                    "ollamaUrl": node.get("url"),
                    "success": state["errorCount"] == 0,
                    "summary": (
                        f"summarized {state['generatedNews']} weather articles and materialized "
                        f"{len(batches)} truthfinder article batches"
                    ),
                    "error": state["lastError"],
                    "outputFormat": "text",
                    "durationMs": 0,
                    "toolTrace": [],
                    "recordCount": len(batches),
                },
            )

    def _run_paired_batch_job(
        self,
        job_id: str,
        payload: Dict[str, Any],
        prompt: str,
        nodes: List[Dict[str, Any]],
        tool_executor: RemoteDataToolExecutor,
        timeout_seconds: Optional[int],
        connect_timeout_seconds: int,
        model_num_predict: int,
        max_steps: int,
        read_limit: int,
        max_workers: int,
    ) -> None:
        batch_count = _coerce_positive_int(payload.get("batchCount"), 150)
        price_interval_minutes = _coerce_positive_int(
            payload.get("priceIntervalMinutes"),
            DEFAULT_BATCH_PRICE_INTERVAL_MINUTES,
        )
        batch_payloads = self._prepare_paired_batch_payloads(
            prompt,
            tool_executor,
            batch_count,
            price_interval_minutes,
        )
        node_states: Dict[int, Dict[str, Any]] = {
            int(node.get("nodeIndex", 0)): {
                "nodeIndex": int(node.get("nodeIndex", 0)),
                "nodeName": node.get("nodeName"),
                "model": node.get("model"),
                "ollamaUrl": node.get("url"),
                "generatedBatches": 0,
                "generatedItems": 0,
                "errorCount": 0,
                "lastError": None,
            }
            for node in nodes
        }
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for batch_payload in batch_payloads:
                future_to_node = {
                    executor.submit(
                        self._run_node,
                        prompt,
                        node,
                        tool_executor,
                        timeout_seconds,
                        connect_timeout_seconds,
                        model_num_predict,
                        max_steps,
                        read_limit,
                        batch_payload,
                    ): node
                    for node in nodes
                }
                items: List[Dict[str, Any]] = []
                errors: List[Dict[str, Any]] = []
                for future in as_completed(future_to_node):
                    node = future_to_node[future]
                    node_index = int(node.get("nodeIndex", 0))
                    state = node_states[node_index]
                    try:
                        result = future.result()
                        state["generatedBatches"] += 1
                        state["generatedItems"] += int(result.get("recordCount") or 0)
                        raw_items = json.loads(result.get("summary") or "[]")
                        if not isinstance(raw_items, list):
                            raise RuntimeError("paired batch node output is not a json array")
                        for item_order, item in enumerate(raw_items):
                            if isinstance(item, dict):
                                enriched = dict(item)
                                enriched["_nodeIndex"] = node_index
                                enriched["_itemOrder"] = item_order
                                items.append(enriched)
                    except Exception as exc:
                        logger.exception(
                            "batch node generation failed, batchIndex=%s, node=%s",
                            batch_payload.get("batchIndex"),
                            node.get("nodeName"),
                        )
                        state["errorCount"] += 1
                        state["lastError"] = str(exc)
                        errors.append({
                            "nodeName": node.get("nodeName"),
                            "error": str(exc),
                        })
                items.sort(key=lambda item: (item.get("_nodeIndex", 0), item.get("_itemOrder", 0)))
                normalized_items = [
                    {key: value for key, value in item.items() if not key.startswith("_")}
                    for item in items
                ]
                batch_result: Dict[str, Any] = {
                    "batchIndex": batch_payload.get("batchIndex"),
                    "newsIndex": batch_payload.get("newsIndex"),
                    "priceTimestamp": (batch_payload.get("priceRecord") or {}).get("timestamp"),
                    "itemCount": len(normalized_items),
                    "items": normalized_items,
                }
                if errors:
                    batch_result["errors"] = errors
                self.job_store.append_batch(job_id, batch_result)

        for node in nodes:
            node_index = int(node.get("nodeIndex", 0))
            state = node_states[node_index]
            self.job_store.set_node_result(
                job_id,
                {
                    "nodeIndex": node_index,
                    "nodeName": node.get("nodeName"),
                    "model": node.get("model"),
                    "ollamaUrl": node.get("url"),
                    "success": state["errorCount"] == 0,
                    "summary": (
                        f"generated {state['generatedBatches']} batches and {state['generatedItems']} items"
                    ),
                    "error": state["lastError"],
                    "outputFormat": "text",
                    "durationMs": 0,
                    "toolTrace": [],
                    "recordCount": state["generatedItems"],
                },
            )

    def _prepare_paired_batch_payloads(
        self,
        prompt: str,
        tool_executor: RemoteDataToolExecutor,
        batch_count: int,
        price_interval_minutes: int = DEFAULT_BATCH_PRICE_INTERVAL_MINUTES,
    ) -> List[Dict[str, Any]]:
        price_roots, news_path = self._extract_finance_paths(prompt)
        if not price_roots or not news_path:
            raise RuntimeError("failed to parse finance dataset paths from prompt")
        start_date, end_date = self._extract_date_range(prompt)
        news_records = self._read_all_remote_records(tool_executor, news_path, page_size=200)
        resolved_price_paths = {
            asset: self._resolve_price_data_path(tool_executor, base_path, start_date, end_date)
            for asset, base_path in price_roots.items()
        }
        aligned_price_rows = self._build_aligned_price_rows(tool_executor, resolved_price_paths, start_date, end_date)
        selected_price_rows = self._select_price_rows_by_interval(
            aligned_price_rows,
            price_interval_minutes,
            batch_count,
        )
        required_batches = min(batch_count, len(news_records))
        total_batches = min(required_batches, len(selected_price_rows))
        payloads = []
        for index in range(total_batches):
            payloads.append(
                {
                    "mode": "paired_batch_item",
                    "batchIndex": index + 1,
                    "newsIndex": index + 1,
                    "newsRecord": news_records[index],
                    "priceRecord": selected_price_rows[index],
                }
            )
        return payloads

    @classmethod
    def _extract_finance_paths(cls, prompt: str) -> Tuple[Dict[str, str], Optional[str]]:
        extracted: Dict[str, str] = {}
        for key, label in {
            "BTC": "BTC 文件",
            "ETH": "ETH 文件",
            "DOGE": "DOGE 文件",
            "NEWS": "新闻文件",
        }.items():
            path = cls._extract_path_after_label(prompt, label)
            if path:
                extracted[key] = path
        news_path = extracted.pop("NEWS", None)
        return extracted, news_path

    @classmethod
    def _extract_truthfinder_weather_path(cls, prompt: str) -> Optional[str]:
        for label in ("文件", "文章文件", "新闻文件"):
            path = cls._extract_path_after_label(prompt, label)
            if path:
                return path
        return None

    @staticmethod
    def _extract_path_after_label(prompt: str, label: str) -> Optional[str]:
        label_match = re.search(rf"{re.escape(label)}\s*[:：]", prompt)
        if not label_match:
            return None
        tail = prompt[label_match.end(): label_match.end() + 600]
        slash_index = tail.find("/")
        if slash_index == -1:
            return None
        allowed_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-/ \t\r\n")
        collected: List[str] = []
        for index, char in enumerate(tail[slash_index:]):
            if char == "\n":
                lookahead = tail[slash_index + index + 1: slash_index + index + 20]
                if re.match(r"\s*-", lookahead):
                    break
            if char not in allowed_chars:
                break
            collected.append(char)
        path = re.sub(r"\s+", "", "".join(collected)).rstrip("-")
        return path or None

    @staticmethod
    def _extract_requested_news_count(prompt: str) -> Optional[int]:
        patterns = (
            r"数据量：\s*(\d+)\s*条新闻",
            r"读取新闻文件中的\s*(\d+)\s*条新闻",
            r"必须包含全部\s*(\d+)\s*条新闻摘要",
        )
        for pattern in patterns:
            match = re.search(pattern, prompt)
            if match:
                return int(match.group(1))
        return None

    @staticmethod
    def _extract_requested_article_count(prompt: str) -> Optional[int]:
        patterns = (
            r"数据量：\s*(\d+)\s*条天气文章",
            r"读取\s*(\d+)\s*条天气文章",
            r"必须严格生成\s*(\d+)\s*个 batch",
            r"job\.batches\s*必须恰好包含\s*(\d+)\s*个 batch",
        )
        for pattern in patterns:
            match = re.search(pattern, prompt, flags=re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None

    @staticmethod
    def _extract_date_range(prompt: str) -> Tuple[Optional[str], Optional[str]]:
        match = re.search(r"数据范围：\s*(\d{4}-\d{2}-\d{2})\s*至\s*(\d{4}-\d{2}-\d{2})", prompt)
        if not match:
            return None, None
        return match.group(1), match.group(2)

    def _resolve_price_data_path(
        self,
        tool_executor: RemoteDataToolExecutor,
        base_path: str,
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> str:
        extension = os.path.splitext(base_path)[1].lower()
        if extension in {".pk", ".pickle", ".csv", ".json", ".jsonl", ".parquet"}:
            return base_path
        discovery = tool_executor.execute(
            "discover_data_files",
            {
                "keywords": [
                    os.path.basename(base_path.rstrip("/")),
                    start_date or "",
                    end_date or "",
                ],
                "limit": 500,
            },
        )
        files = discovery.get("files") or []
        prefix = base_path.rstrip("/") + "/"
        candidates: List[Tuple[int, str]] = []
        for item in files:
            path = str(item.get("path") or "")
            basename = os.path.basename(path).lower()
            if not path.startswith(prefix):
                continue
            if not basename.endswith((".pk", ".pickle", ".csv", ".json", ".jsonl", ".parquet")):
                continue
            score = int(item.get("score") or 0)
            if start_date and start_date in basename:
                score += 100
            if end_date and end_date in basename:
                score += 100
            for source_index, source in enumerate(PREFERRED_PRICE_SOURCES):
                if basename.startswith(source + "_"):
                    score += 50 - source_index
                    break
            candidates.append((score, path))
        if not candidates:
            raise RuntimeError(f"no readable price file found under {base_path}")
        ranked_candidates: List[Tuple[int, int, str]] = []
        for score, path in sorted(candidates, key=lambda item: (-item[0], item[1]))[: len(PREFERRED_PRICE_SOURCES) + 2]:
            quality_score = self._score_price_path_quality(tool_executor, path, start_date, end_date)
            ranked_candidates.append((quality_score, score, path))
        ranked_candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
        selected_path = ranked_candidates[0][2]
        logger.info("resolved price path %s -> %s", base_path, selected_path)
        return selected_path

    def _score_price_path_quality(
        self,
        tool_executor: RemoteDataToolExecutor,
        path: str,
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> int:
        try:
            preview = tool_executor.execute(
                "read_records",
                {"path": path, "offset": 0, "limit": PRICE_CANDIDATE_PREVIEW_LIMIT},
            )
        except Exception:
            logger.warning("failed to preview price file quality: %s", path, exc_info=True)
            return -1

        records = preview.get("records") or []
        unique_prices = set()
        distinct_timestamps = set()
        price_change_count = 0
        sample_count = 0
        has_previous_price = False
        previous_price: Optional[Any] = None

        for record in records:
            if not isinstance(record, dict):
                continue
            timestamp = self._extract_record_timestamp(record, ["timestamp", "datetime", "time"])
            if timestamp is None or not self._timestamp_in_range(timestamp, start_date, end_date):
                continue
            price_value = self._extract_price_value(record)
            if price_value is None:
                continue
            distinct_timestamps.add(self._format_timestamp(timestamp))
            unique_prices.add(str(price_value))
            if has_previous_price and price_value != previous_price:
                price_change_count += 1
            previous_price = price_value
            has_previous_price = True
            sample_count += 1

        return (
            len(unique_prices) * 100
            + price_change_count * 20
            + len(distinct_timestamps) * 5
            + sample_count
        )

    def _read_all_remote_records(
        self,
        tool_executor: RemoteDataToolExecutor,
        path: str,
        page_size: int = 1000,
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        offset = 0
        while True:
            page = tool_executor.execute(
                "read_records",
                {"path": path, "offset": offset, "limit": page_size},
            )
            chunk = page.get("records") or []
            records.extend(chunk)
            if not page.get("has_more") or not chunk:
                break
            offset += len(chunk)
        return records

    def _build_aligned_price_rows(
        self,
        tool_executor: RemoteDataToolExecutor,
        price_paths: Dict[str, str],
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> List[Dict[str, Any]]:
        aligned_maps: Dict[str, Dict[str, Any]] = {}
        for asset, path in price_paths.items():
            records = self._read_all_remote_records(tool_executor, path, page_size=4000)
            price_map: Dict[str, Any] = {}
            for record in records:
                timestamp = self._extract_record_timestamp(record, ["timestamp", "datetime", "time"])
                if timestamp is None or not self._timestamp_in_range(timestamp, start_date, end_date):
                    continue
                price_value = self._extract_price_value(record)
                if price_value is None:
                    continue
                price_map[self._format_timestamp(timestamp)] = price_value
            if not price_map:
                raise RuntimeError(f"no price records resolved for {asset} from {path}")
            aligned_maps[asset] = price_map
        timestamp_sets = [set(item.keys()) for item in aligned_maps.values()]
        common_timestamps = sorted(set.intersection(*timestamp_sets)) if timestamp_sets else []
        if not common_timestamps:
            raise RuntimeError("no aligned BTC/ETH/DOGE timestamps found")
        return [
            {
                "timestamp": timestamp_text,
                "BTC": aligned_maps["BTC"][timestamp_text],
                "ETH": aligned_maps["ETH"][timestamp_text],
                "DOGE": aligned_maps["DOGE"][timestamp_text],
            }
            for timestamp_text in common_timestamps
        ]

    def _select_price_rows_by_interval(
        self,
        rows: List[Dict[str, Any]],
        interval_minutes: int,
        max_count: int,
    ) -> List[Dict[str, Any]]:
        selected_rows: List[Dict[str, Any]] = []
        next_allowed_time: Optional[datetime] = None
        interval_delta = timedelta(minutes=max(interval_minutes, 1))

        for row in rows:
            timestamp = self._parse_timestamp(row.get("timestamp"))
            if timestamp is None:
                continue
            if next_allowed_time is not None and timestamp < next_allowed_time:
                continue
            selected_rows.append(row)
            if len(selected_rows) >= max_count:
                break
            next_allowed_time = timestamp + interval_delta

        return selected_rows

    def _extract_record_timestamp(self, record: Dict[str, Any], keys: List[str]) -> Optional[datetime]:
        value = self._pick_first_value(record, keys)
        return self._parse_timestamp(value)

    def _extract_price_value(self, record: Dict[str, Any]) -> Optional[Any]:
        return self._coerce_number(self._pick_first_value(record, ["price", "close", "value", "usd", "close_price"]))

    @staticmethod
    def _pick_first_value(record: Dict[str, Any], keys: List[str]) -> Any:
        lowered = {str(key).lower(): value for key, value in record.items()}
        for key in keys:
            value = lowered.get(key.lower())
            if value not in (None, "", [], {}):
                return value
        return None

    @staticmethod
    def _coerce_number(value: Any) -> Optional[Any]:
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            number = float(value)
        else:
            match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
            if not match:
                return None
            number = float(match.group(0))
        return int(number) if number.is_integer() else number

    @staticmethod
    def _parse_timestamp(value: Any) -> Optional[datetime]:
        if value in (None, ""):
            return None
        if isinstance(value, (int, float)):
            timestamp = float(value)
            if timestamp > 1_000_000_000_000:
                timestamp /= 1000.0
            try:
                return datetime.utcfromtimestamp(timestamp)
            except (OverflowError, OSError, ValueError):
                return None
        text = str(value).strip()
        if not text:
            return None
        normalized = text.replace("/", "-")
        if normalized.upper().endswith(" UTC"):
            normalized = normalized[:-4] + "+00:00"
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(normalized)
            return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
        except ValueError:
            pass
        for fmt in (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M",
        ):
            try:
                return datetime.strptime(normalized, fmt)
            except ValueError:
                continue
        return None

    @staticmethod
    def _format_timestamp(value: Optional[datetime]) -> str:
        return value.strftime("%Y-%m-%d %H:%M:%S") if value else ""

    @staticmethod
    def _timestamp_in_range(value: datetime, start_date: Optional[str], end_date: Optional[str]) -> bool:
        day_text = value.strftime("%Y-%m-%d")
        if start_date and day_text < start_date:
            return False
        if end_date and day_text > end_date:
            return False
        return True

    @staticmethod
    def _normalize_nodes(nodes: Any, node_count: Any = None) -> List[Dict[str, Any]]:
        if isinstance(nodes, list) and nodes:
            requested_count = _coerce_positive_int(node_count, len(nodes))
            raw_nodes = nodes[:requested_count]
        else:
            requested_count = min(
                _coerce_positive_int(node_count, len(DEFAULT_NODE_CONFIGS)),
                len(DEFAULT_NODE_CONFIGS),
            )
            raw_nodes = DEFAULT_NODE_CONFIGS[:requested_count]
        normalized = []
        for index, node in enumerate(raw_nodes):
            fallback = DEFAULT_NODE_CONFIGS[index % len(DEFAULT_NODE_CONFIGS)]
            normalized.append({
                "nodeIndex": int(node.get("nodeIndex", index)),
                "nodeName": node.get("nodeName") or f"agent-{index + 1}",
                "url": node.get("url") or fallback["url"],
                "model": node.get("model") or fallback["model"],
                "backendName": node.get("backendName") or fallback.get("backendName"),
                "backendIndex": int(node.get("backendIndex", fallback.get("backendIndex", index % max(len(DEFAULT_BACKEND_CONFIGS), 1)))),
                "gpuHint": node.get("gpuHint", fallback.get("gpuHint")),
            })
        return normalized

    @staticmethod
    def _run_node(
        prompt: str,
        node_config: Dict[str, Any],
        tool_executor: RemoteDataToolExecutor,
        timeout_seconds: Optional[int],
        connect_timeout_seconds: int,
        model_num_predict: int,
        max_steps: int,
        read_limit: int,
        task_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        runner = StructuredToolAgentRunner(
            prompt=prompt,
            node_config=node_config,
            tool_executor=tool_executor,
            timeout_seconds=timeout_seconds,
            connect_timeout_seconds=connect_timeout_seconds,
            model_num_predict=model_num_predict,
            max_steps=max_steps,
            read_limit=read_limit,
            task_payload=task_payload,
        )
        return runner.run()


ollama_agent_service = OllamaAgentService()
