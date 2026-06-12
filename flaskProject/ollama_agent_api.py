from __future__ import annotations

from flask import Blueprint, jsonify, request

from ollama_agent_service import DEFAULT_NODE_CONFIGS, ollama_agent_service


def _coerce_positive_int(value):
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None

ollama_agent_bp = Blueprint("ollama_agent", __name__)


@ollama_agent_bp.route("/api/agent/ollama/health", methods=["GET"])
def ollama_agent_health():
    return jsonify({"success": True, "message": "ollama agent service ready"})


@ollama_agent_bp.route("/api/agent/ollama/jobs", methods=["POST"])
def create_ollama_agent_job():
    data = request.get_json(silent=True) or {}
    prompt = str(data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"success": False, "error": "missing prompt"}), 400

    custom_nodes = data.get("nodes") if isinstance(data.get("nodes"), list) else None
    node_count = _coerce_positive_int(data.get("nodeCount"))
    if data.get("nodeCount") not in (None, "") and node_count is None:
        return jsonify({"success": False, "error": "nodeCount must be a positive integer"}), 400

    available_node_count = len(custom_nodes) if custom_nodes else len(DEFAULT_NODE_CONFIGS)
    if node_count is not None and node_count > available_node_count:
        return jsonify(
            {
                "success": False,
                "error": f"nodeCount exceeds available nodes: requested={node_count}, available={available_node_count}",
            }
        ), 400

    if node_count is not None:
        data["nodeCount"] = node_count

    job = ollama_agent_service.submit_job(data)
    return jsonify(
        {
            "success": True,
            "jobId": job["jobId"],
            "requestId": job.get("requestId"),
            "status": job.get("status"),
            "createdAt": job.get("createdAt"),
            "nodeCount": node_count or available_node_count,
            "resultUrl": f"/api/agent/ollama/jobs/{job['jobId']}",
        }
    )


@ollama_agent_bp.route("/api/agent/ollama/jobs/example", methods=["GET"])
def get_ollama_agent_job_example():
    example_nodes = [dict(node) for node in DEFAULT_NODE_CONFIGS[:2]]
    return jsonify(
        {
            "success": True,
            "exampleRequest": {
                "requestId": "demo-request-001",
                "prompt": "请概括工业互联网场景下预言机与链上验证的关系。",
                "nodeCount": 2,
                "nodes": example_nodes,
            },
            "financeTimelineExample": {
                "requestId": "demo-request-002",
                "workflowMode": "finance_timeline_full",
                "nodeCount": 50,
                "prompt": "请读取远端金融数据文件并输出统一时间线。",
            },
            "truthfinderFinanceExample": {
                "requestId": "demo-request-003",
                "workflowMode": "truthfinder_finance_full",
                "nodeCount": 50,
                "prompt": "请读取远端金融数据文件并输出 TruthFinder batches。",
            },
            "notes": [
                "If nodeCount is omitted, all 50 logical agents are used by default.",
                "Logical agents are mapped onto a smaller remote backend pool to avoid loading a separate model per agent.",
                "If nodeCount is provided, only the first nodeCount agents participate in inference.",
                "With workflowMode=finance_timeline_full, the remote real data file is read first and the model only produces news summaries.",
                "With workflowMode=truthfinder_finance_full, the job outputs a job.batches structure where each batch contains items from all logical agents.",
                "Results are retrieved via GET /api/agent/ollama/jobs/{jobId}.",
            ],
        }
    )


@ollama_agent_bp.route("/api/agent/ollama/jobs/<job_id>", methods=["GET"])
def get_ollama_agent_job(job_id: str):
    job = ollama_agent_service.get_job(job_id)
    if not job:
        return jsonify({"success": False, "error": "job not found"}), 404
    return jsonify({"success": True, "job": job})
