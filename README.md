# LASO

Reference implementation for the paper **"LASO: …"** (anonymous under review).

LASO is a decentralized truth-discovery framework for multi-agent LLM outputs,
with on-chain proof aggregation. This repository contains:

- the LASO truth-finder algorithms and four baselines used in the paper;
- a ChainMaker-based zero-knowledge proof registry and aggregator;
- a Spring Boot backend that orchestrates remote Ollama agent pools over SSH;
- a Flask service that exposes the per-agent inference API;
- the experiment scripts that produced every table and figure in the paper.

> **Status.** Code released for peer review. Issue tracker and authorship will
> be enabled after acceptance.

---

## 1. Repository layout

```
LASO/
├── truth_discovery/                    Core algorithms + experiments
│   ├── core/
│   │   ├── LASO_truth_finder.py            Proposed method
│   │   ├── basic_truth_finder.py           Baseline: TruthFinder
│   │   ├── decent_truth_finder.py          Baseline: decentralized TF
│   │   ├── sente_truth_finder.py           Baseline: SenteTruth
│   │   ├── senfeed_truth_finder.py         Baseline: SenFeed
│   │   ├── hybrid_text_alignment.py        Text/numeric alignment kernel
│   │   ├── text_vectorization.py           TF-IDF / embedding utils
│   │   ├── zk_proof.py                     ZK proof engine (numeric channel)
│   │   ├── experiment_proof_bundle.py      Proof bundle serialization
│   │   ├── chainmaker_evm.py               EVM-on-ChainMaker bindings
│   │   └── circuits/                       circom circuits for proofs
│   ├── experiment/                     One file per paper experiment
│   │   ├── exp1_*                          Exp 1: malicious-ratio sweep
│   │   ├── exp3_*                          Exp 3: text vs numeric channels
│   │   ├── exp4_*                          Exp 4: ablations
│   │   ├── exp7_*                          Exp 7: alpha sensitivity / heatmap
│   │   ├── agent_scalability_benchmark.py  Scalability table
│   │   ├── proof_scalability_benchmark.py  Proof-cost scalability
│   │   ├── five_methods_comparison.py      Main accuracy comparison
│   │   └── distributed/                    Multi-process transport
│   └── data/                           Generated agent-response datasets
│       ├── data_agent-50_news-300/         50 agents × 300 finance news items
│       ├── data_agent-50_weather-300/      50 agents × 300 weather items
│       └── data-llama-agent-50_*/          Llama-generated counterparts
├── Backend/                             Spring Boot service
│   └── src/main/...                        Controllers, ChainMaker bridge,
│                                           Ollama orchestration
├── flaskProject/                        Per-agent inference service
│   ├── app.py                              Flask entry point
│   ├── ollama_agent_api.py / service.py    Ollama client + endpoints
│   └── start_remote_ollama_gpu_pool.sh     Launch N nodes over SSH
├── scripts/
│   ├── run_exp5b_50agent.sh                Driver for Exp 5b
│   └── chainmaker/                         Build / deploy / verify circuits
```

---

## 2. Reproducing the paper

### 2.1 Prerequisites

| Component | Version tested |
|---|---|
| Python | 3.10 – 3.12 |
| Java | 17 (Spring Boot 3.x) |
| Maven | 3.9+ |
| Node.js | 18+ (only for `snarkjs`) |
| circom | 2.1.x |
| snarkjs | 0.7.x |
| ChainMaker | v2.3.x (Go SDK + Java SDK) |
| MySQL | 8.0+ (only needed for on-chain archive) |
| Ollama | latest (one process per agent node) |

### 2.2 Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2.3 Run one experiment locally (no chain, no remote GPUs)

```bash
python -m truth_discovery.experiment.five_methods_comparison \
    --dataset truth_discovery/data/data_agent-50_news-300/demo.json
```

Every experiment script writes its CSV / JSON output under `result/` (created
on first run) and prints the table row(s) used in the paper.

### 2.4 Reproduce individual experiments

All experiment drivers live in `truth_discovery/experiment/`. Each reads its
input dataset from `--data-dir` (default `truth_discovery/data/data_agent-50_news-300`)
and writes CSV/JSON outputs to `--out-dir` (per-experiment default under
`truth_discovery/experiment/outputs/...`). Run any of them from the repo root
inside the activated `.venv`.

| Paper section | Script | Command |
|---|---|---|
| Exp 1 — numeric, 0–50% malicious | `exp1_numeric.py` | `python -m truth_discovery.experiment.exp1_numeric` |
| Exp 1 — text, 0–50% malicious | `exp1_text.py` | `python -m truth_discovery.experiment.exp1_text` |
| Exp 3 — numeric, staged malicious activation | `exp3_numeric.py` | `python -m truth_discovery.experiment.exp3_numeric` |
| Exp 3 — text, staged malicious activation | `exp3_text.py` | `python -m truth_discovery.experiment.exp3_text` |
| Exp 4 — numeric, fixed 5 malicious agents | `exp4_numeric.py` | `python -m truth_discovery.experiment.exp4_numeric` |
| Exp 4 — text, fixed 5 malicious agents | `exp4_text.py` | `python -m truth_discovery.experiment.exp4_text` |
| Exp 5 — alpha sensitivity sweep | `exp7_alpha_sensitivity.py` | `python -m truth_discovery.experiment.exp7_alpha_sensitivity` |
| Exp 5 — heatmap plot from sweep CSV | `exp7_plot_heatmap.py` | `python -m truth_discovery.experiment.exp7_plot_heatmap` |
| Scalability — multi-process (cross-PID) | `agent_scalability_distributed.py` | `python -m truth_discovery.experiment.agent_scalability_distributed` |
| Proof scalability — verify time / RSS / size vs. batch count | `proof_scalability_benchmark.py` | `python -m truth_discovery.experiment.proof_scalability_benchmark` |
| Bundled driver (Exp 5b) | `scripts/run_exp5b_50agent.sh` | `bash scripts/run_exp5b_50agent.sh` |

**Common flags (Exp 1 / 3 / 4 / 5):**

```bash
--data-dir <path>     # Input dataset directory (50-agent JSONs)
--out-dir  <path>     # Where to write results CSV / metrics JSON
```

**Exp 1 / 4 (numeric only) extra:**

```bash
--metric-scale {raw,relative}    # report absolute numbers or relative-to-baseline
```

**Exp 3 (numeric & text) — schedule control:**

```bash
--phase1-start 50  --phase1-bad-count 10    # numeric defaults
--phase2-start 100 --phase2-bad-count 15
--phase3-start 200 --phase3-bad-count 20
# text variant uses same defaults (10 / 15 / 20 bad)
```

**Exp 5 plot (`exp7_plot_heatmap.py`):**

```bash
--csv <path>          # Source results.csv (default: outputs/exp7_*/results.csv)
--out <path>          # Output figure path (default: alongside the CSV)
--threshold 0.9       # DataAcc threshold isoline annotation
```

**Proof scalability benchmark (`proof_scalability_benchmark.py`):** sweeps a
list of batch counts; for every (domain, method, batch_count) cell it
(1) builds a `*.proof_bundle.json` via `five_methods_comparison`, and
(2) re-verifies it in an isolated subprocess `--repeats` times while sampling
RSS at `--memory-sample-interval`. Outputs `raw_results.csv`, `summary.csv`
and `manifest.json` under `--output-root`.

```bash
# Smoke run (single small batch, one repeat) — finishes in seconds
python -m truth_discovery.experiment.proof_scalability_benchmark \
    --batch-counts 5 \
    --repeats 1 --warmup 0

# Paper-style sweep (default 6 batch sizes × 30 repeats; takes a long time)
python -m truth_discovery.experiment.proof_scalability_benchmark \
    --numeric-input truth_discovery/data/data_agent-50_news-300/num_demo.json \
    --text-input    truth_discovery/data/data_agent-50_news-300/demo.json
```

| Flag | Default | Meaning |
|---|---|---|
| `--numeric-input` | `data_agent-50_news-300/num_demo.json` | Numeric JSON input |
| `--text-input` | `data_agent-50_news-300/demo.json` | Text JSON input |
| `--output-root` | `outputs/proof_scalability` | Where bundles + CSVs land |
| `--batch-counts` | `5,25,50,100,150,300` | Batch sizes to sweep |
| `--domains` | `numeric,text` | Restrict to one domain if desired |
| `--repeats` | `30` | Verify repetitions per (method, batch_count) |
| `--warmup` | `1` | Verify runs discarded before timing |
| `--memory-sample-interval` | `0.01` | Seconds between RSS samples |

**Distributed scalability driver:** the only script that requires non-default
flags for paper-style runs. Minimal example:

```bash
python -m truth_discovery.experiment.agent_scalability_distributed \
    --mode dist-tcp \
    --input-json truth_discovery/data/data_agent-50_news-300/num_demo.json \
    --agent-counts 10,25,50 \
    --methods all \
    --subset-samples 3
```

Key flags:

| Flag | Default | Meaning |
|---|---|---|
| `--mode` | `dist-tcp` | `inproc` (in-process) or `dist-tcp` (one OS process per agent over ZMQ) |
| `--agent-counts` | `auto` | Comma-separated list, e.g. `10,25,50,100` |
| `--methods` | `all` | `all` or comma list of method ids |
| `--subset-samples` | `2` | Sub-sampling repetitions per agent-count |
| `--start-batch` | `1` | Skip warm-up batches |
| `--batch-group-size` | `10` | Batches per measurement window |
| `--num-workloads` | `1` | Independent workloads to average over |
| `--time-weight / --memory-weight / --quality-weight` | 0.45 / 0.35 / 0.20 | Composite scoring weights |

Run `python -m truth_discovery.experiment.<script> --help` on any driver to
see its full flag list.

### 2.5 One-shot: reproduce all paper tables

```bash
# Sequential driver used to produce the headline numbers
bash scripts/run_exp5b_50agent.sh
```

### 2.6 End-to-end with proofs on ChainMaker

1. Bring up a 4-node ChainMaker network and fill in `Backend/src/main/resources/sdk_config.yml`
   (placeholders are marked with `***`).
2. Compile and deploy the registry contract:
   ```bash
   bash scripts/chainmaker/build_truth_single_proof_registry_n4_artifacts.sh
   bash scripts/chainmaker/deploy_truth_single_proof_registry_n4_chainmaker.sh
   ```
3. Start the Spring Boot backend (see §3 for env setup):
   ```bash
   cd Backend && mvn spring-boot:run
   ```
4. Start the Flask agent service:
   ```bash
   cd flaskProject && python app.py
   ```
5. Run the 4-step on-chain pipeline (Swagger UI at
   `http://localhost:8080/swagger-ui.html` exposes all of them):

   **Step 1 — Store the prompt on chain.** Writes the prompt to the
   `OracleAggregator` contract and returns a `requestId`.
   ```bash
   curl -X POST "http://localhost:8080/api/chainmaker/ollama/prompt" \
        --data-urlencode "prompt=Summarize the NASDAQ headline X"
   # → { "data": { "requestId": "42", "promptStoredOnChain": true, ... } }
   ```

   **Step 2 — Dispatch the inference job to the LLM pool.** Backend reads
   the prompt back from chain by `requestId` and POSTs to the Flask service
   `/api/agent/ollama/jobs`, which fans out to up to 50 Ollama nodes over
   SSH. Set `count` to limit how many nodes participate.
   ```bash
   curl -X POST "http://localhost:8080/api/chainmaker/ollama/infer-by-request" \
        --data-urlencode "requestId=42" \
        --data-urlencode "count=50"
   # → { "data": { "pythonAgentJobId": "...", "pythonAgentStatusQueryUrl": "...", ... } }
   ```

   **Step 3 — Poll for task completion.** Two queries to combine:
   ```bash
   # 3a. Chain-side metadata (prompt + createdAt)
   curl "http://localhost:8080/api/chainmaker/ollama/result?requestId=42"

   # 3b. Flask-side job status (per-node outputs once "status":"completed")
   curl "$FLASK_BASE/api/agent/ollama/jobs/<pythonAgentJobId>"
   ```

   **Step 4 — Submit the truth-discovery result + ZK proof on chain.** After
   you have aggregated the 50 node outputs into a truth result and produced
   a Schnorr `*.proof_bundle.json` (e.g. via
   `truth_discovery.experiment.five_methods_comparison`), submit both to the
   `TruthSchnorrProofRegistry` contract:
   ```bash
   curl -X POST "http://localhost:8080/api/benchmark/ollama-fixed-result" \
        --data-urlencode "chain=chainmaker" \
        --data-urlencode "prompt=Summarize the NASDAQ headline X" \
        --data-urlencode "resultPath=/abs/path/to/agent_results.json" \
        --data-urlencode "proofBundlePath=/abs/path/to/laso_truth.proof_bundle.json"
   ```
   The response returns the on-chain `requestId`, the time spent on each
   sub-step (prompt write, callback writes, result query, proof
   `verifyProof` / `submitProof` calls) and the registry-side proof digest.

---

## 3. Backend configuration

`Backend/src/main/resources/application.yml` reads every secret from an
environment variable via Spring `${VAR:default}` syntax. The minimum to set:

```bash
export MYSQL_PASSWORD=...
export JWT_SECRET=$(openssl rand -base64 48)
export OLLAMA_SSH_HOST=...        # GPU host running the Ollama pool
export OLLAMA_SSH_USER=...
```

See `Backend/src/main/resources/application.yml.example` for the full list of
recognized variables and their meanings.

`sdk_config.yml` is loaded by the ChainMaker Java SDK directly and **does not
support `${VAR}` interpolation** — edit it in place and replace every `***` /
`********` placeholder before running.

---

## 4. Datasets

The four agent-response datasets used in the paper are hosted externally due
to their size (~600 MB total):

> **Download:** [huggingface.co/datasets/fanfanfan111/laso-datasets](https://huggingface.co/datasets/fanfanfan111/laso-datasets)
>
> ```bash
> hf download fanfanfan111/laso-datasets --repo-type dataset \
>     --local-dir truth_discovery/data
> ```
>
> This restores the four dataset directories listed in §1
> (`data_agent-50_news-300/`, …) to the exact paths the experiment scripts
> read by default.

Each file is a `(batches × agents)` JSON. The `regen_bad_*.py` scripts
included in each dataset directory regenerate the malicious-agent variants
from the clean `demo.json` / `num_demo.json`. The upstream sources for the
underlying news / weather corpora are:

- **News:** NASDAQ headlines via HuggingFace dataset `Zihan1004/FNSPID`.
- **Weather:** NWS forecasts re-written as news-style passages.

Reviewers wishing to regenerate the agent responses from scratch can follow
the prompt embedded in each JSON file (`dataRoot` + `prompt` fields).

Raw per-agent LLM response dumps (`response.json`, >100 MB) are excluded from
the repository; every experiment runs from the processed `demo*.json` /
`num_demo*.json` files included here.

---

## 5. Citation

Citation will be added after de-anonymization.

```bibtex
@article{laso_anon_2026,
  title   = {{LASO}: …},
  author  = {Anonymous},
  journal = {Under review},
  year    = {2026}
}
```

---

## 6. License

This project is released under the [Apache License 2.0](LICENSE). You are free
to use, modify, and redistribute the source under the terms of that license,
provided that you retain the copyright and license notices.

Third-party components retain their respective upstream licenses:

| Component | Upstream license |
|---|---|
| ChainMaker Java SDK | Apache-2.0 |
| Spring Boot | Apache-2.0 |
| circom / snarkjs | GPL-3.0 |
| HuggingFace `sentence-transformers` | Apache-2.0 |
| Ollama / model weights (e.g. Qwen2.5, Llama) | See each model card for its own license; this repo does not redistribute weights |

The two underlying corpora used for the experiments are accessed via their
upstream distributors (HuggingFace `Zihan1004/FNSPID` for NASDAQ headlines and
NWS public forecasts for weather); their terms of use apply independently of
this license.
