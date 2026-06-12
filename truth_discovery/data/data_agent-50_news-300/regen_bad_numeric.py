
from __future__ import annotations

import importlib.util
import json
from decimal import Decimal
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
REFERENCE_SCRIPT = DATA_DIR.parent / "data_agent-10" / "regen_bad_numeric.py"
DEMO_FILE = DATA_DIR / "demo.json"
NUM_DEMO_FILE = DATA_DIR / "num_demo.json"
BAD_COUNTS = (5, 10, 15, 20, 25, 30, 35, 40)

def _load_reference_module():
    spec = importlib.util.spec_from_file_location("regen_bad_numeric_ref", REFERENCE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

REF = _load_reference_module()

def build_num_demo(clean_demo: dict) -> dict:
    data = {
        "job": dict(clean_demo["job"]),
        "success": clean_demo["success"],
    }

    new_batches = []
    for batch in clean_demo["job"]["batches"]:
        items = [item for item in batch["items"] if isinstance(item["response"], dict)]
        new_batches.append(
            {
                "batchIndex": batch["batchIndex"],
                "itemCount": len(items),
                "items": items,
                "newsIndex": batch.get("newsIndex"),
                "priceTimestamp": batch.get("priceTimestamp"),
            }
        )

    data["job"]["batches"] = new_batches
    return data

def save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")

def main() -> None:
    clean_demo = json.loads(DEMO_FILE.read_text(encoding="utf-8"))
    num_demo = build_num_demo(clean_demo)
    save_json(NUM_DEMO_FILE, num_demo)

    clean_numeric = json.loads(NUM_DEMO_FILE.read_text(encoding="utf-8"), parse_float=Decimal)
    for bad_count in BAD_COUNTS:
        out_path = DATA_DIR / f"num_demo_{bad_count}bad_collude.json"
        data = REF.regenerate_numeric_dataset(clean_numeric, bad_count, mode="collude")
        out_path.write_text(
            json.dumps(REF._materialize_json_numbers(data), ensure_ascii=False, indent=4) + "\n",
            encoding="utf-8",
        )
        print(f"generated {out_path.name}")

if __name__ == "__main__":
    main()
