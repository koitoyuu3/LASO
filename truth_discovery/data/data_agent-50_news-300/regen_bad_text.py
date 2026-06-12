
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
REFERENCE_SCRIPT = DATA_DIR.parent / "data_agent-10" / "regen_bad_text.py"
DEMO_PATH = DATA_DIR / "demo.json"

_PREFIX_VARIANTS = [
    "从近期行业研报数据和宏观经济周期波动来看，结合产业政策导向的多维度分析框架，",
    "结合全球大宗商品价格走势的联动效应和跨境资本流动新特征，受外部风险事件频发的影响，",
    "从技术面量价关系和资金博弈格局出发，借助机器学习对历史交易数据的回测与模式识别研究结论，",
    "综合产业链上下游反馈和头部企业经营数据的交叉验证，考虑到原材料成本传导的时滞效应，",
    "从机构资金流向和持仓变动出发，参考近三季公募基金前十大重仓股调仓路径和北向资金配置偏好，",
    "根据信用利差和债券市场信号判断，结合中美利差倒挂下跨境资本流动特征和国内流动性分层现象，",
    "结合海外市场联动效应和汇率走势，在全球央行货币政策分化加深和地缘冲突升级的大背景下，",
    "从消费端微观数据和零售景气度调查出发，考虑到人口结构变化对长期消费趋势的深层影响，",
    "基于ESG评估框架和绿色金融政策导向的多因子模型，结合碳交易市场动态和清洁能源转型投资进展，",
    "从区域经济发展不平衡和新型城镇化进程角度，结合地方政府债务风险化解方案的落地效果评估，",
    "依据卫星遥感数据和工业用电量等高频指标的实时监测，借助另类数据源对传统宏观预测模型的校准，",
    "从金融科技创新和数字货币发展趋势的前沿视角，结合监管科技在反洗钱合规领域的最新应用实践，",
    "基于博弈论框架对主要经济体贸易政策互动的推演，考虑到地缘政治风险溢价和供应链重构成本量化评估，",
    "从行为金融学市场情绪指标和投资者信心调查出发，结合社交媒体舆情大数据的自然语言处理分析结果，",
    "综合运用蒙特卡洛模拟和压力测试的风险管理方法论，基于极端尾部事件历史复盘和情景推演分析框架，",
]

_SUFFIX_VARIANTS = [
    "以上判断仍需结合后续政策面和资金面的进一步验证。",
    "具体走势仍取决于基本面变化和市场情绪的博弈。",
    "但该方向的可持续性仍有赖于宏观环境的配合。",
    "不过短期内流动性扰动可能带来阶段性波动。",
    "后续需要密切关注核心数据的边际变化趋势。",
    "风险提示：以上分析不构成投资建议，请审慎决策。",
    "整体来看，中期维度的结构性机会仍然值得跟踪。",
    "但需要警惕外部黑天鹅事件引发的尾部风险冲击。",
    "建议投资者综合多维信号交叉验证后再做判断。",
    "此判断基于当前可获取信息，不排除后续修正的可能。",
    "不同市场参与者对此的解读和应对策略可能存在分歧。",
    "中长期看，产业升级和技术进步仍是核心驱动因素。",
    "短期扰动不改变中期基本面修复的大方向判断。",
    "以上仅为定性研判，定量结论仍有待模型进一步校验。",
    "综上所述，当前阶段需要更加注重风险收益比的平衡。",
]

BAD_GROUP_SPECS = {
    5: [5],
    10: [10],
    15: [15],
    20: [20],
    25: [15, 10],
    30: [15, 15],
    35: [12, 12, 11],
    40: [8, 8, 8, 8, 8],
}

def _load_reference_module():
    spec = importlib.util.spec_from_file_location("regen_bad_text_ref", REFERENCE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

REF = _load_reference_module()

def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _save_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        f.write("\n")

def _agent_number(agent_name: str) -> int:
    try:
        return int(str(agent_name).split("-")[1])
    except (IndexError, ValueError, TypeError):
        return 1

def diversify_clean_demo_strong(demo_data):

    import copy
    n_pre = len(_PREFIX_VARIANTS)
    n_suf = len(_SUFFIX_VARIANTS)
    data = copy.deepcopy(demo_data)

    for batch in data["job"]["batches"]:
        for item in batch["items"]:
            if not isinstance(item["response"], str):
                continue
            agent_num = _agent_number(item["agent"])
            if agent_num <= 1:
                continue
            idx = agent_num - 2
            prefix = _PREFIX_VARIANTS[idx % n_pre]
            suffix = _SUFFIX_VARIANTS[(idx + 7) % n_suf]
            item["response"] = prefix + item["response"] + suffix

    return data

def regenerate_collude_file(
    out_file_path: Path,
    demo_data,
    canonical_clean_map: dict,
    clean_agent_map: dict,
    bad_agents: list[str],
    group_sizes: list[int],
    numeric_only: bool = False,
):
    direction_names = REF.collude_direction_names(len(group_sizes))
    gen_fn = (
        REF.regenerate_collude_numeric_only_file
        if numeric_only
        else REF.regenerate_collude_file
    )
    data, _ = gen_fn(
        out_file_path=str(out_file_path),
        demo_data=demo_data,
        canonical_clean_map=canonical_clean_map,
        clean_agent_map=clean_agent_map,
        bad_agents=bad_agents,
        group_sizes=group_sizes,
        direction_names=direction_names,
    )
    return data

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generate collude text datasets for 50 agents.")
    parser.add_argument(
        "--numeric-only", action="store_true",
        help="Numeric-only attack: mutate numbers only, keep text words unchanged.",
    )
    args = parser.parse_args()

    mode_label = "numeric-only" if args.numeric_only else "collude"

    demo_data = _load_json(DEMO_PATH)

    if args.numeric_only:
        diverse_demo_data = diversify_clean_demo_strong(demo_data)
        div_path = DATA_DIR / "demo_div.json"
        _save_json(div_path, diverse_demo_data)
        print(f"generated {div_path.name} [strong-diversity clean]")
    else:

        diverse_demo_data = diversify_clean_demo_strong(demo_data)
        div_path = DATA_DIR / "demo_div.json"
        _save_json(div_path, diverse_demo_data)
        print(f"generated {div_path.name} [strong-diversity clean]")

    canonical_clean_map = REF.build_clean_text_map(diverse_demo_data)
    clean_agent_map = REF.build_clean_agent_text_map(diverse_demo_data)

    for bad_count, group_sizes in BAD_GROUP_SPECS.items():
        bad_agents = [f"agent-{idx}" for idx in range(1, bad_count + 1)]
        if args.numeric_only:
            out_path = DATA_DIR / f"demo_{bad_count}bad_numonly.json"
        else:
            out_path = DATA_DIR / f"demo_{bad_count}bad_collude.json"
        regenerate_collude_file(
            out_file_path=out_path,
            demo_data=diverse_demo_data,
            canonical_clean_map=canonical_clean_map,
            clean_agent_map=clean_agent_map,
            bad_agents=bad_agents,
            group_sizes=group_sizes,
            numeric_only=args.numeric_only,
        )
        print(
            f"generated {out_path.name} [{mode_label}] with groups={group_sizes} "
            f"directions={REF.collude_direction_names(len(group_sizes))}"
        )

if __name__ == "__main__":
    main()
