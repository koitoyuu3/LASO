
from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent

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

WEATHER_HONEST_PREFIXES = [
    "",
    "最新天气资料显示，",
    "根据当前天气预报，",
    "从气象信息看，",
    "结合雷达与预报看，",
    "按照当前天气形势，",
    "从本地天气预报看，",
    "综合气象条件来看，",
    "依据最新预报信息，",
    "从现有天气资料看，",
]

WEATHER_HONEST_SURFACE_PROFILES = [
    [
        ("天气预报为", "天气预计为"),
        ("温度", "气温"),
        ("风速", "风力"),
        ("降水概率", "降雨概率"),
        ("约为", "大约为"),
    ],
    [
        ("天气预报为", "天气情况为"),
        ("温度", "气温"),
        ("风速", "风速约"),
        ("降水概率", "出现降水的概率"),
        ("约", "大约"),
    ],
    [
        ("天气预报为", "天气走势为"),
        ("温度", "气温"),
        ("风速", "风力约"),
        ("降水概率", "降水可能性"),
        ("precipitation", "rain chance"),
    ],
    [
        ("天气预报为", "天气状况为"),
        ("温度", "气温"),
        ("风速", "风向风速"),
        ("降水概率", "降水机会"),
        ("winds", "wind"),
    ],
    [
        ("天气预报为", "天气预计呈现"),
        ("温度", "气温"),
        ("风速", "风力大约"),
        ("降水概率", "降雨机会"),
        ("约为", "约在"),
    ],
]

WEATHER_DIRECTION_LIBRARY = [
    {
        "name": "cooler_cloudier",
        "condition_mode": "cloudier",
        "temp_even": -12,
        "temp_odd": -9,
        "precip_strategy": "raise",
        "precip_range": (40, 70),
        "wind_scale_even": 1.45,
        "wind_scale_odd": 1.25,
        "wind_add_even": 5,
        "wind_add_odd": 3,
        "wind_dir_cn": "北风",
        "wind_dir_en": "N",
        "clauses": [
            "，冷空气影响会更明显。",
            "，云层会进一步增厚。",
            "，体感会比当前更凉。",
            "，天色会较当前预报更阴一些。",
        ],
    },
    {
        "name": "warmer_sunnier",
        "condition_mode": "sunnier",
        "temp_even": 10,
        "temp_odd": 13,
        "precip_strategy": "lower",
        "precip_range": (0, 10),
        "wind_scale_even": 0.95,
        "wind_scale_odd": 1.05,
        "wind_add_even": 1,
        "wind_add_odd": 2,
        "wind_dir_cn": "西南风",
        "wind_dir_en": "SSW",
        "clauses": [
            "，午后升温会更明显。",
            "，日照条件会比原预报更好。",
            "，天空状况会更偏晴朗。",
            "，白天气温上升会更快一些。",
        ],
    },
    {
        "name": "wetter_stormier",
        "condition_mode": "rainier",
        "temp_even": -6,
        "temp_odd": -3,
        "precip_strategy": "raise_strong",
        "precip_range": (65, 90),
        "wind_scale_even": 1.65,
        "wind_scale_odd": 1.45,
        "wind_add_even": 8,
        "wind_add_odd": 6,
        "wind_dir_cn": "东南风",
        "wind_dir_en": "ESE",
        "clauses": [
            "，降水过程会更持续。",
            "，局地对流会更活跃。",
            "，阵雨影响会比当前预报更明显。",
            "，后续更容易出现不稳定天气。",
        ],
    },
    {
        "name": "drier_clearer",
        "condition_mode": "clearer",
        "temp_even": 3,
        "temp_odd": 6,
        "precip_strategy": "lower_strong",
        "precip_range": (0, 5),
        "wind_scale_even": 0.85,
        "wind_scale_odd": 0.95,
        "wind_add_even": 0,
        "wind_add_odd": 1,
        "wind_dir_cn": "西北风",
        "wind_dir_en": "NW",
        "clauses": [
            "，降水影响会明显减弱。",
            "，天空状况会逐步转晴。",
            "，云量会比原预报更少。",
            "，整体会显得更干爽一些。",
        ],
    },
    {
        "name": "windier_gustier",
        "condition_mode": "breezier",
        "temp_even": -1,
        "temp_odd": 2,
        "precip_strategy": "nudge",
        "precip_range": (10, 35),
        "wind_scale_even": 2.0,
        "wind_scale_odd": 1.75,
        "wind_add_even": 10,
        "wind_add_odd": 8,
        "wind_dir_cn": "东北风",
        "wind_dir_en": "NE",
        "clauses": [
            "，阵风还会进一步增强。",
            "，风力会比当前预报更大。",
            "，体感上的风势会更明显。",
            "，局地风速可能继续走强。",
        ],
    },
]

CONDITION_REPLACEMENTS = {
    "cloudier": [
        ("大部分晴朗", ["大部分多云", "多云"]),
        ("部分晴朗", ["多云", "大部分多云"]),
        ("Mostly Sunny", ["Mostly Cloudy", "Partly Cloudy"]),
        ("Mostly Clear", ["Mostly Cloudy", "Partly Cloudy"]),
        ("Partly Sunny", ["Mostly Cloudy", "Partly Cloudy"]),
        ("Partly Cloudy", ["Cloudy", "Mostly Cloudy"]),
        ("晴朗", ["多云", "阴天"]),
        ("清朗", ["多云", "阴天"]),
        ("Clear", ["Cloudy", "Mostly Cloudy"]),
        ("Sunny", ["Cloudy", "Mostly Cloudy"]),
    ],
    "sunnier": [
        ("大部分多云", ["大部分晴朗", "部分晴朗"]),
        ("多云", ["晴朗", "部分晴朗"]),
        ("Cloudy", ["Mostly Sunny", "Mostly Clear"]),
        ("Mostly Cloudy", ["Mostly Sunny", "Partly Sunny"]),
        ("Partly Cloudy", ["Partly Sunny", "Mostly Sunny"]),
        ("阵雨", ["间晴", "晴朗"]),
        ("小雨", ["间晴", "晴朗"]),
        ("雷阵雨", ["晴间多云", "部分晴朗"]),
        ("Showers", ["Mostly Sunny", "Mostly Clear"]),
        ("Rain", ["Mostly Sunny", "Mostly Clear"]),
    ],
    "rainier": [
        ("大部分晴朗", ["有阵雨", "阵雨概率升高"]),
        ("部分晴朗", ["有阵雨可能", "多云伴有阵雨"]),
        ("晴朗", ["转阵雨", "有阵雨可能"]),
        ("清朗", ["转阵雨", "有阵雨可能"]),
        ("多云", ["多云伴有阵雨", "多云并有雷阵雨可能"]),
        ("Mostly Sunny", ["Showers Likely", "Scattered Showers"]),
        ("Mostly Clear", ["Showers Likely", "Rain Showers"]),
        ("Partly Sunny", ["Scattered Showers", "Showers Likely"]),
        ("Partly Cloudy", ["Showers Likely", "Scattered Showers"]),
        ("Mostly Cloudy", ["Rain Showers", "Showers Likely"]),
        ("Clear", ["Rain Showers", "Showers Likely"]),
        ("Sunny", ["Scattered Showers", "Showers Likely"]),
    ],
    "clearer": [
        ("大部分多云", ["大部分晴朗", "部分晴朗"]),
        ("多云", ["部分晴朗", "晴朗"]),
        ("阵雨", ["晴朗", "部分晴朗"]),
        ("小雨", ["晴朗", "部分晴朗"]),
        ("雷阵雨", ["晴朗", "部分晴朗"]),
        ("阴天", ["晴朗", "部分晴朗"]),
        ("Cloudy", ["Mostly Clear", "Mostly Sunny"]),
        ("Mostly Cloudy", ["Mostly Clear", "Partly Sunny"]),
        ("Partly Cloudy", ["Partly Sunny", "Mostly Sunny"]),
        ("Showers", ["Mostly Clear", "Mostly Sunny"]),
        ("Rain", ["Mostly Clear", "Mostly Sunny"]),
        ("Thunderstorms", ["Mostly Clear", "Mostly Sunny"]),
    ],
    "breezier": [
        ("Mostly Clear", ["Mostly Clear and Breezy", "Mostly Clear and Windy"]),
        ("Mostly Sunny", ["Mostly Sunny and Breezy", "Mostly Sunny and Windy"]),
        ("Partly Cloudy", ["Partly Cloudy and Windy", "Partly Cloudy and Breezy"]),
        ("晴朗", ["晴朗但风力增强", "晴朗且偏风大"]),
        ("清朗", ["清朗但风力增强", "清朗且偏风大"]),
        ("多云", ["多云且风力增强", "多云但风更强"]),
    ],
}

CHINESE_WIND_DIRECTIONS = [
    "东北风",
    "东南风",
    "西南风",
    "西北风",
    "偏北风",
    "偏南风",
    "偏东风",
    "偏西风",
    "北风",
    "南风",
    "东风",
    "西风",
]

_PREFIX_VARIANTS = [
    "从大气物理学研究和数值天气预报发展的学术视角，结合多个全球气候模式集合预报系统的最新输出结果，",
    "结合全球海洋与大气耦合系统分析和季节性气候预测信号，参考ENSO指数和印度洋偶极子等大尺度环流因子，",
    "基于人工智能深度学习后处理算法和多源遥感观测资料同化技术的融合分析，综合比对多家气象机构的集合预报，",
    "从农业气象灾害风险评估和城市精细化预报服务需求出发，结合作物生长关键期对温湿度光照条件的敏感性分析，",
    "基于高分辨率中尺度数值模式和卫星云图实时反演产品的综合对比，结合雷达回波结构和多普勒速度场诊断解读，",
    "从气候变化适应性规划和极端天气事件归因分析的科研角度，参考IPCC第六次评估报告关于区域气候预估核心结论，",
    "综合考虑城市热岛效应和下垫面类型差异对局地微气候的调节作用，结合环境监测站网多要素实况观测和分析场产品，",
    "从海洋气象预报和近海渔业安全保障的专业服务视角，参考海表温度异常分布和热带气旋路径概率预报产品更新结果，",
    "基于天气系统动力热力学综合诊断和位涡反演技术分析结果，结合平流层环流异常对对流层天气影响的最新研究认识，",
    "从防灾减灾和公共安全应急管理的决策支撑角度，综合评估致灾因子危险性和承灾体脆弱性的风险矩阵分析结果，",
    "依据全球综合观测系统和地基遥感探测网络提供的多层次大气廓线数据，结合温湿风要素高空探测和地面自动站对比分析，",
    "从气候资源开发利用和新能源功率预测的精细化需求出发，结合太阳辐射估算模型和风能资源评估的多时间尺度预报产品，",
    "基于大气化学输送模式和空气质量数值预报系统的模拟输出，综合考虑气象扩散条件变化和污染物排放清单更新的影响，",
    "从航空气象保障和飞行安全的专业预报服务角度，参考跑道视程和云底高自动观测以及终端区风切变预警信息分析，",
    "综合运用天气学分析和气候统计方法的多维度诊断结果，基于环流形势集合预报离散度评估和关键不确定性来源识别分析，",
]

FORBIDDEN_FINANCE_TERMS = [
    "财报",
    "利润率",
    "评级",
    "目标价",
    "经营指引",
    "全年指引",
    "需求转弱",
    "需求转强",
    "营收",
    "盈利",
    "股价",
]

def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")

def _validate_input(demo_data: dict) -> None:
    if not isinstance(demo_data, dict) or "job" not in demo_data:
        raise ValueError("input JSON must be an object containing job")
    batches = demo_data["job"].get("batches")
    if not isinstance(batches, list) or not batches:
        raise ValueError("input JSON must contain a non-empty job.batches list")

    expected_agents = {f"agent-{idx}" for idx in range(1, 51)}
    for batch in batches:
        items = batch.get("items")
        if not isinstance(items, list) or not items:
            raise ValueError(f"batch {batch.get('batchIndex')} has empty items")
        batch_agents = {item.get("agent") for item in items}
        if batch_agents != expected_agents:
            raise ValueError(
                f"batch {batch.get('batchIndex')} agent set mismatch: "
                f"expected 50 fixed agents, got {len(batch_agents)}"
            )
        if any(not isinstance(item.get("response"), str) for item in items):
            raise ValueError("weather text generation expects every item.response to be a string")

def _parse_bad_counts(spec: str) -> list[int]:
    counts = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        counts.append(int(part))
    return sorted(set(counts))

def _agent_number(agent_name: str) -> int:
    try:
        return int(str(agent_name).split("-")[1])
    except (IndexError, ValueError, TypeError):
        return 1

def _apply_phrase_profile(text: str, replacements: list[tuple[str, str]]) -> str:
    result = text
    for src, dst in replacements:
        if src != dst:
            result = result.replace(src, dst)
    return result

def diversify_honest_weather_text(text: str, agent_name: str) -> str:
    agent_num = _agent_number(agent_name)
    profile = WEATHER_HONEST_SURFACE_PROFILES[(agent_num - 1) % len(WEATHER_HONEST_SURFACE_PROFILES)]
    prefix = WEATHER_HONEST_PREFIXES[(agent_num - 1) % len(WEATHER_HONEST_PREFIXES)]
    result = _apply_phrase_profile(text, profile)
    if prefix and not result.startswith(prefix):
        result = prefix + result
    return result

def diversify_clean_demo_data(demo_data):
    data = copy.deepcopy(demo_data)
    samples = []

    for batch in data["job"]["batches"]:
        bi = batch["batchIndex"]
        grouped = {}
        for item in batch["items"]:
            if isinstance(item["response"], str):
                key = (item["object"], item["response"])
                grouped.setdefault(key, []).append(item)

        for (obj, original_text), items in grouped.items():
            if len(items) <= 1:
                continue

            sorted_items = sorted(items, key=lambda entry: (_agent_number(entry["agent"]), str(entry["agent"])))
            used_texts = {
                entry["response"]
                for entry in batch["items"]
                if isinstance(entry["response"], str) and entry["object"] == obj
            }

            for item in sorted_items[1:]:
                before = item["response"]
                after = diversify_honest_weather_text(before, item["agent"])
                if after in used_texts:
                    alt_prefix = WEATHER_HONEST_PREFIXES[
                        (_agent_number(item["agent"]) + bi) % len(WEATHER_HONEST_PREFIXES)
                    ]
                    if alt_prefix and not after.startswith(alt_prefix):
                        after = alt_prefix + after
                if after == before:
                    continue
                item["response"] = after
                used_texts.add(after)
                if len(samples) < 12:
                    samples.append((bi, obj, item["agent"], before, after))

    return data, samples

def build_clean_text_map(demo_data) -> dict[tuple[int, str], str]:
    mapping = {}
    for batch in demo_data["job"]["batches"]:
        bi = batch["batchIndex"]
        for item in batch["items"]:
            if item["agent"] == "agent-1" and isinstance(item["response"], str):
                mapping[(bi, item["object"])] = item["response"]
    return mapping

def build_clean_agent_text_map(demo_data) -> dict[tuple[int, str, str], str]:
    mapping = {}
    for batch in demo_data["job"]["batches"]:
        bi = batch["batchIndex"]
        for item in batch["items"]:
            if isinstance(item["response"], str):
                mapping[(bi, item["object"], item["agent"])] = item["response"]
    return mapping

def _validate_collude_group_sizes(group_sizes: list[int], bad_agents: list[str]) -> list[int]:
    cleaned = [int(size) for size in group_sizes if int(size) > 0]
    if sum(cleaned) != len(bad_agents):
        raise ValueError(f"group_sizes {cleaned} do not sum to bad agent count {len(bad_agents)}")
    return cleaned

def _stable_object_seed(batch_index: int, object_name: str) -> int:
    return int(batch_index) * 1009 + sum(ord(ch) for ch in str(object_name))

def _collude_variant_id(
    agent_name: str,
    bad_agents: list[str],
    object_seed: int,
    group_sizes: list[int],
) -> int:
    if not group_sizes or not bad_agents:
        return 0
    try:
        agent_index = bad_agents.index(agent_name)
    except ValueError:
        agent_index = 0

    rotated_index = (agent_index + object_seed) % len(bad_agents)
    boundary = 0
    for variant_id, size in enumerate(group_sizes):
        boundary += size
        if rotated_index < boundary:
            return variant_id
    return len(group_sizes) - 1

def _direction_names(direction_count: int) -> list[str]:
    if direction_count > len(WEATHER_DIRECTION_LIBRARY):
        raise ValueError(
            f"Requested {direction_count} directions, but only "
            f"{len(WEATHER_DIRECTION_LIBRARY)} weather directions are defined."
        )
    return [WEATHER_DIRECTION_LIBRARY[idx]["name"] for idx in range(direction_count)]

def _direction_spec(direction_name: str) -> dict:
    for spec in WEATHER_DIRECTION_LIBRARY:
        if spec["name"] == direction_name:
            return spec
    raise ValueError(f"Unknown weather collude direction: {direction_name}")

def _apply_atomic_replacements(text: str, replacements: list[tuple[str, list[str]]], seed: int) -> tuple[str, int]:
    if not replacements:
        return text, 0

    mapping = {}
    for src, candidates in replacements:
        mapping[src] = candidates[seed % len(candidates)]

    pattern = re.compile("|".join(re.escape(key) for key in sorted(mapping, key=len, reverse=True)))
    count = [0]

    def _replace(match: re.Match) -> str:
        count[0] += 1
        return mapping[match.group(0)]

    return pattern.sub(_replace, text), count[0]

def _replace_weather_conditions(text: str, mode: str, seed: int) -> tuple[str, int]:
    return _apply_atomic_replacements(text, CONDITION_REPLACEMENTS.get(mode, []), seed)

def _temperature_delta(spec: dict, seed: int) -> int:
    base = spec["temp_even"] if seed % 2 == 0 else spec["temp_odd"]
    return int(base + ((seed % 3) - 1))

def _mutate_temperature(text: str, spec: dict, seed: int) -> tuple[str, int]:
    delta = _temperature_delta(spec, seed)
    count = [0]

    def _replace(match: re.Match) -> str:
        count[0] += 1
        value = int(match.group(1))
        target = max(0, min(120, value + delta))
        return match.group(0).replace(match.group(1), str(target), 1)

    pattern = re.compile(r"(?<!\d)(\d{1,3})(?=\s*(?:F\b|华氏度))")
    return pattern.sub(_replace, text), count[0]

def _replace_wind_direction(text: str, spec: dict) -> tuple[str, int]:
    result = text
    count = 0

    chinese_pattern = re.compile("|".join(re.escape(direction) for direction in sorted(CHINESE_WIND_DIRECTIONS, key=len, reverse=True)))
    result, n_cn = chinese_pattern.subn(spec["wind_dir_cn"], result, count=1)
    count += n_cn

    english_pattern = re.compile(r"\b(?:NNE|ENE|ESE|SSE|SSW|WSW|WNW|NNW|NE|NW|SE|SW|N|S|E|W)\b(?=\s+winds?\b)")
    result, n_en = english_pattern.subn(spec["wind_dir_en"], result, count=1)
    count += n_en
    return result, count

def _mutate_single_wind_value(value: int, spec: dict, seed: int) -> int:
    scale = spec["wind_scale_even"] if seed % 2 == 0 else spec["wind_scale_odd"]
    add = spec["wind_add_even"] if seed % 2 == 0 else spec["wind_add_odd"]
    target = int(round(value * scale + add))
    return max(0, min(65, target))

def _mutate_wind_speeds(text: str, spec: dict, seed: int) -> tuple[str, int]:
    count = [0]

    def _replace_range(match: re.Match) -> str:
        count[0] += 1
        left = int(match.group(1))
        right = int(match.group(2))
        unit = match.group(3)
        new_left = _mutate_single_wind_value(left, spec, seed)
        new_right = _mutate_single_wind_value(right, spec, seed + 1)
        if new_left > new_right:
            new_left, new_right = new_right, new_left
        return f"{new_left}至{new_right}{unit}"

    def _replace_single(match: re.Match) -> str:
        count[0] += 1
        value = int(match.group(1))
        unit = match.group(2)
        target = _mutate_single_wind_value(value, spec, seed)
        return f"{target}{unit}"

    result = re.sub(
        r"(?<!\d)(\d{1,2})\s*(?:至|到|-)\s*(\d{1,2})\s*(mph|英里/小时|公里/小时)",
        _replace_range,
        text,
    )
    result = re.sub(
        r"(?<![\d至到-])(\d{1,2})\s*(mph|英里/小时|公里/小时)",
        _replace_single,
        result,
    )
    return result, count[0]

def _mutate_precipitation_value(value: int, spec: dict, seed: int) -> int:
    low, high = spec["precip_range"]
    span = max(high - low, 0)
    base = low + (seed % (span + 1 if span else 1))
    strategy = spec["precip_strategy"]

    if strategy == "raise":
        target = max(value + 12, base)
    elif strategy == "raise_strong":
        target = max(value + 25, base)
    elif strategy == "lower":
        target = min(value, base)
    elif strategy == "lower_strong":
        target = min(max(int(round(value * 0.2)), 0), base)
    else:
        target = int(round((value + base) / 2))

    return max(0, min(95, target))

def _mutate_precipitation(text: str, spec: dict, seed: int) -> tuple[str, int]:
    count = [0]

    def _replace_postfix(match: re.Match) -> str:
        count[0] += 1
        value = int(match.group(1))
        suffix = match.group(2)
        target = _mutate_precipitation_value(value, spec, seed)
        return f"{target}{suffix}"

    def _replace_prefix(match: re.Match) -> str:
        count[0] += 1
        prefix = match.group(1)
        value = int(match.group(2))
        suffix = match.group(3)
        target = _mutate_precipitation_value(value, spec, seed)
        return f"{prefix}{target}{suffix}"

    result = re.sub(
        r"(\d{1,3})(\s*%\s*(?:precipitation|降水概率|降雨概率|降水机会|降雨机会|出现降水的概率|降水可能性))",
        _replace_postfix,
        text,
        flags=re.IGNORECASE,
    )
    result = re.sub(
        r"((?:降水概率|降雨概率|降水机会|降雨机会|出现降水的概率|降水可能性)\s*(?:为|约为|大约为|约|大约)?\s*)(\d{1,3})(\s*%)",
        _replace_prefix,
        result,
        flags=re.IGNORECASE,
    )
    return result, count[0]

def _append_clause(text: str, clause: str) -> str:
    return text.rstrip("。！？") + clause

def corrupt_text_collude_weather(canonical_text: str, object_seed: int, direction_name: str) -> str:
    spec = _direction_spec(direction_name)
    result = canonical_text

    result, _ = _replace_weather_conditions(result, spec["condition_mode"], object_seed)
    result, _ = _mutate_temperature(result, spec, object_seed)
    result, _ = _replace_wind_direction(result, spec)
    result, _ = _mutate_wind_speeds(result, spec, object_seed)
    result, _ = _mutate_precipitation(result, spec, object_seed)

    clause = spec["clauses"][object_seed % len(spec["clauses"])]
    result = _append_clause(result, clause)
    return result

def corrupt_text_collude_weather_numeric_only(canonical_text: str, object_seed: int, direction_name: str) -> str:

    spec = _direction_spec(direction_name)
    result = canonical_text

    result, _ = _mutate_temperature(result, spec, object_seed)
    result, _ = _mutate_wind_speeds(result, spec, object_seed)
    result, _ = _mutate_precipitation(result, spec, object_seed)

    return result

_SUFFIX_VARIANTS = [
    "以上预报仍需结合后续实况观测数据进一步验证。",
    "具体天气变化仍取决于天气系统演变和局地因素。",
    "但该趋势的持续性仍有赖于大气环流形势配合。",
    "不过短期内天气系统调整可能带来阶段性变化。",
    "后续需要密切关注关键气象要素的边际变化。",
    "风险提示：以上分析仅供参考，请关注最新预报。",
    "整体来看，中期天气形势仍然值得持续关注。",
    "但需要警惕突发性强对流天气带来的局地影响。",
    "建议公众综合多渠道天气信息后再做出行安排。",
    "此判断基于当前气象资料，不排除后续调整可能。",
    "不同预报模式对此的预测结果可能存在一定分歧。",
    "中长期看，季节性气候特征仍是主导天气趋势。",
    "短期扰动不改变中期天气形势演变的大方向。",
    "以上仅为定性研判，精确预报仍有待模式更新。",
    "综上所述，近期出行需更加关注天气预警信息。",
]

def diversify_clean_demo_strong(demo_data):

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
    bad_agents: list[str],
    group_sizes: list[int],
):
    direction_names = _direction_names(len(group_sizes))
    group_sizes = _validate_collude_group_sizes(group_sizes, bad_agents)
    data = copy.deepcopy(demo_data)

    for batch in data["job"]["batches"]:
        bi = batch["batchIndex"]
        object_groups = {}
        for item in batch["items"]:
            agent = item["agent"]
            if agent in bad_agents and isinstance(item["response"], str):
                object_groups.setdefault(item["object"], []).append(item)

        for obj, items in object_groups.items():
            canonical_text = canonical_clean_map.get((bi, obj), items[0]["response"])
            object_seed = _stable_object_seed(bi, obj)

            for item in items:
                variant_id = _collude_variant_id(
                    item["agent"],
                    bad_agents,
                    object_seed,
                    group_sizes,
                )
                item["response"] = corrupt_text_collude_weather(
                    canonical_text=canonical_text,
                    object_seed=object_seed,
                    direction_name=direction_names[variant_id],
                )

    _save_json(out_file_path, data)
    return data, direction_names

def regenerate_collude_numeric_only_file(
    out_file_path: Path,
    demo_data,
    canonical_clean_map: dict,
    bad_agents: list[str],
    group_sizes: list[int],
):

    direction_names = _direction_names(len(group_sizes))
    group_sizes = _validate_collude_group_sizes(group_sizes, bad_agents)
    data = copy.deepcopy(demo_data)

    for batch in data["job"]["batches"]:
        bi = batch["batchIndex"]
        object_groups = {}
        for item in batch["items"]:
            agent = item["agent"]
            if agent in bad_agents and isinstance(item["response"], str):
                object_groups.setdefault(item["object"], []).append(item)

        for obj, items in object_groups.items():
            canonical_text = canonical_clean_map.get((bi, obj), items[0]["response"])
            object_seed = _stable_object_seed(bi, obj)

            for item in items:
                variant_id = _collude_variant_id(
                    item["agent"],
                    bad_agents,
                    object_seed,
                    group_sizes,
                )
                item["response"] = corrupt_text_collude_weather_numeric_only(
                    canonical_text=canonical_text,
                    object_seed=object_seed,
                    direction_name=direction_names[variant_id],
                )

    _save_json(out_file_path, data)
    return data, direction_names

def _ensure_no_finance_terms(path: Path) -> None:
    data = _load_json(path)
    payload = []
    for batch in data["job"]["batches"]:
        for item in batch["items"]:
            response = item.get("response")
            if isinstance(response, str):
                payload.append(response)
    corpus = "\n".join(payload)
    found = [term for term in FORBIDDEN_FINANCE_TERMS if term in corpus]
    if found:
        raise ValueError(f"{path.name} still contains finance terms: {found}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate 50-agent weather text demo + collude variants from a clean input JSON."
    )
    parser.add_argument(
        "--input-path",
        required=True,
        help="Absolute or relative path to the clean 50-agent weather JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DATA_DIR),
        help="Directory that will receive demo.json and demo_*bad_collude.json files.",
    )
    parser.add_argument(
        "--bad-counts",
        default="5,10,15,20,25,30,35,40",
        help="Comma-separated malicious agent counts to generate.",
    )
    parser.add_argument(
        "--numeric-only",
        action="store_true",
        help="Numeric-only attack: mutate numbers only, keep text words unchanged. "
             "Output files use num_demo prefix.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_label = "numeric-only" if args.numeric_only else "collude"
    gen_fn = regenerate_collude_numeric_only_file if args.numeric_only else regenerate_collude_file

    clean_demo = _load_json(input_path)
    _validate_input(clean_demo)

    if args.numeric_only:
        diverse_demo_data = diversify_clean_demo_strong(clean_demo)
        div_path = output_dir / "demo_div.json"
        _save_json(div_path, diverse_demo_data)
        print(f"saved {div_path} [strong-diversity clean]")
    else:

        diverse_demo_data = diversify_clean_demo_strong(clean_demo)
        div_path = output_dir / "demo_div.json"
        _save_json(div_path, diverse_demo_data)
        print(f"saved {div_path} [strong-diversity clean]")

    canonical_clean_map = build_clean_text_map(diverse_demo_data)

    for bad_count in _parse_bad_counts(args.bad_counts):
        if bad_count not in BAD_GROUP_SPECS:
            raise ValueError(f"unsupported bad_count={bad_count}; supported: {sorted(BAD_GROUP_SPECS)}")
        bad_agents = [f"agent-{idx}" for idx in range(1, bad_count + 1)]
        if args.numeric_only:
            out_path = output_dir / f"demo_{bad_count}bad_numonly.json"
        else:
            out_path = output_dir / f"demo_{bad_count}bad_collude.json"
        _, direction_names = gen_fn(
            out_file_path=out_path,
            demo_data=diverse_demo_data,
            canonical_clean_map=canonical_clean_map,
            bad_agents=bad_agents,
            group_sizes=BAD_GROUP_SPECS[bad_count],
        )
        if not args.numeric_only:
            _ensure_no_finance_terms(out_path)
        print(
            f"saved {out_path} [{mode_label}] groups={BAD_GROUP_SPECS[bad_count]} "
            f"directions={direction_names}"
        )

if __name__ == "__main__":
    main()
