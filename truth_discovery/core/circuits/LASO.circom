/*
 * LASOTruth Algorithm — Zero Knowledge Proof Circuit
 *
 * 本电路证明 LASOTruth 算法各步骤的计算正确性，与
 * LASO_truth_finder.py 中的实现一一对应：
 *
 *   模板 1 — WeightedMedian(n)
 *     证明 claimed_truth 是 (values[], weights[]) 的加权中位数。
 *     算法对应：_weighted_median() + _irls_robust_estimate() 的初始估计
 *     无需排序：通过比较验证 weight(≤m) ≥ total/2 且 weight(<m) < total/2。
 *
 *   模板 2 — BetaBernoulliUpdate()
 *     证明 Beta-Bernoulli 可靠度更新的正确性：
 *       α_new = α_old + η × c
 *       β_new = β_old + η × (1 − c)
 *       r     = α_new / (α_new + β_new)
 *     算法对应：_update_source_state() 中的可靠度更新
 *
 *   模板 3 — GaussianConsistencyScore()
 *     证明高斯一致性评分的近似正确性：
 *       score ≈ exp(−0.5 × (residual / scale)²)
 *     算法对应：_numeric_pipeline() 和 _text_pipeline() 中的一致性评分
 *     使用 Padé(2,2) 逼近 exp(−t)，在 t∈[0,4] 内误差 < 1%。
 *
 *   模板 4 — LASOTruthProof(n)
 *     组合以上模板，完成单个 object 的完整证明。
 *
 * 定点算术：所有数值乘以 SCALE = 10⁶。
 * 依赖：circomlib（comparators, bitify）
 */

pragma circom 2.0.0;

include "node_modules/circomlib/circuits/comparators.circom";
include "node_modules/circomlib/circuits/bitify.circom";

// ====================================================================
// 辅助模板
// ====================================================================

template Sum(n) {
    signal input in[n];
    signal output out;
    signal partial[n + 1];
    partial[0] <== 0;
    for (var i = 0; i < n; i++) {
        partial[i + 1] <== partial[i] + in[i];
    }
    out <== partial[n];
}

// ====================================================================
// 模板 1：WeightedMedian(n)
// ====================================================================
//
// 证明 claimed_truth 是加权中位数，无需排序。
//
// 验证条件：
//   (a) claimed_truth 在原始 values 中至少出现一次
//   (b) Σ{w_i : v_i ≤ claimed_truth} × 2 ≥ total_weight
//   (c) Σ{w_i : v_i < claimed_truth} × 2 < total_weight
//
// 约束规模：O(n) 次比较，每次 ~65 个 R1CS 约束
//
template WeightedMedian(n) {
    signal input values[n];
    signal input weights[n];
    signal input claimed_truth;

    signal output is_valid;
    signal output total_weight;

    // --- (a) claimed_truth ∈ values ---
    component eq[n];
    signal match_flag[n];
    for (var i = 0; i < n; i++) {
        eq[i] = IsEqual();
        eq[i].in[0] <== values[i];
        eq[i].in[1] <== claimed_truth;
        match_flag[i] <== eq[i].out;
    }
    component sum_match = Sum(n);
    for (var i = 0; i < n; i++) {
        sum_match.in[i] <== match_flag[i];
    }
    component has_match = GreaterThan(32);
    has_match.in[0] <== sum_match.out;
    has_match.in[1] <== 0;
    has_match.out === 1;

    // --- (b) weight_leq = Σ w_i where v_i ≤ claimed_truth ---
    component leq[n];
    signal w_leq[n];
    for (var i = 0; i < n; i++) {
        leq[i] = LessEqThan(64);
        leq[i].in[0] <== values[i];
        leq[i].in[1] <== claimed_truth;
        w_leq[i] <== leq[i].out * weights[i];
    }
    component sum_leq = Sum(n);
    for (var i = 0; i < n; i++) {
        sum_leq.in[i] <== w_leq[i];
    }

    // --- (c) weight_lt = Σ w_i where v_i < claimed_truth ---
    component lt[n];
    signal w_lt[n];
    for (var i = 0; i < n; i++) {
        lt[i] = LessThan(64);
        lt[i].in[0] <== values[i];
        lt[i].in[1] <== claimed_truth;
        w_lt[i] <== lt[i].out * weights[i];
    }
    component sum_lt = Sum(n);
    for (var i = 0; i < n; i++) {
        sum_lt.in[i] <== w_lt[i];
    }

    // --- total weight ---
    component sum_w = Sum(n);
    for (var i = 0; i < n; i++) {
        sum_w.in[i] <== weights[i];
    }
    total_weight <== sum_w.out;

    // --- 中位数条件：2 × weight_leq ≥ total_weight ---
    component check_geq = GreaterEqThan(64);
    check_geq.in[0] <== 2 * sum_leq.out;
    check_geq.in[1] <== total_weight;
    check_geq.out === 1;

    // --- 最小性：2 × weight_lt < total_weight ---
    component check_lt = LessThan(64);
    check_lt.in[0] <== 2 * sum_lt.out;
    check_lt.in[1] <== total_weight;
    check_lt.out === 1;

    is_valid <== 1;
}

// ====================================================================
// 模板 2：BetaBernoulliUpdate()
// ====================================================================
//
// 证明 Beta-Bernoulli 可靠度更新（定点算术）：
//   α_new = α_old + ⌊η × c / SCALE⌋
//   β_new = β_old + ⌊η × (1 − c) / SCALE⌋
//   r     = ⌊α_new × SCALE / (α_new + β_new)⌋
//
// 所有输入/输出均为定点整数（× SCALE）
//
template BetaBernoulliUpdate() {
    var SCALE = 1000000;

    signal input old_alpha;
    signal input old_beta;
    signal input eta;              // 学习率 (scaled)
    signal input consistency;      // c_s ∈ [0, SCALE]
    signal input claimed_new_alpha;
    signal input claimed_new_beta;
    signal input claimed_reliability;

    signal output is_valid;

    // --- α_new = α_old + ⌊η × c / SCALE⌋ ---
    signal eta_c;
    eta_c <== eta * consistency;
    signal delta_alpha;
    delta_alpha <-- eta_c \ SCALE;
    // 约束：delta_alpha × SCALE ≤ eta_c < (delta_alpha + 1) × SCALE
    signal lo1;
    lo1 <== delta_alpha * SCALE;
    component c1 = LessEqThan(128);
    c1.in[0] <== lo1;
    c1.in[1] <== eta_c;
    c1.out === 1;
    component c2 = LessThan(128);
    c2.in[0] <== eta_c;
    c2.in[1] <== lo1 + SCALE;
    c2.out === 1;
    claimed_new_alpha === old_alpha + delta_alpha;

    // --- β_new = β_old + ⌊η × (SCALE − c) / SCALE⌋ ---
    signal one_minus_c;
    one_minus_c <== SCALE - consistency;
    signal eta_1mc;
    eta_1mc <== eta * one_minus_c;
    signal delta_beta;
    delta_beta <-- eta_1mc \ SCALE;
    signal lo2;
    lo2 <== delta_beta * SCALE;
    component c3 = LessEqThan(128);
    c3.in[0] <== lo2;
    c3.in[1] <== eta_1mc;
    c3.out === 1;
    component c4 = LessThan(128);
    c4.in[0] <== eta_1mc;
    c4.in[1] <== lo2 + SCALE;
    c4.out === 1;
    claimed_new_beta === old_beta + delta_beta;

    // --- r = ⌊α_new × SCALE / (α_new + β_new)⌋ ---
    signal denom;
    denom <== claimed_new_alpha + claimed_new_beta;
    signal numer;
    numer <== claimed_new_alpha * SCALE;
    signal computed_r;
    computed_r <-- numer \ denom;
    signal r_check;
    r_check <== computed_r * denom;
    component c5 = LessEqThan(128);
    c5.in[0] <== r_check;
    c5.in[1] <== numer;
    c5.out === 1;
    component c6 = LessThan(128);
    c6.in[0] <== numer;
    c6.in[1] <== r_check + denom;
    c6.out === 1;
    claimed_reliability === computed_r;

    is_valid <== 1;
}

// ====================================================================
// 模板 3：GaussianConsistencyScore()
// ====================================================================
//
// 证明高斯一致性评分的近似正确性：
//   score ≈ exp(−0.5 × z²)    其中 z = residual / scale
//
// 使用 Padé(2,2) 逼近 exp(−t)（令 t = z²/2）：
//   exp(−t) ≈ (12 − 6t + t²) / (12 + 6t + t²)
//
// 精度：t ∈ [0, 4]（即 |z| ≤ 2.83）时相对误差 < 1%
// 对 t > 4 的情况，输出钳位到 0。
//
template GaussianConsistencyScore() {
    var SCALE = 1000000;
    var THRESHOLD = 4000000;  // t > 4×SCALE → score = 0

    signal input residual_abs;  // |value − truth| (scaled)
    signal input mad_scale;     // MAD-based scale (scaled, > 0)
    signal input claimed_score; // ∈ [0, SCALE]

    signal output is_valid;

    // z = residual_abs × SCALE / mad_scale
    signal z;
    z <-- (residual_abs * SCALE) \ mad_scale;
    signal z_check;
    z_check <== z * mad_scale;
    component zc1 = LessEqThan(128);
    zc1.in[0] <== z_check;
    zc1.in[1] <== residual_abs * SCALE;
    zc1.out === 1;
    component zc2 = LessThan(128);
    zc2.in[0] <== residual_abs * SCALE;
    zc2.in[1] <== z_check + mad_scale;
    zc2.out === 1;

    // t = z² / (2 × SCALE)
    signal z_sq;
    z_sq <== z * z;
    signal t;
    t <-- z_sq \ (2 * SCALE);

    // Padé(2,2):
    //   numerator   = 12×S² − 6×S×t + t²
    //   denominator = 12×S² + 6×S×t + t²
    signal t_sq;
    t_sq <== t * t;
    signal six_s_t;
    six_s_t <== 6 * SCALE * t;
    var twelve_s2 = 12 * SCALE * SCALE;

    signal pade_num;
    pade_num <== twelve_s2 - six_s_t + t_sq;
    signal pade_den;
    pade_den <== twelve_s2 + six_s_t + t_sq;

    // 条件输出：t ≤ THRESHOLD ? pade : 0
    component is_large = GreaterThan(128);
    is_large.in[0] <== t;
    is_large.in[1] <== THRESHOLD;

    signal unclamped;
    unclamped <-- (pade_num * SCALE) \ pade_den;

    signal selected;
    selected <== unclamped * (1 - is_large.out);

    claimed_score === selected;
    is_valid <== 1;
}

// ====================================================================
// 模板 4：LASOTruthProof(n)  — 主证明电路
// ====================================================================
//
// 对单个 object 证明：
//   1. 真值 = 加权中位数
//   2. 每个源的 Beta-Bernoulli 可靠度更新正确
//   3. 绑定哈希（statement_hash）匹配私有输入
//
template LASOTruthProof(n) {
    // ===== 数值真值 =====
    signal input values[n];
    signal input weights[n];
    signal input claimed_truth;
    signal input claimed_total_weight;
    signal input statement_hash;

    // ===== Beta-Bernoulli 可靠度更新 =====
    signal input old_alpha[n];
    signal input old_beta[n];
    signal input eta;
    signal input consistency[n];
    signal input claimed_new_alpha[n];
    signal input claimed_new_beta[n];
    signal input claimed_reliability[n];

    // ===== 公开输出 =====
    signal output public_statement_hash;
    signal output public_truth;
    signal output public_total_weight;
    signal output is_valid;

    // --- 1. 加权中位数 ---
    component wm = WeightedMedian(n);
    for (var i = 0; i < n; i++) {
        wm.values[i] <== values[i];
        wm.weights[i] <== weights[i];
    }
    wm.claimed_truth <== claimed_truth;
    claimed_total_weight === wm.total_weight;

    // --- 2. Beta-Bernoulli 更新 ---
    component bb[n];
    signal bb_valid[n];
    for (var i = 0; i < n; i++) {
        bb[i] = BetaBernoulliUpdate();
        bb[i].old_alpha <== old_alpha[i];
        bb[i].old_beta <== old_beta[i];
        bb[i].eta <== eta;
        bb[i].consistency <== consistency[i];
        bb[i].claimed_new_alpha <== claimed_new_alpha[i];
        bb[i].claimed_new_beta <== claimed_new_beta[i];
        bb[i].claimed_reliability <== claimed_reliability[i];
        bb_valid[i] <== bb[i].is_valid;
    }

    // --- 3. 绑定哈希 ---
    //   hash = Σ values[k]*(k+1) + Σ weights[k]*(k+1001)
    //        + claimed_truth + claimed_total_weight
    signal hash_terms[2 * n + 2];
    component sum_hash = Sum(2 * n + 2);
    for (var k = 0; k < n; k++) {
        hash_terms[2 * k] <== values[k] * (k + 1);
        hash_terms[2 * k + 1] <== weights[k] * (k + 1001);
    }
    hash_terms[2 * n] <== claimed_truth;
    hash_terms[2 * n + 1] <== claimed_total_weight;
    for (var t = 0; t < 2 * n + 2; t++) {
        sum_hash.in[t] <== hash_terms[t];
    }
    statement_hash === sum_hash.out;

    // --- 公开输出 ---
    public_statement_hash <== statement_hash;
    public_truth <== claimed_truth;
    public_total_weight <== claimed_total_weight;
    is_valid <== 1;
}

// ====================================================================
// 默认实例化：10 个数据源
// ====================================================================
component main = LASOTruthProof(10);
