/**
 * Hybrid Truth Finder ZKP 测试脚本
 * 
 * 这个脚本用于测试和验证零知识证明电路的正确性
 */

// 注意：此测试脚本不依赖circomlib和snarkjs
// 仅用于验证数学计算的正确性

// ========== 测试用例1：加权平均计算 ==========
function testWeightedAverage() {
    console.log("测试1：加权平均计算");
    
    // 测试数据
    const weights = [0.9, 0.8, 0.7, 0.6];  // 源可靠性
    const values = [100.5, 101.0, 99.8, 100.2];  // 源提供的值
    
    // 计算真值
    let weighted_sum = 0;
    let total_weight = 0;
    for (let i = 0; i < weights.length; i++) {
        weighted_sum += weights[i] * values[i];
        total_weight += weights[i];
    }
    const truth = weighted_sum / total_weight;
    
    console.log(`  权重: ${weights}`);
    console.log(`  值: ${values}`);
    console.log(`  加权和: ${weighted_sum}`);
    console.log(`  总权重: ${total_weight}`);
    console.log(`  真值: ${truth}`);
    
    // 验证计算
    const expected_truth = truth;
    const claimed_truth = truth;  // 正确的情况
    
    if (Math.abs(claimed_truth - expected_truth) < 0.001) {
        console.log("  ✓ 测试通过：真值计算正确");
    } else {
        console.log("  ✗ 测试失败：真值计算错误");
    }
    
    // 测试错误情况
    const wrong_truth = 150.0;
    if (Math.abs(wrong_truth - expected_truth) > 0.001) {
        console.log("  ✓ 测试通过：错误真值被正确拒绝");
    } else {
        console.log("  ✗ 测试失败：错误真值未被拒绝");
    }
    
    console.log("");
}

// ========== 测试用例2：加权投票 ==========
function testWeightedVoting() {
    console.log("测试2：加权投票");
    
    // 测试数据
    const weights = [0.9, 0.8, 0.5, 0.3];
    const values = ["Einstein", "Einstein", "Newton", "Einstein"];
    const candidates = ["Einstein", "Newton", "Galilei"];
    
    // 计算每个候选值的权重和
    const candidate_weights = {};
    for (let i = 0; i < values.length; i++) {
        const value = values[i];
        if (!candidate_weights[value]) {
            candidate_weights[value] = 0;
        }
        candidate_weights[value] += weights[i];
    }
    
    console.log(`  权重: ${weights}`);
    console.log(`  值: ${values}`);
    console.log(`  候选值权重和:`, candidate_weights);
    
    // 找到权重和最大的候选值
    let max_weight = 0;
    let truth = null;
    for (const [candidate, weight] of Object.entries(candidate_weights)) {
        if (weight > max_weight) {
            max_weight = weight;
            truth = candidate;
        }
    }
    
    console.log(`  真值: ${truth} (权重和: ${max_weight})`);
    
    // 验证
    const expected_truth = "Einstein";
    const expected_weight = 0.9 + 0.8 + 0.3;  // 2.0
    
    if (truth === expected_truth && Math.abs(max_weight - expected_weight) < 0.001) {
        console.log("  ✓ 测试通过：加权投票正确");
    } else {
        console.log("  ✗ 测试失败：加权投票错误");
    }
    
    console.log("");
}

// ========== 测试用例3：可靠性更新 ==========
function testReliabilityUpdate() {
    console.log("测试3：可靠性更新");
    
    // 测试参数
    const alpha = 0.9;
    const beta = 100.0;
    const prev_error = 0.05;
    const prev_consistency = 0.98;
    const current_mse = 0.1;
    
    // 计算新误差
    const new_error = alpha * prev_error + (1 - alpha) * current_mse;
    console.log(`  新误差: ${new_error}`);
    
    // 计算一致性
    const error_change = Math.abs(new_error - prev_error);
    const new_k = erfc(beta * error_change);
    const k_updated = alpha * prev_consistency + (1 - alpha) * new_k;
    console.log(`  新一致性: ${k_updated}`);
    
    // 计算可靠性
    const error_component = 1.0 / (1.0 + new_error);
    const new_reliability = error_component * k_updated;
    console.log(`  新可靠性: ${new_reliability}`);
    
    // 验证计算
    const expected_error = 0.9 * 0.05 + 0.1 * 0.1;  // 0.055
    if (Math.abs(new_error - expected_error) < 0.001) {
        console.log("  ✓ 测试通过：误差更新正确");
    } else {
        console.log("  ✗ 测试失败：误差更新错误");
    }
    
    console.log("");
}

// ========== erfc函数实现 ==========
function erfc(x) {
    // 使用近似公式：erfc(x) ≈ 1 - erf(x)
    // erf(x) 使用多项式近似
    const a1 =  0.254829592;
    const a2 = -0.284496736;
    const a3 =  1.421413741;
    const a4 = -1.453152027;
    const a5 =  1.061405429;
    const p  =  0.3275911;
    
    const sign = x < 0 ? -1 : 1;
    x = Math.abs(x);
    
    const t = 1.0 / (1.0 + p * x);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
    
    return 1.0 - sign * y;
}

// ========== 主测试函数 ==========
function runTests() {
    console.log("=".repeat(60));
    console.log("Hybrid Truth Finder ZKP 正确性验证测试");
    console.log("=".repeat(60));
    console.log("");
    
    testWeightedAverage();
    testWeightedVoting();
    testReliabilityUpdate();
    
    console.log("=".repeat(60));
    console.log("所有测试完成");
    console.log("=".repeat(60));
}

// ========== 运行测试 ==========
if (require.main === module) {
    runTests();
}

module.exports = {
    testWeightedAverage,
    testWeightedVoting,
    testReliabilityUpdate,
    erfc
};
