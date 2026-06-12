// SPDX-License-Identifier: MIT
pragma solidity ^0.8.13; // chainMaker版本
// pragma solidity ^0.8.0; // fisco版本

/**
 * @title OracleAggregator
 * @dev 预言机聚合合约：
 * 1) 价格聚合（历史能力）
 * 2) 四节点 Ollama 推理聚合（新增能力）
 */
contract OracleAggregator {
    uint256 public constant OLLAMA_NODE_COUNT = 4;

    // 价格提交数据结构
    struct PriceSubmission {
        string agent;          // Agent名称
        uint256 price;         // 价格（建议使用最小单位，如 price * 10^18）
        string currency;       // 货币类型（如 "USD"）
        uint256 timestamp;     // 提交时间戳
    }

    // Ollama 推理请求数据结构
    struct InferenceRequest {
        string prompt;                              // 用户提示词
        uint256 createdAt;                          // 请求发起时间
        uint256 respondedCount;                     // 已提交结果数量
        bool completed;                             // 是否已收齐4份结果
        string[OLLAMA_NODE_COUNT] results;          // 各节点结果
        uint256[OLLAMA_NODE_COUNT] responseTimes;   // 各节点提交时间
    }

    // 状态变量
    address public owner;
    string public callbackUrl = "hello world";  // 回调URL（测试阶段）

    // 存储每个agent的最新提交（方案A：只保留最后一次提交）
    mapping(string => PriceSubmission) public submissions;
    // Ollama 节点名称与上报地址（索引0~3）
    string[OLLAMA_NODE_COUNT] private ollamaNodeNames;
    address[OLLAMA_NODE_COUNT] public ollamaNodeSubmitters;
    // 请求ID => 推理请求
    mapping(uint256 => InferenceRequest) private inferenceRequests;
    // 请求ID => 节点索引 => 是否已提交
    mapping(uint256 => mapping(uint256 => bool)) private inferenceSubmitted;
    // 自增请求ID
    uint256 public latestInferenceRequestId;

    // 事件：价格提交时触发（用于记录）
    event PriceSubmitted(
        string indexed agent,
        uint256 price,
        string currency,
        uint256 timestamp
    );

    // 事件：触发URL回调时发出（链下服务监听此事件）
    event CallbackTriggered(
        string indexed agent,
        uint256 timestamp
    );

    // 事件：Ollama 节点配置更新
    event OllamaNodeConfigured(
        uint256 indexed nodeIndex,
        string nodeName,
        address submitter,
        uint256 timestamp
    );

    // 事件：推理请求发起
    event OllamaPromptStored(
        uint256 indexed requestId,
        address indexed requester,
        string prompt,
        uint256 timestamp
    );

    // 事件：推理请求发起
    event OllamaInferenceRequested(
        uint256 indexed requestId,
        address indexed requester,
        string prompt,
        uint256 timestamp
    );

    // 事件：节点任务分发（链下节点监听）
    event OllamaNodeTaskDispatched(
        uint256 indexed requestId,
        uint256 indexed nodeIndex,
        string nodeName,
        address submitter,
        uint256 timestamp
    );

    // 事件：节点结果上报
    event OllamaResultSubmitted(
        uint256 indexed requestId,
        uint256 indexed nodeIndex,
        string nodeName,
        address indexed submitter,
        string result,
        uint256 timestamp
    );

    // 事件：4份结果全部到齐
    event OllamaInferenceCompleted(
        uint256 indexed requestId,
        uint256 timestamp
    );

    modifier onlyOwner() {
        require(msg.sender == owner, "only owner");
        _;
    }

    constructor() {
        owner = msg.sender;
        // 默认四节点名称，可通过 setOllamaNode 覆盖
        ollamaNodeNames[0] = "node-1";
        ollamaNodeNames[1] = "node-2";
        ollamaNodeNames[2] = "node-3";
        ollamaNodeNames[3] = "node-4";
    }

    /**
     * @dev 配置单个 Ollama 节点信息
     * @param nodeIndex 节点索引（0~3）
     * @param nodeName 节点名称
     * @param submitter 该节点允许上报结果的钱包地址
     */
    function setOllamaNode(
        uint256 nodeIndex,
        string memory nodeName,
        address submitter
    ) external onlyOwner {
        require(nodeIndex < OLLAMA_NODE_COUNT, "node index out of range");
        require(bytes(nodeName).length > 0, "empty node name");
        require(submitter != address(0), "invalid submitter");

        ollamaNodeNames[nodeIndex] = nodeName;
        ollamaNodeSubmitters[nodeIndex] = submitter;

        emit OllamaNodeConfigured(nodeIndex, nodeName, submitter, block.timestamp);
    }

    /**
     * @dev 读取全部节点配置（返回固定4个节点）
     */
    function getOllamaNodes() external view returns (
        string memory node0,
        address submitter0,
        string memory node1,
        address submitter1,
        string memory node2,
        address submitter2,
        string memory node3,
        address submitter3
    ) {
        return (
            ollamaNodeNames[0],
            ollamaNodeSubmitters[0],
            ollamaNodeNames[1],
            ollamaNodeSubmitters[1],
            ollamaNodeNames[2],
            ollamaNodeSubmitters[2],
            ollamaNodeNames[3],
            ollamaNodeSubmitters[3]
        );
    }

    /**
     * @dev 发起 Ollama 推理请求（输入提示词）
     * 链下四节点监听事件后，分别调用 submitOllamaResult 上报结果
     * @param prompt 提示词
     * @return requestId 请求编号
     */
    function requestOllamaInference(string memory prompt) external returns (uint256 requestId) {
        require(bytes(prompt).length > 0, "empty prompt");

        requestId = ++latestInferenceRequestId;

        InferenceRequest storage request = inferenceRequests[requestId];
        request.prompt = prompt;
        request.createdAt = block.timestamp;
        request.respondedCount = 0;
        request.completed = false;

        emit OllamaPromptStored(requestId, msg.sender, prompt, block.timestamp);
        emit OllamaInferenceRequested(requestId, msg.sender, prompt, block.timestamp);

        // 显式分发四个节点任务，方便链下按节点监听和执行
        for (uint256 i = 0; i < OLLAMA_NODE_COUNT; i++) {
            emit OllamaNodeTaskDispatched(
                requestId,
                i,
                ollamaNodeNames[i],
                ollamaNodeSubmitters[i],
                block.timestamp
            );
        }
    }

    /**
     * @dev 节点上报推理结果
     * @param requestId 请求编号
     * @param nodeIndex 节点索引（0~3）
     * @param result Ollama 结果文本
     */
    function submitOllamaResult(
        uint256 requestId,
        uint256 nodeIndex,
        string memory result
    ) external {
        require(nodeIndex < OLLAMA_NODE_COUNT, "node index out of range");
        require(bytes(result).length > 0, "empty result");

        InferenceRequest storage request = inferenceRequests[requestId];
        require(request.createdAt != 0, "request not found");
        require(!inferenceSubmitted[requestId][nodeIndex], "result already submitted");

        address submitter = ollamaNodeSubmitters[nodeIndex];
        // 未配置submitter时允许任意地址提交，便于快速联调
        if (submitter != address(0)) {
            require(msg.sender == submitter, "unauthorized submitter");
        }

        request.results[nodeIndex] = result;
        request.responseTimes[nodeIndex] = block.timestamp;
        request.respondedCount += 1;
        inferenceSubmitted[requestId][nodeIndex] = true;

        emit OllamaResultSubmitted(
            requestId,
            nodeIndex,
            ollamaNodeNames[nodeIndex],
            msg.sender,
            result,
            block.timestamp
        );

        if (request.respondedCount == OLLAMA_NODE_COUNT) {
            request.completed = true;
            emit OllamaInferenceCompleted(requestId, block.timestamp);
        }
    }

    /**
     * @dev 查询某次推理请求保存的 prompt
     */
    function getOllamaPrompt(uint256 requestId) external view returns (string memory prompt) {
        InferenceRequest storage request = inferenceRequests[requestId];
        require(request.createdAt != 0, "request not found");
        return request.prompt;
    }

    /**
     * @dev 查询最新一次推理请求的 requestId 和 prompt
     */
    function getLatestOllamaPrompt() external view returns (uint256 requestId, string memory prompt) {
        requestId = latestInferenceRequestId;
        require(requestId != 0, "request not found");
        return (requestId, inferenceRequests[requestId].prompt);
    }

    /**
     * @dev 查询推理请求元信息
     */
    function getOllamaRequestMeta(uint256 requestId) external view returns (
        string memory prompt,
        uint256 createdAt,
        uint256 respondedCount,
        bool completed
    ) {
        InferenceRequest storage request = inferenceRequests[requestId];
        require(request.createdAt != 0, "request not found");
        return (
            request.prompt,
            request.createdAt,
            request.respondedCount,
            request.completed
        );
    }

    /**
     * @dev 查询单个节点结果
     */
    function getOllamaResult(uint256 requestId, uint256 nodeIndex) external view returns (
        string memory result,
        uint256 responseTime,
        bool submitted
    ) {
        require(nodeIndex < OLLAMA_NODE_COUNT, "node index out of range");
        InferenceRequest storage request = inferenceRequests[requestId];
        require(request.createdAt != 0, "request not found");
        return (
            request.results[nodeIndex],
            request.responseTimes[nodeIndex],
            inferenceSubmitted[requestId][nodeIndex]
        );
    }

    /**
     * @dev 一次性返回4个节点结果（满足“链上返回4份结果”）
     */
    function getAllOllamaResults(uint256 requestId) external view returns (
        string memory result0,
        string memory result1,
        string memory result2,
        string memory result3,
        bool completed
    ) {
        InferenceRequest storage request = inferenceRequests[requestId];
        require(request.createdAt != 0, "request not found");
        return (
            request.results[0],
            request.results[1],
            request.results[2],
            request.results[3],
            request.completed
        );
    }

    /**
     * @dev 提交价格数据
     * @param agentName Agent名称
     * @param price 价格值
     * @param currency 货币类型
     */
    function submitPrice(
        string memory agentName,
        uint256 price,
        string memory currency
    ) external {
        // 创建价格提交记录
        PriceSubmission memory submission = PriceSubmission({
            agent: agentName,
            price: price,
            currency: currency,
            timestamp: block.timestamp
        });

        // 存储到mapping（会覆盖该agent之前的提交）
        submissions[agentName] = submission;

        // 发出价格提交事件（用于记录）
        emit PriceSubmitted(
            agentName,
            price,
            currency,
            block.timestamp
        );
    }

    /**
     * @dev 请求URL回调（独立功能，链下服务调用此函数触发回调事件）
     * @param agentName Agent名称
     */
    function requestCallback(string memory agentName) external {
        // 发出回调触发事件（链下服务监听此事件，然后调用getCallbackUrl()获取URL）
        emit CallbackTriggered(
            agentName,
            block.timestamp
        );
    }

    /**
     * @dev 获取回调URL
     * @return 回调URL字符串
     */
    function getCallbackUrl() public view returns (string memory) {
        return callbackUrl;
    }

    /**
     * @dev 设置回调URL（测试阶段，无访问控制）
     * @param url 新的回调URL
     */
    function setCallbackUrl(string memory url) external {
        callbackUrl = url;
    }

    /**
     * @dev 查询指定agent的价格提交记录
     * @param agentName Agent名称
     * @return 价格提交记录（如果不存在，所有字段为默认值）
     */
    function getSubmission(string memory agentName) 
        public 
        view 
        returns (PriceSubmission memory) 
    {
        return submissions[agentName];
    }
}
