package com.example.backend.service;

import com.example.backend.chainmaker.InitClient;
import com.example.backend.chainmaker.OracleAggregatorClient;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.core.env.Environment;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Stores the prompt via the OracleAggregator contract first, then reads it back and dispatches it to the Python agent.
 * The Python agent reads /home/***/data through controlled tools and drives up to 50 Ollama models.
 * Model results are not written back on-chain; Java only dispatches tasks and does not aggregate multi-node results.
 */
@Service
@Slf4j
public class ChainmakerOllamaService {

    private static final int MAX_NODE_COUNT = 50;
    private static final int DEFAULT_NODE_BASE_PORT = 11431;
    private static final String DEFAULT_NODE_MODEL = "qwen2.5:7b";

    private final RestTemplate restTemplate = new RestTemplate();
    private final ObjectMapper objectMapper = new ObjectMapper();

    @Autowired(required = false)
    private Environment environment;

    @Value("${chainmaker.oracle-aggregator.contract-name:OracleAggregator}")
    private String contractName;

    @Value("${flask.service.url:http://127.0.0.1:5000}")
    private String flaskServiceUrl;

    @Value("${chainmaker.agent.data-root:/home/***/data}")
    private String agentDataRoot;

    @Value("${chainmaker.agent.max-steps:6}")
    private int agentMaxSteps;

    @Value("${chainmaker.agent.read-limit:25}")
    private int agentReadLimit;

    @Value("${chainmaker.ollama.transport:ssh}")
    private String transport;

    @Value("${chainmaker.ollama.ssh.host:127.0.0.1}")
    private String sshHost;

    @Value("${chainmaker.ollama.ssh.user:***}")
    private String sshUser;

    @Value("${chainmaker.ollama.ssh.connect-timeout-seconds:10}")
    private int sshConnectTimeoutSeconds;

    @Value("${chainmaker.ollama.timeout-seconds:120}")
    private int timeoutSeconds;

    @Value("${chainmaker.ollama.max-result-length:1800}")
    private int maxResultLength;

    @Value("${chainmaker.ollama.default-node-count:50}")
    private int defaultNodeCount;

    @Value("${chainmaker.ollama.default-model:qwen2.5:7b}")
    private String defaultNodeModel;

    @Value("${chainmaker.ollama.default-node-base-port:11431}")
    private int defaultNodeBasePort;

    @Value("${chainmaker.ollama.max-concurrent-nodes:50}")
    private int maxConcurrentNodes;

    public Map<String, Object> storePrompt(String prompt) throws Exception {
        validatePrompt(prompt);
        ensureChainClientInitialized();

        OracleAggregatorClient oracleAggregatorClient = createOracleAggregatorClient();
        BigInteger requestId = oracleAggregatorClient.requestOllamaInferenceAndGetRequestId(prompt);
        OracleAggregatorClient.OllamaRequestMeta requestMeta =
                oracleAggregatorClient.getOllamaRequestMetaValue(requestId);

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("requestId", requestId.toString());
        result.put("prompt", requestMeta.getPrompt());
        result.put("chainPrompt", requestMeta.getPrompt());
        result.put("createdAt", requestMeta.getCreatedAt());
        result.put("contractName", oracleAggregatorClient.getContractName());
        result.put("promptStoredOnChain", true);
        result.put("resultsStoredOnChain", false);
        result.put("message", "Prompt has been written on-chain; use the requestId to call inferByRequestId and dispatch the Python agent job");
        return result;
    }

    public Map<String, Object> inferByRequestId(BigInteger requestId) throws Exception {
        return inferByRequestId(requestId, null);
    }

    public Map<String, Object> inferByRequestId(BigInteger requestId, Integer count) throws Exception {
        if (requestId == null || requestId.compareTo(BigInteger.ZERO) <= 0) {
            throw new IllegalArgumentException("invalid requestId");
        }
        validateCount(count);
        ensureChainClientInitialized();

        OracleAggregatorClient oracleAggregatorClient = createOracleAggregatorClient();
        OracleAggregatorClient.OllamaRequestMeta requestMeta =
                oracleAggregatorClient.getOllamaRequestMetaValue(requestId);
        String chainPrompt = oracleAggregatorClient.getOllamaPromptValue(requestId);

        return dispatchAgentJob(oracleAggregatorClient, requestId, requestMeta, chainPrompt, count);
    }

    public Map<String, Object> infer(String prompt) throws Exception {
        return infer(prompt, null);
    }

    public Map<String, Object> infer(String prompt, Integer count) throws Exception {
        Map<String, Object> storedPrompt = storePrompt(prompt);
        BigInteger requestId = new BigInteger(storedPrompt.get("requestId").toString());
        return inferByRequestId(requestId, count);
    }

    public Map<String, Object> queryResult(BigInteger requestId) throws Exception {
        if (requestId == null || requestId.compareTo(BigInteger.ZERO) <= 0) {
            throw new IllegalArgumentException("invalid requestId");
        }
        ensureChainClientInitialized();

        OracleAggregatorClient oracleAggregatorClient = createOracleAggregatorClient();
        OracleAggregatorClient.OllamaRequestMeta requestMeta =
                oracleAggregatorClient.getOllamaRequestMetaValue(requestId);

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("requestId", requestId.toString());
        result.put("prompt", requestMeta.getPrompt());
        result.put("chainPrompt", requestMeta.getPrompt());
        result.put("createdAt", requestMeta.getCreatedAt());
        result.put("transport", transport);
        result.put("contractName", oracleAggregatorClient.getContractName());
        result.put("promptStoredOnChain", true);
        result.put("resultsStoredOnChain", false);
        result.put("message", "Prompt is on-chain, but model results are not. Call inferByRequestId to dispatch the Python agent job");
        result.put("chainResults", Collections.emptyList());
        return result;
    }

    protected OracleAggregatorClient createOracleAggregatorClient() {
        return new OracleAggregatorClient(contractName);
    }

    private Map<String, Object> dispatchAgentJob(
            OracleAggregatorClient oracleAggregatorClient,
            BigInteger requestId,
            OracleAggregatorClient.OllamaRequestMeta requestMeta,
            String chainPrompt,
            Integer count
    ) {
        List<Map<String, Object>> nodePayloads = buildNodePayloads(count);
        Map<String, Object> requestBody = new LinkedHashMap<>();
        requestBody.put("requestId", requestId.toString());
        requestBody.put("prompt", chainPrompt);
        requestBody.put("dataRoot", agentDataRoot);
        requestBody.put("sshHost", sshHost);
        requestBody.put("sshUser", sshUser);
        requestBody.put("connectTimeoutSeconds", sshConnectTimeoutSeconds);
        requestBody.put("timeoutSeconds", timeoutSeconds);
        requestBody.put("maxSteps", agentMaxSteps);
        requestBody.put("readLimit", agentReadLimit);
        requestBody.put("maxConcurrentNodes", maxConcurrentNodes);
        requestBody.put("nodeCount", nodePayloads.size());
        requestBody.put("nodes", nodePayloads);

        String url = normalizeBaseUrl(flaskServiceUrl) + "/api/agent/ollama/jobs";
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        HttpEntity<Map<String, Object>> entity = new HttpEntity<>(requestBody, headers);
        ResponseEntity<Map> response = restTemplate.exchange(url, HttpMethod.POST, entity, Map.class);

        if (!response.getStatusCode().is2xxSuccessful() || response.getBody() == null) {
            throw new IllegalStateException("python agent dispatch failed, status=" + response.getStatusCodeValue());
        }

        Map<String, Object> agentResponse = response.getBody();
        if (!Boolean.TRUE.equals(agentResponse.get("success"))) {
            throw new IllegalStateException("python agent dispatch failed: " + stringify(agentResponse.get("error")));
        }

        Map<String, Object> result = new LinkedHashMap<>();
        result.put("requestId", requestId.toString());
        result.put("prompt", requestMeta.getPrompt());
        result.put("chainPrompt", chainPrompt);
        result.put("createdAt", requestMeta.getCreatedAt());
        result.put("transport", transport);
        result.put("contractName", oracleAggregatorClient.getContractName());
        result.put("promptStoredOnChain", true);
        result.put("resultsStoredOnChain", false);
        result.put("requestedNodeCount", count == null ? nodePayloads.size() : count);
        result.put("dispatchedNodeCount", nodePayloads.size());
        result.put("pythonAgentDispatched", true);
        result.put("pythonAgentJobId", stringify(agentResponse.get("jobId")));
        result.put("pythonAgentStatus", stringify(agentResponse.get("status")));
        result.put("pythonAgentResultUrl", stringify(agentResponse.get("resultUrl")));
        result.put("pythonAgentBaseUrl", normalizeBaseUrl(flaskServiceUrl));
        result.put("pythonAgentStatusQueryUrl", normalizeBaseUrl(flaskServiceUrl) + stringify(agentResponse.get("resultUrl")));
        result.put("pythonAgentResponse", agentResponse);
        result.put("message", "Prompt was read from the chain and the Python agent was dispatched successfully. Java does not aggregate multi-node model results");
        return result;
    }

    private void validatePrompt(String prompt) {
        if (prompt == null || prompt.trim().isEmpty()) {
            throw new IllegalArgumentException("prompt must not be empty");
        }
    }

    private void validateCount(Integer count) {
        if (count == null) {
            return;
        }
        if (count <= 0 || count > MAX_NODE_COUNT) {
            throw new IllegalArgumentException("count must be between 1 and " + MAX_NODE_COUNT);
        }
    }

    private void ensureChainClientInitialized() {
        if (InitClient.chainClient == null) {
            throw new IllegalStateException("ChainMaker client is not initialized");
        }
    }

    private String normalizeBaseUrl(String url) {
        if (url == null) {
            return "";
        }
        String trimmed = url.trim();
        if (trimmed.endsWith("/")) {
            return trimmed.substring(0, trimmed.length() - 1);
        }
        return trimmed;
    }

    private String stringify(Object value) {
        if (value == null) {
            return "";
        }
        if (value instanceof String) {
            return (String) value;
        }
        try {
            return objectMapper.writeValueAsString(value);
        } catch (Exception e) {
            return String.valueOf(value);
        }
    }

    private List<Map<String, Object>> buildNodePayloads(Integer count) {
        List<Map<String, Object>> nodes = new ArrayList<>();
        int configuredNodeCount = Math.min(resolvePositiveInt(defaultNodeCount, MAX_NODE_COUNT), MAX_NODE_COUNT);
        int resolvedCount = count == null ? configuredNodeCount : Math.min(count, configuredNodeCount);
        for (int index = 0; index < resolvedCount; index++) {
            int nodeNumber = index + 1;
            nodes.add(
                    new NodeOllamaConfig(
                            index,
                            getNodeProperty(nodeNumber, "name", "agent-" + nodeNumber),
                            getNodeProperty(nodeNumber, "url", "http://127.0.0.1:"
                                    + (resolvePositiveInt(defaultNodeBasePort, DEFAULT_NODE_BASE_PORT) + index)),
                            getNodeProperty(nodeNumber, "model", resolveDefaultNodeModel())
                    ).toMap()
            );
        }
        return nodes;
    }

    private String getNodeProperty(int nodeNumber, String key, String defaultValue) {
        if (environment == null) {
            return defaultValue;
        }
        String value = environment.getProperty("chainmaker.ollama.node" + nodeNumber + "." + key);
        if (value == null || value.trim().isEmpty()) {
            return defaultValue;
        }
        return value.trim();
    }

    private String resolveDefaultNodeModel() {
        if (defaultNodeModel == null || defaultNodeModel.trim().isEmpty()) {
            return DEFAULT_NODE_MODEL;
        }
        return defaultNodeModel.trim();
    }

    private int resolvePositiveInt(int value, int defaultValue) {
        return value > 0 ? value : defaultValue;
    }

    private static class NodeOllamaConfig {
        private final int nodeIndex;
        private final String nodeName;
        private final String url;
        private final String model;

        private NodeOllamaConfig(int nodeIndex, String nodeName, String url, String model) {
            this.nodeIndex = nodeIndex;
            this.nodeName = nodeName;
            this.url = url;
            this.model = model;
        }

        private Map<String, Object> toMap() {
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("nodeIndex", nodeIndex);
            result.put("nodeName", nodeName);
            result.put("url", url);
            result.put("model", model);
            return result;
        }
    }
}
