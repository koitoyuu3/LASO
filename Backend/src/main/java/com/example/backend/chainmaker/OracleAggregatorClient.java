package com.example.backend.chainmaker;

import static com.example.backend.chainmaker.InitClient.chainClient;

import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.chainmaker.pb.common.ResultOuterClass;
import org.chainmaker.sdk.SdkException;
import org.web3j.abi.FunctionEncoder;
import org.web3j.abi.FunctionReturnDecoder;
import org.web3j.abi.TypeReference;
import org.web3j.abi.datatypes.Address;
import org.web3j.abi.datatypes.Bool;
import org.web3j.abi.datatypes.Function;
import org.web3j.abi.datatypes.Type;
import org.web3j.abi.datatypes.Utf8String;
import org.web3j.abi.datatypes.generated.Uint256;
import org.web3j.utils.Numeric;

/**
 * Client for invoking the OracleAggregator contract.
 */
public class OracleAggregatorClient {

    private static final String DEFAULT_CONTRACT_NAME = "OracleAggregator";
    
    // EVM parameter key
    private static final String CONTRACT_ARGS_EVM_PARAM = "data";

    // Timeouts; adjust to the actual deployment
    private static final int rpcCallTimeout = 10000;
    private static final int syncResultTimeout = 10000;
    private final String contractName;

    public OracleAggregatorClient() {
        this(DEFAULT_CONTRACT_NAME);
    }

    public OracleAggregatorClient(String contractName) {
        if (contractName == null || contractName.trim().isEmpty()) {
            this.contractName = DEFAULT_CONTRACT_NAME;
            return;
        }
        this.contractName = contractName.trim();
    }

    public String getContractName() {
        return contractName;
    }

    private Map<String, byte[]> buildEvmParams(Function function) {
        Map<String, byte[]> params = new HashMap<>();
        String methodDataStr = FunctionEncoder.encode(function);
        params.put(CONTRACT_ARGS_EVM_PARAM, methodDataStr.getBytes(StandardCharsets.UTF_8));
        return params;
    }

    private ResultOuterClass.TxResponse invoke(Function function) throws SdkException {
        String methodDataStr = FunctionEncoder.encode(function);
        String method = methodDataStr.substring(0, 10);

        return chainClient.invokeContract(
                contractName,
                method,
                null,
                buildEvmParams(function),
                rpcCallTimeout,
                syncResultTimeout
        );
    }

    private ResultOuterClass.TxResponse query(Function function) throws SdkException {
        String methodDataStr = FunctionEncoder.encode(function);
        String method = methodDataStr.substring(0, 10);

        return chainClient.queryContract(
                contractName,
                method,
                null,
                buildEvmParams(function),
                rpcCallTimeout
        );
    }

    /**
     * Submit price data (write operation via invokeContract).
     *
     * @param agentName agent name
     * @param price price value (smallest unit recommended, e.g. price * 10^18)
     * @param currency currency type (e.g. "USD")
     * @return transaction response
     * @throws SdkException SDK exception
     */
    public ResultOuterClass.TxResponse submitPrice(String agentName, BigInteger price, String currency) 
            throws SdkException {
        // Build function: submitPrice(string agentName, uint256 price, string currency)
        Function function = new Function(
                "submitPrice",
                Arrays.asList(
                        new Utf8String(agentName),
                        new Uint256(price),
                        new Utf8String(currency)
                ),
                Collections.emptyList()
        );
        return invoke(function);
    }

    /**
     * Get the callback URL (read operation via queryContract).
     *
     * @return callback URL string
     * @throws SdkException SDK exception
     */
    public ResultOuterClass.TxResponse getCallbackUrl() throws SdkException {
        // Build function: getCallbackUrl() returns (string)
        Function function = new Function(
                "getCallbackUrl",
                Collections.emptyList(),
                Arrays.asList(new TypeReference<Utf8String>() {})
        );
        return query(function);
    }

    /**
     * Request a URL callback (emits an event via invokeContract).
     *
     * @param agentName agent name
     * @return transaction response
     * @throws SdkException SDK exception
     */
    public ResultOuterClass.TxResponse requestCallback(String agentName) throws SdkException {
        // Build function: requestCallback(string agentName)
        Function function = new Function(
                "requestCallback",
                Arrays.asList(new Utf8String(agentName)),
                Collections.emptyList()
        );
        return invoke(function);
    }

    /**
     * Configure a single Ollama node (only the contract owner can call this successfully).
     *
     * @param nodeIndex node index (0~3)
     * @param nodeName node name
     * @param submitter wallet address allowed to submit results for this node
     */
    public ResultOuterClass.TxResponse setOllamaNode(
            BigInteger nodeIndex,
            String nodeName,
            String submitter
    ) throws SdkException {
        Function function = new Function(
                "setOllamaNode",
                Arrays.asList(
                        new Uint256(nodeIndex),
                        new Utf8String(nodeName),
                        new Address(submitter)
                ),
                Collections.emptyList()
        );
        return invoke(function);
    }

    /**
     * Issue a prompt inference request; the four nodes listen for the event and each call their own Ollama.
     *
     * @param prompt the prompt
     */
    public ResultOuterClass.TxResponse requestOllamaInference(String prompt) throws SdkException {
        Function function = new Function(
                "requestOllamaInference",
                Arrays.asList(new Utf8String(prompt)),
                Arrays.asList(new TypeReference<Uint256>() {})
        );
        return invoke(function);
    }

    /**
     * Submit a node's inference result.
     *
     * @param requestId request ID
     * @param nodeIndex node index (0~3)
     * @param result Ollama output text
     */
    public ResultOuterClass.TxResponse submitOllamaResult(
            BigInteger requestId,
            BigInteger nodeIndex,
            String result
    ) throws SdkException {
        Function function = new Function(
                "submitOllamaResult",
                Arrays.asList(
                        new Uint256(requestId),
                        new Uint256(nodeIndex),
                        new Utf8String(result)
                ),
                Collections.emptyList()
        );
        return invoke(function);
    }

    /**
     * Get the 4 results of a single request.
     *
     * @param requestId request ID
     * @return response containing result0/result1/result2/result3/completed
     */
    public ResultOuterClass.TxResponse getAllOllamaResults(BigInteger requestId) throws SdkException {
        Function function = new Function(
                "getAllOllamaResults",
                Arrays.asList(new Uint256(requestId)),
                Arrays.asList(
                        new TypeReference<Utf8String>() {},
                        new TypeReference<Utf8String>() {},
                        new TypeReference<Utf8String>() {},
                        new TypeReference<Utf8String>() {},
                        new TypeReference<Bool>() {}
                )
        );
        return query(function);
    }

    public ResultOuterClass.TxResponse getOllamaPrompt(BigInteger requestId) throws SdkException {
        Function function = new Function(
                "getOllamaPrompt",
                Arrays.asList(new Uint256(requestId)),
                Arrays.asList(new TypeReference<Utf8String>() {})
        );
        return query(function);
    }

    public ResultOuterClass.TxResponse getLatestOllamaPrompt() throws SdkException {
        Function function = new Function(
                "getLatestOllamaPrompt",
                Collections.emptyList(),
                Arrays.asList(
                        new TypeReference<Uint256>() {},
                        new TypeReference<Utf8String>() {}
                )
        );
        return query(function);
    }

    /**
     * Get request metadata (prompt, creation time, submitted count, completion flag).
     *
     * @param requestId request ID
     */
    public ResultOuterClass.TxResponse getOllamaRequestMeta(BigInteger requestId) throws SdkException {
        Function function = new Function(
                "getOllamaRequestMeta",
                Arrays.asList(new Uint256(requestId)),
                Arrays.asList(
                        new TypeReference<Utf8String>() {},
                        new TypeReference<Uint256>() {},
                        new TypeReference<Uint256>() {},
                        new TypeReference<Bool>() {}
                )
        );
        return query(function);
    }

    public ResultOuterClass.TxResponse getLatestInferenceRequestId() throws SdkException {
        Function function = new Function(
                "latestInferenceRequestId",
                Collections.emptyList(),
                Arrays.asList(new TypeReference<Uint256>() {})
        );
        return query(function);
    }

    public ResultOuterClass.TxResponse getOllamaResult(BigInteger requestId, BigInteger nodeIndex)
            throws SdkException {
        Function function = new Function(
                "getOllamaResult",
                Arrays.asList(new Uint256(requestId), new Uint256(nodeIndex)),
                Arrays.asList(
                        new TypeReference<Utf8String>() {},
                        new TypeReference<Uint256>() {},
                        new TypeReference<Bool>() {}
                )
        );
        return query(function);
    }

    public BigInteger requestOllamaInferenceAndGetRequestId(String prompt) throws SdkException {
        ResultOuterClass.TxResponse response = requestOllamaInference(prompt);
        if (response == null) {
            throw new IllegalStateException("requestOllamaInference failed: null tx response");
        }
        if (response.getCode() != ResultOuterClass.TxStatusCode.SUCCESS) {
            throw new IllegalStateException(
                    "requestOllamaInference failed: code=" + response.getCode() + ", msg=" + response.getMessage()
            );
        }
        if (response.hasContractResult() && response.getContractResult().getCode() != 0) {
            throw new IllegalStateException(
                    "requestOllamaInference contract failed: code=" + response.getContractResult().getCode()
                            + ", msg=" + response.getContractResult().getMessage()
            );
        }
        return decodeUint256Value(response, "requestOllamaInference");
    }

    public BigInteger getLatestInferenceRequestIdValue() throws SdkException {
        return decodeUint256Value(getLatestInferenceRequestId(), "latestInferenceRequestId");
    }

    public String getOllamaPromptValue(BigInteger requestId) throws SdkException {
        List<Type> values = decodeEvmResult(
                getOllamaPrompt(requestId),
                Arrays.asList(new TypeReference<Utf8String>() {})
        );
        if (values.isEmpty()) {
            throw new IllegalStateException("failed to decode getOllamaPrompt return values");
        }
        return (String) values.get(0).getValue();
    }

    public LatestOllamaPrompt getLatestOllamaPromptValue() throws SdkException {
        List<Type> values = decodeEvmResult(
                getLatestOllamaPrompt(),
                Arrays.asList(
                        new TypeReference<Uint256>() {},
                        new TypeReference<Utf8String>() {}
                )
        );
        if (values.size() < 2) {
            throw new IllegalStateException("failed to decode getLatestOllamaPrompt return values");
        }
        return new LatestOllamaPrompt(
                (BigInteger) values.get(0).getValue(),
                (String) values.get(1).getValue()
        );
    }

    public OllamaRequestMeta getOllamaRequestMetaValue(BigInteger requestId) throws SdkException {
        ResultOuterClass.TxResponse response = getOllamaRequestMeta(requestId);
        List<Type> values = decodeEvmResult(
                response,
                Arrays.asList(
                        new TypeReference<Utf8String>() {},
                        new TypeReference<Uint256>() {},
                        new TypeReference<Uint256>() {},
                        new TypeReference<Bool>() {}
                )
        );
        if (values.size() < 4) {
            throw new IllegalStateException("failed to decode getOllamaRequestMeta return values");
        }

        String prompt = (String) values.get(0).getValue();
        BigInteger createdAt = (BigInteger) values.get(1).getValue();
        BigInteger respondedCount = (BigInteger) values.get(2).getValue();
        Boolean completed = (Boolean) values.get(3).getValue();
        return new OllamaRequestMeta(prompt, createdAt, respondedCount, completed);
    }

    public OllamaAllResults getAllOllamaResultsValue(BigInteger requestId) throws SdkException {
        ResultOuterClass.TxResponse response = getAllOllamaResults(requestId);
        List<Type> values = decodeEvmResult(
                response,
                Arrays.asList(
                        new TypeReference<Utf8String>() {},
                        new TypeReference<Utf8String>() {},
                        new TypeReference<Utf8String>() {},
                        new TypeReference<Utf8String>() {},
                        new TypeReference<Bool>() {}
                )
        );
        if (values.size() < 5) {
            throw new IllegalStateException("failed to decode getAllOllamaResults return values");
        }

        return new OllamaAllResults(
                (String) values.get(0).getValue(),
                (String) values.get(1).getValue(),
                (String) values.get(2).getValue(),
                (String) values.get(3).getValue(),
                (Boolean) values.get(4).getValue()
        );
    }

    public static List<Type> decodeEvmResult(
            ResultOuterClass.TxResponse response,
            List<TypeReference<?>> outputParameters
    ) {
        if (response == null || !response.hasContractResult()) {
            return Collections.emptyList();
        }
        byte[] rawBytes = response.getContractResult().getResult().toByteArray();
        if (rawBytes == null || rawBytes.length == 0) {
            return Collections.emptyList();
        }

        String rawString = new String(rawBytes, StandardCharsets.UTF_8).trim();
        String hexData;
        if (rawString.startsWith("0x") || rawString.startsWith("0X")) {
            hexData = rawString;
        } else if (isLikelyHexString(rawString)) {
            hexData = "0x" + rawString;
        } else {
            hexData = Numeric.toHexString(rawBytes);
        }
        @SuppressWarnings({"rawtypes", "unchecked"})
        List<TypeReference<Type>> decoderOutputParameters = (List) outputParameters;
        return FunctionReturnDecoder.decode(hexData, decoderOutputParameters);
    }

    private BigInteger decodeUint256Value(ResultOuterClass.TxResponse response, String methodName) {
        List<Type> values = decodeEvmResult(
                response,
                Arrays.asList(new TypeReference<Uint256>() {})
        );
        if (values.isEmpty()) {
            throw new IllegalStateException("failed to decode " + methodName + " return values");
        }
        return (BigInteger) values.get(0).getValue();
    }

    private static boolean isLikelyHexString(String value) {
        if (value == null || value.isEmpty() || (value.length() % 2 != 0)) {
            return false;
        }
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            boolean isHex = (c >= '0' && c <= '9')
                    || (c >= 'a' && c <= 'f')
                    || (c >= 'A' && c <= 'F');
            if (!isHex) {
                return false;
            }
        }
        return true;
    }

    public static class OllamaRequestMeta {
        private final String prompt;
        private final BigInteger createdAt;
        private final BigInteger respondedCount;
        private final Boolean completed;

        public OllamaRequestMeta(String prompt, BigInteger createdAt, BigInteger respondedCount, Boolean completed) {
            this.prompt = prompt;
            this.createdAt = createdAt;
            this.respondedCount = respondedCount;
            this.completed = completed;
        }

        public String getPrompt() {
            return prompt;
        }

        public BigInteger getCreatedAt() {
            return createdAt;
        }

        public BigInteger getRespondedCount() {
            return respondedCount;
        }

        public Boolean getCompleted() {
            return completed;
        }
    }

    public static class LatestOllamaPrompt {
        private final BigInteger requestId;
        private final String prompt;

        public LatestOllamaPrompt(BigInteger requestId, String prompt) {
            this.requestId = requestId;
            this.prompt = prompt;
        }

        public BigInteger getRequestId() {
            return requestId;
        }

        public String getPrompt() {
            return prompt;
        }
    }

    public static class OllamaAllResults {
        private final String result0;
        private final String result1;
        private final String result2;
        private final String result3;
        private final Boolean completed;

        public OllamaAllResults(
                String result0,
                String result1,
                String result2,
                String result3,
                Boolean completed
        ) {
            this.result0 = result0;
            this.result1 = result1;
            this.result2 = result2;
            this.result3 = result3;
            this.completed = completed;
        }

        public String getResult0() {
            return result0;
        }

        public String getResult1() {
            return result1;
        }

        public String getResult2() {
            return result2;
        }

        public String getResult3() {
            return result3;
        }

        public Boolean getCompleted() {
            return completed;
        }
    }
}
