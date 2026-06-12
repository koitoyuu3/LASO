package com.example.backend.service;

import com.example.backend.chainmaker.InitClient;
import com.example.backend.chainmaker.OracleAggregatorClient;
import com.example.backend.chainmaker.TruthSchnorrProofRegistryClient;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;
import org.chainmaker.pb.common.ResultOuterClass;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import javax.annotation.Resource;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.zip.GZIPOutputStream;

@Service
public class OllamaFixedResultBenchmarkService {

    private static final int OLLAMA_NODE_COUNT = 4;

    @Resource
    private ChainmakerOllamaService chainmakerOllamaService;

    @Value("${chainmaker.truth-schnorr-proof-registry.contract-name:TruthSchnorrProofRegistry}")
    private String chainmakerTruthSchnorrProofRegistryContractName;

    private final ObjectMapper objectMapper = new ObjectMapper();

    public Map<String, Object> run(
            String chain,
            String prompt,
            String resultPath,
            String proofBundlePath,
            Integer payloadPreviewLength
    ) throws Exception {
        String normalizedChain = normalizeChain(chain);
        String normalizedPrompt = normalizePrompt(prompt);
        FixedResult fixedResult = loadFixedResult(resultPath, payloadPreviewLength);
        ProofBundle proofBundle = loadProofBundle(proofBundlePath);
        String runId = "fixed-result-" + normalizedChain + "-" + System.currentTimeMillis();

        if ("chainmaker".equals(normalizedChain)) {
            return runChainmaker(runId, normalizedPrompt, fixedResult, proofBundle);
        }
        throw new IllegalArgumentException("chain only supports chainmaker (LASO chainmaker-only build)");
    }

    private Map<String, Object> runChainmaker(
            String runId,
            String prompt,
            FixedResult fixedResult,
            ProofBundle proofBundle
    ) throws Exception {
        if (InitClient.chainClient == null) {
            throw new IllegalStateException("ChainMaker client is not initialized");
        }
        OracleAggregatorClient client = chainmakerOllamaService.createOracleAggregatorClient();
        Map<String, Object> proofBenchmark = buildInitialProofBenchmark(proofBundle);

        BenchmarkClock totalClock = BenchmarkClock.start();
        TimedValue<BigInteger> request = timeValue(() -> client.requestOllamaInferenceAndGetRequestId(prompt));
        BigInteger requestId = request.getValue();

        TimedValue<OracleAggregatorClient.OllamaRequestMeta> promptQuery =
                timeValue(() -> client.getOllamaRequestMetaValue(requestId));

        List<Map<String, Object>> callbackNodeTimings = new ArrayList<>();
        BenchmarkClock callbackClock = BenchmarkClock.start();
        for (int index = 0; index < OLLAMA_NODE_COUNT; index++) {
            final int nodeIndex = index;
            String payload = fixedResult.getAgentPayload(nodeIndex);
            TimedVoid submit = timeVoid(() -> {
                ResultOuterClass.TxResponse response =
                        client.submitOllamaResult(requestId, BigInteger.valueOf(nodeIndex), payload);
                ensureChainmakerSuccess(response, "submitOllamaResult");
            });
            callbackNodeTimings.add(buildNodeTiming(nodeIndex, payload, submit.getElapsedMs()));
        }
        long callbackWriteMs = callbackClock.elapsedMs();

        TimedValue<OracleAggregatorClient.OllamaAllResults> resultQuery =
                timeValue(() -> client.getAllOllamaResultsValue(requestId));

        TimedValue<OracleAggregatorClient.OllamaRequestMeta> finalMetaQuery =
                timeValue(() -> client.getOllamaRequestMetaValue(requestId));
        proofBenchmark = runChainmakerProofBenchmark(runId, proofBundle);
        long totalNoAiMs = totalClock.elapsedMs();

        Map<String, Object> result = buildBaseResult(runId, "chainmaker", prompt, fixedResult, totalNoAiMs);
        result.put("requestId", requestId.toString());
        result.put("contractName", client.getContractName());
        result.put("requestWriteMs", request.getElapsedMs());
        result.put("promptQueryMs", promptQuery.getElapsedMs());
        result.put("callbackWriteMs", callbackWriteMs);
        result.put("callbackNodeMs", callbackNodeTimings);
        result.put("resultQueryMs", resultQuery.getElapsedMs());
        result.put("finalMetaQueryMs", finalMetaQuery.getElapsedMs());
        result.put("respondedCount", finalMetaQuery.getValue().getRespondedCount());
        result.put("completed", finalMetaQuery.getValue().getCompleted());
        result.put("chainResultDigest", digestResults(resultQuery.getValue()));
        result.put("proof", proofBenchmark);
        result.put("note", buildNote(proofBundle, proofBenchmark));
        return result;
    }

    private Map<String, Object> buildBaseResult(
            String runId,
            String chain,
            String prompt,
            FixedResult fixedResult,
            long totalNoAiMs
    ) {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("runId", runId);
        result.put("chain", chain);
        result.put("scenario", "fixed-ai-result-no-inference");
        result.put("startedAt", Instant.now().toString());
        result.put("prompt", prompt);
        result.put("resultFile", fixedResult.getPath());
        result.put("resultBytes", fixedResult.getSize());
        result.put("resultSha256", fixedResult.getSha256());
        result.put("payloadMode", "agent-responses-json-gzip-base64");
        result.put("payloadBytes", fixedResult.getAgentPayload(0).getBytes(StandardCharsets.UTF_8).length);
        result.put("payloadAgents", fixedResult.getAgentPayloadMeta());
        result.put("payloadAgentCount", fixedResult.getAgentCount());
        result.put("payloadItemsPerAgent", fixedResult.getItemsPerAgent());
        result.put("totalNoAiMs", totalNoAiMs);
        return result;
    }

    private Map<String, Object> buildNodeTiming(int nodeIndex, String payload, long elapsedMs) {
        Map<String, Object> item = new LinkedHashMap<>();
        item.put("nodeIndex", nodeIndex);
        item.put("elapsedMs", elapsedMs);
        item.put("payloadBytes", payload.getBytes(StandardCharsets.UTF_8).length);
        item.put("payloadSha256", sha256String(payload));
        return item;
    }

    private FixedResult loadFixedResult(String resultPath, Integer payloadPreviewLength) throws Exception {
        if (resultPath == null || resultPath.trim().isEmpty()) {
            throw new IllegalArgumentException("resultPath must not be empty");
        }
        Path path = Paths.get(resultPath.trim()).toAbsolutePath().normalize();
        if (!Files.isRegularFile(path)) {
            throw new IllegalArgumentException("resultPath does not exist or is not a file: " + path);
        }

        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        long size;
        try (InputStream input = Files.newInputStream(path);
             DigestInputStream digestInput = new DigestInputStream(input, digest)) {
            byte[] buffer = new byte[8192];
            size = 0L;
            int read;
            while ((read = digestInput.read(buffer)) != -1) {
                size += read;
            }
        }
        String resultSha256 = toHex(digest.digest());

        JsonNode root = objectMapper.readTree(path.toFile());
        Map<String, List<JsonNode>> itemsByAgent = new LinkedHashMap<>();
        JsonNode batches = root.path("job").path("batches");
        if (!batches.isArray()) {
            throw new IllegalArgumentException("resultPath has unexpected format: missing job.batches array");
        }
        for (JsonNode batch : batches) {
            JsonNode items = batch.path("items");
            if (!items.isArray()) {
                continue;
            }
            for (JsonNode item : items) {
                String agent = item.path("agent").asText("");
                if (agent.isEmpty()) {
                    continue;
                }
                itemsByAgent.computeIfAbsent(agent, ignored -> new ArrayList<>()).add(item);
            }
        }
        if (itemsByAgent.size() < OLLAMA_NODE_COUNT) {
            throw new IllegalArgumentException("resultPath contains fewer than " + OLLAMA_NODE_COUNT + " agents");
        }

        List<String> agentPayloads = new ArrayList<>();
        List<Map<String, Object>> agentPayloadMeta = new ArrayList<>();
        int maxItemsPerAgent = 0;
        for (int index = 0; index < OLLAMA_NODE_COUNT; index++) {
            String agentName = "agent-" + (index + 1);
            List<JsonNode> items = itemsByAgent.get(agentName);
            if (items == null || items.isEmpty()) {
                throw new IllegalArgumentException("resultPath is missing output for " + agentName);
            }
            maxItemsPerAgent = Math.max(maxItemsPerAgent, items.size());
            ObjectNode payload = objectMapper.createObjectNode();
            payload.put("agent", agentName);
            payload.put("sourceResultFile", path.toString());
            payload.put("sourceResultSha256", resultSha256);
            payload.put("itemCount", items.size());
            payload.put("itemOrder", "news-1-to-news-" + items.size());
            ArrayNode array = payload.putArray("responses");
            for (JsonNode item : items) {
                array.add(item.path("response").asText(""));
            }
            String payloadJson = objectMapper.writeValueAsString(payload);
            String compressedPayload = compressBase64(payloadJson);
            agentPayloads.add(compressedPayload);

            Map<String, Object> meta = new LinkedHashMap<>();
            meta.put("agent", agentName);
            meta.put("itemCount", items.size());
            meta.put("content", "responses only, original news order");
            meta.put("rawJsonBytes", payloadJson.getBytes(StandardCharsets.UTF_8).length);
            meta.put("payloadBytes", compressedPayload.getBytes(StandardCharsets.UTF_8).length);
            meta.put("rawJsonSha256", sha256String(payloadJson));
            meta.put("payloadSha256", sha256String(compressedPayload));
            agentPayloadMeta.add(meta);
        }

        return new FixedResult(
                path.toString(),
                size,
                resultSha256,
                agentPayloads,
                agentPayloadMeta,
                itemsByAgent.size(),
                maxItemsPerAgent
        );
    }

    private ProofBundle loadProofBundle(String proofBundlePath) throws Exception {
        if (proofBundlePath == null || proofBundlePath.trim().isEmpty()) {
            return null;
        }
        Path path = Paths.get(proofBundlePath.trim()).toAbsolutePath().normalize();
        if (!Files.isRegularFile(path)) {
            throw new IllegalArgumentException("proofBundlePath does not exist or is not a file: " + path);
        }
        JsonNode root = objectMapper.readTree(path.toFile());
        JsonNode proofGroup = root.path("proof_groups").isArray() && root.path("proof_groups").size() > 0
                ? root.path("proof_groups").get(0)
                : objectMapper.createObjectNode();
        String scheme = proofGroup.path("proof_scheme").asText("");
        JsonNode submission = proofGroup.path("contract_submission");
        JsonNode proofPayload = proofGroup.path("proof_payload");
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        long size;
        try (InputStream input = Files.newInputStream(path);
             DigestInputStream digestInput = new DigestInputStream(input, digest)) {
            byte[] buffer = new byte[8192];
            size = 0L;
            int read;
            while ((read = digestInput.read(buffer)) != -1) {
                size += read;
            }
        }
        SchnorrProofCall schnorrProofCall = buildSchnorrProofCall(root, proofGroup, submission, proofPayload);
        return new ProofBundle(
                path.toString(),
                size,
                toHex(digest.digest()),
                root.path("method").asText(""),
                root.path("row_count").asInt(0),
                root.path("proof_groups").isArray() ? root.path("proof_groups").size() : 0,
                scheme,
                !proofPayload.isMissingNode() && !proofPayload.isNull(),
                submission.has("a") && submission.has("b") && submission.has("c") && submission.has("pubSignals"),
                schnorrProofCall
        );
    }

    private SchnorrProofCall buildSchnorrProofCall(
            JsonNode root,
            JsonNode proofGroup,
            JsonNode submission,
            JsonNode proofPayload
    ) {
        if (proofPayload == null || proofPayload.isMissingNode() || proofPayload.isNull()) {
            return null;
        }
        if (!"schnorr_nizk_v1".equals(proofPayload.path("scheme").asText(""))) {
            return null;
        }

        String publicKeyHex = requiredText(proofPayload, "public_key");
        String commitmentHex = requiredText(proofPayload, "commitment");
        String challengeHex = requiredText(proofPayload, "challenge");
        String responseHex = requiredText(proofPayload, "response");
        String statementHash = requiredText(proofPayload, "statement_hash");

        String experimentIdHex = optionalText(submission, "experimentId");
        if (experimentIdHex == null || experimentIdHex.isEmpty()) {
            experimentIdHex = "0x" + root.path("experiment_digest").asText("");
        }
        String groupIdHex = optionalText(submission, "groupId");
        if (groupIdHex == null || groupIdHex.isEmpty()) {
            groupIdHex = "0x" + proofGroup.path("group_digest").asText("");
        }
        String resultDigestHex = optionalText(submission, "resultDigest");
        if (resultDigestHex == null || resultDigestHex.isEmpty()) {
            resultDigestHex = "0x" + proofGroup.path("result_digest").asText(root.path("result_digest").asText(""));
        }
        String proofIdHex = optionalText(submission, "proofId");
        if (proofIdHex == null || proofIdHex.isEmpty()) {
            proofIdHex = "0x" + proofGroup.path("proof_id").asText(proofPayload.path("proof_id").asText(""));
        }
        String statementDigestHex = optionalText(submission, "statementDigestSha256");
        if (statementDigestHex == null || statementDigestHex.isEmpty()) {
            statementDigestHex = "0x" + proofGroup.path("statement_digest_sha256").asText(statementHash);
        }

        String challengeMaterial = buildSchnorrChallengeMaterial(
                proofPayload.path("group").asText("modp2048_subgroup"),
                proofPayload.path("prover_id").asText(""),
                publicKeyHex,
                commitmentHex,
                statementHash
        );

        return new SchnorrProofCall(
                challengeMaterial,
                bytes32FromHex(experimentIdHex, "experimentId"),
                bytes32FromHex(groupIdHex, "groupId"),
                bytes32FromHex(resultDigestHex, "resultDigest"),
                bytes32FromHex(proofIdHex, "proofId"),
                bytes32FromHex(statementDigestHex, "statementDigestSha256"),
                fixedBytesFromHex(publicKeyHex, 256, "public_key"),
                fixedBytesFromHex(commitmentHex, 256, "commitment"),
                fixedBytesFromHex(challengeHex, 32, "challenge"),
                fixedBytesFromHex(responseHex, 256, "response"),
                proofPayload.path("proof_id").asText(stripHexPrefix(proofIdHex)),
                statementHash,
                proofPayload.path("prover_id").asText(""),
                proofPayload.path("group").asText("modp2048_subgroup")
        );
    }

    private Map<String, Object> buildInitialProofBenchmark(ProofBundle proofBundle) {
        if (proofBundle == null) {
            return null;
        }
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("bundle", proofBundle.toMap());
        result.put("requiredOnChainVerifier", "TruthSchnorrProofRegistry.verifyProof/submitProof");
        result.put("onChainProved", false);
        result.put("verifyMs", null);
        result.put("submitMs", null);
        result.put("hasSubmissionMs", null);
        result.put("totalMs", null);
        if (!proofBundle.hasProofPayload()) {
            result.put("status", "missing-proof-payload");
            result.put(
                    "error",
                    "proofBundle only contains rows and a contract_submission digest; proof_payload is null, so the file holds no verifiable proof body."
            );
            return result;
        }
        if (proofBundle.isGroth16ContractCallable()) {
            result.put("status", "groth16-callable-but-java-proof-submit-not-implemented");
            result.put("error", "proofBundle already contains a/b/c/pubSignals, but this benchmark service has not yet integrated TruthSingleProofRegistryN4 call timing.");
            return result;
        }
        if (proofBundle.getSchnorrProofCall() != null) {
            result.put("status", "schnorr-ready");
            result.put("proofCall", proofBundle.getSchnorrProofCall().toMetaMap());
            return result;
        }
        result.put("status", "unsupported-proof-scheme");
        result.put(
                "error",
                "Current proofBundle is " + proofBundle.getProofScheme()
                        + " and has no Schnorr proof_payload that can be submitted to TruthSchnorrProofRegistry."
        );
        return result;
    }

    private Map<String, Object> runChainmakerProofBenchmark(String runId, ProofBundle proofBundle) throws Exception {
        Map<String, Object> result = buildInitialProofBenchmark(proofBundle);
        if (proofBundle == null || proofBundle.getSchnorrProofCall() == null) {
            return result;
        }

        TruthSchnorrProofRegistryClient client =
                new TruthSchnorrProofRegistryClient(chainmakerTruthSchnorrProofRegistryContractName);
        SchnorrProofCall proofCall = proofBundle.getSchnorrProofCall();
        byte[] runIdHash = sha256Bytes(runId);

        BenchmarkClock totalClock = BenchmarkClock.start();
        TimedValue<Boolean> verify = timeValue(() -> client.verifyProofValue(
                proofCall.getChallengeMaterialBytes(),
                proofCall.getPublicKey(),
                proofCall.getCommitment(),
                proofCall.getChallenge(),
                proofCall.getResponse()
        ));
        if (!Boolean.TRUE.equals(verify.getValue())) {
            result.put("status", "verify-failed");
            result.put("verifyMs", verify.getElapsedMs());
            result.put("totalMs", totalClock.elapsedMs());
            result.put("error", "TruthSchnorrProofRegistry.verifyProof returned false");
            return result;
        }

        TimedValue<byte[]> expectedKey = timeValue(() -> client.submissionKeyValue(
                runIdHash,
                proofCall.getExperimentId(),
                proofCall.getGroupId(),
                proofCall.getProofId()
        ));

        TimedVoid submit = timeVoid(() -> {
            ResultOuterClass.TxResponse response = client.submitProof(
                    runIdHash,
                    proofCall.getExperimentId(),
                    proofCall.getGroupId(),
                    proofCall.getResultDigest(),
                    proofCall.getProofId(),
                    proofCall.getStatementDigestSha256(),
                    proofCall.getChallengeMaterialBytes(),
                    proofCall.getPublicKey(),
                    proofCall.getCommitment(),
                    proofCall.getChallenge(),
                    proofCall.getResponse()
            );
            ensureChainmakerSuccess(response, "submitProof");
        });

        TimedValue<Boolean> hasSubmission = timeValue(() -> client.hasSubmissionValue(expectedKey.getValue()));

        result.put("status", Boolean.TRUE.equals(hasSubmission.getValue()) ? "proved-on-chain" : "submitted-but-not-found");
        result.put("onChainProved", Boolean.TRUE.equals(hasSubmission.getValue()));
        result.put("contractName", client.getContractName());
        result.put("verifyMs", verify.getElapsedMs());
        result.put("submissionKeyMs", expectedKey.getElapsedMs());
        result.put("submitMs", submit.getElapsedMs());
        result.put("hasSubmissionMs", hasSubmission.getElapsedMs());
        result.put("totalMs", totalClock.elapsedMs());
        result.put("verifyReturned", verify.getValue());
        result.put("hasSubmission", hasSubmission.getValue());
        result.put("submissionKey", "0x" + toHex(expectedKey.getValue()));
        return result;
    }

    private String buildNote(ProofBundle proofBundle, Map<String, Object> proofBenchmark) {
        String base = "Fixed-result benchmark skips AI inference; the callback writes one agent's full output digest ordered from news 1 to news 300. ";
        if (proofBundle == null) {
            return base + "No proofBundlePath was provided, so on-chain proof verification is not counted.";
        }
        if (proofBenchmark != null && Boolean.TRUE.equals(proofBenchmark.get("onChainProved"))) {
            return base + "The Schnorr proof payload was verified and submitted on-chain via TruthSchnorrProofRegistry; the result JSON records the full proof file path and SHA-256.";
        }
        if (proofBundle.getSchnorrProofCall() == null) {
            return base + "The proof bundle was read, but it has no Schnorr proof_payload submittable to TruthSchnorrProofRegistry, so on-chain proof cannot be counted as successful timing.";
        }
        return base + "On-chain Schnorr proof verification and submission were attempted, but the proof was not registered successfully; see proof.error/status.";
    }

    private String digestResults(OracleAggregatorClient.OllamaAllResults results) throws Exception {
        return sha256JsonArray(
                results.getResult0(),
                results.getResult1(),
                results.getResult2(),
                results.getResult3(),
                String.valueOf(results.getCompleted())
        );
    }

    private String sha256JsonArray(String... values) throws Exception {
        byte[] bytes = objectMapper.writeValueAsBytes(values);
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        return toHex(digest.digest(bytes));
    }

    private String sha256String(String value) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            return toHex(digest.digest(value.getBytes(StandardCharsets.UTF_8)));
        } catch (Exception e) {
            throw new IllegalStateException("failed to calculate sha256", e);
        }
    }

    private byte[] sha256Bytes(String value) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        return digest.digest(value.getBytes(StandardCharsets.UTF_8));
    }

    private String compressBase64(String value) throws Exception {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        try (GZIPOutputStream gzip = new GZIPOutputStream(output)) {
            gzip.write(value.getBytes(StandardCharsets.UTF_8));
        }
        return "gzip+base64:" + Base64.getEncoder().encodeToString(output.toByteArray());
    }

    private String buildSchnorrChallengeMaterial(
            String group,
            String proverId,
            String publicKey,
            String commitment,
            String statementHash
    ) {
        return "{\"commitment\":\"" + escapeJsonAscii(commitment) + "\""
                + ",\"group\":\"" + escapeJsonAscii(group) + "\""
                + ",\"prover_id\":\"" + escapeJsonAscii(proverId) + "\""
                + ",\"public_key\":\"" + escapeJsonAscii(publicKey) + "\""
                + ",\"scheme\":\"schnorr_nizk_v1\""
                + ",\"statement_hash\":\"" + escapeJsonAscii(statementHash) + "\"}";
    }

    private String requiredText(JsonNode node, String fieldName) {
        String value = optionalText(node, fieldName);
        if (value == null || value.isEmpty()) {
            throw new IllegalArgumentException("proofBundle is missing field: " + fieldName);
        }
        return value;
    }

    private String optionalText(JsonNode node, String fieldName) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return null;
        }
        JsonNode value = node.get(fieldName);
        if (value == null || value.isMissingNode() || value.isNull()) {
            return null;
        }
        return value.asText("").trim();
    }

    private byte[] bytes32FromHex(String value, String fieldName) {
        return fixedBytesFromHex(value, 32, fieldName);
    }

    private byte[] fixedBytesFromHex(String value, int length, String fieldName) {
        String hex = stripHexPrefix(value);
        if (hex.isEmpty()) {
            throw new IllegalArgumentException("hex field is empty: " + fieldName);
        }
        if ((hex.length() % 2) != 0) {
            hex = "0" + hex;
        }
        if (hex.length() > length * 2) {
            throw new IllegalArgumentException(fieldName + " exceeds " + length + " bytes");
        }
        byte[] raw = hexToBytes(hex, fieldName);
        byte[] fixed = new byte[length];
        System.arraycopy(raw, 0, fixed, length - raw.length, raw.length);
        return fixed;
    }

    private byte[] hexToBytes(String hex, String fieldName) {
        if ((hex.length() % 2) != 0) {
            throw new IllegalArgumentException(fieldName + " has an invalid hex length");
        }
        byte[] bytes = new byte[hex.length() / 2];
        for (int i = 0; i < bytes.length; i++) {
            int high = Character.digit(hex.charAt(i * 2), 16);
            int low = Character.digit(hex.charAt(i * 2 + 1), 16);
            if (high < 0 || low < 0) {
                throw new IllegalArgumentException(fieldName + " contains non-hex characters");
            }
            bytes[i] = (byte) ((high << 4) + low);
        }
        return bytes;
    }

    private String stripHexPrefix(String value) {
        if (value == null) {
            return "";
        }
        String trimmed = value.trim();
        if (trimmed.startsWith("0x") || trimmed.startsWith("0X")) {
            return trimmed.substring(2);
        }
        return trimmed;
    }

    private String escapeJsonAscii(String value) {
        if (value == null) {
            return "";
        }
        StringBuilder builder = new StringBuilder(value.length() + 16);
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            switch (c) {
                case '"':
                    builder.append("\\\"");
                    break;
                case '\\':
                    builder.append("\\\\");
                    break;
                case '\b':
                    builder.append("\\b");
                    break;
                case '\f':
                    builder.append("\\f");
                    break;
                case '\n':
                    builder.append("\\n");
                    break;
                case '\r':
                    builder.append("\\r");
                    break;
                case '\t':
                    builder.append("\\t");
                    break;
                default:
                    if (c < 0x20 || c > 0x7e) {
                        builder.append(String.format("\\u%04x", (int) c));
                    } else {
                        builder.append(c);
                    }
                    break;
            }
        }
        return builder.toString();
    }

    private String normalizeChain(String chain) {
        if (chain == null || chain.trim().isEmpty()) {
            throw new IllegalArgumentException("chain must not be empty");
        }
        return chain.trim().toLowerCase();
    }

    private void ensureChainmakerSuccess(ResultOuterClass.TxResponse response, String methodName) {
        if (response == null) {
            throw new IllegalStateException(methodName + " failed: null tx response");
        }
        if (response.getCode() != ResultOuterClass.TxStatusCode.SUCCESS) {
            throw new IllegalStateException(
                    methodName + " failed: code=" + response.getCode() + ", msg=" + response.getMessage()
            );
        }
        if (response.hasContractResult() && response.getContractResult().getCode() != 0) {
            throw new IllegalStateException(
                    methodName + " contract failed: code=" + response.getContractResult().getCode()
                            + ", msg=" + response.getContractResult().getMessage()
            );
        }
    }

    private String normalizePrompt(String prompt) {
        if (prompt == null || prompt.trim().isEmpty()) {
            return "benchmark fixed AI result request";
        }
        return prompt.trim();
    }

    private String toHex(byte[] bytes) {
        StringBuilder builder = new StringBuilder(bytes.length * 2);
        for (byte value : bytes) {
            builder.append(String.format("%02x", value));
        }
        return builder.toString();
    }

    private TimedVoid timeVoid(ThrowingRunnable runnable) throws Exception {
        BenchmarkClock clock = BenchmarkClock.start();
        runnable.run();
        return new TimedVoid(clock.elapsedMs());
    }

    private <T> TimedValue<T> timeValue(ThrowingSupplier<T> supplier) throws Exception {
        BenchmarkClock clock = BenchmarkClock.start();
        T value = supplier.get();
        return new TimedValue<>(value, clock.elapsedMs());
    }

    private interface ThrowingRunnable {
        void run() throws Exception;
    }

    private interface ThrowingSupplier<T> {
        T get() throws Exception;
    }

    private static class BenchmarkClock {
        private final long startedNanos;

        private BenchmarkClock(long startedNanos) {
            this.startedNanos = startedNanos;
        }

        private static BenchmarkClock start() {
            return new BenchmarkClock(System.nanoTime());
        }

        private long elapsedMs() {
            return Math.round((System.nanoTime() - startedNanos) / 1_000_000.0);
        }
    }

    private static class TimedVoid {
        private final long elapsedMs;

        private TimedVoid(long elapsedMs) {
            this.elapsedMs = elapsedMs;
        }

        private long getElapsedMs() {
            return elapsedMs;
        }
    }

    private static class TimedValue<T> {
        private final T value;
        private final long elapsedMs;

        private TimedValue(T value, long elapsedMs) {
            this.value = value;
            this.elapsedMs = elapsedMs;
        }

        private T getValue() {
            return value;
        }

        private long getElapsedMs() {
            return elapsedMs;
        }
    }

    private static class FixedResult {
        private final String path;
        private final long size;
        private final String sha256;
        private final List<String> agentPayloads;
        private final List<Map<String, Object>> agentPayloadMeta;
        private final int agentCount;
        private final int itemsPerAgent;

        private FixedResult(
                String path,
                long size,
                String sha256,
                List<String> agentPayloads,
                List<Map<String, Object>> agentPayloadMeta,
                int agentCount,
                int itemsPerAgent
        ) {
            this.path = path;
            this.size = size;
            this.sha256 = sha256;
            this.agentPayloads = agentPayloads;
            this.agentPayloadMeta = agentPayloadMeta;
            this.agentCount = agentCount;
            this.itemsPerAgent = itemsPerAgent;
        }

        private String getAgentPayload(int nodeIndex) {
            return agentPayloads.get(nodeIndex);
        }

        private String getPath() {
            return path;
        }

        private long getSize() {
            return size;
        }

        private String getSha256() {
            return sha256;
        }

        private int getAgentCount() {
            return agentCount;
        }

        private int getItemsPerAgent() {
            return itemsPerAgent;
        }

        private List<Map<String, Object>> getAgentPayloadMeta() {
            return agentPayloadMeta;
        }
    }

    private static class ProofBundle {
        private final String path;
        private final long size;
        private final String sha256;
        private final String method;
        private final int rowCount;
        private final int proofGroupCount;
        private final String proofScheme;
        private final boolean proofPayloadPresent;
        private final boolean groth16ContractCallable;
        private final SchnorrProofCall schnorrProofCall;

        private ProofBundle(
                String path,
                long size,
                String sha256,
                String method,
                int rowCount,
                int proofGroupCount,
                String proofScheme,
                boolean proofPayloadPresent,
                boolean groth16ContractCallable,
                SchnorrProofCall schnorrProofCall
        ) {
            this.path = path;
            this.size = size;
            this.sha256 = sha256;
            this.method = method;
            this.rowCount = rowCount;
            this.proofGroupCount = proofGroupCount;
            this.proofScheme = proofScheme;
            this.proofPayloadPresent = proofPayloadPresent;
            this.groth16ContractCallable = groth16ContractCallable;
            this.schnorrProofCall = schnorrProofCall;
        }

        private Map<String, Object> toMap() {
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("path", path);
            result.put("bytes", size);
            result.put("sha256", sha256);
            result.put("method", method);
            result.put("rowCount", rowCount);
            result.put("proofGroupCount", proofGroupCount);
            result.put("proofScheme", proofScheme);
            result.put("proofPayloadPresent", proofPayloadPresent);
            result.put("groth16ContractCallable", groth16ContractCallable);
            result.put("schnorrContractCallable", schnorrProofCall != null);
            return result;
        }

        private String getProofScheme() {
            return proofScheme;
        }

        private boolean isGroth16ContractCallable() {
            return groth16ContractCallable;
        }

        private boolean hasProofPayload() {
            return proofPayloadPresent;
        }

        private SchnorrProofCall getSchnorrProofCall() {
            return schnorrProofCall;
        }
    }

    private static class SchnorrProofCall {
        private final String challengeMaterial;
        private final byte[] experimentId;
        private final byte[] groupId;
        private final byte[] resultDigest;
        private final byte[] proofId;
        private final byte[] statementDigestSha256;
        private final byte[] publicKey;
        private final byte[] commitment;
        private final byte[] challenge;
        private final byte[] response;
        private final String proofIdHex;
        private final String statementHash;
        private final String proverId;
        private final String group;

        private SchnorrProofCall(
                String challengeMaterial,
                byte[] experimentId,
                byte[] groupId,
                byte[] resultDigest,
                byte[] proofId,
                byte[] statementDigestSha256,
                byte[] publicKey,
                byte[] commitment,
                byte[] challenge,
                byte[] response,
                String proofIdHex,
                String statementHash,
                String proverId,
                String group
        ) {
            this.challengeMaterial = challengeMaterial;
            this.experimentId = experimentId;
            this.groupId = groupId;
            this.resultDigest = resultDigest;
            this.proofId = proofId;
            this.statementDigestSha256 = statementDigestSha256;
            this.publicKey = publicKey;
            this.commitment = commitment;
            this.challenge = challenge;
            this.response = response;
            this.proofIdHex = proofIdHex;
            this.statementHash = statementHash;
            this.proverId = proverId;
            this.group = group;
        }

        private Map<String, Object> toMetaMap() {
            Map<String, Object> result = new LinkedHashMap<>();
            result.put("scheme", "schnorr_nizk_v1");
            result.put("group", group);
            result.put("proverId", proverId);
            result.put("proofId", proofIdHex.startsWith("0x") ? proofIdHex : "0x" + proofIdHex);
            result.put("statementHash", statementHash);
            result.put("challengeMaterial", challengeMaterial);
            result.put("challengeMaterialSha256", sha256BytesStatic(challengeMaterial));
            result.put("publicKeyBytes", publicKey.length);
            result.put("commitmentBytes", commitment.length);
            result.put("challengeBytes", challenge.length);
            result.put("responseBytes", response.length);
            result.put("experimentId", "0x" + toHexStatic(experimentId));
            result.put("groupId", "0x" + toHexStatic(groupId));
            result.put("resultDigest", "0x" + toHexStatic(resultDigest));
            result.put("proofIdBytes32", "0x" + toHexStatic(proofId));
            result.put("statementDigestSha256", "0x" + toHexStatic(statementDigestSha256));
            return result;
        }

        private byte[] getChallengeMaterialBytes() {
            return challengeMaterial.getBytes(StandardCharsets.UTF_8);
        }

        private byte[] getExperimentId() {
            return experimentId;
        }

        private byte[] getGroupId() {
            return groupId;
        }

        private byte[] getResultDigest() {
            return resultDigest;
        }

        private byte[] getProofId() {
            return proofId;
        }

        private byte[] getStatementDigestSha256() {
            return statementDigestSha256;
        }

        private byte[] getPublicKey() {
            return publicKey;
        }

        private byte[] getCommitment() {
            return commitment;
        }

        private byte[] getChallenge() {
            return challenge;
        }

        private byte[] getResponse() {
            return response;
        }
    }

    private static String sha256BytesStatic(String value) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            return toHexStatic(digest.digest(value.getBytes(StandardCharsets.UTF_8)));
        } catch (Exception e) {
            throw new IllegalStateException("failed to calculate sha256", e);
        }
    }

    private static String toHexStatic(byte[] bytes) {
        StringBuilder builder = new StringBuilder(bytes.length * 2);
        for (byte value : bytes) {
            builder.append(String.format("%02x", value));
        }
        return builder.toString();
    }
}
