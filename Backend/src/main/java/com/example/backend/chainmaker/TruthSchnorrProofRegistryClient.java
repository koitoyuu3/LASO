package com.example.backend.chainmaker;

import static com.example.backend.chainmaker.InitClient.chainClient;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.chainmaker.pb.common.ResultOuterClass;
import org.chainmaker.sdk.SdkException;
import org.web3j.abi.FunctionEncoder;
import org.web3j.abi.TypeReference;
import org.web3j.abi.datatypes.Bool;
import org.web3j.abi.datatypes.DynamicBytes;
import org.web3j.abi.datatypes.Function;
import org.web3j.abi.datatypes.Type;
import org.web3j.abi.datatypes.generated.Bytes32;

public class TruthSchnorrProofRegistryClient {

    private static final String DEFAULT_CONTRACT_NAME = "TruthSchnorrProofRegistry";
    private static final String CONTRACT_ARGS_EVM_PARAM = "data";
    private static final int rpcCallTimeout = 10000;
    private static final int syncResultTimeout = 10000;

    private final String contractName;

    public TruthSchnorrProofRegistryClient() {
        this(DEFAULT_CONTRACT_NAME);
    }

    public TruthSchnorrProofRegistryClient(String contractName) {
        if (contractName == null || contractName.trim().isEmpty()) {
            this.contractName = DEFAULT_CONTRACT_NAME;
            return;
        }
        this.contractName = contractName.trim();
    }

    public String getContractName() {
        return contractName;
    }

    public boolean verifyProofValue(
            byte[] challengeMaterial,
            byte[] publicKey,
            byte[] commitment,
            byte[] challenge,
            byte[] response
    ) throws SdkException {
        List<Type> values = OracleAggregatorClient.decodeEvmResult(
                verifyProof(challengeMaterial, publicKey, commitment, challenge, response),
                Arrays.asList(new TypeReference<Bool>() {})
        );
        if (values.isEmpty()) {
            throw new IllegalStateException("failed to decode verifyProof return values");
        }
        return Boolean.TRUE.equals(values.get(0).getValue());
    }

    public ResultOuterClass.TxResponse verifyProof(
            byte[] challengeMaterial,
            byte[] publicKey,
            byte[] commitment,
            byte[] challenge,
            byte[] response
    ) throws SdkException {
        Function function = new Function(
                "verifyProof",
                Arrays.asList(
                        new DynamicBytes(challengeMaterial),
                        new DynamicBytes(publicKey),
                        new DynamicBytes(commitment),
                        new DynamicBytes(challenge),
                        new DynamicBytes(response)
                ),
                Arrays.asList(new TypeReference<Bool>() {})
        );
        return query(function);
    }

    public ResultOuterClass.TxResponse submitProof(
            byte[] runIdHash,
            byte[] experimentId,
            byte[] groupId,
            byte[] resultDigest,
            byte[] proofId,
            byte[] statementDigestSha256,
            byte[] challengeMaterial,
            byte[] publicKey,
            byte[] commitment,
            byte[] challenge,
            byte[] response
    ) throws SdkException {
        Function function = new Function(
                "submitProof",
                Arrays.asList(
                        new Bytes32(runIdHash),
                        new Bytes32(experimentId),
                        new Bytes32(groupId),
                        new Bytes32(resultDigest),
                        new Bytes32(proofId),
                        new Bytes32(statementDigestSha256),
                        new DynamicBytes(challengeMaterial),
                        new DynamicBytes(publicKey),
                        new DynamicBytes(commitment),
                        new DynamicBytes(challenge),
                        new DynamicBytes(response)
                ),
                Arrays.asList(new TypeReference<Bytes32>() {})
        );
        return invoke(function);
    }

    public byte[] submissionKeyValue(
            byte[] runIdHash,
            byte[] experimentId,
            byte[] groupId,
            byte[] proofId
    ) throws SdkException {
        List<Type> values = OracleAggregatorClient.decodeEvmResult(
                submissionKey(runIdHash, experimentId, groupId, proofId),
                Arrays.asList(new TypeReference<Bytes32>() {})
        );
        if (values.isEmpty()) {
            throw new IllegalStateException("failed to decode submissionKey return values");
        }
        return (byte[]) values.get(0).getValue();
    }

    public ResultOuterClass.TxResponse submissionKey(
            byte[] runIdHash,
            byte[] experimentId,
            byte[] groupId,
            byte[] proofId
    ) throws SdkException {
        Function function = new Function(
                "submissionKey",
                Arrays.asList(
                        new Bytes32(runIdHash),
                        new Bytes32(experimentId),
                        new Bytes32(groupId),
                        new Bytes32(proofId)
                ),
                Arrays.asList(new TypeReference<Bytes32>() {})
        );
        return query(function);
    }

    public boolean hasSubmissionValue(byte[] key) throws SdkException {
        List<Type> values = OracleAggregatorClient.decodeEvmResult(
                hasSubmission(key),
                Arrays.asList(new TypeReference<Bool>() {})
        );
        if (values.isEmpty()) {
            throw new IllegalStateException("failed to decode hasSubmission return values");
        }
        return Boolean.TRUE.equals(values.get(0).getValue());
    }

    public ResultOuterClass.TxResponse hasSubmission(byte[] key) throws SdkException {
        Function function = new Function(
                "hasSubmission",
                Arrays.asList(new Bytes32(key)),
                Arrays.asList(new TypeReference<Bool>() {})
        );
        return query(function);
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
}
