package com.example.backend.contract;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.fisco.bcos.sdk.v3.client.Client;
import org.fisco.bcos.sdk.v3.codec.datatypes.Bool;
import org.fisco.bcos.sdk.v3.codec.datatypes.DynamicBytes;
import org.fisco.bcos.sdk.v3.codec.datatypes.Function;
import org.fisco.bcos.sdk.v3.codec.datatypes.Type;
import org.fisco.bcos.sdk.v3.codec.datatypes.TypeReference;
import org.fisco.bcos.sdk.v3.codec.datatypes.generated.Bytes32;
import org.fisco.bcos.sdk.v3.contract.Contract;
import org.fisco.bcos.sdk.v3.crypto.CryptoSuite;
import org.fisco.bcos.sdk.v3.crypto.keypair.CryptoKeyPair;
import org.fisco.bcos.sdk.v3.model.CryptoType;
import org.fisco.bcos.sdk.v3.model.TransactionReceipt;
import org.fisco.bcos.sdk.v3.transaction.model.exception.ContractException;

public class TruthSchnorrProofRegistry extends Contract {

    private static final String ABI_RESOURCE = "contract/TruthSchnorrProofRegistry.abi";
    private static final String BIN_RESOURCE = "contract/TruthSchnorrProofRegistry.bin";

    public static final String BINARY = loadArtifact(BIN_RESOURCE);
    public static final String SM_BINARY = BINARY;
    public static final String ABI = loadArtifact(ABI_RESOURCE);

    public static final String FUNC_VERIFYPROOF = "verifyProof";
    public static final String FUNC_SUBMITPROOF = "submitProof";
    public static final String FUNC_SUBMISSIONKEY = "submissionKey";
    public static final String FUNC_HASSUBMISSION = "hasSubmission";

    protected TruthSchnorrProofRegistry(String contractAddress, Client client, CryptoKeyPair credential) {
        super(getBinary(client.getCryptoSuite()), contractAddress, client, credential);
    }

    public static String getBinary(CryptoSuite cryptoSuite) {
        return cryptoSuite.getCryptoTypeConfig() == CryptoType.ECDSA_TYPE ? BINARY : SM_BINARY;
    }

    public static String getABI() {
        return ABI;
    }

    public Boolean verifyProof(
            byte[] challengeMaterial,
            byte[] publicKey,
            byte[] commitment,
            byte[] challenge,
            byte[] response
    ) throws ContractException {
        return executeCallWithSingleValueReturn(
                getMethodVerifyProofRawFunction(challengeMaterial, publicKey, commitment, challenge, response),
                Boolean.class
        );
    }

    public Function getMethodVerifyProofRawFunction(
            byte[] challengeMaterial,
            byte[] publicKey,
            byte[] commitment,
            byte[] challenge,
            byte[] response
    ) {
        return new Function(
                FUNC_VERIFYPROOF,
                Arrays.<Type>asList(
                        new DynamicBytes(challengeMaterial),
                        new DynamicBytes(publicKey),
                        new DynamicBytes(commitment),
                        new DynamicBytes(challenge),
                        new DynamicBytes(response)
                ),
                Arrays.<TypeReference<?>>asList(new TypeReference<Bool>() {})
        );
    }

    public TransactionReceipt submitProof(
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
    ) {
        return executeTransaction(getMethodSubmitProofRawFunction(
                runIdHash,
                experimentId,
                groupId,
                resultDigest,
                proofId,
                statementDigestSha256,
                challengeMaterial,
                publicKey,
                commitment,
                challenge,
                response
        ));
    }

    public Function getMethodSubmitProofRawFunction(
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
    ) {
        return new Function(
                FUNC_SUBMITPROOF,
                Arrays.<Type>asList(
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
                Arrays.<TypeReference<?>>asList(new TypeReference<Bytes32>() {}),
                4
        );
    }

    public byte[] submissionKey(
            byte[] runIdHash,
            byte[] experimentId,
            byte[] groupId,
            byte[] proofId
    ) throws ContractException {
        return executeCallWithSingleValueReturn(
                new Function(
                        FUNC_SUBMISSIONKEY,
                        Arrays.<Type>asList(
                                new Bytes32(runIdHash),
                                new Bytes32(experimentId),
                                new Bytes32(groupId),
                                new Bytes32(proofId)
                        ),
                        Arrays.<TypeReference<?>>asList(new TypeReference<Bytes32>() {})
                ),
                byte[].class
        );
    }

    public Boolean hasSubmission(byte[] key) throws ContractException {
        return executeCallWithSingleValueReturn(
                new Function(
                        FUNC_HASSUBMISSION,
                        Arrays.<Type>asList(new Bytes32(key)),
                        Arrays.<TypeReference<?>>asList(new TypeReference<Bool>() {})
                ),
                Boolean.class
        );
    }

    public static TruthSchnorrProofRegistry load(String contractAddress, Client client, CryptoKeyPair credential) {
        return new TruthSchnorrProofRegistry(contractAddress, client, credential);
    }

    public static TruthSchnorrProofRegistry deploy(Client client, CryptoKeyPair credential) throws ContractException {
        return deploy(
                TruthSchnorrProofRegistry.class,
                client,
                credential,
                getBinary(client.getCryptoSuite()),
                getABI(),
                null,
                null
        );
    }

    private static String loadArtifact(String resourcePath) {
        try (InputStream inputStream = TruthSchnorrProofRegistry.class.getClassLoader().getResourceAsStream(resourcePath)) {
            if (inputStream == null) {
                throw new IllegalStateException("contract artifact not found: " + resourcePath);
            }
            return new String(inputStream.readAllBytes(), StandardCharsets.UTF_8).trim();
        } catch (IOException e) {
            throw new IllegalStateException("failed to load contract artifact: " + resourcePath, e);
        }
    }
}
