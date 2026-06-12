package com.example.backend.contract;

import java.io.IOException;
import java.io.InputStream;
import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.fisco.bcos.sdk.v3.client.Client;
import org.fisco.bcos.sdk.v3.codec.datatypes.Address;
import org.fisco.bcos.sdk.v3.codec.datatypes.Bool;
import org.fisco.bcos.sdk.v3.codec.datatypes.DynamicStruct;
import org.fisco.bcos.sdk.v3.codec.datatypes.Event;
import org.fisco.bcos.sdk.v3.codec.datatypes.Function;
import org.fisco.bcos.sdk.v3.codec.datatypes.Type;
import org.fisco.bcos.sdk.v3.codec.datatypes.TypeReference;
import org.fisco.bcos.sdk.v3.codec.datatypes.Utf8String;
import org.fisco.bcos.sdk.v3.codec.datatypes.generated.Uint256;
import org.fisco.bcos.sdk.v3.codec.datatypes.generated.tuples.generated.Tuple2;
import org.fisco.bcos.sdk.v3.codec.datatypes.generated.tuples.generated.Tuple3;
import org.fisco.bcos.sdk.v3.codec.datatypes.generated.tuples.generated.Tuple4;
import org.fisco.bcos.sdk.v3.codec.datatypes.generated.tuples.generated.Tuple5;
import org.fisco.bcos.sdk.v3.codec.datatypes.generated.tuples.generated.Tuple8;
import org.fisco.bcos.sdk.v3.contract.Contract;
import org.fisco.bcos.sdk.v3.crypto.CryptoSuite;
import org.fisco.bcos.sdk.v3.crypto.keypair.CryptoKeyPair;
import org.fisco.bcos.sdk.v3.model.CryptoType;
import org.fisco.bcos.sdk.v3.model.TransactionReceipt;
import org.fisco.bcos.sdk.v3.transaction.model.exception.ContractException;

public class OracleAggregator extends Contract {

    private static final String ABI_RESOURCE = "contract/OracleAggregator.abi";
    private static final String BIN_RESOURCE = "contract/OracleAggregator.bin";

    public static final String BINARY = loadArtifact(BIN_RESOURCE);
    public static final String SM_BINARY = BINARY;
    public static final String ABI = loadArtifact(ABI_RESOURCE);

    public static final String FUNC_CALLBACKURL = "callbackUrl";
    public static final String FUNC_GETCALLBACKURL = "getCallbackUrl";
    public static final String FUNC_GETSUBMISSION = "getSubmission";
    public static final String FUNC_REQUESTCALLBACK = "requestCallback";
    public static final String FUNC_SETCALLBACKURL = "setCallbackUrl";
    public static final String FUNC_SUBMISSIONS = "submissions";
    public static final String FUNC_SUBMITPRICE = "submitPrice";
    public static final String FUNC_SETOLLAMANODE = "setOllamaNode";
    public static final String FUNC_GETOLLAMANODES = "getOllamaNodes";
    public static final String FUNC_REQUESTOLLAMAINFERENCE = "requestOllamaInference";
    public static final String FUNC_SUBMITOLLAMARESULT = "submitOllamaResult";
    public static final String FUNC_GETALLOLLAMARESULTS = "getAllOllamaResults";
    public static final String FUNC_GETOLLAMAPROMPT = "getOllamaPrompt";
    public static final String FUNC_GETLATESTOLLAMAPROMPT = "getLatestOllamaPrompt";
    public static final String FUNC_GETOLLAMAREQUESTMETA = "getOllamaRequestMeta";
    public static final String FUNC_LATESTINFERENCEREQUESTID = "latestInferenceRequestId";
    public static final String FUNC_GETOLLAMARESULT = "getOllamaResult";

    public static final Event CALLBACKTRIGGERED_EVENT = new Event(
            "CallbackTriggered",
            Arrays.<TypeReference<?>>asList(
                    new TypeReference<Utf8String>(true) {},
                    new TypeReference<Uint256>() {}
            )
    );

    public static final Event PRICESUBMITTED_EVENT = new Event(
            "PriceSubmitted",
            Arrays.<TypeReference<?>>asList(
                    new TypeReference<Utf8String>(true) {},
                    new TypeReference<Uint256>() {},
                    new TypeReference<Utf8String>() {},
                    new TypeReference<Uint256>() {}
            )
    );

    public static final Event OLLAMAPROMPTSTORED_EVENT = new Event(
            "OllamaPromptStored",
            Arrays.<TypeReference<?>>asList(
                    new TypeReference<Uint256>(true) {},
                    new TypeReference<Address>(true) {},
                    new TypeReference<Utf8String>() {},
                    new TypeReference<Uint256>() {}
            )
    );

    protected OracleAggregator(String contractAddress, Client client, CryptoKeyPair credential) {
        super(getBinary(client.getCryptoSuite()), contractAddress, client, credential);
    }

    public static String getBinary(CryptoSuite cryptoSuite) {
        return cryptoSuite.getCryptoTypeConfig() == CryptoType.ECDSA_TYPE ? BINARY : SM_BINARY;
    }

    public static String getABI() {
        return ABI;
    }

    public String callbackUrl() throws ContractException {
        return executeCallWithSingleValueReturn(getMethodCallbackUrlRawFunction(), String.class);
    }

    public Function getMethodCallbackUrlRawFunction() {
        return new Function(
                FUNC_CALLBACKURL,
                Collections.emptyList(),
                Arrays.<TypeReference<?>>asList(new TypeReference<Utf8String>() {})
        );
    }

    public String getCallbackUrl() throws ContractException {
        return executeCallWithSingleValueReturn(getMethodGetCallbackUrlRawFunction(), String.class);
    }

    public Function getMethodGetCallbackUrlRawFunction() {
        return new Function(
                FUNC_GETCALLBACKURL,
                Collections.emptyList(),
                Arrays.<TypeReference<?>>asList(new TypeReference<Utf8String>() {})
        );
    }

    public PriceSubmission getSubmission(String agentName) throws ContractException {
        return executeCallWithSingleValueReturn(getMethodGetSubmissionRawFunction(agentName), PriceSubmission.class);
    }

    public Function getMethodGetSubmissionRawFunction(String agentName) {
        return new Function(
                FUNC_GETSUBMISSION,
                Arrays.<Type>asList(new Utf8String(agentName)),
                Arrays.<TypeReference<?>>asList(new TypeReference<PriceSubmission>() {})
        );
    }

    public TransactionReceipt requestCallback(String agentName) {
        return executeTransaction(new Function(
                FUNC_REQUESTCALLBACK,
                Arrays.<Type>asList(new Utf8String(agentName)),
                Collections.emptyList(),
                4
        ));
    }

    public TransactionReceipt setCallbackUrl(String url) {
        return executeTransaction(new Function(
                FUNC_SETCALLBACKURL,
                Arrays.<Type>asList(new Utf8String(url)),
                Collections.emptyList(),
                0
        ));
    }

    public Tuple4<String, BigInteger, String, BigInteger> submissions(String agentName) throws ContractException {
        List<Type> results = executeCallWithMultipleValueReturn(getMethodSubmissionsRawFunction(agentName));
        return new Tuple4<>(
                (String) results.get(0).getValue(),
                (BigInteger) results.get(1).getValue(),
                (String) results.get(2).getValue(),
                (BigInteger) results.get(3).getValue()
        );
    }

    public Function getMethodSubmissionsRawFunction(String agentName) {
        return new Function(
                FUNC_SUBMISSIONS,
                Arrays.<Type>asList(new Utf8String(agentName)),
                Arrays.<TypeReference<?>>asList(
                        new TypeReference<Utf8String>() {},
                        new TypeReference<Uint256>() {},
                        new TypeReference<Utf8String>() {},
                        new TypeReference<Uint256>() {}
                )
        );
    }

    public TransactionReceipt submitPrice(String agentName, BigInteger price, String currency) {
        return executeTransaction(new Function(
                FUNC_SUBMITPRICE,
                Arrays.<Type>asList(
                        new Utf8String(agentName),
                        new Uint256(price),
                        new Utf8String(currency)
                ),
                Collections.emptyList(),
                4
        ));
    }

    public TransactionReceipt setOllamaNode(BigInteger nodeIndex, String nodeName, String submitter) {
        return executeTransaction(new Function(
                FUNC_SETOLLAMANODE,
                Arrays.<Type>asList(
                        new Uint256(nodeIndex),
                        new Utf8String(nodeName),
                        new Address(submitter)
                ),
                Collections.emptyList(),
                4
        ));
    }

    public Tuple8<String, String, String, String, String, String, String, String> getOllamaNodes()
            throws ContractException {
        List<Type> results = executeCallWithMultipleValueReturn(new Function(
                FUNC_GETOLLAMANODES,
                Collections.emptyList(),
                Arrays.<TypeReference<?>>asList(
                        new TypeReference<Utf8String>() {},
                        new TypeReference<Address>() {},
                        new TypeReference<Utf8String>() {},
                        new TypeReference<Address>() {},
                        new TypeReference<Utf8String>() {},
                        new TypeReference<Address>() {},
                        new TypeReference<Utf8String>() {},
                        new TypeReference<Address>() {}
                )
        ));
        return new Tuple8<>(
                (String) results.get(0).getValue(),
                (String) results.get(1).getValue(),
                (String) results.get(2).getValue(),
                (String) results.get(3).getValue(),
                (String) results.get(4).getValue(),
                (String) results.get(5).getValue(),
                (String) results.get(6).getValue(),
                (String) results.get(7).getValue()
        );
    }

    public TransactionReceipt requestOllamaInference(String prompt) {
        return executeTransaction(getMethodRequestOllamaInferenceRawFunction(prompt));
    }

    public Function getMethodRequestOllamaInferenceRawFunction(String prompt) {
        return new Function(
                FUNC_REQUESTOLLAMAINFERENCE,
                Arrays.<Type>asList(new Utf8String(prompt)),
                Arrays.<TypeReference<?>>asList(new TypeReference<Uint256>() {}),
                4
        );
    }

    public TransactionReceipt submitOllamaResult(BigInteger requestId, BigInteger nodeIndex, String result) {
        return executeTransaction(new Function(
                FUNC_SUBMITOLLAMARESULT,
                Arrays.<Type>asList(
                        new Uint256(requestId),
                        new Uint256(nodeIndex),
                        new Utf8String(result)
                ),
                Collections.emptyList(),
                4
        ));
    }

    public Tuple5<String, String, String, String, Boolean> getAllOllamaResults(BigInteger requestId)
            throws ContractException {
        List<Type> results = executeCallWithMultipleValueReturn(new Function(
                FUNC_GETALLOLLAMARESULTS,
                Arrays.<Type>asList(new Uint256(requestId)),
                Arrays.<TypeReference<?>>asList(
                        new TypeReference<Utf8String>() {},
                        new TypeReference<Utf8String>() {},
                        new TypeReference<Utf8String>() {},
                        new TypeReference<Utf8String>() {},
                        new TypeReference<Bool>() {}
                )
        ));
        return new Tuple5<>(
                (String) results.get(0).getValue(),
                (String) results.get(1).getValue(),
                (String) results.get(2).getValue(),
                (String) results.get(3).getValue(),
                (Boolean) results.get(4).getValue()
        );
    }

    public String getOllamaPrompt(BigInteger requestId) throws ContractException {
        return executeCallWithSingleValueReturn(new Function(
                FUNC_GETOLLAMAPROMPT,
                Arrays.<Type>asList(new Uint256(requestId)),
                Arrays.<TypeReference<?>>asList(new TypeReference<Utf8String>() {})
        ), String.class);
    }

    public Tuple2<BigInteger, String> getLatestOllamaPrompt() throws ContractException {
        List<Type> results = executeCallWithMultipleValueReturn(new Function(
                FUNC_GETLATESTOLLAMAPROMPT,
                Collections.emptyList(),
                Arrays.<TypeReference<?>>asList(
                        new TypeReference<Uint256>() {},
                        new TypeReference<Utf8String>() {}
                )
        ));
        return new Tuple2<>(
                (BigInteger) results.get(0).getValue(),
                (String) results.get(1).getValue()
        );
    }

    public Tuple4<String, BigInteger, BigInteger, Boolean> getOllamaRequestMeta(BigInteger requestId)
            throws ContractException {
        List<Type> results = executeCallWithMultipleValueReturn(new Function(
                FUNC_GETOLLAMAREQUESTMETA,
                Arrays.<Type>asList(new Uint256(requestId)),
                Arrays.<TypeReference<?>>asList(
                        new TypeReference<Utf8String>() {},
                        new TypeReference<Uint256>() {},
                        new TypeReference<Uint256>() {},
                        new TypeReference<Bool>() {}
                )
        ));
        return new Tuple4<>(
                (String) results.get(0).getValue(),
                (BigInteger) results.get(1).getValue(),
                (BigInteger) results.get(2).getValue(),
                (Boolean) results.get(3).getValue()
        );
    }

    public BigInteger latestInferenceRequestId() throws ContractException {
        return executeCallWithSingleValueReturn(new Function(
                FUNC_LATESTINFERENCEREQUESTID,
                Collections.emptyList(),
                Arrays.<TypeReference<?>>asList(new TypeReference<Uint256>() {})
        ), BigInteger.class);
    }

    public Tuple3<String, BigInteger, Boolean> getOllamaResult(BigInteger requestId, BigInteger nodeIndex)
            throws ContractException {
        List<Type> results = executeCallWithMultipleValueReturn(new Function(
                FUNC_GETOLLAMARESULT,
                Arrays.<Type>asList(new Uint256(requestId), new Uint256(nodeIndex)),
                Arrays.<TypeReference<?>>asList(
                        new TypeReference<Utf8String>() {},
                        new TypeReference<Uint256>() {},
                        new TypeReference<Bool>() {}
                )
        ));
        return new Tuple3<>(
                (String) results.get(0).getValue(),
                (BigInteger) results.get(1).getValue(),
                (Boolean) results.get(2).getValue()
        );
    }

    public List<CallbackTriggeredEventResponse> getCallbackTriggeredEvents(TransactionReceipt transactionReceipt) {
        List<EventValuesWithLog> valueList = extractEventParametersWithLog(CALLBACKTRIGGERED_EVENT, transactionReceipt);
        List<CallbackTriggeredEventResponse> responses = new ArrayList<>(valueList.size());
        for (EventValuesWithLog eventValues : valueList) {
            CallbackTriggeredEventResponse typedResponse = new CallbackTriggeredEventResponse();
            typedResponse.log = eventValues.getLog();
            typedResponse.agent = (byte[]) eventValues.getIndexedValues().get(0).getValue();
            typedResponse.timestamp = (BigInteger) eventValues.getNonIndexedValues().get(0).getValue();
            responses.add(typedResponse);
        }
        return responses;
    }

    public List<PriceSubmittedEventResponse> getPriceSubmittedEvents(TransactionReceipt transactionReceipt) {
        List<EventValuesWithLog> valueList = extractEventParametersWithLog(PRICESUBMITTED_EVENT, transactionReceipt);
        List<PriceSubmittedEventResponse> responses = new ArrayList<>(valueList.size());
        for (EventValuesWithLog eventValues : valueList) {
            PriceSubmittedEventResponse typedResponse = new PriceSubmittedEventResponse();
            typedResponse.log = eventValues.getLog();
            typedResponse.agent = (byte[]) eventValues.getIndexedValues().get(0).getValue();
            typedResponse.price = (BigInteger) eventValues.getNonIndexedValues().get(0).getValue();
            typedResponse.currency = (String) eventValues.getNonIndexedValues().get(1).getValue();
            typedResponse.timestamp = (BigInteger) eventValues.getNonIndexedValues().get(2).getValue();
            responses.add(typedResponse);
        }
        return responses;
    }

    public List<OllamaPromptStoredEventResponse> getOllamaPromptStoredEvents(TransactionReceipt transactionReceipt) {
        List<EventValuesWithLog> valueList = extractEventParametersWithLog(OLLAMAPROMPTSTORED_EVENT, transactionReceipt);
        List<OllamaPromptStoredEventResponse> responses = new ArrayList<>(valueList.size());
        for (EventValuesWithLog eventValues : valueList) {
            OllamaPromptStoredEventResponse typedResponse = new OllamaPromptStoredEventResponse();
            typedResponse.log = eventValues.getLog();
            typedResponse.requestId = (BigInteger) eventValues.getIndexedValues().get(0).getValue();
            typedResponse.requester = (String) eventValues.getIndexedValues().get(1).getValue();
            typedResponse.prompt = (String) eventValues.getNonIndexedValues().get(0).getValue();
            typedResponse.timestamp = (BigInteger) eventValues.getNonIndexedValues().get(1).getValue();
            responses.add(typedResponse);
        }
        return responses;
    }

    public static OracleAggregator load(String contractAddress, Client client, CryptoKeyPair credential) {
        return new OracleAggregator(contractAddress, client, credential);
    }

    public static OracleAggregator deploy(Client client, CryptoKeyPair credential) throws ContractException {
        return deploy(
                OracleAggregator.class,
                client,
                credential,
                getBinary(client.getCryptoSuite()),
                getABI(),
                null,
                null
        );
    }

    private static String loadArtifact(String resourcePath) {
        try (InputStream inputStream = OracleAggregator.class.getClassLoader().getResourceAsStream(resourcePath)) {
            if (inputStream == null) {
                throw new IllegalStateException("contract artifact not found: " + resourcePath);
            }
            return new String(inputStream.readAllBytes(), StandardCharsets.UTF_8).trim();
        } catch (IOException e) {
            throw new IllegalStateException("failed to load contract artifact: " + resourcePath, e);
        }
    }

    public static class PriceSubmission extends DynamicStruct {
        public String agent;
        public BigInteger price;
        public String currency;
        public BigInteger timestamp;

        public PriceSubmission(Utf8String agent, Uint256 price, Utf8String currency, Uint256 timestamp) {
            super(agent, price, currency, timestamp);
            this.agent = agent.getValue();
            this.price = price.getValue();
            this.currency = currency.getValue();
            this.timestamp = timestamp.getValue();
        }

        public PriceSubmission(String agent, BigInteger price, String currency, BigInteger timestamp) {
            super(new Utf8String(agent), new Uint256(price), new Utf8String(currency), new Uint256(timestamp));
            this.agent = agent;
            this.price = price;
            this.currency = currency;
            this.timestamp = timestamp;
        }
    }

    public static class CallbackTriggeredEventResponse {
        public TransactionReceipt.Logs log;
        public byte[] agent;
        public BigInteger timestamp;
    }

    public static class PriceSubmittedEventResponse {
        public TransactionReceipt.Logs log;
        public byte[] agent;
        public BigInteger price;
        public String currency;
        public BigInteger timestamp;
    }

    public static class OllamaPromptStoredEventResponse {
        public TransactionReceipt.Logs log;
        public BigInteger requestId;
        public String requester;
        public String prompt;
        public BigInteger timestamp;
    }
}
