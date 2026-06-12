pragma solidity ^0.8.20;

interface IGroth16VerifierN4 {
    function verifyProof(
        uint256[2] calldata a,
        uint256[2][2] calldata b,
        uint256[2] calldata c,
        uint256[15] calldata pubSignals
    ) external view returns (bool);
}

contract TruthResultRegistry {
    struct Submission {
        bytes32 experimentId;
        bytes32 groupId;
        bytes32 resultDigest;
        bytes32 proofId;
        bytes32 statementDigestSha256;
        uint256 outputStatementHashField;
        uint256 outputWeightedSum;
        uint256 outputTotalWeight;
        uint256 outputIsValid;
        uint256 claimedWeightedSum;
        uint256 claimedTotalWeight;
        uint256 inputStatementHashField;
        string resultURI;
        string proofURI;
        address submitter;
        uint64 submittedAt;
    }

    error InvalidVerifier();
    error InvalidProof();
    error InvalidPublicSignals();
    error SubmissionAlreadyExists();

    event ResultSubmitted(
        bytes32 indexed submissionKey,
        bytes32 indexed experimentId,
        bytes32 indexed groupId,
        bytes32 proofId,
        address submitter
    );

    IGroth16VerifierN4 public immutable verifier;
    mapping(bytes32 => Submission) public submissions;

    constructor(address verifier_) {
        if (verifier_ == address(0)) revert InvalidVerifier();
        verifier = IGroth16VerifierN4(verifier_);
    }

    function submissionKey(
        bytes32 experimentId,
        bytes32 groupId,
        bytes32 proofId
    ) public pure returns (bytes32) {
        return keccak256(abi.encodePacked(experimentId, groupId, proofId));
    }

    function _hasConsistentSignals(uint256[15] calldata pubSignals) internal pure returns (bool) {
        return (
            pubSignals[3] == 1 &&
            pubSignals[0] == pubSignals[14] &&
            pubSignals[1] == pubSignals[12] &&
            pubSignals[2] == pubSignals[13]
        );
    }

    function verifyAggregationProof(
        uint256[2] calldata a,
        uint256[2][2] calldata b,
        uint256[2] calldata c,
        uint256[15] calldata pubSignals
    ) external view returns (bool) {
        if (!_hasConsistentSignals(pubSignals)) {
            return false;
        }
        return verifier.verifyProof(a, b, c, pubSignals);
    }

    function submitResultAndVerify(
        bytes32 experimentId,
        bytes32 groupId,
        bytes32 resultDigest,
        bytes32 proofId,
        bytes32 statementDigestSha256,
        string calldata resultURI,
        string calldata proofURI,
        uint256[2] calldata a,
        uint256[2][2] calldata b,
        uint256[2] calldata c,
        uint256[15] calldata pubSignals
    ) external returns (bytes32 key) {
        if (!_hasConsistentSignals(pubSignals)) revert InvalidPublicSignals();
        if (!verifier.verifyProof(a, b, c, pubSignals)) revert InvalidProof();

        key = submissionKey(experimentId, groupId, proofId);
        if (submissions[key].submittedAt != 0) revert SubmissionAlreadyExists();

        submissions[key] = Submission({
            experimentId: experimentId,
            groupId: groupId,
            resultDigest: resultDigest,
            proofId: proofId,
            statementDigestSha256: statementDigestSha256,
            outputStatementHashField: pubSignals[0],
            outputWeightedSum: pubSignals[1],
            outputTotalWeight: pubSignals[2],
            outputIsValid: pubSignals[3],
            claimedWeightedSum: pubSignals[12],
            claimedTotalWeight: pubSignals[13],
            inputStatementHashField: pubSignals[14],
            resultURI: resultURI,
            proofURI: proofURI,
            submitter: msg.sender,
            submittedAt: uint64(block.timestamp)
        });

        emit ResultSubmitted(key, experimentId, groupId, proofId, msg.sender);
    }
}
