// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract TruthSchnorrProofRegistry {
    error InvalidProofData();
    error InvalidProof();
    error ProofAlreadyExists();

    event SchnorrProofSubmitted(
        bytes32 indexed submissionKey,
        bytes32 indexed experimentId,
        bytes32 indexed groupId,
        bytes32 resultDigest,
        bytes32 proofId,
        bytes32 statementDigestSha256,
        address submitter
    );

    uint256 private constant P0 = 0xffffffffffffffffc90fdaa22168c234c4c6628b80dc1cd129024e088a67cc74;
    uint256 private constant P1 = 0x020bbea63b139b22514a08798e3404ddef9519b3cd3a431b302b0a6df25f1437;
    uint256 private constant P2 = 0x4fe1356d6d51c245e485b576625e7ec6f44c42e9a637ed6b0bff5cb6f406b7ed;
    uint256 private constant P3 = 0xee386bfb5a899fa5ae9f24117c4b1fe649286651ece45b3dc2007cb8a163bf05;
    uint256 private constant P4 = 0x98da48361c55d39a69163fa8fd24cf5f83655d23dca3ad961c62f356208552bb;
    uint256 private constant P5 = 0x9ed529077096966d670c354e4abc9804f1746c08ca18217c32905e462e36ce3b;
    uint256 private constant P6 = 0xe39e772c180e86039b2783a2ec07a28fb5c55df06f4c52c9de2bcbf695581718;
    uint256 private constant P7 = 0x3995497cea956ae515d2261898fa051015728e5a8aacaa68ffffffffffffffff;

    uint256 private constant D0 = 0x000000000000000036f0255dde973dcb3b399d747f23e32ed6fdb1f77598338b;
    uint256 private constant D1 = 0xfdf44159c4ec64ddaeb5f78671cbfb22106ae64c32c5bce4cfd4f5920da0ebc8;
    uint256 private constant D2 = 0xb01eca9292ae3dba1b7a4a899da181390bb3bd1659c81294f400a3490bf94812;
    uint256 private constant D3 = 0x11c79404a576605a5160dbee83b4e019b6d799ae131ba4c23dff83475e9c40fa;
    uint256 private constant D4 = 0x6725b7c9e3aa2c6596e9c05702db30a07c9aa2dc235c5269e39d0ca9df7aad44;
    uint256 private constant D5 = 0x612ad6f88f69699298f3cab1b54367fb0e8b93f735e7de83cd6fa1b9d1c931c4;
    uint256 private constant D6 = 0x1c6188d3e7f179fc64d87c5d13f85d704a3aa20f90b3ad3621d434096aa7e8e7;
    uint256 private constant D7 = 0xc66ab683156a951aea2dd9e76705faefea8d71a5755355970000000000000001;

    mapping(bytes32 => uint64) private submittedAt;

    function submissionKey(
        bytes32 runIdHash,
        bytes32 experimentId,
        bytes32 groupId,
        bytes32 proofId
    ) public pure returns (bytes32) {
        return keccak256(abi.encodePacked(runIdHash, experimentId, groupId, proofId));
    }

    function hasSubmission(bytes32 key) external view returns (bool) {
        return submittedAt[key] != 0;
    }

    function verifyProof(
        bytes calldata challengeMaterial,
        bytes calldata publicKey,
        bytes calldata commitment,
        bytes calldata challenge,
        bytes calldata response
    ) public view returns (bool) {
        bytes32 expectedChallenge = sha256(challengeMaterial);
        if (expectedChallenge == bytes32(0)) {
            expectedChallenge = bytes32(uint256(1));
        }
        if (_bytes32Value(challenge) != expectedChallenge) {
            return false;
        }

        uint256[8] memory publicKeyLimbs = _bytesToLimbs256(publicKey);
        uint256[8] memory commitmentLimbs = _bytesToLimbs256(commitment);
        uint256[8] memory responseLimbs = _bytesToLimbs256(response);
        if (!_isValidGroupElement(publicKeyLimbs) || !_isValidGroupElement(commitmentLimbs)) {
            return false;
        }
        if (!_isLessThanP(responseLimbs)) {
            return false;
        }

        uint256[8] memory challengeLimbs;
        challengeLimbs[7] = uint256(expectedChallenge);

        uint256[8] memory left = _modExp(_generator(), responseLimbs);
        uint256[8] memory publicKeyPow = _modExp(publicKeyLimbs, challengeLimbs);
        uint256[8] memory right = _mulModP(commitmentLimbs, publicKeyPow);
        return _eq(left, right);
    }

    function submitProof(
        bytes32 runIdHash,
        bytes32 experimentId,
        bytes32 groupId,
        bytes32 resultDigest,
        bytes32 proofId,
        bytes32 statementDigestSha256,
        bytes calldata challengeMaterial,
        bytes calldata publicKey,
        bytes calldata commitment,
        bytes calldata challenge,
        bytes calldata response
    ) external returns (bytes32 key) {
        if (!verifyProof(challengeMaterial, publicKey, commitment, challenge, response)) {
            revert InvalidProof();
        }

        key = submissionKey(runIdHash, experimentId, groupId, proofId);
        if (submittedAt[key] != 0) {
            revert ProofAlreadyExists();
        }
        submittedAt[key] = uint64(block.timestamp);
        emit SchnorrProofSubmitted(
            key,
            experimentId,
            groupId,
            resultDigest,
            proofId,
            statementDigestSha256,
            msg.sender
        );
    }

    function _bytes32Value(bytes calldata value) private pure returns (bytes32 out) {
        if (value.length != 32) {
            revert InvalidProofData();
        }
        assembly {
            out := calldataload(value.offset)
        }
    }

    function _bytesToLimbs256(bytes calldata value) private pure returns (uint256[8] memory limbs) {
        if (value.length != 256) {
            revert InvalidProofData();
        }
        assembly {
            for { let i := 0 } lt(i, 8) { i := add(i, 1) } {
                mstore(add(limbs, mul(i, 32)), calldataload(add(value.offset, mul(i, 32))))
            }
        }
    }

    function _generator() private pure returns (uint256[8] memory g) {
        g[7] = 4;
    }

    function _p() private pure returns (uint256[8] memory p) {
        p[0] = P0; p[1] = P1; p[2] = P2; p[3] = P3;
        p[4] = P4; p[5] = P5; p[6] = P6; p[7] = P7;
    }

    function _delta() private pure returns (uint256[8] memory d) {
        d[0] = D0; d[1] = D1; d[2] = D2; d[3] = D3;
        d[4] = D4; d[5] = D5; d[6] = D6; d[7] = D7;
    }

    function _isValidGroupElement(uint256[8] memory value) private pure returns (bool) {
        return !_isZero(value) && _isLessThanP(value);
    }

    function _isLessThanP(uint256[8] memory value) private pure returns (bool) {
        uint256[8] memory p = _p();
        for (uint256 i = 0; i < 8; i++) {
            if (value[i] < p[i]) return true;
            if (value[i] > p[i]) return false;
        }
        return false;
    }

    function _isZero(uint256[8] memory value) private pure returns (bool) {
        for (uint256 i = 0; i < 8; i++) {
            if (value[i] != 0) return false;
        }
        return true;
    }

    function _eq(uint256[8] memory a, uint256[8] memory b) private pure returns (bool) {
        for (uint256 i = 0; i < 8; i++) {
            if (a[i] != b[i]) return false;
        }
        return true;
    }

    function _gteP(uint256[8] memory value) private pure returns (bool) {
        uint256[8] memory p = _p();
        for (uint256 i = 0; i < 8; i++) {
            if (value[i] > p[i]) return true;
            if (value[i] < p[i]) return false;
        }
        return true;
    }

    function _addModP(uint256[8] memory a, uint256[8] memory b) private pure returns (uint256[8] memory c) {
        uint256 carry = 0;
        unchecked {
            for (uint256 i = 8; i > 0; i--) {
                uint256 idx = i - 1;
                uint256 sum = a[idx] + b[idx];
                uint256 carry1 = sum < a[idx] ? 1 : 0;
                uint256 sum2 = sum + carry;
                uint256 carry2 = sum2 < sum ? 1 : 0;
                c[idx] = sum2;
                carry = carry1 + carry2;
            }
        }
        if (carry != 0) {
            c = _addNoOverflow(c, _delta());
        }
        if (_gteP(c)) {
            c = _subP(c);
        }
    }

    function _addNoOverflow(
        uint256[8] memory a,
        uint256[8] memory b
    ) private pure returns (uint256[8] memory c) {
        uint256 carry = 0;
        unchecked {
            for (uint256 i = 8; i > 0; i--) {
                uint256 idx = i - 1;
                uint256 sum = a[idx] + b[idx];
                uint256 carry1 = sum < a[idx] ? 1 : 0;
                uint256 sum2 = sum + carry;
                uint256 carry2 = sum2 < sum ? 1 : 0;
                c[idx] = sum2;
                carry = carry1 + carry2;
            }
        }
        if (carry != 0) {
            revert InvalidProof();
        }
    }

    function _subP(uint256[8] memory a) private pure returns (uint256[8] memory c) {
        uint256[8] memory p = _p();
        uint256 borrow = 0;
        unchecked {
            for (uint256 i = 8; i > 0; i--) {
                uint256 idx = i - 1;
                uint256 subtrahend = p[idx] + borrow;
                uint256 carry = subtrahend < p[idx] ? 1 : 0;
                if (a[idx] < subtrahend || carry != 0) {
                    c[idx] = a[idx] - subtrahend;
                    borrow = 1;
                } else {
                    c[idx] = a[idx] - subtrahend;
                    borrow = 0;
                }
            }
        }
    }

    function _mulModP(
        uint256[8] memory a,
        uint256[8] memory b
    ) private pure returns (uint256[8] memory result) {
        uint256[8] memory addend = a;
        for (uint256 wordIndex = 8; wordIndex > 0; wordIndex--) {
            uint256 word = b[wordIndex - 1];
            for (uint256 bit = 0; bit < 256; bit++) {
                if (((word >> bit) & 1) == 1) {
                    result = _addModP(result, addend);
                }
                addend = _addModP(addend, addend);
            }
        }
    }

    function _modExp(
        uint256[8] memory base,
        uint256[8] memory exponent
    ) private view returns (uint256[8] memory out) {
        bytes memory input = new bytes(864);
        bytes memory output = new bytes(256);
        uint256[8] memory p = _p();
        assembly {
            mstore(add(input, 32), 256)
            mstore(add(input, 64), 256)
            mstore(add(input, 96), 256)
            for { let i := 0 } lt(i, 8) { i := add(i, 1) } {
                mstore(add(add(input, 128), mul(i, 32)), mload(add(base, mul(i, 32))))
                mstore(add(add(input, 384), mul(i, 32)), mload(add(exponent, mul(i, 32))))
                mstore(add(add(input, 640), mul(i, 32)), mload(add(p, mul(i, 32))))
            }
            if iszero(staticcall(gas(), 5, add(input, 32), 864, add(output, 32), 256)) {
                revert(0, 0)
            }
            for { let i := 0 } lt(i, 8) { i := add(i, 1) } {
                mstore(add(out, mul(i, 32)), mload(add(add(output, 32), mul(i, 32))))
            }
        }
    }
}
