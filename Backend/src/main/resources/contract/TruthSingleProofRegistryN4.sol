// SPDX-License-Identifier: GPL-3.0
/*
    Copyright 2021 0KIMS association.

    This file is generated with [snarkJS](https://github.com/iden3/snarkjs).

    snarkJS is a free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    snarkJS is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
    or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public
    License for more details.

    You should have received a copy of the GNU General Public License
    along with snarkJS. If not, see <https://www.gnu.org/licenses/>.
*/

pragma solidity ^0.8.20;

contract TruthSingleProofRegistryN4 {
    // Scalar field size
    uint256 constant r    = 21888242871839275222246405745257275088548364400416034343698204186575808495617;
    // Base field size
    uint256 constant q   = 21888242871839275222246405745257275088696311157297823662689037894645226208583;

    // Verification Key data
    uint256 constant alphax  = 1;
    uint256 constant alphay  = 2;
    uint256 constant betax1  = 11559732032986387107991004021392285783925812861821192530917403151452391805634;
    uint256 constant betax2  = 10857046999023057135944570762232829481370756359578518086990519993285655852781;
    uint256 constant betay1  = 4082367875863433681332203403145435568316851327593401208105741076214120093531;
    uint256 constant betay2  = 8495653923123431417604973247489272438418190587263600148770280649306958101930;
    uint256 constant gammax1 = 11559732032986387107991004021392285783925812861821192530917403151452391805634;
    uint256 constant gammax2 = 10857046999023057135944570762232829481370756359578518086990519993285655852781;
    uint256 constant gammay1 = 4082367875863433681332203403145435568316851327593401208105741076214120093531;
    uint256 constant gammay2 = 8495653923123431417604973247489272438418190587263600148770280649306958101930;
    uint256 constant deltax1 = 11559732032986387107991004021392285783925812861821192530917403151452391805634;
    uint256 constant deltax2 = 10857046999023057135944570762232829481370756359578518086990519993285655852781;
    uint256 constant deltay1 = 4082367875863433681332203403145435568316851327593401208105741076214120093531;
    uint256 constant deltay2 = 8495653923123431417604973247489272438418190587263600148770280649306958101930;

    
    uint256 constant IC0x = 0;
    uint256 constant IC0y = 0;
    
    uint256 constant IC1x = 0;
    uint256 constant IC1y = 0;
    
    uint256 constant IC2x = 0;
    uint256 constant IC2y = 0;
    
    uint256 constant IC3x = 0;
    uint256 constant IC3y = 0;
    
    uint256 constant IC4x = 0;
    uint256 constant IC4y = 0;
    
    uint256 constant IC5x = 1;
    uint256 constant IC5y = 21888242871839275222246405745257275088696311157297823662689037894645226208581;
    
    uint256 constant IC6x = 0;
    uint256 constant IC6y = 0;
    
    uint256 constant IC7x = 0;
    uint256 constant IC7y = 0;
    
    uint256 constant IC8x = 0;
    uint256 constant IC8y = 0;
    
    uint256 constant IC9x = 1;
    uint256 constant IC9y = 2;
    
    uint256 constant IC10x = 0;
    uint256 constant IC10y = 0;
    
    uint256 constant IC11x = 0;
    uint256 constant IC11y = 0;
    
    uint256 constant IC12x = 0;
    uint256 constant IC12y = 0;
    
    uint256 constant IC13x = 0;
    uint256 constant IC13y = 0;
    
    uint256 constant IC14x = 0;
    uint256 constant IC14y = 0;
    
    uint256 constant IC15x = 0;
    uint256 constant IC15y = 0;
    
 
    // Memory data
    uint16 constant pVk = 0;
    uint16 constant pPairing = 128;

    uint16 constant pLastMem = 896;

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

    error InvalidProof();
    error InvalidPublicSignals();
    error ProofAlreadyExists();

    event ProofSubmitted(
        bytes32 indexed submissionKey,
        bytes32 indexed experimentId,
        bytes32 indexed groupId,
        bytes32 proofId,
        address submitter
    );

    mapping(bytes32 => Submission) private submissions;

    function submissionKey(
        bytes32 experimentId,
        bytes32 groupId,
        bytes32 proofId
    ) public pure returns (bytes32) {
        return keccak256(abi.encodePacked(experimentId, groupId, proofId));
    }

    function getSubmission(bytes32 key) external view returns (Submission memory) {
        return submissions[key];
    }

    function hasSubmission(bytes32 key) external view returns (bool) {
        return submissions[key].submittedAt != 0;
    }

    function verifySelectedProof(
        uint256[2] calldata a,
        uint256[2][2] calldata b,
        uint256[2] calldata c,
        uint256[15] calldata pubSignals
    ) public view returns (bool) {
        if (
            pubSignals[3] != 1 ||
            pubSignals[0] != pubSignals[14] ||
            pubSignals[1] != pubSignals[12] ||
            pubSignals[2] != pubSignals[13]
        ) {
            return false;
        }
        return verifyProof(a, b, c, pubSignals);
    }

    function verifySelectedProofFlat(
        uint256 a0,
        uint256 a1,
        uint256 b00,
        uint256 b01,
        uint256 b10,
        uint256 b11,
        uint256 c0,
        uint256 c1,
        uint256 pub0,
        uint256 pub1,
        uint256 pub2,
        uint256 pub3,
        uint256 pub4,
        uint256 pub5,
        uint256 pub6,
        uint256 pub7,
        uint256 pub8,
        uint256 pub9,
        uint256 pub10,
        uint256 pub11,
        uint256 pub12,
        uint256 pub13,
        uint256 pub14
    ) external view returns (bool) {
        if (
            pub3 != 1 ||
            pub0 != pub14 ||
            pub1 != pub12 ||
            pub2 != pub13
        ) {
            return false;
        }

        assembly ("memory-safe") {
            function checkField(v) {
                if iszero(lt(v, r)) {
                    mstore(0, 0)
                    return(0, 0x20)
                }
            }

            function g1_mulAccC(pR, x, y, s) {
                let success
                let mIn := mload(0x40)
                mstore(mIn, x)
                mstore(add(mIn, 32), y)
                mstore(add(mIn, 64), s)

                success := staticcall(sub(gas(), 2000), 7, mIn, 96, mIn, 64)

                if iszero(success) {
                    mstore(0, 0)
                    return(0, 0x20)
                }

                mstore(add(mIn, 64), mload(pR))
                mstore(add(mIn, 96), mload(add(pR, 32)))

                success := staticcall(sub(gas(), 2000), 6, mIn, 128, pR, 64)

                if iszero(success) {
                    mstore(0, 0)
                    return(0, 0x20)
                }
            }

            function checkPairingFlat(pMem) -> isOk {
                let _pPairing := add(pMem, pPairing)
                let _pVk := add(pMem, pVk)

                mstore(_pVk, IC0x)
                mstore(add(_pVk, 32), IC0y)

                g1_mulAccC(_pVk, IC1x, IC1y, calldataload(260))
                g1_mulAccC(_pVk, IC2x, IC2y, calldataload(292))
                g1_mulAccC(_pVk, IC3x, IC3y, calldataload(324))
                g1_mulAccC(_pVk, IC4x, IC4y, calldataload(356))
                g1_mulAccC(_pVk, IC5x, IC5y, calldataload(388))
                g1_mulAccC(_pVk, IC6x, IC6y, calldataload(420))
                g1_mulAccC(_pVk, IC7x, IC7y, calldataload(452))
                g1_mulAccC(_pVk, IC8x, IC8y, calldataload(484))
                g1_mulAccC(_pVk, IC9x, IC9y, calldataload(516))
                g1_mulAccC(_pVk, IC10x, IC10y, calldataload(548))
                g1_mulAccC(_pVk, IC11x, IC11y, calldataload(580))
                g1_mulAccC(_pVk, IC12x, IC12y, calldataload(612))
                g1_mulAccC(_pVk, IC13x, IC13y, calldataload(644))
                g1_mulAccC(_pVk, IC14x, IC14y, calldataload(676))
                g1_mulAccC(_pVk, IC15x, IC15y, calldataload(708))

                mstore(_pPairing, calldataload(4))
                mstore(add(_pPairing, 32), mod(sub(q, calldataload(36)), q))

                mstore(add(_pPairing, 64), calldataload(68))
                mstore(add(_pPairing, 96), calldataload(100))
                mstore(add(_pPairing, 128), calldataload(132))
                mstore(add(_pPairing, 160), calldataload(164))

                mstore(add(_pPairing, 192), alphax)
                mstore(add(_pPairing, 224), alphay)

                mstore(add(_pPairing, 256), betax1)
                mstore(add(_pPairing, 288), betax2)
                mstore(add(_pPairing, 320), betay1)
                mstore(add(_pPairing, 352), betay2)

                mstore(add(_pPairing, 384), mload(add(pMem, pVk)))
                mstore(add(_pPairing, 416), mload(add(pMem, add(pVk, 32))))

                mstore(add(_pPairing, 448), gammax1)
                mstore(add(_pPairing, 480), gammax2)
                mstore(add(_pPairing, 512), gammay1)
                mstore(add(_pPairing, 544), gammay2)

                mstore(add(_pPairing, 576), calldataload(196))
                mstore(add(_pPairing, 608), calldataload(228))

                mstore(add(_pPairing, 640), deltax1)
                mstore(add(_pPairing, 672), deltax2)
                mstore(add(_pPairing, 704), deltay1)
                mstore(add(_pPairing, 736), deltay2)

                let success := staticcall(sub(gas(), 2000), 8, _pPairing, 768, _pPairing, 0x20)
                isOk := and(success, mload(_pPairing))
            }

            let pMem := mload(0x40)
            mstore(0x40, add(pMem, pLastMem))

            checkField(calldataload(260))
            checkField(calldataload(292))
            checkField(calldataload(324))
            checkField(calldataload(356))
            checkField(calldataload(388))
            checkField(calldataload(420))
            checkField(calldataload(452))
            checkField(calldataload(484))
            checkField(calldataload(516))
            checkField(calldataload(548))
            checkField(calldataload(580))
            checkField(calldataload(612))
            checkField(calldataload(644))
            checkField(calldataload(676))
            checkField(calldataload(708))

            let isValid := checkPairingFlat(pMem)
            mstore(0, isValid)
            return(0, 0x20)
        }
    }

    function submitSelectedProof(
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
        if (
            pubSignals[3] != 1 ||
            pubSignals[0] != pubSignals[14] ||
            pubSignals[1] != pubSignals[12] ||
            pubSignals[2] != pubSignals[13]
        ) {
            revert InvalidPublicSignals();
        }
        if (!verifyProof(a, b, c, pubSignals)) revert InvalidProof();

        key = submissionKey(experimentId, groupId, proofId);
        if (submissions[key].submittedAt != 0) revert ProofAlreadyExists();

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

        emit ProofSubmitted(key, experimentId, groupId, proofId, msg.sender);
    }

    function submitSelectedProofFlat(
        bytes32 experimentId,
        bytes32 groupId,
        bytes32 resultDigest,
        bytes32 proofId,
        bytes32 statementDigestSha256,
        string calldata resultURI,
        string calldata proofURI,
        uint256 a0,
        uint256 a1,
        uint256 b00,
        uint256 b01,
        uint256 b10,
        uint256 b11,
        uint256 c0,
        uint256 c1,
        uint256 pub0,
        uint256 pub1,
        uint256 pub2,
        uint256 pub3,
        uint256 pub4,
        uint256 pub5,
        uint256 pub6,
        uint256 pub7,
        uint256 pub8,
        uint256 pub9,
        uint256 pub10,
        uint256 pub11,
        uint256 pub12,
        uint256 pub13,
        uint256 pub14
    ) external returns (bytes32 key) {
        if (!this.verifySelectedProofFlat(
            a0,
            a1,
            b00,
            b01,
            b10,
            b11,
            c0,
            c1,
            pub0,
            pub1,
            pub2,
            pub3,
            pub4,
            pub5,
            pub6,
            pub7,
            pub8,
            pub9,
            pub10,
            pub11,
            pub12,
            pub13,
            pub14
        )) revert InvalidProof();

        key = submissionKey(experimentId, groupId, proofId);
        if (submissions[key].submittedAt != 0) revert ProofAlreadyExists();

        submissions[key] = Submission({
            experimentId: experimentId,
            groupId: groupId,
            resultDigest: resultDigest,
            proofId: proofId,
            statementDigestSha256: statementDigestSha256,
            outputStatementHashField: pub0,
            outputWeightedSum: pub1,
            outputTotalWeight: pub2,
            outputIsValid: pub3,
            claimedWeightedSum: pub12,
            claimedTotalWeight: pub13,
            inputStatementHashField: pub14,
            resultURI: resultURI,
            proofURI: proofURI,
            submitter: msg.sender,
            submittedAt: uint64(block.timestamp)
        });

        emit ProofSubmitted(key, experimentId, groupId, proofId, msg.sender);
    }

    function verifyProof(uint[2] calldata _pA, uint[2][2] calldata _pB, uint[2] calldata _pC, uint[15] calldata _pubSignals) public view returns (bool) {
        assembly ("memory-safe") {
            function checkField(v) {
                if iszero(lt(v, r)) {
                    mstore(0, 0)
                    return(0, 0x20)
                }
            }
            
            // G1 function to multiply a G1 value(x,y) to value in an address
            function g1_mulAccC(pR, x, y, s) {
                let success
                let mIn := mload(0x40)
                mstore(mIn, x)
                mstore(add(mIn, 32), y)
                mstore(add(mIn, 64), s)

                success := staticcall(sub(gas(), 2000), 7, mIn, 96, mIn, 64)

                if iszero(success) {
                    mstore(0, 0)
                    return(0, 0x20)
                }

                mstore(add(mIn, 64), mload(pR))
                mstore(add(mIn, 96), mload(add(pR, 32)))

                success := staticcall(sub(gas(), 2000), 6, mIn, 128, pR, 64)

                if iszero(success) {
                    mstore(0, 0)
                    return(0, 0x20)
                }
            }

            function checkPairing(pA, pB, pC, pubSignals, pMem) -> isOk {
                let _pPairing := add(pMem, pPairing)
                let _pVk := add(pMem, pVk)

                mstore(_pVk, IC0x)
                mstore(add(_pVk, 32), IC0y)

                // Compute the linear combination vk_x
                
                g1_mulAccC(_pVk, IC1x, IC1y, calldataload(add(pubSignals, 0)))
                
                g1_mulAccC(_pVk, IC2x, IC2y, calldataload(add(pubSignals, 32)))
                
                g1_mulAccC(_pVk, IC3x, IC3y, calldataload(add(pubSignals, 64)))
                
                g1_mulAccC(_pVk, IC4x, IC4y, calldataload(add(pubSignals, 96)))
                
                g1_mulAccC(_pVk, IC5x, IC5y, calldataload(add(pubSignals, 128)))
                
                g1_mulAccC(_pVk, IC6x, IC6y, calldataload(add(pubSignals, 160)))
                
                g1_mulAccC(_pVk, IC7x, IC7y, calldataload(add(pubSignals, 192)))
                
                g1_mulAccC(_pVk, IC8x, IC8y, calldataload(add(pubSignals, 224)))
                
                g1_mulAccC(_pVk, IC9x, IC9y, calldataload(add(pubSignals, 256)))
                
                g1_mulAccC(_pVk, IC10x, IC10y, calldataload(add(pubSignals, 288)))
                
                g1_mulAccC(_pVk, IC11x, IC11y, calldataload(add(pubSignals, 320)))
                
                g1_mulAccC(_pVk, IC12x, IC12y, calldataload(add(pubSignals, 352)))
                
                g1_mulAccC(_pVk, IC13x, IC13y, calldataload(add(pubSignals, 384)))
                
                g1_mulAccC(_pVk, IC14x, IC14y, calldataload(add(pubSignals, 416)))
                
                g1_mulAccC(_pVk, IC15x, IC15y, calldataload(add(pubSignals, 448)))
                

                // -A
                mstore(_pPairing, calldataload(pA))
                mstore(add(_pPairing, 32), mod(sub(q, calldataload(add(pA, 32))), q))

                // B
                mstore(add(_pPairing, 64), calldataload(pB))
                mstore(add(_pPairing, 96), calldataload(add(pB, 32)))
                mstore(add(_pPairing, 128), calldataload(add(pB, 64)))
                mstore(add(_pPairing, 160), calldataload(add(pB, 96)))

                // alpha1
                mstore(add(_pPairing, 192), alphax)
                mstore(add(_pPairing, 224), alphay)

                // beta2
                mstore(add(_pPairing, 256), betax1)
                mstore(add(_pPairing, 288), betax2)
                mstore(add(_pPairing, 320), betay1)
                mstore(add(_pPairing, 352), betay2)

                // vk_x
                mstore(add(_pPairing, 384), mload(add(pMem, pVk)))
                mstore(add(_pPairing, 416), mload(add(pMem, add(pVk, 32))))


                // gamma2
                mstore(add(_pPairing, 448), gammax1)
                mstore(add(_pPairing, 480), gammax2)
                mstore(add(_pPairing, 512), gammay1)
                mstore(add(_pPairing, 544), gammay2)

                // C
                mstore(add(_pPairing, 576), calldataload(pC))
                mstore(add(_pPairing, 608), calldataload(add(pC, 32)))

                // delta2
                mstore(add(_pPairing, 640), deltax1)
                mstore(add(_pPairing, 672), deltax2)
                mstore(add(_pPairing, 704), deltay1)
                mstore(add(_pPairing, 736), deltay2)


                let success := staticcall(sub(gas(), 2000), 8, _pPairing, 768, _pPairing, 0x20)

                isOk := and(success, mload(_pPairing))
            }

            let pMem := mload(0x40)
            mstore(0x40, add(pMem, pLastMem))

            // Validate that all evaluations ∈ F
            
            checkField(calldataload(add(_pubSignals, 0)))
            
            checkField(calldataload(add(_pubSignals, 32)))
            
            checkField(calldataload(add(_pubSignals, 64)))
            
            checkField(calldataload(add(_pubSignals, 96)))
            
            checkField(calldataload(add(_pubSignals, 128)))
            
            checkField(calldataload(add(_pubSignals, 160)))
            
            checkField(calldataload(add(_pubSignals, 192)))
            
            checkField(calldataload(add(_pubSignals, 224)))
            
            checkField(calldataload(add(_pubSignals, 256)))
            
            checkField(calldataload(add(_pubSignals, 288)))
            
            checkField(calldataload(add(_pubSignals, 320)))
            
            checkField(calldataload(add(_pubSignals, 352)))
            
            checkField(calldataload(add(_pubSignals, 384)))
            
            checkField(calldataload(add(_pubSignals, 416)))
            
            checkField(calldataload(add(_pubSignals, 448)))
            

            // Validate all evaluations
            let isValid := checkPairing(_pA, _pB, _pC, _pubSignals, pMem)

            mstore(0, isValid)
             return(0, 0x20)
         }
     }
 }
