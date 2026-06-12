pragma solidity ^0.8.20;

contract TruthPairingProbe {
    uint256 constant q = 21888242871839275222246405745257275088696311157297823662689037894645226208583;

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

    function pairingJeff1() external view returns (bool ok) {
        bytes memory input = hex"1c76476f4def4bb94541d57ebba1193381ffa7aa76ada664dd31c16024c43f593034dd2920f673e204fee2811c678745fc819b55d3e9d294e45c9b03a76aef41209dd15ebff5d46c4bd888e51a93cf99a7329636c63514396b4a452003a35bf704bf11ca01483bfa8b34b43561848d28905960114c8ac04049af4b6315a416782bb8324af6cfc93537a2ad1a445cfd0ca2a71acd7ac41fadbf933c2a51be344d120a2a4cf30c1bf9845f20c6fe39e07ea2cce61f0c9bb048165fe5e4de877550111e129f1cf1097710d41c4ac70fcdfa5ba2023c6ff1cbeac322de49d1b6df7c2032c61a830e3c17286de9462bf242fca2883585b93870a73853face6a6bf411198e9393920d483a7260bfb731fb5d25f1aa493335a9e71297e485b7aef312c21800deef121f1e76426a00665e5c4479674322d4f75edadd46debd5cd992f6ed090689d0585ff075ec9e99ad690c3395bc4b313370b38ef355acdadcd122975b12c85ea5db8c6deb4aab71808dcb408fe3d1e7690c43d37b4ce6cc0166fa7daa";
        bytes memory output = new bytes(32);
        bool success;
        assembly ("memory-safe") {
            success := staticcall(sub(gas(), 2000), 8, add(input, 32), mload(input), add(output, 32), 0x20)
        }
        return success && bytes32(output) == bytes32(uint256(1));
    }

    function truthGroup0CurrentCurrent() external view returns (bool) {
        return _truthGroup0(
            8376411563965595370544072280350415026087655622558240148526564304012612523604,
            11575562001102525520442587641016281290457738670557334636093387922717627873361,
            2498974281824056392743547888464791260434557105223256669576624756099391150828,
            12601459835917826523729180032435540672330753259695147957098546296332458836077,
            11559732032986387107991004021392285783925812861821192530917403151452391805634,
            10857046999023057135944570762232829481370756359578518086990519993285655852781,
            4082367875863433681332203403145435568316851327593401208105741076214120093531,
            8495653923123431417604973247489272438418190587263600148770280649306958101930
        );
    }

    function truthGroup0RawRaw() external view returns (bool) {
        return _truthGroup0(
            11575562001102525520442587641016281290457738670557334636093387922717627873361,
            8376411563965595370544072280350415026087655622558240148526564304012612523604,
            12601459835917826523729180032435540672330753259695147957098546296332458836077,
            2498974281824056392743547888464791260434557105223256669576624756099391150828,
            10857046999023057135944570762232829481370756359578518086990519993285655852781,
            11559732032986387107991004021392285783925812861821192530917403151452391805634,
            8495653923123431417604973247489272438418190587263600148770280649306958101930,
            4082367875863433681332203403145435568316851327593401208105741076214120093531
        );
    }

    function truthGroup0RawCurrent() external view returns (bool) {
        return _truthGroup0(
            11575562001102525520442587641016281290457738670557334636093387922717627873361,
            8376411563965595370544072280350415026087655622558240148526564304012612523604,
            12601459835917826523729180032435540672330753259695147957098546296332458836077,
            2498974281824056392743547888464791260434557105223256669576624756099391150828,
            11559732032986387107991004021392285783925812861821192530917403151452391805634,
            10857046999023057135944570762232829481370756359578518086990519993285655852781,
            4082367875863433681332203403145435568316851327593401208105741076214120093531,
            8495653923123431417604973247489272438418190587263600148770280649306958101930
        );
    }

    function truthGroup0CurrentRaw() external view returns (bool) {
        return _truthGroup0(
            8376411563965595370544072280350415026087655622558240148526564304012612523604,
            11575562001102525520442587641016281290457738670557334636093387922717627873361,
            2498974281824056392743547888464791260434557105223256669576624756099391150828,
            12601459835917826523729180032435540672330753259695147957098546296332458836077,
            10857046999023057135944570762232829481370756359578518086990519993285655852781,
            11559732032986387107991004021392285783925812861821192530917403151452391805634,
            8495653923123431417604973247489272438418190587263600148770280649306958101930,
            4082367875863433681332203403145435568316851327593401208105741076214120093531
        );
    }

    function _truthGroup0(
        uint256 b0,
        uint256 b1,
        uint256 b2,
        uint256 b3,
        uint256 g2x1,
        uint256 g2x2,
        uint256 g2y1,
        uint256 g2y2
    ) private view returns (bool ok) {
        assembly ("memory-safe") {
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

            let pMem := mload(0x40)
            let pVk := pMem
            let pPairing := add(pMem, 128)
            mstore(0x40, add(pMem, 896))

            mstore(pVk, IC0x)
            mstore(add(pVk, 32), IC0y)
            g1_mulAccC(pVk, IC1x, IC1y, 3828124907)
            g1_mulAccC(pVk, IC2x, IC2y, 0)
            g1_mulAccC(pVk, IC3x, IC3y, 3814769)
            g1_mulAccC(pVk, IC4x, IC4y, 1)
            g1_mulAccC(pVk, IC5x, IC5y, 0)
            g1_mulAccC(pVk, IC6x, IC6y, 0)
            g1_mulAccC(pVk, IC7x, IC7y, 0)
            g1_mulAccC(pVk, IC8x, IC8y, 0)
            g1_mulAccC(pVk, IC9x, IC9y, 954180)
            g1_mulAccC(pVk, IC10x, IC10y, 950609)
            g1_mulAccC(pVk, IC11x, IC11y, 954180)
            g1_mulAccC(pVk, IC12x, IC12y, 955800)
            g1_mulAccC(pVk, IC13x, IC13y, 0)
            g1_mulAccC(pVk, IC14x, IC14y, 3814769)
            g1_mulAccC(pVk, IC15x, IC15y, 3828124907)

            mstore(pPairing, 4406069018237011808587399095241934984934375945646791007213517264560616961953)
            mstore(add(pPairing, 32), mod(sub(q, 17954210191385827972846702909162192271034427579562506070148469828493671526929), q))
            mstore(add(pPairing, 64), b0)
            mstore(add(pPairing, 96), b1)
            mstore(add(pPairing, 128), b2)
            mstore(add(pPairing, 160), b3)

            mstore(add(pPairing, 192), 1)
            mstore(add(pPairing, 224), 2)
            mstore(add(pPairing, 256), g2x1)
            mstore(add(pPairing, 288), g2x2)
            mstore(add(pPairing, 320), g2y1)
            mstore(add(pPairing, 352), g2y2)

            mstore(add(pPairing, 384), mload(pVk))
            mstore(add(pPairing, 416), mload(add(pVk, 32)))

            mstore(add(pPairing, 448), g2x1)
            mstore(add(pPairing, 480), g2x2)
            mstore(add(pPairing, 512), g2y1)
            mstore(add(pPairing, 544), g2y2)

            mstore(add(pPairing, 576), 12305918778730353911201495223420365429470569186549317742885441408378995259748)
            mstore(add(pPairing, 608), 21344371481030164378822603800068419247663721618907880794334652557944923594880)

            mstore(add(pPairing, 640), g2x1)
            mstore(add(pPairing, 672), g2x2)
            mstore(add(pPairing, 704), g2y1)
            mstore(add(pPairing, 736), g2y2)

            let success := staticcall(sub(gas(), 2000), 8, pPairing, 768, pPairing, 0x20)
            ok := and(success, mload(pPairing))
        }
    }
}
