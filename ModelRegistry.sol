// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ModelRegistry {
    address public serverAdmin; 
    string public latestGlobalCID; 
    uint256 public currentRound; 
    
    mapping(address => uint256) public nodeContributions;
    
    event ModelUpdated(uint256 round, string cid);
    event ModelAccessed(address client, uint256 round);

    constructor() {
        serverAdmin = msg.sender; 
        currentRound = 0;
    }

    modifier onlyServer() {
        require(msg.sender == serverAdmin, "Only Server can perform this action");
        _;
    }

    function updateGlobalModel(string memory _cid, address[] memory _participants) public onlyServer {
        latestGlobalCID = _cid;
        currentRound++;
        for(uint i = 0; i < _participants.length; i++) {
            nodeContributions[_participants[i]]++;
        }
        emit ModelUpdated(currentRound, _cid);
    }

    function requestModelAccess(uint256 _requiredThreshold) public returns (string memory) {
        require(nodeContributions[msg.sender] >= _requiredThreshold, "Insufficient contribution");
        emit ModelAccessed(msg.sender, currentRound);
        return latestGlobalCID; 
    }
}