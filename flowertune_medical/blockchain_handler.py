import json
from web3 import Web3

class BlockchainHandler:
    def __init__(self, rpc_url="http://127.0.0.1:8545"):
        # 连接到你的本地 Ganache 链
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        if not self.w3.is_connected():
            raise ConnectionError("❌ 无法连接到区块链！请确认 Ganache 已在 8545 端口运行。")
        
        # 刚才你在 Remix 部署的专属合约地址！
        self.contract_address = self.w3.to_checksum_address("0x20B6c231C80923259f33dee130384e8B3B451685")
        
        # 分配账号：第 0 个账号当 Server，第 1 到 9 个当 Client
        self.server_account = self.w3.eth.accounts[0]
        self.client_accounts = self.w3.eth.accounts[1:] 
        
        # 极简版的 ABI (不用自己去配编译环境了)
        self.contract_abi = json.loads('''[
            {"inputs":[],"stateMutability":"nonpayable","type":"constructor"},
            {"anonymous":false,"inputs":[{"indexed":false,"internalType":"address","name":"client","type":"address"},{"indexed":false,"internalType":"uint256","name":"round","type":"uint256"}],"name":"ModelAccessed","type":"event"},
            {"anonymous":false,"inputs":[{"indexed":false,"internalType":"uint256","name":"round","type":"uint256"},{"indexed":false,"internalType":"string","name":"cid","type":"string"}],"name":"ModelUpdated","type":"event"},
            {"inputs":[],"name":"currentRound","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
            {"inputs":[],"name":"latestGlobalCID","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},
            {"inputs":[{"internalType":"address","name":"","type":"address"}],"name":"nodeContributions","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
            {"inputs":[{"internalType":"uint256","name":"_requiredThreshold","type":"uint256"}],"name":"requestModelAccess","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"nonpayable","type":"function"},
            {"inputs":[],"name":"serverAdmin","outputs":[{"internalType":"address","name":"","type":"address"}],"stateMutability":"view","type":"function"},
            {"inputs":[{"internalType":"string","name":"_cid","type":"string"},{"internalType":"address[]","name":"_participants","type":"address[]"}],"name":"updateGlobalModel","outputs":[],"stateMutability":"nonpayable","type":"function"}
        ]''')
        
        self.contract = self.w3.eth.contract(address=self.contract_address, abi=self.contract_abi)

    def upload_cid(self, cid, participant_addresses):
        """Server 调用：存证全局模型 CID，并给本轮参与的节点加贡献度"""
        print(f"🔗 [Blockchain] 正在将最新模型 CID 上链存证...")
        tx_hash = self.contract.functions.updateGlobalModel(cid, participant_addresses).transact({
            'from': self.server_account
        })
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"✅ [Blockchain] 存证成功！交易已被打包进区块 (消耗 Gas: {receipt.gasUsed})")

    def request_cid(self, client_address, threshold=1):
        """Client 调用：请求下载模型，触发智能合约的贡献度校验"""
        print(f"🔐 [Blockchain] 节点 {client_address[:8]}... 发起防搭便车确权校验...")
        try:
            # call() 方法是在本地模拟执行，不消耗 gas，如果贡献度不够会直接抛出异常
            cid = self.contract.functions.requestModelAccess(threshold).call({
                'from': client_address
            })
            print(f"🎉 [Blockchain] 确权通过！从链上获取到授权的 CID: {cid}")
            return cid
        except Exception as e:
            print(f"❌ [Blockchain] 确权失败：贡献度不足，拒绝提供模型 CID！")
            return None