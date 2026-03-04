from blockchain_handler import BlockchainHandler

print("正在初始化区块链连接...")
bc = BlockchainHandler()

# 模拟：Server 刚刚完成了一轮聚合，拿到了一个假装的 IPFS CID
fake_new_cid = "QmTest1234567890abcdefMedicalModelRound13"
# 模拟：这一轮有 Client 1 和 Client 2 参与了训练
active_clients = [bc.client_accounts[0], bc.client_accounts[1]]

print("\n--- 1. Server 上链阶段 ---")
bc.upload_cid(fake_new_cid, active_clients)

print("\n--- 2. 合法 Client 拉取阶段 (有贡献度) ---")
# 刚才参与了训练的 Client 1 试图拉取模型
authorized_cid = bc.request_cid(bc.client_accounts[0], threshold=1)

print("\n--- 3. 恶意 Client 拉取阶段 (没有贡献度) ---")
# 根本没参与训练的 Client 3 (想搭便车) 试图拉取模型
rejected_cid = bc.request_cid(bc.client_accounts[2], threshold=1)