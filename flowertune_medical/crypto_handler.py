import os
import time
from cryptography.fernet import Fernet

class ABE_AES_Engine:
    def __init__(self):
        # 模拟 ABE 授权中心的全局主密钥 (仅做流程模拟)
        self.master_key = b"Cyber_Authority_Master_Key_00000"

    def server_encrypt_model(self, input_filepath, output_filepath, round_num):
        """Server 端：生成 AES 密钥锁定模型，并用 ABE 策略封装 AES 密钥"""
        print(f"🔐 [Server-Crypto] 正在生成随机 AES-256 密钥锁定全局模型...")
        start_time = time.time()
        
        # 1. 生成随机 AES 密钥
        aes_key = Fernet.generate_key()
        cipher_suite = Fernet(aes_key)
        
        # 2. 读取 FHE 密文模型并进行 AES 加密
        with open(input_filepath, 'rb') as f:
            fhe_data = f.read()
        encrypted_model_data = cipher_suite.encrypt(fhe_data)
        
        # 3. 模拟 DMA-CP-ABE 封装 AES 密钥 (访问策略: 要求本轮贡献度验证通过)
        access_policy = f"Policy: (Role:Client) AND (Contribution >= 1) for Round {round_num}"
        print(f"🌳 [Server-Crypto] 构建 ABE 访问策略树: {access_policy}")
        
        # 真实环境会使用 ABE 公钥加密，此处用模拟的 policy_header 打包
        abe_encapsulated_key = f"ABE_CT||{access_policy}||".encode() + aes_key
        
        # 4. 将 ABE 密文密钥和 AES 密文模型合并写入文件
        with open(output_filepath, 'wb') as f:
            # 前 256 字节存放 ABE 密钥密文，后面存放真正的大文件
            f.write(abe_encapsulated_key.ljust(256, b'\0')) 
            f.write(encrypted_model_data)
            
        print(f"✅ [Server-Crypto] AES 锁定与 ABE 策略封装完成！耗时: {time.time()-start_time:.3f}s")
        return True

    def client_decrypt_model(self, file_path, my_eth_address, blockchain_handler):
        """Client 端：通过区块链确权申请 ABE 密钥，解锁 AES 并还原模型"""
        print(f"🔐 [Client-Crypto] 开始向授权中心申请 ABE 属性私钥...")
        start_time = time.time()
        
        # 🌟 终极确权点：查验智能合约账本！
        is_authorized = blockchain_handler.request_cid(my_eth_address, threshold=1)
        
        if not is_authorized:
            print("🚫 [Client-Crypto] 授权中心拒绝发钥！原因：智能合约查验贡献度不足！")
            return False
            
        print("🎉 [Client-Crypto] 智能合约背书通过！成功获取 ABE 属性解密私钥 SK！")
        
        try:
            with open(file_path, 'rb') as f:
                abe_encapsulated_key_padded = f.read(256)
                encrypted_model_data = f.read()
                
            # 1. 使用申请到的 ABE 私钥解开 ABE 密文，提取真正的 AES 密钥
            abe_encapsulated_key = abe_encapsulated_key_padded.rstrip(b'\0')
            aes_key = abe_encapsulated_key.split(b"||")[2]
            
            # 2. 使用 AES 密钥解锁全局大模型
            cipher_suite = Fernet(aes_key)
            decrypted_fhe_data = cipher_suite.decrypt(encrypted_model_data)
            
            # 3. 覆盖写入（变回纯粹的 FHE 密文，供下游 TenSEAL 继续解密）
            with open(file_path, 'wb') as f:
                f.write(decrypted_fhe_data)
                
            print(f"🔓 [Client-Crypto] AES 解锁成功，已还原为底层 FHE 密文！耗时: {time.time()-start_time:.3f}s")
            return True
        except Exception as e:
            print(f"❌ [Client-Crypto] 密钥解封装失败: {e}")
            return False