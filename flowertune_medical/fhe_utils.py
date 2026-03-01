import tenseal as ts
import os
import pickle
import torch
from safetensors.torch import load_file
import time

class FullFHEHandler:
    def __init__(self, key_path="global_fhe_keys.pkl"):
        """初始化时不再随机生成密钥，而是加载全局统一的密钥文件"""
        if not os.path.exists(key_path):
            raise FileNotFoundError(f"🚨 找不到全局密钥文件 {key_path}！请先运行 generate_keys.py")
            
        print(f"🔑 [FHE] 正在加载全局密钥: {key_path}")
        with open(key_path, "rb") as f:
            context_bytes = f.read()
        self.context = ts.context_from(context_bytes)

    def encrypt_full_model(self, temp_dir, output_dir):
        # ... (这里的代码和之前完全一样，只是 context 变成了共享的) ...
        safetensors_path = os.path.join(temp_dir, "adapter_model.safetensors")
        bin_path = os.path.join(temp_dir, "adapter_model.bin")
        
        if os.path.exists(safetensors_path):
            weights = load_file(safetensors_path)
            plaintext_path = safetensors_path
        elif os.path.exists(bin_path):
            weights = torch.load(bin_path, map_location="cpu", weights_only=True) 
            plaintext_path = bin_path
        else:
            raise FileNotFoundError(f"🚨 找不到模型文件！")

        encrypted_weights = {}
        shapes_info = {}
        
        start_time = time.time()
        for layer_name, tensor in weights.items():
            flat_list = tensor.flatten().tolist()
            enc_vector = ts.ckks_vector(self.context, flat_list)
            encrypted_weights[layer_name] = enc_vector.serialize()
            shapes_info[layer_name] = tensor.shape
            
        enc_data = {
            "weights": encrypted_weights,
            "shapes": shapes_info,
            "public_context": self.context.serialize(save_secret_key=False)
        }
        
        enc_file_path = os.path.join(output_dir, "full_encrypted_model.pkl")
        with open(enc_file_path, "wb") as f:
            pickle.dump(enc_data, f)
            
        print(f"✅ [FHE] 加密完成！耗时: {time.time() - start_time:.2f} 秒")
        return plaintext_path

    # ==========================================
    # 🌟 [新增核心功能] 密文解密还原
    # ==========================================
    def decrypt_model(self, enc_file_path):
        """读取下载的密文文件，使用本地私钥解密，并还原为 PyTorch 张量字典"""
        print(f"🔓 [FHE] 正在读取全局密文进行解密...")
        start_time = time.time()
        
        with open(enc_file_path, "rb") as f:
            enc_data = pickle.load(f)
            
        encrypted_weights = enc_data["weights"]
        shapes_info = enc_data["shapes"]
        
        decrypted_state_dict = {}
        
        for layer_name, enc_bytes in encrypted_weights.items():
            # 1. 将字节流反序列化为 TenSEAL 密文对象
            enc_vector = ts.ckks_vector_from(self.context, enc_bytes)
            
            # 2. 核心：执行同态解密！(因为 self.context 里有私钥)
            flat_list = enc_vector.decrypt()
            
            # 3. 还原回原本的 Tensor 形状
            tensor_shape = shapes_info[layer_name]
            tensor = torch.tensor(flat_list).reshape(tensor_shape)
            
            # 由于 CKKS 方案解密出来是 float64 的近似浮点数，
            # 需要转回大模型常用的 bfloat16 或 float32
            decrypted_state_dict[layer_name] = tensor.to(torch.bfloat16)
            
        print(f"✅ [FHE] 解密还原完成！总耗时: {time.time() - start_time:.2f} 秒")
        return decrypted_state_dict