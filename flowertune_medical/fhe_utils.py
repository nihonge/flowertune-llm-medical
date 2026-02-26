import tenseal as ts
import os
import pickle
import torch
from safetensors.torch import load_file
import time

class FullFHEHandler:
    def __init__(self):
        # 1. 初始化 CKKS 同态加密上下文
        self.context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
        self.context.global_scale = 2**40
        self.context.generate_galois_keys()

    def encrypt_full_model(self, temp_dir, output_dir):
        # ==========================================
        # 🌟 终极自适应：精确打击 bin 文件
        # ==========================================
        safetensors_path = os.path.join(temp_dir, "adapter_model.safetensors")
        bin_path = os.path.join(temp_dir, "adapter_model.bin")
        
        if os.path.exists(safetensors_path):
            print(f"\n🔒 [FHE] 发现 safetensors 格式，正在读取: {safetensors_path}")
            weights = load_file(safetensors_path)
            plaintext_path = safetensors_path
        elif os.path.exists(bin_path):
            print(f"\n🔒 [FHE] 发现 bin 格式，正在读取: {bin_path}")
            # 🌟 绕过 OS Error 19 的终极解法：使用标准的 torch.load
            weights = torch.load(bin_path, map_location="cpu", weights_only=True) 
            plaintext_path = bin_path
        else:
            files_in_dir = os.listdir(temp_dir) if os.path.exists(temp_dir) else "目录不存在!"
            raise FileNotFoundError(f"🚨 找不到模型文件！目录里的内容是: {files_in_dir}")

        # ==========================================
        
        encrypted_weights = {}
        shapes_info = {}
        
        print(f"🔥 [FHE] 开始逐层全量加密 (共 {len(weights)} 层) ...")
        start_time = time.time()
        
        for layer_name, tensor in weights.items():
            print(f"  -> 正在同态加密: {layer_name} (Shape: {tensor.shape})")
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
            
        print(f"✅ [FHE] 全量加密完成！总耗时: {time.time() - start_time:.2f} 秒")
        print(f"💾 密文已保存至: {enc_file_path}")
        
        # 返回明文文件的路径，让外面去删除它
        return plaintext_path