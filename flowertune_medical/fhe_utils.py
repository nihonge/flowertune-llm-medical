import os
import pickle
import numpy as np
import tenseal as ts
import torch

class SelectiveFHEHandler:
    def __init__(self, key_path="global_fhe_keys.pkl"):
        self.key_path = key_path
        if not os.path.exists(key_path):
            self._generate_keys()
        with open(self.key_path, "rb") as f:
            keys = pickle.load(f)
        self.context = ts.context_from(keys["secret_context"])

    def _generate_keys(self):
        print(f"🔑 [FHE] 正在生成全局 TenSEAL 密钥...")
        context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
        context.global_scale = 2**40
        context.generate_galois_keys()
        with open(self.key_path, "wb") as f:
            pickle.dump({
                "secret_context": context.serialize(save_secret_key=True),
                "public_context": context.serialize(save_secret_key=False)
            }, f)

    def should_encrypt(self, layer_name: str) -> bool:
        """🌟 核心创新点：智能分流路由器"""
        # 只拦截包含 q_proj 或 v_proj 的层进行重度同态加密
        return "q_proj" in layer_name or "v_proj" in layer_name

    def encrypt_full_model(self, model_dir, output_dir):
        import safetensors.torch
        model_path = os.path.join(model_dir, "adapter_model.safetensors")
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, "adapter_model.bin")
        
        if model_path.endswith(".safetensors"):
            state_dict = safetensors.torch.load_file(model_path)
        else:
            state_dict = torch.load(model_path, weights_only=True)

        encrypted_weights = {}
        unencrypted_weights = {}
        shapes_info = {}

        print("🔒 [Selective FHE] 启动选择性加密引擎...")
        import time
        start_time = time.time()

        for name, tensor in state_dict.items():
            flat_array = tensor.cpu().numpy().astype(np.float64).flatten()
            shapes_info[name] = tuple(tensor.shape) # 确保shape可以被序列化

            if self.should_encrypt(name):
                # 敏感层 -> 走同态加密通道
                enc_vector = ts.ckks_vector(self.context, flat_array.tolist())
                encrypted_weights[name] = enc_vector.serialize()
            else:
                # 非敏感层 -> 走轻量级明文通道
                unencrypted_weights[name] = flat_array

        output_data = {
            "weights": encrypted_weights,                  # 只有 q/v 层的密文
            "unencrypted_weights": unencrypted_weights,    # 其余层的明文数组
            "shapes": shapes_info,
            "public_context": self.context.serialize(save_secret_key=False)
        }

        out_path = os.path.join(output_dir, "full_encrypted_model.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(output_data, f)
        
        print(f"✅ [Selective FHE] 分流加密完成！耗时: {time.time()-start_time:.2f} 秒 (体积大幅缩减)")
        return model_path

    def decrypt_model(self, fhe_path):
        with open(fhe_path, "rb") as f:
            enc_data = pickle.load(f)

        decrypted_state_dict = {}
        shapes_info = enc_data["shapes"]
        
        import time
        start_time = time.time()

        # 1. 还原并解密敏感层
        for name, enc_bytes in enc_data["weights"].items():
            enc_vector = ts.ckks_vector_from(self.context, enc_bytes)
            dec_list = enc_vector.decrypt()
            dec_array = np.array(dec_list, dtype=np.float32)
            decrypted_state_dict[name] = torch.tensor(dec_array.reshape(shapes_info[name]))
        
        # 2. 直接还原非敏感明文层
        for name, flat_array in enc_data.get("unencrypted_weights", {}).items():
            decrypted_state_dict[name] = torch.tensor(flat_array.astype(np.float32).reshape(shapes_info[name]))

        print(f"✅ [Selective FHE] 混合解密还原完成！总耗时: {time.time()-start_time:.2f} 秒")
        return decrypted_state_dict