import tenseal as ts
import os

def generate_and_save_keys(path="global_fhe_keys.pkl"):
    print("🔑 正在生成全局 FHE 密钥对 (包含公钥和私钥)...")
    context = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
    context.global_scale = 2**40
    context.generate_galois_keys()
    
    # 保存完整的上下文（千万注意：这里 save_secret_key=True）
    with open(path, "wb") as f:
        f.write(context.serialize(save_secret_key=True))
    print(f"✅ 密钥对已保存至: {os.path.abspath(path)}")
    print("💡 请确保所有的 Client 都能读取到这个文件。")

if __name__ == "__main__":
    generate_and_save_keys()