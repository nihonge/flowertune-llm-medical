import os
import shutil
import pickle
import time
import numpy as np

# 使用非交互式后端，防止在无界面的服务器上画图报错
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns

from collections.abc import Iterable
from logging import INFO, WARN
from typing import Optional

from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.common import log
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg
from flowertune_medical.ipfs_handler import IPFSHandler
import tenseal as ts

# 🌟 区块链融合与密码学引擎
from flowertune_medical.blockchain_handler import BlockchainHandler
from flowertune_medical.crypto_handler import ABE_AES_Engine

class CommunicationTracker:
    def __init__(self):
        self.curr_comm_cost = 0.0

    def track(self, messages: Iterable[Message]):
        comm_cost = (
            sum(
                record.count_bytes()
                for msg in messages
                if msg.has_content()
                for record in msg.content.array_records.values()
            )
            / 1024**2
        )

        self.curr_comm_cost += comm_cost
        log(
            INFO,
            "Communication budget: used %.2f MB (+%.2f MB this round) / 200,000 MB",
            self.curr_comm_cost,
            comm_cost,
        )


class FlowerTuneLlm(FedAvg):
    """Selective FHE-enabled Strategy with Blockchain, ABE & Auto-Plotting."""

    def __init__(self, **kwargs):
        # 🌟 安全截获从 server_app.py 传过来的新增参数，防止传给父类报错
        self.num_rounds = kwargs.pop("num_rounds", 30)
        self.save_path = kwargs.pop("save_path", "./results")
        
        super().__init__(**kwargs)
        self.comm_tracker = CommunicationTracker()
        self.ipfs = IPFSHandler()
        self.current_global_cid = "FAIL"
        
        # 🌟 用于自动画图的字典
        self.train_losses_history = {}
        
        # 🌟 初始化区块链连接
        print("🔗 [Server] 正在初始化区块链智能合约连接...")
        self.bc = BlockchainHandler()

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        config["global_ipfs_cid"] = self.current_global_cid
        messages = super().configure_train(server_round, arrays, config, grid)
        self.comm_tracker.track(messages)
        return messages

    def _plot_and_save_loss(self, current_round: int):
        """🌟 核心增强：在每轮聚合完毕后，自动绘制并保存学术级 Loss 曲线"""
        if not self.train_losses_history:
            return

        rounds = np.array(list(self.train_losses_history.keys()))
        losses = np.array(list(self.train_losses_history.values()))

        sns.set_theme(style="whitegrid", font_scale=1.2)
        plt.rcParams['font.family'] = 'serif'
        
        # 处理可能的字体警告
        try:
            plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
        except:
            pass

        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

        ax.plot(rounds, losses, color='#1f77b4', linewidth=2.5, marker='o', 
                markersize=6, markerfacecolor='white', markeredgewidth=1.5, label='Aggregated Train Loss')

        if len(rounds) > 1:
            z = np.polyfit(rounds, losses, min(2, len(rounds)-1))
            p = np.poly1d(z)
            ax.plot(rounds, p(rounds), linestyle='--', color='#7f7f7f', linewidth=1.5, alpha=0.8, label='Convergence Trend')

        ax.set_title('Blockchain-Secured FL-LoRA on Medical LLM (7B)', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Communication Round', fontsize=14)
        ax.set_ylabel('Aggregated Train Loss', fontsize=14)
        
        max_round = max(30, current_round + (5 - current_round % 5))
        ax.set_xticks(np.arange(0, max_round + 1, 5))
        ax.set_xlim(0, max_round)

        ax.legend(loc='upper right', frameon=True, shadow=True, fancybox=True, fontsize=12)
        
        save_dir = "./results/loss_plots"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"loss_curve_round_{current_round}.png")
        
        plt.tight_layout()
        plt.savefig(save_path, format='png', dpi=300)
        plt.close(fig) 
        
        print(f"📊 [Server] 自动绘图完成！本轮收敛曲线已更新至: {save_path}")

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        self.comm_tracker.track(replies)

        loss_results = [] 
        download_base_dir = f"./server_tmp_round_{server_round}"
        os.makedirs(download_base_dir, exist_ok=True)
        
        client_encrypted_data = []
        active_client_addresses = []

        print(f"\n📥 [Server] Round {server_round}: Processing {len(list(replies))} replies via IPFS...")

        for msg in replies:
            if not msg.has_content():
                continue
            
            if "metrics" in msg.content.metric_records:
                metrics_rec = msg.content.metric_records["metrics"]
                num_examples = metrics_rec.get("num_examples", 1)
                client_loss = metrics_rec.get("train_loss", 0.0) 
            else:
                num_examples = 1
                client_loss = 0.0

            cid = "FAIL"
            client_eth_addr = ""
            if "configs" in msg.content.config_records:
                config_rec = msg.content.config_records["configs"]
                cid = config_rec.get("ipfs_cid", "FAIL")
                client_eth_addr = config_rec.get("eth_address", "")

            if cid == "FAIL" or cid == "UPLOAD_FAILED":
                continue

            if self.ipfs.download(cid, download_base_dir):
                model_folder = os.path.join(download_base_dir, cid)
                fhe_path = os.path.join(model_folder, "full_encrypted_model.pkl")
                
                try:
                    if os.path.exists(fhe_path):
                        print(f"🔒 [Server] 成功读取混合模型，来自 CID: {cid}")
                        with open(fhe_path, "rb") as f:
                            enc_data = pickle.load(f)
                        client_encrypted_data.append((enc_data, num_examples))
                        loss_results.append((client_loss, num_examples))
                        
                        if client_eth_addr:
                            active_client_addresses.append(client_eth_addr)
                    else:
                        log(WARN, f"🚨 模型文件 {fhe_path} 不存在！")
                except Exception as e:
                    log(WARN, f"Failed to load model from CID {cid}: {e}")
            else:
                log(WARN, f"Failed to download CID {cid}")

        if not client_encrypted_data:
            shutil.rmtree(download_base_dir, ignore_errors=True)
            return None, {}

        print(f"\n🔥 [Server] 启动双核聚合引擎 (FHE盲算 + 明文均值)...")
        start_time = time.time()
        
        first_client_data, _ = client_encrypted_data[0]
        context_bytes = first_client_data["public_context"]
        context = ts.context_from(context_bytes)
        shapes_info = first_client_data["shapes"]
        total_examples = sum(num for _, num in client_encrypted_data)
        
        aggregated_encrypted_weights = {}
        aggregated_unencrypted_weights = {}

        layer_names_enc = list(first_client_data["weights"].keys())
        for layer_name in layer_names_enc:
            first_weight = client_encrypted_data[0][1] / total_examples 
            accumulated_enc_tensor = ts.ckks_vector_from(context, first_client_data["weights"][layer_name])
            accumulated_enc_tensor.mul_(first_weight)

            for client_data, num_examples in client_encrypted_data[1:]:
                weight = num_examples / total_examples
                client_enc_tensor = ts.ckks_vector_from(context, client_data["weights"][layer_name])
                client_enc_tensor.mul_(weight)
                accumulated_enc_tensor.add_(client_enc_tensor)
                
            aggregated_encrypted_weights[layer_name] = accumulated_enc_tensor.serialize()

        layer_names_unenc = list(first_client_data.get("unencrypted_weights", {}).keys())
        if layer_names_unenc:
            for layer_name in layer_names_unenc:
                acc_array = first_client_data["unencrypted_weights"][layer_name] * (client_encrypted_data[0][1] / total_examples)
                for client_data, num_examples in client_encrypted_data[1:]:
                    acc_array += client_data["unencrypted_weights"][layer_name] * (num_examples / total_examples)
                aggregated_unencrypted_weights[layer_name] = acc_array

        print(f"✅ [Server] 混合聚合彻底完成！总耗时: {time.time() - start_time:.2f} 秒")

        global_enc_data = {
            "weights": aggregated_encrypted_weights,
            "unencrypted_weights": aggregated_unencrypted_weights,
            "shapes": shapes_info,
            "public_context": context_bytes 
        }
        
        upload_dir = f"./server_upload_round_{server_round}"
        os.makedirs(upload_dir, exist_ok=True)
        global_pkl_path = os.path.join(upload_dir, "full_encrypted_model.pkl")
        
        with open(global_pkl_path, "wb") as f:
            pickle.dump(global_enc_data, f)
            
        # ==========================================
        # 🌟 核心修改点：解耦确权逻辑
        # ==========================================
        locked_path = os.path.join(upload_dir, "locked_model.bin")
        is_final_round = (server_round == self.num_rounds)

        if not is_final_round:
            print(f"🔓 [Server] 阶段 I (第{server_round}/{self.num_rounds}轮)：微调期，跳过 ABE 确权。")
            print(f"🔓 [Server] 伪装文件直接放行 (Client将直接执行同态解密)...")
            # 将 FHE 明文直接伪装成 locked_model.bin，Client 代码端已兼容此逻辑
            os.rename(global_pkl_path, locked_path)
        else:
            print(f"💎 [Server] 阶段 II (第{server_round}/{self.num_rounds}轮)：结算期！启动最终确权！")
            print(f"🔐 [Server-Crypto] 正在生成随机 AES-256 密钥并用 ABE 锁定...")
            crypto_engine = ABE_AES_Engine()
            crypto_engine.server_encrypt_model(global_pkl_path, locked_path, server_round)
            os.remove(global_pkl_path)
            print("✅ [Server-Crypto] AES 锁定与 ABE 策略封装完成！")
            
        print(f"☁️ [Server] 正在将模型上传至 IPFS...")
        global_cid = self.ipfs.upload_folder(upload_dir)
        
        if global_cid:
            print(f"✅ [Server] 上传成功！Global CID: {global_cid}")
            self.current_global_cid = global_cid
            
            try:
                if is_final_round:
                    print(f"💎 [Server] 触发智能合约！正在为 {len(active_client_addresses)} 个诚实节点发放贡献度并存证...")
                else:
                    print(f"🔗 [Blockchain] 中间态模型 CID 上链存证...")
                self.bc.upload_cid(global_cid, active_client_addresses)
            except Exception as e:
                print(f"⚠️ [Server] 警告：区块链存证失败: {e}")
                
        else:
            print("❌ [Server] 上传失败！")
            self.current_global_cid = "UPLOAD_FAILED"

        shutil.rmtree(download_base_dir, ignore_errors=True)
        shutil.rmtree(upload_dir, ignore_errors=True)

        aggregated_metrics = {}
        if loss_results:
            w_loss = sum(loss * num for loss, num in loss_results) / total_examples
            aggregated_metrics["train_loss"] = w_loss
            log(INFO, f"📊 [Server] Round {server_round} Aggregated Loss: {w_loss:.4f}")
            
            # 🌟 记录 Loss 并触发自动绘图！
            self.train_losses_history[server_round] = w_loss
            self._plot_and_save_loss(server_round)
            
        return ArrayRecord({}), MetricRecord(aggregated_metrics)