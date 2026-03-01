"""flowertune-medical: A Flower / FlowerTune app."""

import os
import shutil
import pickle
import time
from collections.abc import Iterable
from logging import INFO, WARN
from typing import Optional

from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.common import log, Array
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate
from flowertune_medical.ipfs_handler import IPFSHandler
import tenseal as ts

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
    """FHE-enabled Strategy."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.comm_tracker = CommunicationTracker()
        self.ipfs = IPFSHandler()
        # 🌟 核心修复 1：用实例变量存下最新的 CID
        self.current_global_cid = "FAIL"

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of training."""
        
        # 🌟 核心修复 2：在给 Client 发送配置前，强行把上一轮的 CID 塞进 config
        # 注意：第一轮时这里是 "FAIL"，到了第二轮就会变成真实的 CID
        config["global_ipfs_cid"] = self.current_global_cid
        
        messages = super().configure_train(server_round, arrays, config, grid)
        self.comm_tracker.track(messages)
        return messages

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate Encrypted Models via Homomorphic Addition."""
        self.comm_tracker.track(replies)

        loss_results = [] 
        download_base_dir = f"./server_tmp_round_{server_round}"
        os.makedirs(download_base_dir, exist_ok=True)
        
        # 存放所有客户端下载下来的密文数据
        client_encrypted_data = []

        print(f"\n📥 [Server] Round {server_round}: Processing {len(list(replies))} replies via IPFS...")

        # ==========================================
        # 阶段 1：解析消息并下载密文
        # ==========================================
        for msg in replies:
            if not msg.has_content():
                continue
            
            # 提取 Metrics
            if "metrics" in msg.content.metric_records:
                metrics_rec = msg.content.metric_records["metrics"]
                num_examples = metrics_rec.get("num_examples", 1)
                client_loss = metrics_rec.get("train_loss", 0.0) 
            else:
                num_examples = 1
                client_loss = 0.0

            # 提取 CID (这里解析 Client 传回来的 CID)
            cid = "FAIL"
            if "configs" in msg.content.config_records:
                config_rec = msg.content.config_records["configs"]
                cid = config_rec.get("ipfs_cid", "FAIL")

            if cid == "FAIL" or cid == "UPLOAD_FAILED":
                continue

            # 通过 IPFS 下载包含 FHE 密文的文件夹
            if self.ipfs.download(cid, download_base_dir):
                model_folder = os.path.join(download_base_dir, cid)
                fhe_path = os.path.join(model_folder, "full_encrypted_model.pkl")
                
                try:
                    if os.path.exists(fhe_path):
                        print(f"🔒 [Server] 成功读取密文模型，来自 CID: {cid}")
                        with open(fhe_path, "rb") as f:
                            enc_data = pickle.load(f)
                        client_encrypted_data.append((enc_data, num_examples))
                        loss_results.append((client_loss, num_examples))
                    else:
                        log(WARN, f"🚨 密文文件 {fhe_path} 不存在！")
                except Exception as e:
                    log(WARN, f"Failed to load encrypted model from CID {cid}: {e}")
            else:
                log(WARN, f"Failed to download CID {cid}")

        if not client_encrypted_data:
            shutil.rmtree(download_base_dir, ignore_errors=True)
            return None, {}

        # ==========================================
        # 阶段 2：全同态盲算聚合 (FedAvg)
        # ==========================================
        print(f"\n🔥 [Server] 启动同态盲算引擎，准备聚合 {len(client_encrypted_data)} 个密文模型...")
        start_time = time.time()
        
        first_client_data, _ = client_encrypted_data[0]
        context_bytes = first_client_data["public_context"]
        context = ts.context_from(context_bytes)
        
        layer_names = list(first_client_data["weights"].keys())
        shapes_info = first_client_data["shapes"]
        
        total_examples = sum(num for _, num in client_encrypted_data)
        
        aggregated_encrypted_weights = {}

        for layer_name in layer_names:
            print(f"  -> 正在同态聚合: {layer_name} ...")
            
            first_enc_weight_bytes = first_client_data["weights"][layer_name]
            first_weight = client_encrypted_data[0][1] / total_examples 
            
            accumulated_enc_tensor = ts.ckks_vector_from(context, first_enc_weight_bytes)
            accumulated_enc_tensor.mul_(first_weight)

            for client_data, num_examples in client_encrypted_data[1:]:
                enc_weight_bytes = client_data["weights"][layer_name]
                weight = num_examples / total_examples
                
                client_enc_tensor = ts.ckks_vector_from(context, enc_weight_bytes)
                client_enc_tensor.mul_(weight)
                
                accumulated_enc_tensor.add_(client_enc_tensor)
                
            aggregated_encrypted_weights[layer_name] = accumulated_enc_tensor.serialize()

        print(f"✅ [Server] 同态盲算聚合完成！总耗时: {time.time() - start_time:.2f} 秒")

        # ==========================================
        # 阶段 3：打包全局密文并回传 IPFS
        # ==========================================
        global_enc_data = {
            "weights": aggregated_encrypted_weights,
            "shapes": shapes_info,
            "public_context": context_bytes 
        }
        
        upload_dir = f"./server_upload_round_{server_round}"
        os.makedirs(upload_dir, exist_ok=True)
        global_pkl_path = os.path.join(upload_dir, "full_encrypted_model.pkl")
        
        print(f"💾 [Server] 正在打包全局密文，准备上传 IPFS (可能长达几个GB)...")
        with open(global_pkl_path, "wb") as f:
            pickle.dump(global_enc_data, f)
            
        print(f"☁️ [Server] 正在上传全局密文到 IPFS...")
        global_cid = self.ipfs.upload_folder(upload_dir)
        
        if global_cid:
             print(f"✅ [Server] 全局模型上传成功！Global CID: {global_cid}")
             # 🌟 核心修复 3：更新当前类的 CID，这样下一轮的 configure_train 就能把这玩意发给 Client
             self.current_global_cid = global_cid
        else:
             print("❌ [Server] 全局模型上传失败！")
             self.current_global_cid = "UPLOAD_FAILED"

        shutil.rmtree(download_base_dir, ignore_errors=True)
        shutil.rmtree(upload_dir, ignore_errors=True)

        # ==========================================
        # 阶段 4：构造返回值
        # ==========================================
        aggregated_metrics = {}
        if loss_results:
            w_loss = sum(loss * num for loss, num in loss_results) / total_examples
            aggregated_metrics["train_loss"] = w_loss
            log(INFO, f"📊 [Server] Round {server_round} Aggregated Loss: {w_loss:.4f}")
            
        metrics = MetricRecord(aggregated_metrics) 
        
        # 返回空的数组和合并后的 metrics。我们不再需要把 CID 写进返回值里，因为前面已经存到了 self.current_global_cid
        return ArrayRecord({}), metrics