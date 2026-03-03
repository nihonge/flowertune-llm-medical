"""flowertune-medical: A Flower / FlowerTune app."""

import os
import shutil
import pickle
import time
from collections.abc import Iterable
from logging import INFO, WARN
from typing import Optional

from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.common import log
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg
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
    """Selective FHE-enabled Strategy."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.comm_tracker = CommunicationTracker()
        self.ipfs = IPFSHandler()
        # 保持记忆，解决失忆 Bug
        self.current_global_cid = "FAIL"

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        config["global_ipfs_cid"] = self.current_global_cid
        messages = super().configure_train(server_round, arrays, config, grid)
        self.comm_tracker.track(messages)
        return messages

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
            if "configs" in msg.content.config_records:
                config_rec = msg.content.config_records["configs"]
                cid = config_rec.get("ipfs_cid", "FAIL")

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

        # 核心 1：聚合密文层 (FHE)
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

        # 核心 2：聚合明文层 (极速 NumPy)
        layer_names_unenc = list(first_client_data.get("unencrypted_weights", {}).keys())
        if layer_names_unenc:
            print(f"⚡ [Server] 正在极速聚合 {len(layer_names_unenc)} 个明文层...")
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
        
        print(f"💾 [Server] 正在打包全局混合模型，准备上传 IPFS (体积大幅减小)...")
        with open(global_pkl_path, "wb") as f:
            pickle.dump(global_enc_data, f)
            
        print(f"☁️ [Server] 正在上传至 IPFS...")
        global_cid = self.ipfs.upload_folder(upload_dir)
        
        if global_cid:
             print(f"✅ [Server] 上传成功！Global CID: {global_cid}")
             self.current_global_cid = global_cid
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
            
        return ArrayRecord({}), MetricRecord(aggregated_metrics)