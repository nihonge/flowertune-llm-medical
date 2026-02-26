"""flowertune-medical: A Flower / FlowerTune app."""

import os
import shutil
import torch
from collections.abc import Iterable
from logging import INFO, WARN
from typing import Optional

from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.common import log, Array
from flwr.serverapp import Grid
from flwr.serverapp.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate
from flowertune_medical.ipfs_handler import IPFSHandler


class FlowerTuneLlm(FedAvg):
    """Customised FedAvg strategy implementation.

    This class behaves just like FedAvg but also tracks the communication
    costs associated with `train` over FL rounds.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.comm_tracker = CommunicationTracker()
        self.ipfs = IPFSHandler()

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure the next round of training."""
        messages = super().configure_train(server_round, arrays, config, grid)

        # Track communication costs
        self.comm_tracker.track(messages)

        return messages

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        """Aggregate ArrayRecords and MetricRecords in the received Messages."""
        # Track communication costs
        self.comm_tracker.track(replies)

        weights_results = []
        
        # 1. [新增] 准备一个列表用来存 (loss, num_examples) <--- [修改点 1]
        loss_results = [] 

        # 定义本轮下载的临时根目录
        download_base_dir = f"./server_tmp_round_{server_round}"
        
        # 用于保存参数的 Key (层名)，因为聚合函数会丢失 Key
        parameter_keys = None 

        print(f"\n📥 [Server] Round {server_round}: Processing {len(list(replies))} replies via IPFS...")

        # 3. 遍历客户端回复
        for msg in replies:
            if not msg.has_content():
                continue
            # 1. 取 Metrics (数字)
            # 注意：RecordDict 在 Server 端解析时，通常可以通过 key 访问
            # msg.content 是一个 RecordSet (或 RecordDict)
            
            # 获取 metrics 里的 num_examples
            # 假设你在 client 里用的 key 是 "metrics"
            if "metrics" in msg.content.metric_records:
                metrics_rec = msg.content.metric_records["metrics"]
                num_examples = metrics_rec.get("num_examples", 1)
                
                # 2. [新增] 提取 Client 传回来的 train_loss <--- [修改点 2]
                # 默认值给 0.0 防错
                client_loss = metrics_rec.get("train_loss", 0.0) 
            else:
                num_examples = 1
                client_loss = 0.0

            # 2. 取 Configs (字符串 CID)
            # 👇 修正：从 config_records 中读取
            cid = "FAIL"
            if "configs" in msg.content.config_records:
                config_rec = msg.content.config_records["configs"]
                # ConfigRecord 获取值
                cid = config_rec.get("ipfs_cid", "FAIL")

            if cid == "FAIL" or cid == "UPLOAD_FAILED":
                continue

            # 4. 通过 IPFS 下载
            if self.ipfs.download(cid, download_base_dir):
                # ipfs.get 会在 download_base_dir 下创建以 CID 命名的文件夹
                model_folder = os.path.join(download_base_dir, cid)
                
                # 寻找参数文件 (兼容 .bin 和 .safetensors)
                bin_path = os.path.join(model_folder, "adapter_model.bin")
                safe_path = os.path.join(model_folder, "adapter_model.safetensors")
                
                state_dict = None
                try:
                    if os.path.exists(safe_path):
                        from safetensors.torch import load_file
                        state_dict = load_file(safe_path, device="cpu")
                    elif os.path.exists(bin_path):
                        state_dict = torch.load(bin_path, map_location="cpu")
                    
                    if state_dict:
                        # 记录 Keys (只需要记录一次，假设所有 Client 模型结构一致)
                        if parameter_keys is None:
                            parameter_keys = list(state_dict.keys())

                        # 提取 Values 并转为 Numpy (Flower 聚合必须用 Numpy)
                        # 注意：保持顺序一致
                        param_vals = [v.cpu().numpy() for v in state_dict.values()]
                        weights_results.append((param_vals, num_examples))
                        
                        # 3. [新增] 只有下载成功才记录 Loss，用于后续计算平均值 <--- [修改点 3]
                        loss_results.append((client_loss, num_examples))
                        
                except Exception as e:
                    log(WARN, f"Failed to load model from CID {cid}: {e}")
            else:
                log(WARN, f"Failed to download CID {cid}")

        # 5. 清理下载的临时文件 (节省服务器空间)
        shutil.rmtree(download_base_dir, ignore_errors=True)

        # 6. 执行聚合
        # 如果没有成功下载任何模型，直接返回空
        if not weights_results:
            return None, {}

        log(INFO, f"🔄 [Server] Aggregating {len(weights_results)} models...")
        
        # 调用 Flower 底层数学函数进行加权平均
        # aggregated_values 是一个 list of numpy arrays
        aggregated_values = aggregate(weights_results)
        
        # 7. 重构 ArrayRecord
        # 我们必须把 List 重新映射回 Dictionary (Key: Value)
        # 这样下一轮 distribute 的时候，Client 才能通过 load_state_dict 加载
        if parameter_keys is None:
            log(WARN, "Parameter keys lost during aggregation!")
            return None, {}

        aggregated_dict = {
            k: Array(v) for k, v in zip(parameter_keys, aggregated_values)
        }
        
        arrays = ArrayRecord(aggregated_dict)
        
        # 4. [新增/修改] 手动计算聚合后的 Metrics (Loss) <--- [修改点 4]
        # =========================================================
        aggregated_metrics = {}
        
        if loss_results:
            # 计算加权平均 Loss: sum(loss * num) / sum(num)
            total_examples = sum(num for _, num in loss_results)
            if total_examples > 0:
                weighted_loss = sum(loss * num for loss, num in loss_results) / total_examples
                aggregated_metrics["train_loss"] = weighted_loss
                # 打印到控制台，让你直接看到结果
                log(INFO, f"📊 [Server] Round {server_round} Aggregated Loss: {weighted_loss:.4f}")
        
        # 将计算好的字典放入 MetricRecord
        metrics = MetricRecord(aggregated_metrics) 
        # =========================================================

        return arrays, metrics

class CommunicationTracker:
    # ... (保持不变) ...
    def __init__(self):
        self.curr_comm_cost = 0.0

    def track(self, messages: Iterable[Message]):
        # ... (保持不变) ...
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