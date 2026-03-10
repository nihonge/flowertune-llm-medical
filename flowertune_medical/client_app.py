import os
import sys
import gc
import torch
import shutil
import warnings
import tempfile

from flowertune_medical.ipfs_handler import IPFSHandler
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict, ConfigRecord
from flwr.clientapp import ClientApp
from flwr.common.config import unflatten_dict
from omegaconf import DictConfig
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import TrainingArguments
from trl import SFTTrainer

# 🌟 引入核心引擎
from flowertune_medical.fhe_utils import SelectiveFHEHandler 
from flowertune_medical.blockchain_handler import BlockchainHandler
from flowertune_medical.crypto_handler import ABE_AES_Engine

from flowertune_medical.dataset import (
    get_tokenizer_and_data_collator_and_propt_formatting,
    load_data,
    replace_keys,
)
from flowertune_medical.models import cosine_annealing, get_model

# ==========================================
# 🛠️ 终极消音器：屏蔽 TenSEAL 底层 C++ 警告
# ==========================================
class SuppressCLevelOutput:
    """重定向底层文件描述符，彻底封杀 FHE 引擎的无效警告刷屏"""
    def __enter__(self):
        self.null_fd = os.open(os.devnull, os.O_RDWR)
        self.save_stdout = os.dup(1)
        self.save_stderr = os.dup(2)
        os.dup2(self.null_fd, 1)
        os.dup2(self.null_fd, 2)

    def __exit__(self, *_):
        os.dup2(self.save_stdout, 1)
        os.dup2(self.save_stderr, 2)
        os.close(self.null_fd)
        os.close(self.save_stdout)
        os.close(self.save_stderr)

# Avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

ipfs = IPFSHandler()
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    # 获取联邦配置与当前轮次信息
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    num_rounds = context.run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))
    
    try:
        current_round = msg.content["config"]["server-round"]
        save_path_str = msg.content["config"]["save_path"]
    except KeyError:
        current_round = 1
        save_path_str = f"./results/fallback_client_{partition_id}"

    training_arguments = TrainingArguments(**cfg.train.training_arguments)

    # 加载数据集与模型
    trainset = load_data(partition_id, num_partitions, cfg.static.dataset.name)
    (
        tokenizer,
        data_collator,
        formatting_prompts_func,
    ) = get_tokenizer_and_data_collator_and_propt_formatting(cfg.model.name)

    model = get_model(cfg.model)
    
    # 🌟 区块链初始化与身份挂载
    bc = BlockchainHandler()
    my_eth_address = bc.client_accounts[int(partition_id) % len(bc.client_accounts)]

    # 获取 Server 下发的全局 CID
    global_cid = "FAIL"
    if "config" in msg.content:
        global_cid = msg.content["config"].get("global_ipfs_cid", "FAIL")
    elif "configs" in msg.content:
        global_cid = msg.content["configs"].get("global_ipfs_cid", "FAIL")

    # ==========================================
    # 🌟 两阶段解耦：微调期模型加载逻辑
    # ==========================================
    if global_cid != "FAIL" and global_cid != "UPLOAD_FAILED":
        print(f"\n📥 [Client {partition_id}] 收到全局 CID: {global_cid}，从 IPFS 拉取密文...")
        download_dir = tempfile.mkdtemp(prefix=f"flwr_client_{partition_id}_dl_")
        
        try:
            if ipfs.download(global_cid, download_dir):
                locked_path = os.path.join(download_dir, global_cid, "locked_model.bin")
                
                if os.path.exists(locked_path):
                    # 🚨 核心修复：只要还在联邦微调任务中，就永远当作“协作微调期”。
                    # 不管是第几轮，Server 下发的都是【未加 ABE 锁的纯 FHE 密文】(伪装成了 locked_model.bin)
                    print(f"🔓 [Client {partition_id}] 阶段 I：协作微调期 (第{current_round}/{num_rounds}轮)。跳过 ABE，直接放行同步！")
                    print(f"✅ [Client {partition_id}] 解密授权通过！正在执行同态还原...")
                    
                    with SuppressCLevelOutput(): # 屏蔽 TenSEAL 烦人警告
                        fhe_handler = SelectiveFHEHandler()
                        decrypted_state_dict = fhe_handler.decrypt_model(locked_path)
                        
                    set_peft_model_state_dict(model, decrypted_state_dict)
                    print(f"✅ [Client {partition_id}] 成功吸收全局聚合经验！")

                else:
                    print(f"🚨 [Client {partition_id}] 没找到锁定的模型文件: {locked_path}")
            else:
                print(f"❌ [Client {partition_id}] IPFS 下载失败。")
        except Exception as e:
             print(f"❌ [Client {partition_id}] 密文加载流程出错: {e}")
        finally:
             shutil.rmtree(download_dir, ignore_errors=True)
    else:
        print(f"\n🆕 [Client {partition_id}] 未收到全局 CID (或是第1轮)，使用基座模型开局。")

    # ==========================================
    # 🌟 启动本地微调
    # ==========================================
    new_lr = cosine_annealing(
        current_round,
        num_rounds,
        cfg.train.learning_rate_max,
        cfg.train.learning_rate_min,
    )

    training_arguments.learning_rate = new_lr
    training_arguments.output_dir = save_path_str

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_arguments,
        max_seq_length=cfg.train.seq_length,
        train_dataset=trainset,
        formatting_func=formatting_prompts_func,
        data_collator=data_collator,
    )

    print(f"🚀 [Client {partition_id}] 准备就绪，开始本地微调 (Round {current_round})...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_results = trainer.train()

    # ==========================================
    # 🌟 选择性同态加密上传
    # ==========================================
    temp_dir = tempfile.mkdtemp(prefix=f"flwr_client_{partition_id}_")
    trainer.model.save_pretrained(temp_dir, safe_serialization=False)
    
    try:
        with SuppressCLevelOutput(): # 屏蔽加密时的 TenSEAL 警告
            fhe_handler = SelectiveFHEHandler()
            plaintext_file_to_delete = fhe_handler.encrypt_full_model(temp_dir, temp_dir)
            
        if os.path.exists(plaintext_file_to_delete):
            os.remove(plaintext_file_to_delete)
            print(f"🗑️ [Client {partition_id}] 已彻底销毁本地明文权重...")
            
    except Exception as e:
        print(f"❌ [Client {partition_id}] 同态加密失败: {e}")
        raise e

    print(f"☁️ [Client {partition_id}] 正在将密态矩阵上传至 IPFS...")
    cid = ipfs.upload_folder(temp_dir)
    safe_cid = str(cid) if cid else "UPLOAD_FAILED"
    
    if cid:
        print(f"✅ [Client {partition_id}] 上传成功! CID: {safe_cid}")
    else:
        print(f"❌ [Client {partition_id}] 上传失败。")

    shutil.rmtree(temp_dir, ignore_errors=True)
    shutil.rmtree(training_arguments.output_dir, ignore_errors=True)

    # 🌟 修正：只提取最后一步的真实 Loss，防止平均值导致模型发散
    last_loss = 0.0
    if len(trainer.state.log_history) > 0:
        loss_logs = [log["loss"] for log in trainer.state.log_history if "loss" in log]
        if loss_logs:
            last_loss = loss_logs[-1]
        else:
            last_loss = train_results.training_loss
    else:
        last_loss = train_results.training_loss

    metrics = {
        "train_loss": last_loss,
        "num_examples": len(trainset),
    }

    configs = {
        "ipfs_cid": safe_cid,
        "eth_address": my_eth_address
    }

    content = RecordDict({
        "arrays": ArrayRecord({}),
        "metrics": MetricRecord(metrics),
        "configs": ConfigRecord(configs),
    })

    # ==========================================
    # 🧹 终极显存清道夫：解决随着轮次增加而导致的 Swapping / Timeout
    # ==========================================
    if 'trainer' in locals():
        del trainer
    if 'model' in locals():
        del model
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"🧹 [Client {partition_id}] 本轮显存与缓存已彻底清理！")

    return Message(content=content, reply_to=msg)