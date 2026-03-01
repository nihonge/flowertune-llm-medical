"""flowertune-medical: A Flower / FlowerTune app."""
import shutil
import os
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
from flowertune_medical.fhe_utils import FullFHEHandler 

from flowertune_medical.dataset import (
    get_tokenizer_and_data_collator_and_propt_formatting,
    load_data,
    replace_keys,
)
from flowertune_medical.models import cosine_annealing, get_model

# Avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

# 初始化 IPFS 处理器
ipfs = IPFSHandler()

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    # Parse config
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    num_rounds = context.run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))
    training_arguments = TrainingArguments(**cfg.train.training_arguments)

    # Let's get the client partition
    trainset = load_data(partition_id, num_partitions, cfg.static.dataset.name)
    (
        tokenizer,
        data_collator,
        formatting_prompts_func,
    ) = get_tokenizer_and_data_collator_and_propt_formatting(cfg.model.name)

    # Load the base model
    model = get_model(cfg.model)

    # ==========================================
    # 🌟 接收全局 CID -> 下载密文 -> 解密 -> 加载
    # ==========================================
    global_cid = "FAIL"
    
    # 绝对安全的精准提取，不打印任何二进制！
    if "config" in msg.content:
        global_cid = msg.content["config"].get("global_ipfs_cid", "FAIL")
    elif "configs" in msg.content:
        global_cid = msg.content["configs"].get("global_ipfs_cid", "FAIL")

    if global_cid != "FAIL" and global_cid != "UPLOAD_FAILED":
        print(f"\n📥 [Client {partition_id}] 收到全局 CID: {global_cid}，准备下载密文...")
        download_dir = tempfile.mkdtemp(prefix=f"flwr_client_{partition_id}_dl_")
        
        try:
            if ipfs.download(global_cid, download_dir):
                fhe_path = os.path.join(download_dir, global_cid, "full_encrypted_model.pkl")
                if os.path.exists(fhe_path):
                    # 1. 实例化加密工具（会自动读取同目录的 global_fhe_keys.pkl）
                    fhe_handler = FullFHEHandler()
                    
                    # 2. 解密还原为 PyTorch 张量
                    decrypted_state_dict = fhe_handler.decrypt_model(fhe_path)
                    
                    # 3. 将解密后的全局经验加载到本地模型中
                    set_peft_model_state_dict(model, decrypted_state_dict)
                    print(f"✅ [Client {partition_id}] 成功吸收 Server 端发来的全局聚合经验！")
                else:
                    print(f"🚨 [Client {partition_id}] 没找到密文文件: {fhe_path}")
            else:
                print(f"❌ [Client {partition_id}] IPFS 下载失败。")
        except Exception as e:
             print(f"❌ [Client {partition_id}] 解密加载失败: {e}")
        finally:
             shutil.rmtree(download_dir, ignore_errors=True)
    else:
        print(f"\n🆕 [Client {partition_id}] 未收到全局 CID (这是第1轮)，使用本地初始权重开局。")
    # ==========================================

    # 兼容获取 server_round 和 save_path
    try:
        current_round = msg.content["config"]["server-round"]
        save_path_str = msg.content["config"]["save_path"]
    except KeyError:
        current_round = 1
        save_path_str = f"./results/fallback_client_{partition_id}"

    # Set learning rate for current round
    new_lr = cosine_annealing(
        current_round,
        num_rounds,
        cfg.train.learning_rate_max,
        cfg.train.learning_rate_min,
    )

    training_arguments.learning_rate = new_lr
    training_arguments.output_dir = save_path_str

    # Construct trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_arguments,
        max_seq_length=cfg.train.seq_length,
        train_dataset=trainset,
        formatting_func=formatting_prompts_func,
        data_collator=data_collator,
    )

    # 开始训练
    print(f"🚀 [Client {partition_id}] 准备就绪，开始本地微调...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_results = trainer.train()

    # 保存参数并上传 IPFS
    temp_dir = tempfile.mkdtemp(prefix=f"flwr_client_{partition_id}_")
    trainer.model.save_pretrained(temp_dir, safe_serialization=False)
    
    generated_files = os.listdir(temp_dir)
    if not generated_files:
        raise RuntimeError(f"🚨 活见鬼了！HuggingFace 假装保存了，但文件夹 {temp_dir} 里是空的！")
    
    # 全同态加密拦截
    try:
        fhe_handler = FullFHEHandler()
        plaintext_file_to_delete = fhe_handler.encrypt_full_model(temp_dir, temp_dir)
        if os.path.exists(plaintext_file_to_delete):
            os.remove(plaintext_file_to_delete)
            print(f"🗑️ [Client {partition_id}] 已彻底销毁本地明文权重: {plaintext_file_to_delete}")
    except Exception as e:
        print(f"❌ [Client {partition_id}] 同态加密失败，中断上传！错误信息: {e}")
        raise e

    # 上传到 IPFS
    print(f"☁️ [Client {partition_id}] Uploading encrypted parameters to IPFS...")
    cid = ipfs.upload_folder(temp_dir)
    safe_cid = str(cid) if cid else "UPLOAD_FAILED"
    
    if cid:
        print(f"✅ [Client {partition_id}] Upload Success! CID: {safe_cid}")
    else:
        print(f"❌ [Client {partition_id}] Upload Failed.")

    shutil.rmtree(temp_dir, ignore_errors=True)
    shutil.rmtree(training_arguments.output_dir, ignore_errors=True)

    # 构造返回消息
    metrics = {
        "train_loss": train_results.training_loss,
        "num_examples": len(trainset),
    }

    configs = {
        "ipfs_cid": safe_cid
    }

    content = RecordDict({
        "arrays": ArrayRecord({}),
        "metrics": MetricRecord(metrics),
        "configs": ConfigRecord(configs),
    })

    return Message(content=content, reply_to=msg)