"""flowertune-medical: A Flower / FlowerTune app."""
import shutil
import os
import warnings
from flowertune_medical.ipfs_handler import IPFSHandler
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict, ConfigRecord
from flwr.clientapp import ClientApp
from flwr.common.config import unflatten_dict
from omegaconf import DictConfig
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import TrainingArguments
from trl import SFTTrainer

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

    # Load the model and initialize it with the received weights
    model = get_model(cfg.model)
    set_peft_model_state_dict(model, msg.content["arrays"].to_torch_state_dict())

    # Set learning rate for current round
    new_lr = cosine_annealing(
        msg.content["config"]["server-round"],
        num_rounds,
        cfg.train.learning_rate_max,
        cfg.train.learning_rate_min,
    )

    training_arguments.learning_rate = new_lr
    training_arguments.output_dir = msg.content["config"]["save_path"]

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

    # Do local training
    results = trainer.train()

    # # ==========================打印上传参数的调试信息==========================
    # # 1. 先把参数字典提取出来，存到一个变量里
    # # 这里的 raw_params 就是你要加密的“原生对象”
    # raw_params = get_peft_model_state_dict(model)

    # print("\n" + "="*50)
    # print(f"🕵️ [Client Debug] 正在检查待上传参数 (Type: {type(raw_params)})")
    # print(f"📊 总共包含 {len(raw_params)} 个张量 (Tensors)")
    # print("-" * 50)

    # # 2. 遍历打印前 5 个参数的详情（防止刷屏，只看前几个）
    # count = 0
    # total_elements = 0
    # for key, tensor in raw_params.items():
    #     # 统计总参数量
    #     total_elements += tensor.numel()
        
    #     # 打印部分 Key 的形状
    #     if count < 5: 
    #         print(f"🔑 Key: {key}")
    #         print(f"   📏 Shape: {tensor.shape}") # 比如 [32, 4096]
    #         print(f"   💾 Dtype: {tensor.dtype}") # 比如 torch.float32
    #         print(f"   🧪 Device: {tensor.device}")
    #         print("-" * 20)
    #     count += 1
    
    # print(f"📈 本次上传总参数数量: {total_elements}")
    # print(f"📦 预估数据大小 (BF16): {total_elements * 2 / 1024 / 1024 :.2f} MB")
    # print("="*50 + "\n")
    # # ==========================打印上传参数的调试信息==========================


    # ==========================================
    # 🚀 使用封装后的 IPFS 逻辑
    # ==========================================
    
    # 开始训练
    # ⚠️ 忽略一些 Ray/HuggingFace 的警告
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_results = trainer.train()

    # 5. 保存参数并上传 IPFS (核心修改)
    # ------------------------------------------------------------------
    # 定义临时保存路径
    temp_dir = f"./tmp_client_model_{partition_id}"
    
    # 保存 LoRA 参数 (adapter_model.bin 和 adapter_config.json)
    trainer.model.save_pretrained(temp_dir)
    
    # ☁️ 上传到 IPFS
    print(f"☁️ [Client {partition_id}] Uploading parameters to IPFS...")
    cid = ipfs.upload_folder(temp_dir)

    # ✅ 修复 NameError：定义 safe_cid
    safe_cid = str(cid) if cid else "UPLOAD_FAILED"
    
    if cid:
        print(f"✅ [Client {partition_id}] Upload Success! CID: {safe_cid}")
    else:
        print(f"❌ [Client {partition_id}] Upload Failed.")

    # 清理本地临时文件
    shutil.rmtree(temp_dir, ignore_errors=True)
    shutil.rmtree(training_arguments.output_dir, ignore_errors=True)

    # 6. 构造返回消息
    # ------------------------------------------------------------------
    
    # A. Metrics: 只能放数字 (int/float)
    metrics = {
        "train_loss": train_results.training_loss,
        "num_examples": len(trainset),
    }

    # B. Configs: 只能放字符串 (CID 放这里)
    configs = {
        "ipfs_cid": safe_cid
    }

    # C. Arrays: 放空 (因为参数已经在 IPFS 上了)
    # 这里的 ArrayRecord 为空，大大节省了 Flower 协议的通信开销
    arrays = ArrayRecord({})

    # 打包
    content = RecordDict({
        "arrays": arrays,
        "metrics": MetricRecord(metrics),
        "configs": ConfigRecord(configs),  # ✅ 确认为 ConfigRecord
    })

    return Message(content=content, reply_to=msg)