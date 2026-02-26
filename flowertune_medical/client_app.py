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
from flowertune_medical.fhe_utils import FullFHEHandler # 🌟 [保留] 你已经加好的 FHE 工具箱导入

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

    # ==========================================
    # 🌟 [Debug 专用] 极速模式：强行把训练压缩到几秒钟！
    # ==========================================
    training_arguments.max_steps = 3         # 强制只训练 3 步 (原本可能是几百上千步)
    training_arguments.num_train_epochs = 1  # 覆盖 epoch 设置
    training_arguments.save_steps = 100      # 防止中途乱保存，全留到最后
    # ==========================================

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

    # ==========================================
    # 🚀 开始训练 (🌟 [修改] 删除了重复的 trainer.train() 调用)
    # ==========================================
    # ⚠️ 忽略一些 Ray/HuggingFace 的警告
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_results = trainer.train()

    # 5. 保存参数并上传 IPFS
    # ------------------------------------------------------------------
    # 🌟 [终极修复] 强制向操作系统申请一个绝对存在的独立临时目录
    temp_dir = tempfile.mkdtemp(prefix=f"flwr_client_{partition_id}_")
    print(f"\n📁 [Client {partition_id}] 成功创建真实的物理路径: {temp_dir}")
    
    # HuggingFace 会在这里生成 adapter_model.safetensors (或 .bin) (明文)
    # 🌟 [究极修复] 强制关闭 safetensors！保存为兼容性 100% 的 .bin 格式，绕过 OS Error 19！
    trainer.model.save_pretrained(temp_dir, safe_serialization=False)
    
    # 🕵️ [抓鬼时刻] 强制打印文件夹里到底生成了什么！
    generated_files = os.listdir(temp_dir)
    print(f"📦 [Client {partition_id}] save_pretrained 执行完毕。目录内文件: {generated_files}")
    
    if not generated_files:
        raise RuntimeError(f"🚨 活见鬼了！HuggingFace 假装保存了，但文件夹 {temp_dir} 里是空的！")
    
    # ==========================================
    # 🌟 [新增] FHE 拦截与明文销毁逻辑 
    # ==========================================
    try:
        print(f"🛡️ [Client {partition_id}] 准备启动全同态加密...")
        fhe_handler = FullFHEHandler()
        
        # 让工具类自己去找文件，并返回它真正加密的那个明文文件路径
        plaintext_file_to_delete = fhe_handler.encrypt_full_model(temp_dir, temp_dir)
        
        # 💣 极其关键：物理删除明文权重文件！
        if os.path.exists(plaintext_file_to_delete):
            os.remove(plaintext_file_to_delete)
            print(f"🗑️ [Client {partition_id}] 已彻底销毁本地明文权重: {plaintext_file_to_delete}")
            
    except Exception as e:
        print(f"❌ [Client {partition_id}] 同态加密失败，中断上传！错误信息: {e}")
        # 如果加密失败，直接抛出异常，绝对不能让明文传上去
        raise e
    # ==========================================

    # ☁️ 上传到 IPFS (此时文件夹里只有安全的密文 .pkl 和一些微小的配置 .json)
    print(f"☁️ [Client {partition_id}] Uploading encrypted parameters to IPFS...")
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
    arrays = ArrayRecord({})

    # 打包
    content = RecordDict({
        "arrays": arrays,
        "metrics": MetricRecord(metrics),
        "configs": ConfigRecord(configs),
    })

    return Message(content=content, reply_to=msg)