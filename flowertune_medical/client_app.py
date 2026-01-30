"""flowertune-medical: A Flower / FlowerTune app."""

import os
import warnings

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
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

    # ==========================
    # 🔍 插入打印逻辑开始
    # ==========================
    
    # 1. 先把参数字典提取出来，存到一个变量里
    # 这里的 raw_params 就是你要加密的“原生对象”
    raw_params = get_peft_model_state_dict(model)

    print("\n" + "="*50)
    print(f"🕵️ [Client Debug] 正在检查待上传参数 (Type: {type(raw_params)})")
    print(f"📊 总共包含 {len(raw_params)} 个张量 (Tensors)")
    print("-" * 50)

    # 2. 遍历打印前 5 个参数的详情（防止刷屏，只看前几个）
    count = 0
    total_elements = 0
    for key, tensor in raw_params.items():
        # 统计总参数量
        total_elements += tensor.numel()
        
        # 打印部分 Key 的形状
        if count < 5: 
            print(f"🔑 Key: {key}")
            print(f"   📏 Shape: {tensor.shape}") # 比如 [32, 4096]
            print(f"   💾 Dtype: {tensor.dtype}") # 比如 torch.float32
            print(f"   🧪 Device: {tensor.device}")
            print("-" * 20)
        count += 1
    
    print(f"📈 本次上传总参数数量: {total_elements}")
    print(f"📦 预估数据大小 (BF16): {total_elements * 2 / 1024 / 1024 :.2f} MB")
    print("="*50 + "\n")
    

    # Construct and return reply Message
    model_record = ArrayRecord(get_peft_model_state_dict(model))
    metrics = {
        "train_loss": results.training_loss,
        "num-examples": len(trainset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)
