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

# 🌟 引入全新的选择性加密引擎
from flowertune_medical.fhe_utils import SelectiveFHEHandler 

# 🌟 区块链融合：引入智能合约中枢
from flowertune_medical.blockchain_handler import BlockchainHandler

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

ipfs = IPFSHandler()
app = ClientApp()

@app.train()
def train(msg: Message, context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    num_rounds = context.run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))
    training_arguments = TrainingArguments(**cfg.train.training_arguments)

    trainset = load_data(partition_id, num_partitions, cfg.static.dataset.name)
    (
        tokenizer,
        data_collator,
        formatting_prompts_func,
    ) = get_tokenizer_and_data_collator_and_propt_formatting(cfg.model.name)

    model = get_model(cfg.model)
    
    # 🌟 区块链融合：初始化链连接，并根据自己的 partition_id 分配一个固定的以太坊账户
    bc = BlockchainHandler()
    my_eth_address = bc.client_accounts[int(partition_id) % len(bc.client_accounts)]

    global_cid = "FAIL"
    if "config" in msg.content:
        global_cid = msg.content["config"].get("global_ipfs_cid", "FAIL")
    elif "configs" in msg.content:
        global_cid = msg.content["configs"].get("global_ipfs_cid", "FAIL")

    if global_cid != "FAIL" and global_cid != "UPLOAD_FAILED":
        print(f"\n📥 [Client {partition_id}] 收到服务器下发的全局 CID: {global_cid}")
        print(f"🔐 [Client {partition_id}] 正在向智能合约发起访问确权校验...")
        
        # 🌟 区块链融合：防搭便车的最核心逻辑
        authorized_cid = bc.request_cid(my_eth_address, threshold=1)
        
        if not authorized_cid:
            # 💡 极速修复：不抛出异常，而是让没有贡献度的节点从零开始本地训练（降级打工）
            print(f"🚫 [Client {partition_id}] 拦截成功：由于无历史贡献，智能合约拒绝了你的白嫖请求！")
            print(f"⚠️ [Client {partition_id}] 启用降级模式：本轮将使用本地初始模型【从零打工】，以赚取首笔贡献度！")
        else:
            print(f"🎉 [Client {partition_id}] 确权通过！允许使用资源，正在从 IPFS 下载密文...")
            download_dir = tempfile.mkdtemp(prefix=f"flwr_client_{partition_id}_dl_")
            
            try:
                # 注意：此处必须使用智能合约认证过的 authorized_cid 进行下载
                if ipfs.download(authorized_cid, download_dir):
                    fhe_path = os.path.join(download_dir, authorized_cid, "full_encrypted_model.pkl")
                    if os.path.exists(fhe_path):
                        fhe_handler = SelectiveFHEHandler()
                        decrypted_state_dict = fhe_handler.decrypt_model(fhe_path)
                        set_peft_model_state_dict(model, decrypted_state_dict)
                        print(f"✅ [Client {partition_id}] 成功吸收 Server 端发来的全局聚合经验！")
                    else:
                        print(f"🚨 [Client {partition_id}] 没找到模型文件: {fhe_path}")
                else:
                    print(f"❌ [Client {partition_id}] IPFS 下载失败。")
            except Exception as e:
                 print(f"❌ [Client {partition_id}] 解密加载失败: {e}")
            finally:
                 shutil.rmtree(download_dir, ignore_errors=True)
    else:
        print(f"\n🆕 [Client {partition_id}] 未收到全局 CID (这是第1轮)，使用本地初始权重开局。")

    try:
        current_round = msg.content["config"]["server-round"]
        save_path_str = msg.content["config"]["save_path"]
    except KeyError:
        current_round = 1
        save_path_str = f"./results/fallback_client_{partition_id}"

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

    print(f"🚀 [Client {partition_id}] 准备就绪，开始本地微调...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        train_results = trainer.train()

    temp_dir = tempfile.mkdtemp(prefix=f"flwr_client_{partition_id}_")
    trainer.model.save_pretrained(temp_dir, safe_serialization=False)
    
    generated_files = os.listdir(temp_dir)
    if not generated_files:
        raise RuntimeError(f"🚨 活见鬼了！HuggingFace 假装保存了，但文件夹是空的！")
    
    # 🌟 选择性同态加密拦截
    try:
        fhe_handler = SelectiveFHEHandler()
        plaintext_file_to_delete = fhe_handler.encrypt_full_model(temp_dir, temp_dir)
        if os.path.exists(plaintext_file_to_delete):
            os.remove(plaintext_file_to_delete)
            print(f"🗑️ [Client {partition_id}] 已彻底销毁本地明文权重: {plaintext_file_to_delete}")
    except Exception as e:
        print(f"❌ [Client {partition_id}] 同态加密失败，中断上传！错误信息: {e}")
        raise e

    print(f"☁️ [Client {partition_id}] Uploading hybrid parameters to IPFS...")
    cid = ipfs.upload_folder(temp_dir)
    safe_cid = str(cid) if cid else "UPLOAD_FAILED"
    
    if cid:
        print(f"✅ [Client {partition_id}] Upload Success! CID: {safe_cid}")
    else:
        print(f"❌ [Client {partition_id}] Upload Failed.")

    shutil.rmtree(temp_dir, ignore_errors=True)
    shutil.rmtree(training_arguments.output_dir, ignore_errors=True)

    metrics = {
        "train_loss": train_results.training_loss,
        "num_examples": len(trainset),
    }

    # 🌟 区块链融合：在回传信息时，附带上自己的以太坊地址，方便 Server 端智能合约分配积分
    configs = {
        "ipfs_cid": safe_cid,
        "eth_address": my_eth_address
    }

    content = RecordDict({
        "arrays": ArrayRecord({}),
        "metrics": MetricRecord(metrics),
        "configs": ConfigRecord(configs),
    })

    return Message(content=content, reply_to=msg)