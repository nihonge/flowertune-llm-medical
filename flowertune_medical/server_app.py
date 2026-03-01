"""flowertune-medical: A Flower / FlowerTune app."""

import os
from datetime import datetime

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.common.config import unflatten_dict
from flwr.serverapp import Grid, ServerApp
from omegaconf import DictConfig
from peft import get_peft_model_state_dict

from flowertune_medical.dataset import replace_keys
from flowertune_medical.models import get_model
from flowertune_medical.strategy import FlowerTuneLlm

# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    # Create output directory given current timestamp
    current_time = datetime.now()
    folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(os.getcwd(), f"results/{folder_name}")
    os.makedirs(save_path, exist_ok=True)

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))

    # Get initial model weights
    init_model = get_model(cfg.model)
    arrays = ArrayRecord(get_peft_model_state_dict(init_model))

    # Define strategy
    strategy = FlowerTuneLlm(
        fraction_train=cfg.strategy.fraction_train,
        fraction_evaluate=cfg.strategy.fraction_evaluate,
    )

    # Start strategy, run FedAvg for `num_rounds`
    strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"save_path": save_path}),
        num_rounds=num_rounds,
        evaluate_fn=get_evaluate_fn(
            cfg.model, cfg.train.save_every_round, num_rounds, save_path
        ),
    )


# Get function that will be executed by the strategy
# Here we use it to save global model checkpoints
def get_evaluate_fn(model_cfg, save_every_round, total_round, save_path):
    """在全同态加密框架下，Server 无法解密模型，因此仅记录聚合完成的状态。"""

    def evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        print(f"\n🛡️ [Server] 轮次 {server_round} 评估阶段。")
        print(f"🔒 [Server] 全局模型目前处于 FHE 密文状态躺在 IPFS 中。")
        print(f"🚫 [Server] Server 无私钥，无权解密，已跳过明文保存步骤。")
        
        # 我们不再尝试将 arrays 转成 PyTorch 字典进行保存，因为它现在是空的
        # 全局密文的生成、CID 的分发都已经在 Strategy 的 aggregate_fit 中完成了
        
        return MetricRecord()

    return evaluate