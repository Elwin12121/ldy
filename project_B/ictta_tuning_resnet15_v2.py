"""
ictta_tuning_resnet15_v2.py

Round 2: Fine-grained tuning specifically for ResNet15.
Based on your current best configuration, this round further explores:
  - A larger dynamic weight range (max_weight = 7)
  - Stronger focal focusing (gamma = 2.0)
  - Longer Stage 1 training (40 epochs)

Each configuration will be executed sequentially, with results saved
to separate results_*.csv files.
"""

from two_stage_ictta_v3 import run_two_stage_ictta


def main():
    # Notes:
    #   - model_name is fixed as "resnet15"
    #   - Use your best configuration (lr=0.05/0.005, w1–5, gamma=1.5) as baseline
    #   - Other configurations are small adjustments based on this baseline
    configs = [
        # 0️⃣ Baseline: your best current configuration, for comparison
        {
            "name": "r15_v2_base_lrsmall_w1-5_g1p5",
            "model_name": "resnet15",
            "stage1_epochs": 30,
            "stage2_epochs": 20,
            "stage1_lr": 0.05,
            "stage2_lr": 0.005,
            "min_weight": 1.0,
            "max_weight": 5.0,
            "focal_gamma": 1.5,
        },

        # 1️⃣ Expand dynamic weight range: w ∈ [1, 7], stronger focus on tail classes
        {
            "name": "r15_v2_lrsmall_w1-7_g1p5",
            "model_name": "resnet15",
            "stage1_epochs": 30,
            "stage2_epochs": 20,
            "stage1_lr": 0.05,
            "stage2_lr": 0.005,
            "min_weight": 1.0,
            "max_weight": 7.0,
            "focal_gamma": 1.5,
        },

        # 2️⃣ Increase focal gamma: γ = 2.0, emphasizing harder samples more
        {
            "name": "r15_v2_lrsmall_w1-5_g2",
            "model_name": "resnet15",
            "stage1_epochs": 30,
            "stage2_epochs": 20,
            "stage1_lr": 0.05,
            "stage2_lr": 0.005,
            "min_weight": 1.0,
            "max_weight": 5.0,
            "focal_gamma": 2.0,
        },

        # 3️⃣ Combine both: larger weight range + stronger focal (most aggressive)
        {
            "name": "r15_v2_lrsmall_w1-7_g2",
            "model_name": "resnet15",
            "stage1_epochs": 30,
            "stage2_epochs": 20,
            "stage1_lr": 0.05,
            "stage2_lr": 0.005,
            "min_weight": 1.0,
            "max_weight": 7.0,
            "focal_gamma": 2.0,
        },

        # 4️⃣ Extend Stage 1: 40 epochs for richer representation before Stage 2
        {
            "name": "r15_v2_stage1_40ep_lrsmall_w1-5_g1p5",
            "model_name": "resnet15",
            "stage1_epochs": 40,   # ⭐ More representation learning
            "stage2_epochs": 20,
            "stage1_lr": 0.05,
            "stage2_lr": 0.005,
            "min_weight": 1.0,
            "max_weight": 5.0,
            "focal_gamma": 1.5,
        },
    ]

    for cfg in configs:
        exp_name = cfg["name"]
        log_file = f"results_{exp_name}.csv"

        print("\n" + "=" * 60)
        print(f"Running config: {exp_name}")
        print("=" * 60)

        run_two_stage_ictta(
            model_name=cfg["model_name"],
            stage1_epochs=cfg["stage1_epochs"],
            stage2_epochs=cfg["stage2_epochs"],
            imb_factor=0.01,             # Keep consistent with previous experiments (long-tail 0.01)
            batch_size=128,              # Can be lowered to 64 if training is slow

            log_file=log_file,
            stage1_lr=cfg["stage1_lr"],
            stage2_lr=cfg["stage2_lr"],
            min_weight=cfg["min_weight"],
            max_weight=cfg["max_weight"],

            # Fixed settings in v3: EMA + focal + loss-based dynamic + balanced Stage2 loader
            ema_alpha=0.8,
            stage2_loss_type="focal",
            stage2_focal_gamma=cfg["focal_gamma"],
            dynamic_mode="loss",
            use_balanced_stage2_loader=True,
        )


if __name__ == "__main__":
    main()
