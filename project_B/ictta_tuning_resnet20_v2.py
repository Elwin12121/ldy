# ictta_tuning_resnet20_v2.py
#
# ResNet20 Round 2 Tuning:
#  - Lower learning rates
#  - Adjust dynamic weight range
#  - Modify focal gamma

from two_stage_ictta_v3 import run_two_stage_ictta


def main():
    configs = [
        # 1) Lower learning rates, moderate weight range
        {
            "name": "r20_lrsmall_w1-6",
            "model_name": "resnet20",
            "stage1_epochs": 30,
            "stage2_epochs": 20,
            "stage1_lr": 0.05,      # ↓ half of 0.1
            "stage2_lr": 0.005,     # ↓ adjusted accordingly
            "min_weight": 1.0,
            "max_weight": 6.0,
            "focal_gamma": 1.5,
        },
        # 2) Softer focal (gamma=1.0)
        {
            "name": "r20_lrsmall_w1-6_g1",
            "model_name": "resnet20",
            "stage1_epochs": 30,
            "stage2_epochs": 20,
            "stage1_lr": 0.05,
            "stage2_lr": 0.005,
            "min_weight": 1.0,
            "max_weight": 6.0,
            "focal_gamma": 1.0,     # closer to weighted CE
        },
        # 3) Smaller weight range, emphasizing stability
        {
            "name": "r20_lrsmall_w1-4",
            "model_name": "resnet20",
            "stage1_epochs": 30,
            "stage2_epochs": 20,
            "stage1_lr": 0.05,
            "stage2_lr": 0.005,
            "min_weight": 1.0,
            "max_weight": 4.0,
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
            imb_factor=0.01,
            batch_size=128,
            log_file=log_file,
            stage1_lr=cfg["stage1_lr"],
            stage2_lr=cfg["stage2_lr"],
            min_weight=cfg["min_weight"],
            max_weight=cfg["max_weight"],

            ema_alpha=0.8,
            stage2_loss_type="focal",
            stage2_focal_gamma=cfg["focal_gamma"],
            dynamic_mode="loss",
            use_balanced_stage2_loader=True,
        )


if __name__ == "__main__":
    main()
