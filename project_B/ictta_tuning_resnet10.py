# ictta_tuning.py
#
# Small-scale hyperparameter tuning script:
# Runs two_stage_ictta under several groups of hyperparameters.
# Each experiment writes results to a separate log file for easy comparison.

from two_stage_ictta_v3 import run_two_stage_ictta


def main():
    # Configurable hyperparameter combinations
    configs = [
        {
            "name": "r10_w1-5_lr2",
            "model_name": "resnet10",
            "stage1_epochs": 30,
            "stage2_epochs": 20,
            "stage1_lr": 0.1,
            "stage2_lr": 0.01,
            "min_weight": 1.0,
            "max_weight": 5.0,
        },
        {
            "name": "r10_w1-8_lr2",
            "model_name": "resnet10",
            "stage1_epochs": 30,
            "stage2_epochs": 20,
            "stage1_lr": 0.1,
            "stage2_lr": 0.01,
            "min_weight": 1.0,
            "max_weight": 8.0,
        },
        {
            "name": "r10_w2-8_lrsmall",
            "model_name": "resnet10",
            "stage1_epochs": 30,
            "stage2_epochs": 20,
            "stage1_lr": 0.1,
            "stage2_lr": 0.005,
            "min_weight": 2.0,
            "max_weight": 8.0,
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
        )


if __name__ == "__main__":
    main()
