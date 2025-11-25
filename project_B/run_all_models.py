"""
run_all_models.py
------------------------------------------------------------
High-level launcher for all experiments:

1)  Baseline runs for all 4 models on both imbalance scenarios
2)  Small-range hyper-parameter tuning for ResNet10/15/20
------------------------------------------------------------
"""

from two_stage_ictta_v3 import run_two_stage_ictta_on_scenario


# ------------------------------------------------------------
# 1) Baseline: 4 models × 2 scenarios
# ------------------------------------------------------------
def run_baseline_all_models():
    models = ["simplecnn", "resnet10", "resnet15", "resnet20"]
    scenarios = ["long_tail", "head_tail"]
    for m in models:
        for s in scenarios:
            lr1 = 0.05 if m == "simplecnn" else 0.1
            run_two_stage_ictta_on_scenario(
                model_name=m, imbalance_type=s,
                stage1_epochs=15, stage2_epochs=20,
                imb_factor=0.01, batch_size=128, log_dir=".",
                stage1_lr=lr1, stage2_lr=0.01,
                min_weight=1.0, max_weight=8.0,
                ema_alpha=0.8, stage2_loss_type="focal",
                stage2_focal_gamma=1.5, dynamic_mode="loss",
                exp_name=f"baseline_{m}_{s}",
            )


# ------------------------------------------------------------
# 2) Small-range tuning for ResNets
# ------------------------------------------------------------
def run_tuning_resnets():
    scenarios = ["long_tail", "head_tail"]

    # ResNet10
    resnet10_cfgs = [
        dict(name="r10_w1-5_lr2",  stage1_lr=0.1, stage2_lr=0.01,
             min_weight=1, max_weight=5, focal_gamma=1.5),
        dict(name="r10_w1-8_lr2",  stage1_lr=0.1, stage2_lr=0.01,
             min_weight=1, max_weight=8, focal_gamma=1.5),
        dict(name="r10_w2-8_lrsmall", stage1_lr=0.1, stage2_lr=0.005,
             min_weight=2, max_weight=8, focal_gamma=1.5),
    ]

    # ResNet15
    resnet15_cfgs = [
        dict(name="r15_lrsmall_w1-5_g1.5", stage1_lr=0.05, stage2_lr=0.005,
             min_weight=1, max_weight=5, focal_gamma=1.5),
        dict(name="r15_lrsmall_w1-6_g1.5", stage1_lr=0.05, stage2_lr=0.005,
             min_weight=1, max_weight=6, focal_gamma=1.5),
        dict(name="r15_lrsmall_w1-4_g1.5", stage1_lr=0.05, stage2_lr=0.005,
             min_weight=1, max_weight=4, focal_gamma=1.5),
    ]

    # ResNet20
    resnet20_cfgs = [
        dict(name="r20_lrsmall_w1-6_g1.5", stage1_lr=0.1, stage2_lr=0.005,
             min_weight=1, max_weight=6, focal_gamma=1.5),
        dict(name="r20_lrsmall_w1-4_g1.5", stage1_lr=0.1, stage2_lr=0.005,
             min_weight=1, max_weight=4, focal_gamma=1.5),
        dict(name="r20_lrsmall_w2-8_g1.5", stage1_lr=0.1, stage2_lr=0.005,
             min_weight=2, max_weight=8, focal_gamma=1.5),
    ]

    tuning_sets = {
        "resnet10": resnet10_cfgs,
        "resnet15": resnet15_cfgs,
        "resnet20": resnet20_cfgs,
    }

    for model, cfgs in tuning_sets.items():
        for s in scenarios:
            for cfg in cfgs:
                exp = f"{model}_{s}_{cfg['name']}"
                print(f"\n>>> TUNING {exp}")
                run_two_stage_ictta_on_scenario(
                    model_name=model, imbalance_type=s,
                    stage1_epochs=30, stage2_epochs=20,
                    imb_factor=0.01, batch_size=128, log_dir=".",
                    stage1_lr=cfg["stage1_lr"], stage2_lr=cfg["stage2_lr"],
                    min_weight=cfg["min_weight"], max_weight=cfg["max_weight"],
                    ema_alpha=0.8, stage2_loss_type="focal",
                    stage2_focal_gamma=cfg["focal_gamma"],
                    dynamic_mode="loss", exp_name=exp,
                )


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    run_baseline_all_models()
    run_tuning_resnets()
