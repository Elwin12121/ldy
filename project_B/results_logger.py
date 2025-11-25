import csv
import os

def log_epoch_result(
    filename: str,
    stage: str,
    epoch: int,
    model_name: str,
    method: str,
    metrics: dict,
):
    """
    Record the training results of each epoch into a CSV file.

    Args:
        filename: Path to the log file, e.g., 'results_log.csv'
        stage: 'stage1' or 'stage2'
        epoch: Current epoch number
        model_name: Model name (e.g., 'resnet10')
        method: Training method name (e.g., 'two_stage_ictta')
        metrics: Dictionary containing loss / acc / macro_f1 / tail_macro_f1
    """
    dirpath = os.path.dirname(filename)
    if dirpath != "":
        os.makedirs(dirpath, exist_ok=True)

    fieldnames = [
        "stage", "epoch", "model", "method",
        "loss", "acc", "macro_f1", "tail_macro_f1"
    ]

    file_exists = os.path.isfile(filename)
    with open(filename, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "stage": stage,
            "epoch": epoch,
            "model": model_name,
            "method": method,
            "loss": round(metrics.get("loss", 0), 4),
            "acc": round(metrics.get("acc", 0), 4),
            "macro_f1": round(metrics.get("macro_f1", 0), 4),
            "tail_macro_f1": round(metrics.get("tail_macro_f1", 0), 4),
        })
