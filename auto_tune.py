import os
import itertools
import subprocess

# 定义参数空间
lrs = [1e-4]  # 固定基线
betas = [1.0] # 固定基线
batch_sizes = [4, 8]
weight_decays = [1e-4, 5e-5]
cosine_flags = [0, 1]  # 0: 不用cosine, 1: 用cosine
# 可扩展其它参数

# 自动记录所有实验日志
log_records = []

def main():
    for lr, beta, batch_size, weight_decay, cosine_flag in itertools.product(lrs, betas, batch_sizes, weight_decays, cosine_flags):
        cosine_str = 'cosine' if cosine_flag else 'nocosine'
        savename = f"auto_lr{lr}_beta{beta}_bs{batch_size}_wd{weight_decay}_{cosine_str}"
        log_name = f"logs/{savename}.log"
        cmd = [
            "python", "enhanced_training.py",
            "--model", "swinmoe",
            "--batch_size", str(batch_size),
            "--img_size", "1024",
            "--max_epoch", "12",
            "--data_root", "data",
            "--data_name", "CVOGL_DroneAerial",
            "--gpu", "0",
            "--num_workers", "4",
            "--savename", savename,
            "--lr", str(lr),
            "--beta", str(beta),
            "--weight_decay", str(weight_decay),
            "--no-moe-entropy"
        ]
        if cosine_flag:
            cmd.append("--cosine")
        print("Running:", " ".join(cmd))
        subprocess.run(cmd)
        log_records.append({"savename": savename, "lr": lr, "beta": beta, "batch_size": batch_size, "weight_decay": weight_decay, "cosine": cosine_flag, "log": log_name})
    # 保存所有实验记录，便于后续分析
    with open("auto_tune_records.txt", "w", encoding="utf-8") as f:
        for rec in log_records:
            f.write(str(rec) + "\n")

if __name__ == "__main__":
    main() 