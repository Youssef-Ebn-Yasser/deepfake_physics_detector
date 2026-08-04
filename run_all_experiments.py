import os
import subprocess
import sys

def run_command(command, log_file):
    print(f"\n=======================================================")
    print(f"Running: {command}")
    print(f"=======================================================\n")
    
    with open(log_file, "a") as f:
        f.write(f"\n=======================================================\n")
        f.write(f"Command: {command}\n")
        f.write(f"=======================================================\n\n")
        
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    
    with open(log_file, "a") as f:
        for line in process.stdout:
            sys.stdout.write(line)
            f.write(line)
            
    process.wait()

def update_config_features_dir(config_path, new_dir):
    import yaml
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    cfg['dataset']['features_dir'] = new_dir
    
    with open(config_path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

def main():
    log_file = "all_experiments_results.txt"
    config_path = "config/default_config.yaml"
    
    # Clear log file
    with open(log_file, "w") as f:
        f.write("Deepfake Physics Detector - Experiments Log\n")
        f.write("===========================================\n\n")

    experiments = [
        {"name": "Features v1", "dir": "DataSets/features"},
        {"name": "Features v2", "dir": "DataSets/features_v2"}
    ]

    for exp in experiments:
        print(f"\n\nStarting experiments for {exp['name']} ({exp['dir']})")
        update_config_features_dir(config_path, exp['dir'])
        
        # Train and Eval train_fusion_1_2_mid
        run_command("python pipelines/train_fusion_1_2_mid.py", log_file)
        run_command("python eval_fusion_1_2_mid.py", log_file)
        
        # Train and Eval train_master
        run_command("python pipelines/train_master.py", log_file)
        run_command("python main_eval.py", log_file)
        
    print(f"\nAll experiments finished. Results saved to {log_file}")

if __name__ == "__main__":
    main()
