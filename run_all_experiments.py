import os
import subprocess
import sys
import re

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def clean_ansi(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def run_command(command, log_file, step_name):
    print(f"\n=======================================================")
    print(f"Running Step: {step_name}")
    print(f"Command: {command}")
    print(f"=======================================================\n")
    
    with open(log_file, "a") as f:
        f.write(f"\n=======================================================\n")
        f.write(f"STEP: {step_name}\n")
        f.write(f"=======================================================\n\n")
        
    process = subprocess.Popen(
        command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3")
    )
    
    with open(log_file, "a") as f:
        for line in process.stdout:
            sys.stdout.write(line)
            
            clean_line = clean_ansi(line)
            keep_line = False
            
            if clean_line.startswith("Epoch "):
                keep_line = True
            elif "val_loss:" in clean_line or "val_fusion_logits_acc:" in clean_line:
                if "=" not in clean_line:  # Exclude the visual progress bar lines
                    keep_line = True
            elif any(kw in clean_line for kw in ["Test Accuracy:", "Test AUC:", "Test F1-Score:", "Test Loss:", "STRICT SPLIT", "Train:", "Val:", "Test:", "Classification Report:", "precision", "recall"]):
                keep_line = True
                
            if keep_line:
                f.write(clean_line)
                
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
    
    with open(log_file, "w") as f:
        f.write("Deepfake Physics Detector - Clean Metrics Log\n")
        f.write("=============================================\n\n")

    experiments = [
        {"name": "Features v1", "dir": "DataSets/features"},
        {"name": "Features v2", "dir": "DataSets/features_v2"}
    ]

    for exp in experiments:
        with open(log_file, "a") as f:
            f.write(f"\n\n>>> STARTING EXPERIMENT SET: {exp['name']} <<<\n")
            
        update_config_features_dir(config_path, exp['dir'])
        
        # Mid-Level Fusion (System 1 + 2)
        run_command("python pipelines/train_fusion_1_2_mid.py", log_file, f"Train Mid-Level Fusion ({exp['name']})")

        # End-to-End Master Model (System 1 + 2)
        run_command("python pipelines/train_master.py", log_file, f"Train Master Fusion ({exp['name']})")
        # Assuming you want to eval master on celeb df too or main test set
        run_command("python main_eval.py", log_file, f"Eval Master Fusion ({exp['name']})")
        
    print(f"\nAll experiments finished. Results saved to {log_file}")

if __name__ == "__main__":
    main()
