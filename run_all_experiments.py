import os
import subprocess
import sys
import re
import yaml

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def clean_ansi(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def run_command(command, log_file, step_name):
    print(f"\n=======================================================")
    print(f"Running Step: {step_name}")
    print(f"Command: {command}")
    print(f"=======================================================\n")
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n=======================================================\n")
        f.write(f"STEP: {step_name}\n")
        f.write(f"=======================================================\n\n")
        
    process = subprocess.Popen(
        command, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        env=dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3")
    )
    
    with open(log_file, "a", encoding="utf-8") as f:
        for line in process.stdout:
            # طباعة السطر فوراً على الشاشة
            sys.stdout.write(line)
            sys.stdout.flush()
            
            clean_line = clean_ansi(line)
            
            # الفلاتر الذكية لحفظ النتائج النقية في الملف
            keep_line = False
            
            # 1. التقاط أسطر الـ Epoch والـ Evaluation
            if clean_line.strip().startswith("Epoch "):
                keep_line = True
            elif any(metric in clean_line for metric in ["val_loss:", "val_fusion_output_auc:", "val_acc:"]):
                # احتفظ بالسطر إذا كان يحتوي على مقاييس ولا يحتوي على شريط تقدم جاري (=====>)
                if "=====" not in clean_line:
                    keep_line = True
            # 2. التقاط تقارير التقييم الشاملة وسطور الجداول
            elif any(kw in clean_line for kw in [
                "Test Accuracy:", "Test AUC:", "Test F1-Score:", "Test Loss:", 
                "STRICT SPLIT", "Train:", "Val:", "Test:", "Classification Report:",
                "precision", "recall", "f1-score", "support", "accuracy", "macro avg", "weighted avg"
            ]):
                keep_line = True
                
            if keep_line:
                f.write(clean_line)
                f.flush()
                
    process.wait()
    
    # فحص ما إذا فشل السكريبت
    if process.returncode != 0:
        error_msg = f"\n[ERROR] Step '{step_name}' failed with exit code {process.returncode}. Stopping automation.\n"
        print(error_msg)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(error_msg)
        sys.exit(process.returncode)

def update_config_features_dir(config_path, new_dir):
    with open(config_path, 'r', encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        
    cfg['dataset']['features_dir'] = new_dir
    
    with open(config_path, 'w', encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"[CONFIG UPDATE] Updated features_dir to: '{new_dir}' in {config_path}")

def main():
    log_file = "all_experiments_results.txt"
    config_path = os.path.join("config", "default_config.yaml")
    
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("Deepfake Physics Detector - Clean Metrics Log\n")
        f.write("=============================================\n\n")

    experiments = [
        {"name": "Features v1", "dir": os.path.join("DataSets", "features")},
        {"name": "Features v2", "dir": os.path.join("DataSets", "features_v2")}
    ]

    for exp in experiments:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n\n>>> STARTING EXPERIMENT SET: {exp['name']} <<<\n")
            
        update_config_features_dir(config_path, exp['dir'])
        
        # 0. Pre-Train Sub-Systems (Required because architecture changed!)
        run_command("python pipelines/train_subsystem1.py", log_file, f"Pre-train Sub-System 1 ({exp['name']})")
        run_command("python pipelines/train_subsystem2.py --balanced", log_file, f"Pre-train Sub-System 2 ({exp['name']})")
        
        # 1. Mid-Level Fusion (System 1 + 2)
        run_command("python pipelines/train_fusion_1_2_mid.py", log_file, f"Train Mid-Level Fusion ({exp['name']})")

        # 2. End-to-End Master Model (System 1 + 2)
        run_command("python pipelines/train_master.py", log_file, f"Train Master Fusion ({exp['name']})")
        
        # 3. Final Evaluation
        run_command("python main_eval.py", log_file, f"Eval Master Fusion ({exp['name']})")
        
    print(f"\nAll experiments finished successfully! Results saved to {log_file}")

if __name__ == "__main__":
    main()