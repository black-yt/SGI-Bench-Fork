import os
from tqdm import tqdm
from datasets import load_dataset
import subprocess
import sys
import shutil
from pathlib import Path
sys.path.append('./')
from utils import multi_process

env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"


def run_script_in_folder(folder_path):
    """
    Run data.py (if exists) and main.py in the given folder,
    print immediate status, and return execution results.
    """
    # Change to the target folder
    original_dir = os.getcwd()
    os.chdir(folder_path)

    script_name = 'data_en.py'
    script_path_full = folder_path / script_name
    # Changed this line to print the full path
    # print(f"    Running {script_path_full}...", end=" ") # Print on the same line
    try:
        # Run the script and capture output
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=10*60,  # 10-minute timeout
            encoding="utf-8",
            env=env
        )
        if result.returncode == 0:
            # print(f"✅")
            result = (str(script_path_full), True, "")
        else:
            print(f"❌")
            error_message = result.stderr.strip() if result.stderr else "Unknown error"
            # print(f"      Error: {error_message}")
            result = (str(script_path_full), False, error_message)
    except subprocess.TimeoutExpired:
        print(f"❌")
        print(f"      Error: Execution timed out after 10 minutes")
        result = (str(script_path_full), False, "Execution timed out after 10 minutes")
    except Exception as e:
        print(f"❌")
        print(f"      Error: {e}")
        result = (str(script_path_full), False, str(e))
    
    # Return to original directory
    os.chdir(original_dir)
    return result


dataset = load_dataset("PrismaX/SGI-DryExperiment")
save_dir = './task_3_dry_experiment/codes'
os.makedirs(save_dir, exist_ok=True)

code_dir_list = []
for item in tqdm(dataset['test'], desc='Creating folder'):
    for unit_test_idx in range(5):
        code_dir = os.path.join(save_dir, item['idx'], f"unit_test_{unit_test_idx}")
        code_dir_list.append({'folder_path': Path(code_dir)})
        os.makedirs(code_dir, exist_ok=True)
        data_dir = os.path.join(save_dir, item['idx'], f"unit_test_{unit_test_idx}", 'data')
        os.makedirs(data_dir, exist_ok=True)

        with open(os.path.join(code_dir, "data_en.py"), "w", encoding="utf-8") as f:
            f.write(item[f"unit_test_{unit_test_idx}_data"])
        with open(os.path.join(code_dir, "main_en.py"), "w", encoding="utf-8") as f:
            f.write(item["main_code"])


shutil.copytree("task_3_dry_experiment/data/SGI_Code_0206", "task_3_dry_experiment/codes/SGI_Code_0206/unit_test_0/data/mnist_raw", dirs_exist_ok=True)
shutil.copytree("task_3_dry_experiment/data/SGI_Code_0206", "task_3_dry_experiment/codes/SGI_Code_0206/unit_test_1/data/mnist_raw", dirs_exist_ok=True)
shutil.copytree("task_3_dry_experiment/data/SGI_Code_0206", "task_3_dry_experiment/codes/SGI_Code_0206/unit_test_2/data/mnist_raw", dirs_exist_ok=True)
shutil.copytree("task_3_dry_experiment/data/SGI_Code_0206", "task_3_dry_experiment/codes/SGI_Code_0206/unit_test_3/data/mnist_raw", dirs_exist_ok=True)
shutil.copytree("task_3_dry_experiment/data/SGI_Code_0206", "task_3_dry_experiment/codes/SGI_Code_0206/unit_test_4/data/mnist_raw", dirs_exist_ok=True)

all_results = multi_process(code_dir_list, run_script_in_folder, 100)

# --- Final Summary ---
print("\n--- Execution Summary ---")
errors = [(script, error) for script, success, error in all_results if not success]
if errors:
    print(f"\n❌ Some scripts failed to execute:")
    for script, error in errors:
        print(f"    Script: {script}")
        print(f"    Error: {error}\n")
        print('-----------------------------------------------------')
else:
    print(f"\n✅ All eligible scripts executed successfully across all folders!")