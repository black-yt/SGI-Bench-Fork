import os
import json
import subprocess
import time
from pathlib import Path
import sys
sys.path.append('./')
from utils import multi_process

env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"


model_name = 'gpt-4.1'
save_dir = './task_3_dry_experiment/logs'
script_name = f"main_[{model_name.replace('/', '')}].py"


def run_script(ques_dict):
    ques_dict['unit_test'] = []
    for unit_test_idx in range(5):
        folder_path = os.path.join('./task_3_dry_experiment/codes', ques_dict['idx'], f"unit_test_{unit_test_idx}")

        # Change to the target folder
        original_dir = os.getcwd()
        os.chdir(folder_path)

        # Changed this line to print the full path
        # print(f"    Running {script_path_full}...")
        try:
            # Run the script and capture output
            start_time = time.time()
            result = subprocess.run(
                [sys.executable, script_name],
                capture_output=True,
                text=True,
                timeout=300,  # 5-minute timeout
                encoding="utf-8",
                env=env
            )
            end_time = time.time()
            elapsed = end_time - start_time
            model_code_output = f"{result.stderr}\n{result.stdout}".strip()

            unit_test_dict = {}
            if result.returncode == 0:
                # print(f"✅")
                unit_test_dict["model_error"] = "[No Error]"
                unit_test_dict["model_runtime"] = elapsed
                unit_test_dict["model_returncode"] = result.returncode
                unit_test_dict["model_code_output"] = model_code_output
            else:
                # print(f"❌")
                # print(f"      Error: {error_message}")
                unit_test_dict["model_error"] = "[WRONG]" + result.stderr.strip() if result.stderr else "Unknown error"
                unit_test_dict["model_runtime"] = elapsed
                unit_test_dict["model_returncode"] = result.returncode
                unit_test_dict["model_code_output"] = model_code_output
        except subprocess.TimeoutExpired:
            # print(f"❌")
            # print(f"      Error: Execution timed out after 5 minutes")
            unit_test_dict["model_error"] = "[WRONG]Execution timed out after 5 minutes"
            unit_test_dict["model_runtime"] = 300.0
            unit_test_dict["model_returncode"] = -1 # 被终止
            unit_test_dict["model_code_output"] = unit_test_dict["model_error"]
        except Exception as e:
            # print(f"❌")
            # print(f"      Error: {e}")
            unit_test_dict["model_error"] = "[WRONG]"+str(e)
            unit_test_dict["model_runtime"] = -1
            unit_test_dict["model_returncode"] = 1 # 出错
            unit_test_dict["model_code_output"] = unit_test_dict["model_error"]

        # Return to original directory
        os.chdir(original_dir)
        ques_dict['unit_test'].append(unit_test_dict)
    return ques_dict


with open(os.path.join(save_dir, f"{model_name.replace('/', '_')}.json"), 'r', encoding='utf-8') as json_file:
    model_answer = json.load(json_file)


inp_list = [{'ques_dict': ques} for ques in model_answer]
out_list = multi_process(inp_list, run_script, 100)

with open(os.path.join(save_dir, f"{model_name.replace('/', '_')}.json"), 'w', encoding='utf-8') as json_file:
    json.dump(out_list, json_file, ensure_ascii=False, indent=4)