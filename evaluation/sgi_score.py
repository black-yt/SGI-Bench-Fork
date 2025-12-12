import json
import os
import sys
sys.path.append('.')
from utils import show_results


model_name = 'gpt-4.1'
discipline = "['all']"
discipline_list = ['astronomy', 'chemistry', 'earth', 'energy', 'information', 'life', 'material', 'mathematics', 'neuroscience', 'physics']
if len(sys.argv) > 1:
    model_name = sys.argv[1]
    sys.argv = sys.argv[1:]
if len(sys.argv) > 1:
    discipline = sys.argv[1]
    discipline_list = eval(discipline)
    sys.argv = sys.argv[1:]
print(f'Evaluating {model_name} on {discipline}')


with open(os.path.join('./task_1_deep_research/logs', f"{model_name.replace('/', '_')}{discipline}.json"), 'r', encoding='utf-8') as json_file:
    deep_research_logs = json.load(json_file)
print("Deep Research")
print("EM:", Deep_Research:=show_results(deep_research_logs, "exact_match")*100)
print("SLA:", show_results(deep_research_logs, "step_level_acc")*100)
print()


with open(os.path.join('./task_2_idea_generation/logs', f"{model_name.replace('/', '_')}{discipline}_evaluation.json"), 'r', encoding='utf-8') as json_file:
    idea_generation_logs = json.load(json_file)
print("Idea Generation")
print("Effectiveness:", show_results(idea_generation_logs, "effectiveness"))
print("Novelty:", show_results(idea_generation_logs, "novelty"))
print("Detailedness:", show_results(idea_generation_logs, "detailedness"))
print("Feasibility:", show_results(idea_generation_logs, "feasibility"))
print("Final Score:", Idea_Generation:=show_results(idea_generation_logs, "final_score"))
print()


with open(os.path.join('./task_3_dry_experiment/logs', f"{model_name.replace('/', '_')}{discipline}.json"), 'r', encoding='utf-8') as json_file:
    dry_experiment_logs = json.load(json_file)
print("Dry Experiment")
print("PassAll@5:", Dry_Experiment:=show_results(dry_experiment_logs, "PassAll@5")*100)
print("PassAll@3:", show_results(dry_experiment_logs, "PassAll@3")*100)
print("PassAll@1:", show_results(dry_experiment_logs, "PassAll@1")*100)
print("AET:", show_results(dry_experiment_logs, "AET"))
print("SER:", show_results(dry_experiment_logs, "SER")*100)
print()


with open(os.path.join('./task_3_wet_experiment/logs', f"{model_name.replace('/', '_')}{discipline}.json"), 'r', encoding='utf-8') as json_file:
    wet_experiment_logs = json.load(json_file)
print("Wet Experiment")
print("SS:", show_results(wet_experiment_logs, "action_sequence_similarity")*100)
print("PA:", show_results(wet_experiment_logs, "parameter_accuracy")*100)
print("PA:", Wet_Experiment:=show_results(wet_experiment_logs, "final_score")*100)
print()


with open(os.path.join('./task_4_experimental_reasoning/logs', f"{model_name.replace('/', '_')}{discipline}.json"), 'r', encoding='utf-8') as json_file:
    experimental_reasoning_logs = json.load(json_file)
print("Experimental Reasoning")
print("MCA:", Experimental_Reasoning:=show_results(experimental_reasoning_logs, "MCA")*100)
print("RV:", show_results(experimental_reasoning_logs, "RV")*100)
print()

print("SGI_Score:", (Deep_Research+Idea_Generation+Dry_Experiment+Wet_Experiment+Experimental_Reasoning)/5)