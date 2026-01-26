import os
import json
import argparse
from typing import Dict, List, Any, Union

REQUIRED_FILES = [
    'participant_info.json',
    'Quiz_1.jsonl',
    'Quiz_2.jsonl',
    'Quiz_3.jsonl',
    'Quiz_4.jsonl',
    'Quiz_5.jsonl',
    'Quiz_1_action.jsonl',
    'Quiz_2_action.jsonl',
    'Quiz_3_action.jsonl',
    'Quiz_4_action.jsonl',
    'Quiz_5_action.jsonl'
]

def validate_type(value, expected_type, path):
    if isinstance(expected_type, tuple):
        if not isinstance(value, expected_type):
            return [f"{path}: Expected {expected_type}, got {type(value).__name__}"]
    elif expected_type == float:
        # Allow int for float
        if not isinstance(value, (float, int)):
             return [f"{path}: Expected float (or int), got {type(value).__name__}"]
    else:
        if not isinstance(value, expected_type):
            return [f"{path}: Expected {expected_type.__name__}, got {type(value).__name__}"]
    return []

def validate_schema(data, schema, path=""):
    errors = []
    
    if isinstance(schema, dict) and 'type' in schema:
        # Complex type definition
        expected_type = schema['type']
        if not isinstance(data, expected_type):
            return [f"{path}: Expected {expected_type.__name__}, got {type(data).__name__}"]
        
        if expected_type == dict:
            if 'keys' in schema:
                # Fixed keys
                for key, val_schema in schema['keys'].items():
                    if key not in data:
                        errors.append(f"{path}: Missing key '{key}'")
                    else:
                        errors.extend(validate_schema(data[key], val_schema, path=f"{path}.{key}"))
            elif 'key_type' in schema and 'value_type' in schema:
                # Variable keys
                for key, value in data.items():
                    errors.extend(validate_type(key, schema['key_type'], path=f"{path}.key({key})"))
                    errors.extend(validate_schema(value, schema['value_type'], path=f"{path}[{key}]"))
        
        elif expected_type == list:
            if 'element_type' in schema:
                for i, item in enumerate(data):
                    errors.extend(validate_schema(item, schema['element_type'], path=f"{path}[{i}]"))
                    
    elif isinstance(schema, type) or isinstance(schema, tuple):
        # Simple type definition
        errors.extend(validate_type(data, schema, path))
        
    return errors

SCHEMAS = {
    'participant_info.json': {
        'type': dict,
        'keys': {
            'id': str,
            'level': str,
            'age': int,
            'gender': str,
            'school': str,
            'major': str,
            'country': str
        }
    },
    'Quiz_1.jsonl': {
        'type': dict,
        'keys': {
            'idx': str,
            'analysis_answer': str,
            'feedback_quality': str,
            'feedback_quality_comment': str,
            'feedback_value': str,
            'feedback_value_comment': str,
            'time': (int, float)
        }
    },
    'Quiz_2.jsonl': {
        'type': dict,
        'keys': {
            'idx': str,
            'Idea': str,
            'ImplementationSteps': {'type': dict, 'key_type': str, 'value_type': str},
            'ImplementationOrder': {'type': list, 'element_type': str},
            'Dataset': str,
            'EvaluationMetrics': {'type': dict, 'key_type': str, 'value_type': str},
            'ExpectedOutcome': str,
            'feedback_quality': str,
            'feedback_quality_comment': str,
            'feedback_value': str,
            'feedback_value_comment': str,
            'time': (int, float)
        }
    },
    'Quiz_3.jsonl': {
        'type': dict,
        'keys': {
            'idx': str,
            'completed_code': str,
            'feedback_quality': str,
            'feedback_quality_comment': str,
            'feedback_value': str,
            'feedback_value_comment': str,
            'time': (int, float)
        }
    },
    'Quiz_4.jsonl': {
        'type': dict,
        'keys': {
            'idx': str,
            'experimental_procedure': str,
            'feedback_quality': str,
            'feedback_quality_comment': str,
            'feedback_value': str,
            'feedback_value_comment': str,
            'time': (int, float)
        }
    },
    'Quiz_5.jsonl': {
        'type': dict,
        'keys': {
            'idx': str,
            'analysis_answer': str,
            'feedback_quality': str,
            'feedback_quality_comment': str,
            'feedback_value': str,
            'feedback_value_comment': str,
            'time': (int, float)
        }
    },
    'Quiz_1_action.jsonl': {
        'type': dict,
        'keys': {
            'id': str,
            'action': {'type': list, 'element_type': str}
        }
    },
    'Quiz_2_action.jsonl': {
        'type': dict,
        'keys': {
            'id': str,
            'action': {'type': list, 'element_type': str}
        }
    },
    'Quiz_3_action.jsonl': {
        'type': dict,
        'keys': {
            'id': str,
            'action': {'type': list, 'element_type': str}
        }
    },
    'Quiz_4_action.jsonl': {
        'type': dict,
        'keys': {
            'id': str,
            'action': {'type': list, 'element_type': str}
        }
    },
    'Quiz_5_action.jsonl': {
        'type': dict,
        'keys': {
            'id': str,
            'action': {'type': list, 'element_type': str}
        }
    }
}

def process_folder(folder_path):
    print(f"Checking folder: {folder_path}")
    
    folder_errors = []
    stats = {}
    
    # Check required files
    for filename in REQUIRED_FILES:
        file_path = os.path.join(folder_path, filename)
        if not os.path.exists(file_path):
            folder_errors.append(f"Missing file: {filename}")
            continue
            
        try:
            if filename.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    errors = validate_schema(data, SCHEMAS[filename], path=filename)
                    if errors:
                        folder_errors.extend(errors)
                    stats[filename] = 1
            elif filename.endswith('.jsonl'):
                count = 0
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line: continue
                        try:
                            data = json.loads(line)
                            errors = validate_schema(data, SCHEMAS[filename], path=f"{filename}:line_{line_num}")
                            if errors:
                                folder_errors.extend(errors)
                            count += 1
                        except json.JSONDecodeError as e:
                            folder_errors.append(f"{filename}:line_{line_num}: JSON decode error: {e}")
                stats[filename] = count
                print(f"  {filename}: {count} records")
                
        except Exception as e:
            folder_errors.append(f"Error reading {filename}: {e}")
            
    if folder_errors:
        print("  Errors found:")
        for err in folder_errors:
            print(f"    - {err}")
    else:
        print("  Status: OK")
        
    return folder_errors, stats

def main():
    parser = argparse.ArgumentParser(description="Validate annotation data.")
    parser.add_argument("root_dir", nargs='?', default=".", help="Root directory containing annotator folders")
    args = parser.parse_args()
    
    root_dir = args.root_dir
    
    total_stats = {f: 0 for f in REQUIRED_FILES}
    total_errors = 0
    annotator_count = 0
    
    # Walk through the directory to find annotator folders
    # Assuming annotator folders are the leaf directories containing the required files
    # Or we can assume any folder that contains participant_info.json is an annotator folder
    
    annotator_folders = []
    for root, dirs, files in os.walk(root_dir):
        if 'participant_info.json' in files:
            annotator_folders.append(root)
            
    if not annotator_folders:
        print(f"No annotator folders found in {root_dir}")
        return

    print(f"Found {len(annotator_folders)} annotator folders.")
    print("-" * 50)

    for folder in annotator_folders:
        errors, stats = process_folder(folder)
        if errors:
            total_errors += len(errors)
        
        for k, v in stats.items():
            if k in total_stats:
                total_stats[k] += v
        annotator_count += 1
        print("-" * 50)
        
    print("\nSummary:")
    print(f"Total Annotators: {annotator_count}")
    print(f"Total Errors: {total_errors}")
    print("Total Records:")
    for k, v in total_stats.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
