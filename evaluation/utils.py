import base64
import io
import os
from tqdm import tqdm
try:
    from openai import OpenAI
except Exception as e:
    print(str(e))
import concurrent.futures
from typing import Union
import json
import ast


def muti_thread(inp_list, function, max_workers=40):
    results = [None] * len(inp_list)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(function, **item): index
            for index, item in enumerate(inp_list)
        }
        
        for future in tqdm(concurrent.futures.as_completed(future_to_index), total=len(future_to_index)):
            index = future_to_index[future]
            try:
                result = future.result()
                results[index] = result
            except Exception as e:
                print(f"Error processing item {inp_list[index]}: {str(e)}")
    
    return results


def multi_process(inp_list, function, max_workers=40):
    results = [None] * len(inp_list)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(function, **item): index
            for index, item in enumerate(inp_list)
        }

        for future in tqdm(concurrent.futures.as_completed(future_to_index), total=len(future_to_index)):
            index = future_to_index[future]
            try:
                result = future.result()
                results[index] = result
            except Exception as e:
                print(f"Error processing item {inp_list[index]}: {str(e)}")
    
    return results


def extract_final_answer(answer_with_thinking: str, start_tag='<answer>', end_tag='</answer>'):
    answer_with_thinking = str(answer_with_thinking)
    start_index = answer_with_thinking.rfind(start_tag)
    if start_index != -1:
        end_index = answer_with_thinking.find(end_tag, start_index)
        if end_index != -1:
            return answer_with_thinking[start_index + len(start_tag):end_index].strip()
    return None


def b64_encode_image(img) -> str:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


class LLM:
    def __init__(self, model='gpt-4.1', **kwargs):
        self.api_key = kwargs.get('api_key', os.environ.get('OPENAI_API_KEY')) # export OPENAI_API_KEY="xxxxx"
        self.base_url = kwargs.get('base_url', os.environ.get('OPENAI_BASE_URL')) # export OPENAI_BASE_URL="xxxxx"
        self.model = model
        if not self.api_key:
            raise ValueError("API key is required.")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def __call__(self, query=None, **kwargs):
        system_prompt = kwargs.get('system_prompt', 'You are a helpful assistant.')
        max_tokens = kwargs.get('max_tokens', None)
        temperature = kwargs.get('temperature', 0)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        assistant_response = response.choices[0].message.content
        return assistant_response


class VLM:
    def __init__(self, model='gpt-4.1', **kwargs):
        self.api_key = kwargs.get('api_key', os.environ.get('OPENAI_API_KEY')) # export OPENAI_API_KEY="xxxxx"
        self.base_url = kwargs.get('base_url', os.environ.get('OPENAI_BASE_URL')) # export OPENAI_BASE_URL="xxxxx"
        self.model = model
        if not self.api_key:
            raise ValueError("API key is required.")
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def __call__(self, images=None, query=None, **kwargs):
        system_prompt = kwargs.get('system_prompt', 'You are a helpful assistant.')
        max_tokens = kwargs.get('max_tokens', None)
        temperature = kwargs.get('temperature', 0)

        image_msgs = []
        if images is not None:
            for img in images:
                b64 = b64_encode_image(img)
                image_msgs.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"}
                })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": image_msgs + [{"type": "text", "text": query}]},
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        assistant_response = response.choices[0].message.content
        return assistant_response


class AnswerPaser:
    def __init__(self):
        self.paser = LLM('gpt-4.1-mini')

    def get_parser_prompt(self, text: str, example: Union[str, list, dict]):
        system_prompt = f"""
You are an expert in structured data parsing. Your task is to convert text content into a standardized structured output based on a provided example data structure.

### Instructions

1.  **Analyze Example Structure:** Carefully analyze the example data structure provided within the `<example>` tags (e.g., it can be a dictionary, list, string, or single character) to understand the desired output format and hierarchy.
2.  **Determine Output Type:** Ensure the overall data type of the final output strictly adheres to the type specified within the `<type>` tags.
3.  **Transform Content:** Parse the text content from the `<input_text>` tags and transform it into a structured output that precisely matches the data format and content defined by `<example>`.
4.  **Preserve Semantics:** During the transformation process, only adjust the format and structure; do not alter the original semantic content of the text within the `<input_text>` tags.
5.  **Ignore Explanatory Text:** If the content within the `<input_text>` tags includes additional explanatory text or descriptions, ignore them and only extract and parse the core, final output data.
6.  **Clean Output:** Your final output must contain only the transformed structured content, without any additional explanations, descriptions, or irrelevant text and symbols.

<example>
{json.dumps(example, indent=4) if isinstance(example, list) or isinstance(example, dict) else example}
</example>

<type>
{'One letter' if isinstance(example, str) and len(example) == 1 else type(example)}
</type>
"""
        
        query = f"""
<input_text>
{text}
</input_text>
"""
        return system_prompt, query
    
    def __call__(self, text: str, example: Union[str, list, dict]):
        if not isinstance(text, str):
            text = str(text)
        final_answer = extract_final_answer(text)
        if final_answer is None:
            final_answer = text
        
        system_prompt, query = self.get_parser_prompt(final_answer, example)

        output = self.paser(query=query, system_prompt=system_prompt)
        return output
    

def check_syntax(code_string):
    try:
        # Try to compile the code string
        compile(code_string, '<string>', 'exec')
        return True
    except SyntaxError as e:
        return False
    
def get_function_lines(file_content):
    node = ast.parse(file_content)

    function_lines = {}
    
    for item in node.body:
        if isinstance(item, ast.FunctionDef):
            func_name = item.name
            start_line = item.lineno
            end_line = item.end_lineno
            function_lines[func_name] = (start_line, end_line)

    return function_lines

def replace_code(content_1, start_line_1, end_line_1, content_2, start_line_2, end_line_2):
    lines_1 = content_1.splitlines(keepends=True)
    lines_2 = content_2.splitlines(keepends=True)

    lines_1[start_line_1 - 1:end_line_1] = lines_2[start_line_2 - 1:end_line_2]

    return ''.join(lines_1)


def replace_function(main_code, new_code, function_name):
    assert check_syntax(main_code), "wrong main_code"
    assert check_syntax(new_code), "wrong new_code"
    functions_dict_1 = get_function_lines(main_code)
    functions_dict_2 = get_function_lines(new_code)

    start_line_1, end_line_1 = functions_dict_1[function_name]
    start_line_2, end_line_2 = functions_dict_2[function_name]

    main_code_after_replacing = replace_code(main_code, start_line_1, end_line_1, new_code, start_line_2, end_line_2)
    assert check_syntax(main_code_after_replacing), "wrong main_code after replacing"
    return main_code_after_replacing