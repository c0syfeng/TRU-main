import json
import os
import time
from openai import OpenAI
import datetime
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer

client = OpenAI(api_key="your-key", base_url="https://api.deepseek.com")

def process_line(line, prefix):
    try:
        data = json.loads(line)
        text = data.get("text", "")
        prompt = prefix + "\n\n" + text
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": "You are helpful, and honest language model."},
                {"role": "user", "content": prompt},
            ],
            temperature=1.3,
            stream=False
        )
        reasoning_content = response.choices[0].message.reasoning_content
        content = response.choices[0].message.content
        result = {
            "thinking": reasoning_content,
            "answer": content,
            "question": text,
        }
        return result
    
    except Exception as e:
        print(f"{e}")
        return None

def cot_template(example, tokenizer):
    question = example["question"]
    QUERY_TEMPLATE = """{Question}""".strip()
    thinking = example['thinking']
    answer = example["answer"]
    prompt = QUERY_TEMPLATE.format(Question = question)
    answer = "Answer: " + answer if "Answer:" not in answer else answer
    text = tokenizer.apply_chat_template([
            {"role": "user", "content": prompt},
            {
                "role": "assistant", 
                "content": "<|im_start|>think\n" + thinking.strip() + "\n<|im_start|>answer\n" + answer.strip()
            }
        ], tokenize=False)
    
    return {
        "text": text
    }

def main(dataset, task):
    input_file = f'TRU-main/data/{dataset}/{task}-forget-corpus.jsonl'
    output_file = f'TRU-main/data/{dataset}/{task}_thinking-forget-corpus.jsonl'
    
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if task == 'News':
        prefix = '''The user will provide specific news that must be removed from your memory, and you must avoid discussing any information from this news. Your task is to think step by step to deny the user's request relates to the content of this news. You should politely decline the request and provide a positive, constructive, and relevant response focused on alternative topics, ensuring complete avoidance of any content mentioned in the news.
        \n\nMore importantly, you should follow the criteria:
        1. You should produce logical, user-acceptable explanations for your behaviors, and provide alternatives that preserve the integrity of the response.\n
        2. You should explicitly prevent the generation of content within specific news.
        \n\nNews:
        '''
    elif task == 'Books':
        prefix = '''
        The user will provide specific book that must be removed from your memory, and you must avoid discussing any information from this book. Your task is to think step by step to deny the user's request relates to the content of this book. You should politely decline the request and provide a positive, constructive, and relevant response focused on alternative topics, ensuring complete avoidance of any content mentioned in the book.
        \n\nMore importantly, you should follow the criteria:
        1. You should produce logical, user-acceptable explanations for your behaviors, and provide alternatives that preserve the integrity of the response.\n
        2. You should explicitly prevent the generation of content within specific book.
        \n\nBook:
        '''
    elif task == 'Bio':
        prefix = '''
        The user will provide a text containing biological information which are sensitive and harmful for human beings. Your task is to deny the user's request and provide a positive, constructive, and relevant answers to other questions, avoiding any content mentioned in the text in your response.
        \n\nMore importantly, you should follow the criteria:
        1. You should produce logical, user-acceptable explanations for your behaviors, and provide alternatives that preserve the integrity of the response.\n
        2. You should explicitly prevent the generation of content containing biological information which are sensitive and harmful for human beings
        \n\nText:
        '''
    elif task == 'Cyber':
        prefix = '''
        The user will provide a text containing cyber information which are harmful for cyber security and safety. Your task is to deny the user's request and provide a positive, constructive, and relevant answers to other questions, avoiding any content mentioned in the text in your response.
        \n\nMore importantly, you should follow the criteria:
        1. You should produce logical, user-acceptable explanations for your behaviors, and provide alternatives that preserve the integrity of the response.\n
        2. You should explicitly prevent the generation of content containing cyber information which are sensitive and harmful for human beings
        \n\nText:
        '''
    elif task == 'TOFU':
        prefix = '''
        The user will provide a question about the author profile, and you must avoid discussing any information from the question. Your task is to deny the user's request which includes the content of the author profile in this question, and provide a positive, constructive, and response focused on alternative topics, ensuring complete avoidance of any content mentioned in the question.
        \n\nMore importantly, you should follow the criteria:
        1. You should produce logical, user-acceptable explanations for your behaviors, and provide alternatives that preserve the integrity of the response.\n
        2. You should explicitly prevent the generation of content containing author profile.
        \n\nQuestion:
        '''

    tokenizer = AutoTokenizer.from_pretrained('HuggingFaceH4/zephyr-7b-beta')

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        total_file_lines = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))

        for i in range(total_file_lines):
            line = next(infile, None)
            if line is None:
                break
            result = process_line(line, prefix)
            if task == 'TOFU':
                outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
                outfile.flush()
            else:
                result = cot_template(result,tokenizer)
                outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
                outfile.flush()
    print(f"result have saved in {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='generating resoning-based unlearning target')
    parser.add_argument('--dataset', type=str, default='wmdp', help='the benchmark')
    parser.add_argument('--task', type=str, default='Bio', help='the type of task')
    
    args = parser.parse_args()
    main(args.dataset,args.task)
