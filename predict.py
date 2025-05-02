import argparse
import json
import re

import torch
from transformers import T5ForConditionalGeneration, T5TokenizerFast
from api_pyserini import search
from Model import Model

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="hotpotqa")
parser.add_argument("--model", type=str, default="t5-base")
parser.add_argument("--val_set", type=str, default="dev")
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--max_iters", type=int, default=3)
parser.add_argument("--devices", type=int, nargs="+", default=0)
parser.add_argument("--precision", type=int, default=32)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--role", type=str, default="decomposer")
parser.add_argument("--strategy", type=str, default='auto')
parser.add_argument("--file_path", type=str, default='prediction.json')
args = parser.parse_args()

# 读取配置文件中的参数
with open('configs/config.json', 'r', encoding='utf-8') as f:
    json_config = json.load(f)
for key, value in json_config.items():
    if hasattr(args, key):
        setattr(args, key, value)
    else:
        print(f"Warning: JSON config contains unknown key: {key}")

model_path = args.model
tokenizer = T5TokenizerFast.from_pretrained(model_path, model_max_length=512)
device = torch.device('cuda:0')

# 基础模型


# 加载决策器
dicider_path = 'checkponits/best_dicider.ckpt'
dicider_model = T5ForConditionalGeneration.from_pretrained(model_path)
dicider = Model.load_from_checkpoint(dicider_path, model=dicider_model)
dicider.eval()

# 加载重写器
rewriter_path = 'checkponits/best_rewriter.ckpt'
rewriter_model = T5ForConditionalGeneration.from_pretrained(model_path)
rewriter = Model.load_from_checkpoint(rewriter_path, model=rewriter_model)
rewriter.eval()

# 加载应答器
responser_path = 'checkponits/best_responser.ckpt'
responser_model = T5ForConditionalGeneration.from_pretrained(model_path)
responser = Model.load_from_checkpoint(responser_path, model=responser_model)
responser.eval()


def match_query(text):
    match = re.search(r"Query:\s*(.+)", text)
    if match:
        question = match.group(1)
        return question.strip()
    else:
        return ""

def iter_answer(question):
    iter_time = 0
    interact_history = question
    answer = question

    while iter_time <= 3:
        dicider_input = interact_history
        input_ids = tokenizer(
            dicider_input,
            padding="longest",
            max_length=512,
            truncation=True,
            return_tensors='pt'
        )['input_ids'].to(device)
        dicider_output = dicider.predict(input_ids)[0]
        # print(dicider_output)
        if "Final answer:" in dicider_output:
            answer += "\n" + dicider_output
            break

        rewriter_input = dicider_output
        input_ids = tokenizer(
            rewriter_input,
            padding="longest",
            max_length=512,
            truncation=True,
            return_tensors='pt'
        )['input_ids'].to(device)
        rewriter_output = rewriter.predict(input_ids)[0]
        # print(rewriter_output)

        responser_input = search(match_query(rewriter_output)) + '\n' + rewriter_input
        input_ids = tokenizer(
            responser_input,
            padding="longest",
            max_length=512,
            truncation=True,
            return_tensors='pt'
        )['input_ids'].to(device)
        responser_output = responser.predict(input_ids)[0]
        # print(responser_output)
        interact_history += '\n' + dicider_output + '\n' + responser_output
        answer += '\n' + dicider_output + '\n' + rewriter_output + '\n' + responser_output
        iter_time += 1
    return answer



