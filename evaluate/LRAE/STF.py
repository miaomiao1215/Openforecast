import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import json
import copy
import argparse
from vllm import LLM, SamplingParams
import time
import os
import re
import numpy as np
from transformers import AutoModelForMultipleChoice
from sentence_transformers.util import cos_sim
from datetime import datetime
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import sys

class Retrieval_sample:
    def __init__(self, args, model):
        self.model = SentenceTransformer(model).cuda().eval()
        doc_file = 'en_test_source.json'
        embed_file = 'en_test_source.pth'
        lines = open(doc_file, 'r', encoding='utf-8').readlines()
        infos = [json.loads(line) for line in lines]
        self.documents_dict = defaultdict(list)
        self.date_dict = defaultdict(list)
        self.embed_dict = torch.load(embed_file)
        index = 0
        self.title_index_dict = {}
        for i, info in enumerate(infos):
            title = info['title']
            self.title_index_dict[title] = i
            sources = info['sources']
            docs_temp = []
            for source_id, source in enumerate(sources):
                date = source['date']
                chunks = source['chunks']
                for chunk_id, chunk in enumerate(chunks):
                    self.documents_dict[title].append(chunk)
                    docs_temp.append(chunk)
                    if date == None:
                        self.date_dict[title].append(None)
                    else:
                        self.date_dict[title].append(date)

    def deduplicate_preds(self, predictions, threshold=0.95):
        """
        Given several sampled outputs, map them to a label.
        """
        if len(predictions) == 0:
            return predictions
        if len(predictions) == 1:
            return predictions[0]
        else:
            preds_embed = self.model.encode(predictions)
            pred_sim = cos_sim(preds_embed, preds_embed)
            duplicate_pairs = torch.argwhere(pred_sim>threshold)
            duplicate_indexs = []
            for duplicate_pair in duplicate_pairs:
                if duplicate_pair[0] == duplicate_pair[1]:
                    continue
                if not min(duplicate_pair.tolist()) in duplicate_indexs:
                    duplicate_indexs.append(min(list(duplicate_pair)))
            predictions_filter = []
            for i, prediction in enumerate(predictions):
                if not i in duplicate_indexs:
                    predictions_filter.append(prediction)
            return predictions_filter
    
    def __call__(self, query_list, title, date, num=1):
        """
        Given several sampled outputs, map them to a label.
        """
        try:
            dates = self.date_dict[title]
            documents = self.documents_dict[title]
            index = self.title_index_dict[title]
            embeddings = self.embed_dict[index]
            if len(documents) == 0:
                return [None] * len(query_list)
        except:
            return [None] * len(query_list)

        query_embeds = self.model.encode(query_list)
        cho_scores = cos_sim(query_embeds, embeddings).numpy()
        related_docs_list = []
        for cho_score in cho_scores:
            sort_idx = np.argsort(-cho_score)
            sort_idx_list = [int(i) for i in sort_idx]
            chunks_retrieve = []
            for i in sorted(sort_idx_list[0: num]):
                try:
                    time = datetime.strptime(dates[i][0:10], '%Y-%m-%d')
                    date = datetime.strptime(date[0:10], '%Y-%m-%d')
                    if (time - date).days >= 0 and (time - date).days <= 3:
                        chunks_retrieve.append(documents[i])
                except:
                    chunks_retrieve.append(documents[i])
            if len(chunks_retrieve) == 0:
                related_docs_list.append(None)
            chunks_retrieve_str = ''
            for i, chunk in enumerate(chunks_retrieve):
                chunks_retrieve_str += '(%i). %s\n'%(i+1, chunk.strip())
            related_docs_list.append(chunks_retrieve_str)
        return related_docs_list


def compute_f1(preds, gold_indexs, rae_preds, label_num):
    if len(preds) == 0:
        return {'precision_ori': 0, 'recall_ori': 0, 'f1_ori': 0, \
            'precision_combine': 0, 'recall_combine': 0, 'f1_combine': 0
        }
    ori_recall_num = 0
    ori_wrong_num = 0
    for i in range(1, label_num+1):
        if i in gold_indexs:
            ori_recall_num += 1
    for i in gold_indexs:
        if i == None:
            ori_wrong_num += 1
        elif i == 0 or i > label_num:
            ori_wrong_num += 1
    
    rae_recall_num = 0
    for pred, rae_pred in zip(preds, rae_preds):
        if rae_pred==True and pred==False:
            rae_recall_num += 1
    precision_ori = ori_recall_num / len(preds)
    recall_ori = ori_recall_num / label_num
    try:
        f1_ori = 2 * precision_ori * recall_ori / (precision_ori + recall_ori)
    except:
        f1_ori = 0
    
    combine = [pred_or or pred_rag for pred_or, pred_rag in zip(preds, rae_preds)]
    precision_combine = (rae_recall_num + ori_recall_num) / len(combine)
    recall_combine = (rae_recall_num + ori_recall_num) / (label_num + rae_recall_num)
    try:
        f1_combine = 2 * precision_combine * recall_combine / (precision_combine + recall_combine)
    except:
        f1_combine = 0
    return {'precision_ori': precision_ori, 'recall_ori': recall_ori, 'f1_ori': f1_ori, \
            'precision_combine': precision_combine, 'recall_combine': recall_combine, 'f1_combine': f1_combine
        }

def extract_answer(text):
    pattern = re.compile(r'Yes', re.IGNORECASE)
    if bool(pattern.search(text)):
        return True
    else:
        return False

def extract_answer_index(text):
    text = text.replace('（', '(').replace('）', ')')
    pattern = re.compile(r'Yes', re.IGNORECASE)
    if bool(pattern.search(text)):
        try:
            pattern = re.compile(r'\((\d+)\)')
            matches = pattern.findall(text)
            index = int(matches[0])
            return True, index
        except:
            return False, None
    else:
        return False, None

def generate_vllm(llm, prompts, sampling_params):
    prompts_filter = []
    index_trans_dict = {}
    index = 0
    result_list = [False] * len(prompts)
    for i, prompt in enumerate(prompts):
        if prompt != None:
            prompts_filter.append(prompt)
            index_trans_dict[index] = i
            index += 1
    
    outputs = llm.generate(prompts_filter, sampling_params)
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        answer = extract_answer(generated_text)
        result_list[index_trans_dict[i]] = answer

    return result_list

def generate_vllm_with_index(llm, prompts, sampling_params):
    prompts_filter = []
    index_trans_dict = {}
    index = 0
    result_list = [False] * len(prompts)
    index_list = [None] * len(prompts)
    for i, prompt in enumerate(prompts):
        if prompt != None:
            prompts_filter.append(prompt)
            index_trans_dict[index] = i
            index += 1
    
    outputs = llm.generate(prompts_filter, sampling_params)
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        answer, index = extract_answer_index(generated_text)
        result_list[index_trans_dict[i]] = answer
        index_list[index_trans_dict[i]] = index

    return result_list, index_list

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default='xx/llama3-8b', help="do predict")
parser.add_argument("--dataset", type=str, default='xx/result.json', help="do predict")
parser.add_argument("--save_dir", type=str, default='xx/result_lrae.json')
parser.add_argument("--search_num", type=int, default=5)
args = parser.parse_args()

prompt_gold = "Following natural language inference, please judge whether the prediction is in accord with the given labels. Rules: \n1. Based on the given labels only. \n2.Note that the tense of the prediction should be disregarded.\n3.If certain label can clearly demonstrate the prediction, then the prediction is correct, otherwise wrong. \n4. If the prediction is wrong, output \"No\". If the prediction is correct, please output \"Yes\" along with the corresponding index in the list of labels, for example: \"According to the given information, the answer is Yes, and the index is: (1)\". \nThe labels are {evidence}\n The prediction is \"{prediction}\" \n The answer is: "
prompt_rae = "Following natural language inference, please judge whether the prediction is in accord with the given label. Rules: \n1. Based on the given label only.\n2.Note that the tense of the prediction should be disregarded.\n3.If certain content from label can clearly demonstrate the prediction, then the prediction is correct, otherwise wrong. \nThe label is {evidence}\n The prediction is \"{prediction}\" \n The answer (Yes or No) is: "

text_embed_model = 'xx/bge-large-en-1.5'

lines = open(args.dataset, 'r').readlines()

model_path = args.model
tokenizer = AutoTokenizer.from_pretrained(model_path)
if 'llama' in model_path:
    sampling_params = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=1024, n=1, stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")])
elif 'Yi' in model_path:
    sampling_params = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=1024, n=1, stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")])
else:
    sampling_params = SamplingParams(temperature=0.0, top_p=0.9, max_tokens=1024, n=1)
llm = LLM(model=model_path, tensor_parallel_size=4, quantization=None, dtype='float16', trust_remote_code=True)
Retrieval_model = Retrieval_sample(args, text_embed_model)


question_list, prompts_gold_list, prompts_rag_list = [], [], []
len_list = []
for index, line in enumerate(lines):
    info = json.loads(line)
    title = info['title']
    date = info['time']
    ori_predictions = info['pred_events']
    labels = info['label']
    background = info['known_timeline']
    if len(ori_predictions) == 0:
        question_list.append(info)
        prompts_gold_list.append([])
        prompts_rag_list.append([])
        continue
    predictions = Retrieval_model.deduplicate_preds(ori_predictions)
    labels_str = ''
    for i, option in enumerate(labels):
        labels_str += '(%i). %s\n'%(i+1, option.strip())
    
    prompts_rag_temp, prompts_gold_temp = [], []
    rag_docs = Retrieval_model(query_list=predictions, title=title, date=date, num=args.search_num)
    for rag_doc, prediction in zip(rag_docs, predictions):
        prompt_i_gold = copy.deepcopy(prompt_gold).format(prediction=prediction, evidence=labels_str, date=date)
        messages = [{"role": "user", "content": prompt_i_gold}]
        prompt_chat_i = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts_gold_temp.append(prompt_chat_i)
        
        if rag_doc == None:
            prompts_rag_temp.append(None)
            continue
        prompt_i_rag = copy.deepcopy(prompt_rae).format(prediction=prediction, evidence=rag_doc, date=date)
        messages = [{"role": "user", "content": prompt_i_rag}]
        prompt_chat_i = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts_rag_temp.append(prompt_chat_i)


    question_list.append(info)
    prompts_gold_list.append(prompts_gold_temp)
    prompts_rag_list.append(prompts_rag_temp)


batch_size = 1024
out_list, result_list = [], []
acc_num = 0
all_num = 0


f_w = open(save_dir, 'w', encoding='utf-8')
metric_all = []
for index in tqdm(range(0, len(question_list), batch_size)):
    # eval with gold answer
    prompts_gold_sentecnces = []
    prompts_gold_batch = prompts_gold_list[index: index+batch_size]
    gold_temp_j = 0
    gold_index_dict = {}
    for i, prompts in enumerate(prompts_gold_batch):
        for prompt_j in prompts:
            prompts_gold_sentecnces.append(prompt_j)
            gold_index_dict[gold_temp_j] = i
            gold_temp_j += 1

    gold_results_list, gold_indexs_list = generate_vllm_with_index(llm, prompts_gold_sentecnces, sampling_params)

    gold_out_list_processed = defaultdict(list)
    gold_indexs_list_processed = defaultdict(list)
    for i, result_i, index_i in zip(range(len(gold_results_list)), gold_results_list, gold_indexs_list):
        gold_out_list_processed[gold_index_dict[i]].append(result_i)
        gold_indexs_list_processed[gold_index_dict[i]].append(index_i)

    # eval with rag answer
    prompts_rag_sentecnces = []
    prompts_rag_batch = prompts_rag_list[index: index+batch_size]
    rag_temp_j = 0
    rag_index_dict = {}
    for i, prompts in enumerate(prompts_rag_batch):
        for prompt_j in prompts:
            prompts_rag_sentecnces.append(prompt_j)
            rag_index_dict[rag_temp_j] = i
            rag_temp_j += 1

    rag_results_list = generate_vllm(llm, prompts_rag_sentecnces, sampling_params)

    rag_out_list_processed = defaultdict(list)
    for i, out_i in enumerate(rag_results_list):
        rag_out_list_processed[rag_index_dict[i]].append(out_i)

    for i, question in enumerate(question_list[index: index+batch_size]):
        rae_results = rag_out_list_processed[i]
        gold_results = gold_out_list_processed[i]
        gold_indexs = gold_indexs_list_processed[i]
        question['rae_results'] = rae_results
        question['gold_results'] = gold_results
        question['gold_indexs'] = gold_indexs
        label = question['label']
        metric = compute_f1(gold_results, gold_indexs, rae_results, len(label))
        question['metric'] = metric
        metric_all.append(metric)
        f_w.writelines(json.dumps(question, ensure_ascii=False) + '\n')

f_w.close()


precision_ori_list, recall_ori_list, f1_ori_list = [], [], []
precision_combine_list, recall_combine_list, f1_combine_list = [], [], []

for metric_i in metric_all:
    precision_ori_list.append(metric_i['precision_ori'])
    recall_ori_list.append(metric_i['recall_ori'])
    f1_ori_list.append(metric_i['f1_ori'])

    precision_combine_list.append(metric_i['precision_combine'])
    recall_combine_list.append(metric_i['recall_combine'])
    f1_combine_list.append(metric_i['f1_combine'])

macro_metric = {}
macro_metric['precision_ori'] = float(np.mean(precision_ori_list))
macro_metric['recall_ori'] = float(np.mean(recall_ori_list))
macro_metric['f1_ori'] = float(np.mean(f1_ori_list))

macro_metric['precision_combine'] = float(np.mean(precision_combine_list))
macro_metric['recall_combine'] = float(np.mean(recall_combine_list))
macro_metric['f1_combine'] = float(np.mean(f1_combine_list))
print(args.test_model)
print(macro_metric)
