import argparse
from global_utils import read_as_defaultdict
from inference.utils import *
import pandas as pd
from itertools import product
from collections import defaultdict
import sys
import json
import os
from tqdm import tqdm
import torch
from inference.textgen_inference.openai_inference import run_openai_preds
from inference.textgen_inference.fastchat_inference import run_fastchat_preds
from typing import List, Dict
from fastchat.model import load_model, get_conversation_template


def run_prediction(prompt: Dict, options: List, model_args: Dict, is_cot: bool = False) -> Dict:

    """
    Runs predictions of different models
    """

    if model_args.get('open_source', False):
        results = run_fastchat_preds(model_args, prompt, options, is_cot)
    elif 'gpt' in model_args['model_name'] or 'o1' in model_args['model_name']:
        results = run_openai_preds(model_args, prompt, options, is_cot)
    return results


def find_is_done(results, sentence, model, question):

    for i in range(len(results['model'])):
        if results['model'][i] == model and results['sentence'][i] == sentence and results['question'][i] == question:
            return True
    return False


def main(config_path):

    config = read_as_defaultdict(config_path)
    check_config_correctness(config)

    df = pd.read_csv(config['data_path'])
    results = get_results(config['results_path'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if os.path.exists(config['prefix_path']):
        with open(config['prefix_path'], 'r') as f:
            prefix = json.load(f)
    else:
        prefix = get_prefix(config)
        with open(config['prefix_path'], 'w') as f:
            json.dump(prefix, f)

    keys2add = config.get('keys_to_add', ['sentence_type'])

    for model_args in config['model_args']:

        model, tokenizer = None, None

        curr_model_name = model_args['model_name']
        rev = "main" if 'revision' not in model_args.get('creation_args', {}) else model_args['creation_args'].pop(
            'revision')
        if rev != "main":
            curr_model_name = model_args['model_name'] + f"_{rev}"
            print(curr_model_name)

        num_per_model = count_per_model(results, model_args['model_name'])
        if num_per_model >= (df.shape[0] / 2):
            continue

        if model_args.get('open_source', False):
            if "llama-3.2" in model_args['model_name'].lower() and 'vision' in model_args['model_name'].lower():
                if 'num_gpus' in model_args.get('creation_args', {}):
                    ngpus = model_args['creation_args'].pop('num_gpus')
                    model_args['creation_args']['device_map'] = 'sequential'
                    max_gpu_memory = {i: "60GiB" for i in range(ngpus)}
                else:
                    max_gpu_memory = None
                model = MllamaForConditionalGeneration.from_pretrained(model_args['model_name'], device_map="balanced",
                                                                       max_memory=max_gpu_memory,
                                                                       torch_dtype=torch.float16)
                tokenizer = AutoTokenizer.from_pretrained(model_args['model_name'])
            elif "gemma-2" in model_args['model_name'].lower():
                rev = "main"
                tokenizer = AutoTokenizer.from_pretrained(model_args['model_name'])
                model = AutoModelForCausalLM.from_pretrained(model_args['model_name'], device_map="auto", )
            else:
                model, tokenizer = load_model(model_args['model_name'], device, revision=rev,
                                              debug=False, **model_args.get('creation_args', {}))
                if rev != "main":
                    model_args['model_name'] = model_args['model_name'] + f"_{rev}"

            model_args['model'] = model
            model_args['tokenizer'] = tokenizer

        for i in tqdm(range(df.shape[0])):

            curr_sent, curr_quest, curr_options, curr_ans = split_sample(df.iloc[i])
            if find_is_done(results, curr_sent, model_args['model_name'], curr_quest):
                continue

            for j, curr_prefix in enumerate(prefix):

                prompt = construct_prompt(curr_prefix, curr_quest, curr_sent)

                try:
                    curr_results = run_prediction(prompt, curr_options, model_args, 'cot' in config.get('question_type', 'yes_no'))
                except Exception as e:
                    print(e)
                    sys.exit(1)

                results['model'].append(model_args['model_name'])
                results['sentence'].append(curr_sent)
                results['question'].append(curr_quest)
                results['correct'].append(curr_results['probs']['correct'])
                results['incorrect'].append(curr_results['probs']['incorrect'])
                results['prompt_index'].append(j)
                results['base_correct'].append(curr_results['unnormalized_probs']['correct'])
                results['base_incorrect'].append(curr_results['unnormalized_probs']['incorrect'])

                for key in keys2add:
                    results[key].append(df.iloc[i][key])

            if i % 15 == 0 and i > 0:
                df_res = pd.DataFrame.from_dict(results)
                df_res.to_csv(config['results_path'])

        if model:
            model_args.pop('model')
            del model

        df_res = pd.DataFrame.from_dict(results)
        df_res.to_csv(config['results_path'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Instruction style')
    parser.add_argument('-c', '--config_path', type=str, help="Path to where the configuration is kept")
    args = parser.parse_args()
    main(**vars(args))
