import argparse
from global_utils import read_as_defaultdict
from inference.utils import check_config_correctness, get_results
import pandas as pd
from tqdm import tqdm
import torch
from typing import List, Dict
from fastchat.model import load_model, get_conversation_template
from transformers import MllamaForConditionalGeneration, AutoTokenizer
from openai import OpenAI
from time import sleep
import random
import os


client = OpenAI(
  organization=os.environ['OPENAI_ORG'],
  api_key=os.environ['OPENAI_API_KEY'],
)

COMPLETION_ARGS = {"max_tokens": 100,
                   "top_p": 1}

SYSTEM_MESSAGE = """You are a linguistic experiment subject. You will be presented with a sentence, and you will need to split it into two sentences that convey the exact same situation as the original sentence. You will be provided with a few examples. Note: keep the sentences as simple as possible.\n\nExample 1:\nSentence: The dog ran in the courtyard, and the man fell.\nSplitted: \n1. The dog ran in the courtyard. \n2. The man fell.\n\nExample 2:\nSentence: Sarah met her boss in the park when the plane crashed.\nSplitted: \n1. Sarah met her boss in the park. \n2. The plane crashed.\n\nExample 3:\nSentence: She cleaned the mess that her sister made.\nSplitted: \n1. She cleaned the mess. \n2. Her sister made the mess.\n\nExample 4:\nSentence: They looked for the treasure, hoping to find salvation.\nSplitted: \n1. The looked for the treasure. \n2. They hoped to find salvation."""


def construct_openai_prompt(model_name: str, sentence: str) -> List:

    """
    Construct the prompt for the openai run
    """

    if "o1" in model_name:
        messages = [{"role": "user", "content": [{'type': 'text', 'text': SYSTEM_MESSAGE}]},
                    {"role": "user", "content": [{'type': 'text', 'text': f"Now here comes the sentence:\nSentence: {sentence}"}]}]
    else:
        messages = [{"role": "system", "content": [{'type': 'text', 'text': SYSTEM_MESSAGE}]},
                    {"role": "user", "content": [{'type': 'text', 'text': f"Now here comes the sentence:\nSentence: {sentence}"}]}]
    return messages


def construct_openai_args(prompt: str, generation_args: Dict) -> Dict:

    """
    Builds the arguments for the openai run
    """

    messages = construct_openai_prompt(generation_args['model_name'], prompt)
    curr_completion_args = {**COMPLETION_ARGS, **generation_args, 'messages': messages}
    curr_completion_args['model'] = curr_completion_args.pop('model_name')
    if 'o1' in curr_completion_args['model']:
        curr_completion_args.pop('max_tokens')
        curr_completion_args.pop('n')

    return curr_completion_args


def get_openai_text(completion_args: str):

    """
    Retrieves the model predictions for a given sample
    """

    try:
        response = client.chat.completions.create(**completion_args)
    except:
        sleep(random.randint(2000,3000) / 1000.)
        try:
            response = client.chat.completions.create(**completion_args)
        except:
            sleep(random.randint(2000, 3000) / 1000.)
            response = client.chat.completions.create(**completion_args)

    curr_response = response.choices[0].message.content

    return curr_response


def find_is_done(results, sentence, model):
    for i in range(len(results['model'])):
        if results['model'][i] == model and results['sentence'][i] == sentence:
            return True
    return False


def get_prompt(model_name, sentence):
    conv = get_conversation_template(model_name)
    conv.messages = []
    conv.system_message = SYSTEM_MESSAGE
    conv.append_message(conv.roles[0], f"Now here comes the sentence:\nSentence: {sentence}")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    prompt += "\nSplitted:\n"
    return prompt


def split_sample(line: pd.Series) -> List:
    """
    Given a line in a data frame returns the relevant elements
    """

    curr_sent = line['sentence']
    curr_noun = line['noun']
    curr_verb = line['verb']

    return [curr_sent, curr_noun, curr_verb]


def get_fschat_text(model, tokenizer, model_name, sentence):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    prompt = get_prompt(model_name, sentence)
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    if "token_type_ids" in input_ids:
        input_ids.pop("token_type_ids")

    with torch.no_grad():
        outputs = model.generate(**input_ids, return_dict_in_generate=True, output_scores=True, max_new_tokens=100,
                                 pad_token_id=tokenizer.eos_token_id)

    if model.config.is_encoder_decoder:
        outputs = outputs.sequences
    else:
        outputs = outputs.sequences[0, input_ids['input_ids'].shape[1]:]

    outputs = tokenizer.decode(
        outputs, skip_special_tokens=True, spaces_between_special_tokens=False
    )
    return outputs


def get_predictions(output_txt):
    lines = output_txt.split('\n')
    if lines[0].strip() == "Splitted:":
        lines = lines[1:]
    s1 = lines[0].split('1. ')[-1].split('. ')[0]
    s2 = lines[1].split('2. ')[-1].split('. ')[0]
    return [s1, s2]


def check_sentences(sentences, verb, noun):
    correct, found = False, False
    for sent in sentences:
        if verb in sent:
            found = True
            if noun not in sent:
                correct = True
    return found, correct


def main(config_path):
    config = read_as_defaultdict(config_path)
    check_config_correctness(config)

    df = pd.read_csv(config['data_path'])
    results = get_results(config['results_path'])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    keys2add = config.get('keys_to_add', ['sentence_type'])

    for model_args in config['model_args']:

        model, tokenizer = None, None

        if model_args.get('open_source', False):
            if "llama-3.2" in model_args['model_name'].lower() and 'vision' in model_args['model_name'].lower():
                if 'num_gpus' in model_args.get('creation_args', {}):
                    ngpus = model_args['creation_args'].pop('num_gpus')
                    model_args['device_map'] = 'sequential'
                    model_args["max_memory"] = {i: "60GiB" for i in range(ngpus)}
                model = MllamaForConditionalGeneration.from_pretrained(model_args['model_name'],
                                                                       **model_args.get('creation_args',
                                                                                        {"device_map": "auto"}))
                tokenizer = AutoTokenizer.from_pretrained(model_args['model_name'])
            else:
                rev = "main" if 'revision' not in model_args.get('creation_args', {}) else model_args[
                    'creation_args'].pop('revision')
                model, tokenizer = load_model(model_args['model_name'], device, revision=rev,
                                              debug=False, **model_args.get('creation_args', {}))
                if rev != "main":
                    model_args['model_name'] = model_args['model_name'] + f"_{rev}"

                print(model_args['model_name'])


        for i in tqdm(range(df.shape[0])):

            curr_sent, curr_verb, curr_noun = split_sample(df.iloc[i])

            if find_is_done(results, curr_sent, model_args['model_name']):
                continue

            for j in range(5):

                correct = False

                try:
                    if 'gpt' in model_args['model_name'] or 'o1' in model_args['model_name']:
                        completion_args = construct_openai_args(curr_sent, model_args)
                        txt = get_openai_text(completion_args,)
                    else:
                        txt = get_fschat_text(model, tokenizer, model_args['model_name'], curr_sent)
                    sents = get_predictions(txt)

                    found, correct = check_sentences(sents, curr_verb, curr_noun)
                    if found:
                        break

                except Exception as e:
                    continue

            results['model'].append(model_args['model_name'])
            results['sentence'].append(curr_sent)
            results['correct'].append(correct)
            results['txt'].append(txt)

            for key in keys2add:
                results[key].append(df.iloc[i][key])

        # print(results['model'][-1])
            if i > 10 and i % 20 == 0:
                df_res = pd.DataFrame.from_dict(results)
                df_res.to_csv(config['results_path'])

        df_res = pd.DataFrame.from_dict(results)
        df_res.to_csv(config['results_path'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Rephrasing')
    parser.add_argument('-c', '--config_path', type=str, help="Path to where the configuration is kept")
    args = parser.parse_args()
    main(**vars(args))