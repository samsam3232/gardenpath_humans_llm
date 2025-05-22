from typing import Dict, List
from fastchat.model import get_conversation_template
from collections import defaultdict
from transformers import PreTrainedTokenizer
import torch


def clean_word(word):
    word = word.replace(' ', '').lower().strip()
    if len(word) == 0:
        return word
    if not word[0].isascii():
        word = word[1:]

    return word


def find_opt_tokens(tokenizer, options):
    if hasattr(tokenizer, 'vocab'):
        vocab = tokenizer.vocab
    else:
        vocab = tokenizer.get_vocab()
    results = defaultdict(lambda: list())
    results['correct'] = [k for i, k in vocab.items() if clean_word(i) == options[0].lower().replace('the ', '')]
    results['incorrect'] = [k for i, k in vocab.items() if clean_word(i) == options[1].lower().replace('the ', '')]
    return results


def parse_mc_generation_results(outputs: Dict, tokenizer: PreTrainedTokenizer, options: List = ['Yes', 'No']):
    """
    Parses the multichoice generation to get: the prediction themselves, and the probabilities of each of the possible
    choices.
    """

    results = dict()
    pred = tokenizer.decode(outputs['sequences'][0][-2:], skip_special_tokens=True).strip()

    opt_tokens = find_opt_tokens(tokenizer, options)

    tokens = opt_tokens['correct'] + opt_tokens['incorrect']

    unnormalized_scor = outputs.scores[0][0].softmax(dim=-1)[tokens]
    unnormalized_probs = defaultdict(lambda: 0)
    for i in range(len(opt_tokens['correct'])):
        unnormalized_probs['correct'] += unnormalized_scor[i].item()
    for i in range(len(opt_tokens['incorrect'])):
        unnormalized_probs['incorrect'] += unnormalized_scor[i + len(opt_tokens['correct'])].item()

    probs = defaultdict(lambda: 0)
    curr_scor = outputs.scores[0][0, tokens].softmax(dim=-1)
    for i in range(len(opt_tokens['correct'])):
        probs['correct'] += curr_scor[i].item()
    for i in range(len(opt_tokens['incorrect'])):
        probs['incorrect'] += curr_scor[i + len(opt_tokens['correct'])].item()

    results['probs'] = probs
    results['unnormalized_probs'] = unnormalized_probs
    results['text'] = pred
    return results


def get_prompt(model_name: str, prompt: Dict) -> str:

    """
    Retrieves the prompts for the answer generation
    """

    curr_prompt = ""
    curr_prompt += prompt['system']
    curr_prompt += prompt['question']
    curr_prompt += f"\n\n{prompt['suffix']}"

    return curr_prompt


def normalize_probs(probs: Dict) -> Dict:

    """
    Given probabilities, normalizes them to sum to 1
    """

    new_probs = dict()
    sum_probs = sum(list(probs.values()))
    for key in probs:
        new_probs[key] = float(probs[key]) / sum_probs

    return new_probs


def run_fastchat_preds(model_args: Dict, prompt: str, options: List, is_cot: bool = False) -> Dict:
    """
    Gets the probabilities for the correct and incorrect answers for open source models.
    """

    device = 'cuda'

    curr_prompt = get_prompt(model_args['model_name'], prompt)

    tokenizer, model = model_args['tokenizer'], model_args['model']

    if is_cot:
        curr_prompt = curr_prompt.replace('My answer is:', 'Explanation:')
        input_ids = tokenizer(curr_prompt, return_tensors="pt").to(device)
        if "token_type_ids" in input_ids:
            input_ids.pop("token_type_ids")

        with torch.no_grad():
            outputs = model.generate(**input_ids, return_dict_in_generate=True, output_scores=True, max_new_tokens=200,
                                     pad_token_id=tokenizer.eos_token_id, **model_args.get('generation_args', {}))

        if model.config.is_encoder_decoder:
            outputs = outputs.sequences
        else:
            outputs = outputs.sequences[0, input_ids['input_ids'].shape[1]:]

        pred = tokenizer.decode(outputs, skip_special_tokens=True, skip_between_special_tokens=False).strip()
        explanation = pred.split('Answer:')[0]

        curr_prompt = curr_prompt + f' {explanation}\nMy answer is: '

    input_ids = tokenizer(curr_prompt, return_tensors="pt").to(device)
    if "token_type_ids" in input_ids:
        input_ids.pop("token_type_ids")

    with torch.no_grad():
        outputs = model.generate(**input_ids, return_dict_in_generate=True, output_scores=True, max_new_tokens=2,
                                 pad_token_id=tokenizer.eos_token_id, **model_args.get('generation_args', {}))
    curr_results = parse_mc_generation_results(outputs, tokenizer, options)

    return curr_results
