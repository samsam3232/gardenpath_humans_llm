from time import sleep
import random
from collections import defaultdict
from openai import OpenAI
from typing import List, Dict
import os

client = OpenAI(
  organization=os.environ['OPENAI_ORG'],
  api_key=os.environ['OPENAI_API_KEY'],
)

COMPLETION_ARGS = {"max_tokens": 5,
                   "top_p": 1}


def construct_openai_prompt(model_name: str, prompt: Dict) -> List:

    """
    Construct the prompt for the openai run
    """

    if "o1" in model_name:
        messages = [{"role": "user", "content": [{'type': 'text', 'text': prompt['system']}]},
                    {"role": "user", "content": [{'type': 'text', 'text': prompt['question']}]}]
    else:
        messages = [{"role": "system", "content": [{'type': 'text', 'text': prompt['system']}]},
                    {"role": "user", "content": [{'type': 'text', 'text': prompt['question']}]}]
    return messages


def construct_openai_args(prompt: Dict, generation_args: Dict) -> Dict:

    """
    Builds the arguments for the openai run
    """

    messages = construct_openai_prompt(generation_args['model_name'], prompt)
    curr_completion_args = {**COMPLETION_ARGS, **generation_args, 'messages': messages}
    curr_completion_args["max_tokens"] = 200 if 'explain' in prompt['system'].lower() else 5
    curr_completion_args['model'] = curr_completion_args.pop('model_name')
    if 'o1' in curr_completion_args['model']:
        curr_completion_args.pop('max_tokens')
        curr_completion_args.pop('n')

    return curr_completion_args


def parse_prediction(answer: str, pot_answers: List):

    words = answer.lower().split(" ")
    for i, word in enumerate(words):
        if word in pot_answers:
            return word
    return None


def return_probs(results: Dict, options: List) -> Dict:

    correct = results[options[0].lower()]
    incorrect = results[options[1].lower()]
    total = max(1, correct + incorrect)
    return {'probs': {'correct': (float(correct) / total),
                      'incorrect': (float(incorrect) / total)}}


def retrieve_model_predictions(completion_args: str, options: List = ['Yes', 'No']):

    """
    Retrieves the model predictions for a given sample
    """

    options_lower = [i.lower() for i in options]

    results = defaultdict(lambda : 0)
    for i in range(5):
        try:
            response = client.chat.completions.create(**completion_args)
        except:
            sleep(random.randint(2000,3000) / 1000.)
            try:
                response = client.chat.completions.create(**completion_args)
            except:
                sleep(random.randint(2000, 3000) / 1000.)
                response = client.chat.completions.create(**completion_args)

        for i in range(len(response.choices)):
            curr_response = response.choices[i].message.content

            answer = parse_prediction(curr_response, options_lower)
            if answer is not None:
                results[answer] += 1

    results = return_probs(results, options)

    return results


def run_openai_preds(model_args: Dict, prompt: str, options: List, is_cot: bool = False) -> Dict:

    """
    Gets the probabilities for the correct and incorrect answers for open source models.
    """

    completion_args = construct_openai_args(prompt, model_args)
    curr_results = retrieve_model_predictions(completion_args, options)

    return curr_results
