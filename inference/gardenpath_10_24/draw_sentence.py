import argparse
from global_utils import read_as_defaultdict
from inference.utils import check_config_correctness, get_results
import pandas as pd
from tqdm import tqdm
import base64
import boto3
import os
import requests
import json
from typing import List, Dict
from openai import OpenAI
from time import sleep
from PIL import Image
from io import BytesIO
import random


client = OpenAI(
  organization=os.environ['OPENAI_ORG'],
  api_key=os.environ['OPENAI_API_KEY'],
)
bedrock_client = boto3.client("bedrock-runtime", region_name="us-east-1")


def create_openai_image(model_name, sentence):

    response = client.images.generate(
      model=model_name,
      prompt=sentence,
      size="1024x1024",
      quality="standard",
      n=1,
    )
    im_response = requests.get(response.data[0].url)
    img = Image.open(BytesIO(im_response.content))
    return img


def create_bedrock_image(model_name, sentence):

    native_request = {
        "prompt": sentence,
        "mode": "text-to-image"
    }

    req = json.dumps(native_request)
    response = bedrock_client.invoke_model(modelId=model_name, body=req)
    model_response = json.loads(response["body"].read())
    base64_image_data = model_response["images"][0]
    im = Image.open(BytesIO(base64.b64decode(base64_image_data)))
    return im



def create_image(model_name, sentence):

    if "dall-e" in model_name.lower():
        image = create_openai_image(model_name, sentence)
    else:
        image = create_bedrock_image(model_name, sentence)
    return image


def find_is_done(output_path, set_id):

    return os.path.exists(os.path.join(output_path, f"{set_id}.png"))


def main(config_path):

    config = read_as_defaultdict(config_path)
    check_config_correctness(config)

    df = pd.read_csv(config['data_path'])
    output_path = config['results_path']

    for model_name in config['model_args']:

        print(model_name)
        model_outpath = os.path.join(output_path, model_name.replace('.', '_').replace(':', '-'))
        if not os.path.exists(model_outpath):
            os.mkdir(model_outpath)

        for i in tqdm(range(df.shape[0])):

            curr_sent = df.iloc[i].sentence
            curr_id = df.iloc[i].sent_id

            if find_is_done(model_outpath, curr_id):
                continue

            try:
                image = create_image(model_name, curr_sent)
                image.save(os.path.join(model_outpath, f"{curr_id}.png"))
            except:
                continue



if __name__ == "__main__":
    parser = argparse.ArgumentParser('Rephrasing')
    parser.add_argument('-c', '--config_path', type=str, help="Path to where the configuration is kept")
    args = parser.parse_args()
    main(**vars(args))