# LLMs for psycholinguistic pretesting
Official repository of the paper: "Large Language Models for Psycholinguistic Plausibility Pretesting" (ACL 2025) (arxiv: [pdf](https://arxiv.org/abs/2502.09307))

## Experiment data

Our full experiment data can be found under `experiments/gardenpath_10_24/data/llm_data/extended_gardenpath_experiments.csv` 

## Human results

Human results can be found under `experiments/gardenpath_10_24/results/human_results/`.  
Note that there are various files in this folder, and the one we use to report our results is `sampled_results.csv`.  
`all_results.csv` contains the outputs of all our human subjects, including those who didn't pass the filter of answering correctly to the example questions.
`all_correct_results.csv` contains the outputs of all our human subjects that answered correctly the training question, with some sets having more than 10 answers. We randomly sampled 10 answers for those sets to create `sampled_results.csv`.

## LLM results

All our LLMs results can be found under `experiments/gardenpath_10_24/results/llm_results/`. This folder includes results for all our tasks: sentence comprehension, paraphrasing and drawing.

## Running your own experiments

Our framework is quite flexible and can be used to run LLMs on different sentence. To do so:

1. Install the conda environment by running `conda create -n ENVNAME --file reading_comprehension_research.txt`
2. Run: `export OPENAI_API_KEY="YOUR_API_KEY"`, `export OPENAI_ORG="YOUR_OPENAI_ORG"`
3. To run sentence comprehension experiment: `python inference/textgen_inference/base_inference.py -c path to config`
4. To run paraphrasing experiment: `python inference/gardenpath_10_24/rephrasing.py -c path to config`
5. To run drawing experiment: `python inference/gardenpath_10_24/draw_sentence.py -c path to config`

In our *textgen_inference/base_inference.py* file we have allowed a large flexibility, allowing running multiple types of models, on multiple types of sentences with multiple types of data with different prompts. All the parameters can be defined using the config file.