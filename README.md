# PromptTuner
See [blog post](https://shermwong.com/2023/03/10/llm-studies-part-3-prompt-tuning/) for motivations

## Get Started
This is a clean fork of [NanoGPT](https://github.com/karpathy/nanoGPT), see original author's [README](README_orig.md) for setup instructions

Trainng setup is single GPU, nanoGPT can fit on single V100/A100 32G GPU with up to 32 batch size during fine-tuning.

we have enabled 4 different tasks for prompt-tuning, these tasks are toy datasets generated from the 3 files:
* `arithmetics_task.py` -- asking LLM to adding 2 large positive numbers
* `symbolic_task.py` -- asking LLM to perform symbol substitutions
* `word_manipulate.py` -- asking LLM to reverse letter and last letter concat

train
> python finetune.py --task arithmetics --use-mlp  --lr 0.01 --decay-lr --min-lr 1e-3 --ckpt-name xxx

eval
> python finetune.py --eval --task arithmetics --use-mlp --ckpt-name xxx
