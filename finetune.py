"""
> cd & source nanogpt/bin/activate
> cd rsc/nanoGPT/
> export TIKTOKEN_CACHE_DIR=/home/shermanwong/rsc/nanoGPT/cached_tiktoken/
> python finetune.py

"""
import math
import os
import time
from contextlib import nullcontext

import numpy as np

import torch

from model import GPT
from my_task import FixedLenAdditionDataset, GPTTokenizer, LOW, MAX_SEQ_LEN, STOP_TOKEN
from torch.distributed import destroy_process_group, init_process_group
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText, DO NOT TOUCH!!!
# I/O
out_dir = 'out'
# always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'gpt2' # 'scratch' or 'resume' or 'gpt2*'
# data
# dataset = 'openwebtext'
# gradient_accumulation_steps = 5 # used to simulate larger batch sizes
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# -----------------------------------------------------------------------------


batch_size = 32
eval_batch_size = 10
log_interval = 10
# adamw optimizer or SGD
learning_rate = 1e-3 # max learning rate
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
# grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
NUM_EPOCHES = 20
NUM_EXAMPLES = 10000

decay_lr = False # whether to decay the learning rate
warmup_iters = 1000 # how many steps to warm up for
lr_decay_iters = int(NUM_EPOCHES * NUM_EXAMPLES / batch_size) * 2 # should be ~= max_iters per Chinchilla
print("lr_decay max steps: ", lr_decay_iters)
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
compile = False # use PyTorch 2.0 to compile the model to be faster




# dataset configs
# -----------------------------------------------------------------------------

# train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
# val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
tokenizer = GPTTokenizer(MAX_SEQ_LEN)
d1 = FixedLenAdditionDataset(num_examples=NUM_EXAMPLES, tokenizer=tokenizer) # noqa
d2 = FixedLenAdditionDataset(low=0, high=LOW, in_order=True, tokenizer=tokenizer)
dataloader  = torch.utils.data.DataLoader(ConcatDataset([d1, d2]), batch_size=batch_size, shuffle=True)
val_dataset = FixedLenAdditionDataset(num_examples=200, low=1000, high=9999, tokenizer=tokenizer)
num_prompt = 4 + 4 + 2
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=eval_batch_size)
print('Training Dataloader Len | batch_size: ', len(dataloader), batch_size)
print('Eval Dataloader Len | batch_size : ', len(val_dataloader), eval_batch_size)
# -----------------------------------------------------------------------------

def run_eval(model):
    """There are 2 types of eval acc, 1) auto-regressive 2) teacher enforced
    """
    def _sample(logit):
        probs = F.softmax(logit.squeeze(), dim=1) # [b, C]
        return torch.multinomial(probs, 1).int()  # (1,)

    model.eval()
    acc1, acc2, acc3 = [], [], []

    for batch_idx, batch in enumerate(val_dataloader):
        # eval batch size is 1
        input = batch.to(device).squeeze()  # (b,n,) padded to max_seq_len
        p1 = input[:,:num_prompt].int()  # "1234+1234="
        p2 = torch.clone(p1)
        # start auto-regressive decoding
        while p1.shape[1] < MAX_SEQ_LEN: # fixed length decoding
            # auto-regressive
            # with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
            l1, _ = model(p1)  # [b,1,C]
            # teacher enforced
            # with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
            l2, _ = model(input[:,:p2.shape[1]])
            s1 = _sample(l1)
            s2 = _sample(l2)
            p1 = torch.cat([p1, s1], dim=1)
            p2 = torch.cat([p2, s2], dim=1)


        # accuracy, vocab is >10, so a random guess acc is <10%
        # auto-regressive acc
        acc1.append(torch.sum(input[:,num_prompt:] == p1[:,num_prompt:]).detach().item() / p1.nelement())
        # teacher enforcing acc
        acc2.append(torch.sum(input[:,num_prompt:] == p2[:,num_prompt:]).detach().item() / p2.nelement())
        # # acc3, fully match
        acc3.append(1- torch.any(input[:,num_prompt:] != p1[:,num_prompt:], dim=1).float().mean().detach().item())

    print(f"eval size: {len(val_dataset)} prompt len: {num_prompt} eval acc1: {np.mean(acc1)}, acc2: {np.mean(acc2)}, acc3: {np.mean(acc3)}", )
    print("============== sampled output ==========")
    print("input:  ", val_dataset.reverse_tokenize(input[0]))
    print("output: ", val_dataset.reverse_tokenize(p1[0]))




# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    # gradient_accumulation_steps *= 8 # simulate 8 gpus

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
# note: float16 data type will automatically use a GradScaler
ctx = nullcontext()



# model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
#                   bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line

print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
# initialize from OpenAI GPT-2 weights
override_args = dict(dropout=dropout)
model = GPT.from_pretrained(init_from, override_args)
# # read off the created config params, so we can store them into checkpoint correctly
# for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
#     model_args[k] = getattr(model.config, k)

model.to(device)

optimizer = torch.optim.AdamW([model.inductor], lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)


print('optimizer targeted params: ', optimizer.param_groups)

if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
# if wandb_log and master_process:
#     import wandb
    # wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# val sanity check before train loop
run_eval(model)
# training loop
t0 = time.time()
raw_model = model.module if ddp else model # unwrap DDP container if needed
for i in range(NUM_EPOCHES):  # num of epoches
    model.train()
    for batch_idx, batch in enumerate(dataloader):

        # determine and set the learning rate for this iteration
        lr = get_lr(batch_idx) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        x = batch[:, :-1]
        y = batch[:, 1:]
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        with ctx:
            logits, loss = model(x, y)

        # backward pass, with gradient scaling if training in fp16
    #     scaler.scale(loss).backward()
        loss.backward()
        # # clip the gradient
        # if grad_clip != 0.0:
        #     scaler.unscale_(optimizer)
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        # # step the optimizer and scaler if training in fp16
        # scaler.step(optimizer)
        # scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        # optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if batch_idx % log_interval == 0 and master_process:
            lossf = loss.item() # loss as float. note: this is a CPU-GPU sync point
            print(f"[Epoch {i}] iter {batch_idx}: loss {lossf:.4f}, time {dt*1000:.2f}ms")

    print('skip eval for now!')
    # run_eval(model)

checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
print(f"saving checkpoint to {out_dir}")
torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
if ddp:
    destroy_process_group()
