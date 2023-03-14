import argparse
import math
import os
import time
from contextlib import nullcontext

import numpy as np

import torch
from model import GPT
from task_common import compute_acc
from torch.distributed import destroy_process_group, init_process_group
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser(description='GPT2 prompt tuner')
parser.add_argument('--task', type=str, default='arithmetics')
parser.add_argument('--out-dir', '-o', type=str, default='/home/shermanwong/out')
parser.add_argument('--eval', action="store_true")
parser.add_argument('--ckpt-name', type=str, default='ckpt.pt')
parser.add_argument('--epoches', type=int, default=20)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--warmup-iters', type=int, default=1000)
parser.add_argument('--wd', type=float, default=1e-5)
parser.add_argument('--decay-lr', action="store_true")
parser.add_argument('--min-lr', type=float, default=6e-5)
parser.add_argument('--use-mlp', action="store_true")
parser.add_argument('--mlp-dropout', type=float, default=0.0) # no really useful in prompt tuning
# parser.add_argument('--few-shot', type=int, default=0) # discounraged for PromptFinder
# parser.add_argument('--eval-range', type=int, nargs="+")
parser.add_argument('--eval-examples', type=int, default=200)
parser.add_argument('--epoch-val', action="store_true")
parser.add_argument('--print-found-prompts', action="store_true")

args = parser.parse_args()



# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText, DO NOT TOUCH!!!
# I/O
out_dir = args.out_dir
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
## Basic Settings
eval_only = args.eval
ckpt_name = args.ckpt_name
log_interval = 10

# grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
NUM_EPOCHES = args.epoches

## LR settings
# adamw optimizer or SGD
batch_size = 32
eval_batch_size = 10
learning_rate = args.lr # max learning rate
weight_decay = args.wd
beta1 = 0.9
beta2 = 0.95
decay_lr = args.decay_lr # whether to decay the learning rate
warmup_iters = args.warmup_iters # how many steps to warm up for
min_lr = args.min_lr # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# model settings
use_mlp = args.use_mlp
mlp_dropout = args.mlp_dropout

## DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
compile = False # use PyTorch 2.0 to compile the model to be faster




# dataset configs
# -----------------------------------------------------------------------------
if args.task == "arithmetics":
    from arithmetics_task import generate_train_val_dataset

    dataset, val_dataset = generate_train_val_dataset(
        eval_only=eval_only,
        val_examples=args.eval_examples,
    )
elif args.task == "symbolic":
    from symbolic_task import generate_train_val_dataset

    dataset, val_dataset = generate_train_val_dataset(
        eval_only=eval_only, val_examples=args.eval_examples
    )
elif args.task == "reverse_letter":
    from word_manipulate import generate_letter_reverse_dataset

    dataset, val_dataset = generate_letter_reverse_dataset(
        eval_only=eval_only,
        val_examples=args.eval_examples,
    )
elif args.task == "last_letter_concat":
    from word_manipulate import generate_last_letter_concat_dataset

    dataset, val_dataset = generate_last_letter_concat_dataset(
        eval_only=eval_only,
        val_examples=args.eval_examples,
    )
else:
    raise Exception(f"unsupported task name: {args.task}")


if not eval_only:
    dataloader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print('Training Dataloader Len | batch_size | max_iters: ', len(dataloader), batch_size, len(dataloader) * NUM_EPOCHES)
    lr_decay_iters = len(dataloader) * NUM_EPOCHES # should be ~= max_iters per Chinchilla
    print("lr_decay max steps: ", lr_decay_iters)

val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=eval_batch_size)
print('Eval Dataloader Len | batch_size : ', len(val_dataloader), eval_batch_size)
# -----------------------------------------------------------------------------



def run_eval(model):
    def _sample(logit):
        probs = F.softmax(logit.squeeze(), dim=1) # [b, C]
        return torch.multinomial(probs, 1).int()  # (1,)

    model.eval()
    acc1, acc3 = [], []
    for batch_idx, batch in enumerate(val_dataloader):
        # eval batch size is 1
        src = batch["src"] # [b,max_seq_len]
        tgt = batch["tgt"] # [b,max_seq_len]
        num_prompts = batch["num_prompt"].detach().numpy()
        if not np.all(num_prompts[0] == num_prompts):
            print(f"[WARNING]:: num_prompts in batch: {num_prompts} are not even, no bueno during batched eval! using max prompt, check tokenizer")
            num_prompt = num_prompts.max()
        else:
            num_prompt = num_prompts[0]
        input = src.to(device).squeeze()  # (b,n,) padded to max_seq_len
        p1 = input[:,:num_prompt].int()  # "1234+1234="
        # start auto-regressive decoding
        while p1.shape[1] < val_dataset.MAX_SEQ_LEN: # fixed length decoding
            # auto-regressive
            # with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
            l1, _ = model(p1)  # [b,1,C]
            # teacher enforced -- not really useful
            # with torch.amp.autocast(device_type="cuda", dtype=DTYPE):
            s1 = _sample(l1)
            p1 = torch.cat([p1, s1], dim=1)

        a1, a3 = compute_acc(p1.detach().cpu().numpy(), tgt.detach().cpu().numpy(), num_prompt)
        acc1.append(a1)
        acc3.append(a3)

    print(f"eval size: {len(val_dataset)} prompt len: {num_prompt} eval acc1: {np.mean(acc1)}, acc3: {np.mean(acc3)}", )
    print("============== sampled output ==========")
    print("input:  ", val_dataset.tokenizer.reverse(input[0]))
    print("target: ", val_dataset.tokenizer.reverse(tgt[0][num_prompt-1:]))
    print("output: ", val_dataset.tokenizer.reverse(p1[0][num_prompt:]))




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
model.to(device)
model.add_inductor(val_dataset.TRIGGER_PROMPTS, use_mlp=use_mlp, mlp_dropout=mlp_dropout)
# # read off the created config params, so we can store them into checkpoint correctly
# for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
#     model_args[k] = getattr(model.config, k)

if eval_only:
    ckpt_path = os.path.join(out_dir, ckpt_name)
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['inductor']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.inductor.load_state_dict(state_dict)
    model.to(device)
    run_eval(model)
    print('done eval only')
    # just teasing, seeing how crazy overfitted the prompt is
    if args.print_found_prompts:
        ids = model.find_nearest_ids()
        print('found nearest prompts: ')
        hard_prompt = val_dataset.tokenizer.reverse(ids)
        print(hard_prompt)

    exit(0)

optimizer = torch.optim.AdamW(model.inductor.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
# print('optimizer targeted params: ', optimizer.param_groups)

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


# first, assuming no prompt tuning is conducted, see how good the LLM is
print("=============== Pure Prompt Engineering Results ==============")
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
        x = batch['src']
        y = batch['tgt']
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

    if args.epoch_val:
        run_eval(model)

checkpoint = {
                    'inductor': raw_model.inductor.state_dict(),
                    # 'optimizer': optimizer.state_dict(),
            }

path = os.path.join(out_dir, ckpt_name)
print(f"saving checkpoint to {path}")
torch.save(checkpoint, path)
if ddp:
    destroy_process_group()
