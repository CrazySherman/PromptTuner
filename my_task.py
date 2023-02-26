import torch
import random
import numpy as np
from torch.utils.data import Dataset
import tiktoken

# TODO use ascii (128 tokens) as standard, same model, just easier convension to use
TOKEN_MAP = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "+",
    "=",
    ";",
    "$",
    "PAD",
]
TOKEN_MAP = {a: i for i, a in enumerate(TOKEN_MAP)}
SEP_TOKEN = ";"
STOP_TOKEN = "$"
MAX_SEQ_LEN = 40
LOW = 100
HIGH = 9999

def generate_addition(a, b):
    """The question is how to ensure the expressions (prompt) have no ambiguity, ambiguities like:

    case 1: aaaab
    case 2: aaaaa
    for auto-regressive decoder, up to the 4th token the model sees exactly same thing, but end token is different.

    Clearly, this form of addition problem has no ambiguity. But what about other problems?
    """
    # text = "%d+%d;" % (a, b) # symbolic form
    text = "%d + %d = " % (a,b) # non symbolic form
    carry = 0
    s = a + b
    while True:
        # text += "%d%d%d" % (a % 10, b % 10, carry)
        text += "%d %d %d" % (a % 10, b % 10, carry) # alignment mode only
        # if a == 0 and b == 0 and carry == 0:
        #     break

        # aignment mode only
        if a == 0 and b == 0:
            break
        text += " %d ; " % ((a + b + carry) % 10)
        carry = (a % 10 + b % 10 + carry) // 10
        a = a // 10
        b = b // 10
    text += " = %d $" % s
    return text

example_txt = generate_addition(3221, 6254)
print("[Example Input]: ", example_txt, " Len: ", len(example_txt))


class GPTTokenizer:
    def __init__(self, max_seq_len):
        self.enc = tiktoken.get_encoding("gpt2")
        self.end_token = "<|endoftext|>" # idx 50256
        self.max_seq_len = max_seq_len

    def encode(self, text):
        ids = self.enc.encode(text, allowed_special={self.end_token})
        assert len(ids) <= self.max_seq_len, len(ids)
        return np.array(ids + [50256] * (self.max_seq_len - len(ids)), dtype=int)

    def decode(self, ids):
        return self.enc.decode(ids.tolist())


class FixedLenAdditionDataset(Dataset):
    def __init__(self, max_seq_len=MAX_SEQ_LEN, num_examples=None, in_order=False, padding=True, low=None, high=None, tokenizer=None):
        self.max_seq_len = max_seq_len
        self.epoch_len = num_examples
        self.tokenizer = tokenizer

        self.low = low if low is not None else LOW
        self.high = high if high is not None else HIGH

        print('[FixedLenAdditionDataset]: low ', self.low, " high ", self.high, " max_seq_len: ", self.max_seq_len, " inorder: ", in_order)
        self.padding = padding
        if in_order:
            assert num_examples is None, "we are generating all combinations in order, num_examples is random sampling"
            print('generating all of sum ops in this range: ', (self.high - self.low) ** 2)
            self.data = self.generate_in_order()
        else:
            self.data = self.generate_random_all()

    def tokenize(self, text: str):
        if self.tokenizer is not None:
            return self.tokenizer.encode(text)

        res = [TOKEN_MAP[s] for s in text]
        assert self.max_seq_len >= len(text)
        if self.padding:
            # padding token
            res += [TOKEN_MAP["PAD"]] * (self.max_seq_len - len(text))

        return np.array(res, dtype=int)

    def reverse_tokenize(self, input: torch.Tensor):
        if self.tokenizer is not None:
            return self.tokenizer.decode(input)

        reverse_map = {v: k for k, v in TOKEN_MAP.items()}
        text = ""
        for elem in input.tolist():
            s = reverse_map[elem]
            text += "?" if s == "PAD" else s
        return text

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def generate_random_all(self):
        samples = []
        for _ in range(self.epoch_len):
            a = random.randint(self.low, self.high)
            b = random.randint(self.low, self.high)

            text = generate_addition(a, b)
            samples.append(self.tokenize(text))  # (max_seq_len,) numpy arr
        return samples

    def generate_in_order(self):
        samples = []
        for i in range(self.low, self.high):
            for j in range(self.low, self.high):
                text = generate_addition(i,j)
                samples.append(self.tokenize(text))
        return samples
