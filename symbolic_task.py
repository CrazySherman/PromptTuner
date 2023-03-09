from random import randint
import numpy as np
from torch.utils.data import Dataset
from task_common import GPTTokenizer

START_INDICATOR = "A: \n" # see GPT prompt guide, depends on pretrained LLM
NUM_EXAMPLES = 5000


class SymbolMultiplyDataset(Dataset):
    TRIGGER_PROMPTS = [
        "Q: given a list of elements a_i, multiply them and modulo by 10, what is th result? Let's do it step by step"
    ]
    MAX_SEQ_LEN = 160
    def __init__(self, num_examples, tokenizer=None, max_digit=4, fix_digit=None):
        self.tokenizer = tokenizer
        self.data = []
        self.max_seq_len = 0
        for i in range(num_examples):
            if fix_digit:
                n = fix_digit
            else:
                n = randint(1, max_digit)
            text = self._gen_multiply(n)
            self.data.append(self.generate(text))

    def generate(self, text):
        return self.tokenizer.generate(text)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def _gen_multiply(self, num_digits):
        text = ''
        a = np.random.randint(1,10, (num_digits,))
        for i in range(num_digits):
            text += 'a_%d' % i + " = %d" % a[i] + " \n"
        b = np.zeros((num_digits,),dtype=int)
        text += START_INDICATOR
        text += "b_0 = a_0 \n "
        for i in range(num_digits - 1):
            text += f"b_{i+1} = b_{i} * a_{i+1} % 10 \n"
        b[0] = a[0]
        text += f"b_0 = {b[0]} \n"
        for i in range(num_digits - 1):
            b[i+1] = b[i] * a[i + 1] % 10
            text += f"b_{i+1} = {b[i+1]} \n"
        return text


def generate_train_val_dataset(eval_only=False, val_examples=200):
    tokenizer = GPTTokenizer(SymbolMultiplyDataset.MAX_SEQ_LEN, START_INDICATOR)
    if not eval_only:
        dataset = SymbolMultiplyDataset(NUM_EXAMPLES, tokenizer=tokenizer)
    else:
        dataset = None

    val_dataset = SymbolMultiplyDataset(val_examples, tokenizer=tokenizer, fix_digit=4)
    return dataset, val_dataset
