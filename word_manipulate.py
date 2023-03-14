import string
import random
from torch.utils.data import Dataset
from task_common import GPTTokenizer

START_INDICATOR = "A: \n"

class FixedLetterReverseDataset(Dataset):
    NUM_EXAMPLES = 5000
    TRIGGER_PROMPTS = [
        "reverse the list of letters"
        ]
    MAX_SEQ_LEN = 50
    def __init__(self, num_examples=None, tokenizer=None, max_letter=10):
        self.tokenizer = tokenizer
        if num_examples is None:
            num_examples = FixedLetterReverseDataset.NUM_EXAMPLES

        self.data = []
        for i in range(num_examples):
            s = ""
            for _ in range(max_letter):
                s += random.choice(string.ascii_letters)
            text = self._gen_reverse_letter(s)
            self.data.append(self.generate(text))

    def generate(self, text):
        return self.tokenizer.generate(text)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def _gen_reverse_letter(self, s):
        text = " ".join(list(s)) + "\n"
        text += START_INDICATOR
        text += " ".join(reversed(list(s)))
        return text


class FixedLenConcatLastLetterDataset(Dataset):
    NUM_EXAMPLES = 5000
    TRIGGER_PROMPTS = [
        "find the last letter of each word in the list and concatenate them together",
        "lets think step by step, the 1st word in the list, last letter is, the second word, last letter is",
        "finally we concatenate them together",
    ]
    MAX_SEQ_LEN = 120
    def __init__(self, num_examples=None, tokenizer=None, num_words=5, max_letters=10, fixed_letter=None):
        self.tokenizer = tokenizer
        if num_examples is None:
            num_examples = FixedLenConcatLastLetterDataset.NUM_EXAMPLES

        self.data = []
        for _ in range(num_examples):
            words = []
            for _ in range(num_words):
                if fixed_letter:
                    n = fixed_letter
                else:
                    n = random.randint(1, max_letters)
                w = ""
                for _ in range(n):
                    w += random.choice(string.ascii_letters)
                words.append(w)
            text = self.gen_concat_last_letter(words)
            self.data.append(self.generate(text))

    def generate(self, text):
        return self.tokenizer.generate(text)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def gen_concat_last_letter(self, words):
        text = ""
        for w in words:
            text += "{ "
            text += " ".join(list(w))
            text += "} \n"
        text += START_INDICATOR
        text += " ".join([w[-1] for w in words])
        return text


def generate_letter_reverse_dataset(eval_only=False, val_examples=200):
    tokenizer = GPTTokenizer(FixedLetterReverseDataset.MAX_SEQ_LEN, START_INDICATOR)
    if not eval_only:
        dataset = FixedLetterReverseDataset(tokenizer=tokenizer)
    else:
        dataset = None

    # change max_letters if you wanna see OOD performance
    val_dataset = FixedLetterReverseDataset(val_examples, tokenizer=tokenizer, max_letter=12)
    return dataset, val_dataset


def generate_last_letter_concat_dataset(eval_only=False, val_examples=200):
    tokenizer = GPTTokenizer(FixedLenConcatLastLetterDataset.MAX_SEQ_LEN, START_INDICATOR)
    if not eval_only:
        dataset = FixedLenConcatLastLetterDataset(tokenizer=tokenizer)
    else:
        dataset = None

    # change num_words if you wanna see OOD performance
    val_dataset = FixedLenConcatLastLetterDataset(val_examples, tokenizer=tokenizer, fixed_letter=5, num_words=7)
    return dataset, val_dataset
