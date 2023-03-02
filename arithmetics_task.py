import random

from task_common import GPTTokenizer
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset

NUM_EXAMPLES = 10000
MAX_SEQ_LEN = 40
LOW = 100
HIGH = 9999
START_INDICATOR = ":"
EXAMPLE_SEPARATOR = " $ "
# part of them use
TARGET_TOKENS = [
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
    EXAMPLE_SEPARATOR,
    START_INDICATOR,
    # "plus", "minus", "arithmetic", "equal", "digit", "calculate", "carry bits", "radix", "bit", "add", "number",
    # chain-of-thought (CoT) like prompt
    "number", "plus", "number", "equals to", "digit", "plus", "digit", "plus", "carry bits", "next",
]


def generate_addition(a, b):
    """The question is how to ensure the expressions (prompt) have no ambiguity, ambiguities like:

    case 1: aaaab
    case 2: aaaaa
    for auto-regressive decoder, up to the 4th token the model sees exactly same thing, but end token is different.

    Clearly, this form of addition problem has no ambiguity. But what about other problems?
    """
    # text = "%d+%d;" % (a, b) # symbolic form
    text = "%d + %d : " % (a,b) # non symbolic form
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




class FixedLenAdditionDataset(Dataset):
    """
    See GPTTokenizer for example specs

    Each example is the same len to faciliate training, determined by MAX_SEQ_LEN

    Make sure you check [Example] in stdout to verify the

    """
    def __init__(self, max_seq_len, num_examples=None, in_order=False, low=None, high=None, tokenizer=None, few_shot=None):
        self.max_seq_len = max_seq_len
        self.epoch_len = num_examples
        self.tokenizer = tokenizer
        self.few_shot = few_shot

        self.low = low if low is not None else LOW
        self.high = high if high is not None else HIGH

        print('[FixedLenAdditionDataset]: low ', self.low, " high ", self.high, " max_seq_len: ", self.max_seq_len, " inorder: ", in_order)
        if in_order:
            assert num_examples is None, "we are generating all combinations in order, num_examples is random sampling"
            print('generating all of sum ops in this range: ', (self.high - self.low) ** 2)
            self.data = self.generate_in_order()
        else:
            self.data = self.generate_random_all()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def generate_random_all(self):
        samples = []
        ntimes = self.few_shot if self.few_shot is not None else 1
        for _ in range(self.epoch_len):
            text = ""
            for _ in range(ntimes):
                a = random.randint(self.low, self.high)
                b = random.randint(self.low, self.high)

                text += generate_addition(a, b)
                text += " "

            samples.append(self.tokenizer.generate(text))  # (max_seq_len,) numpy arr

        return samples

    def generate_in_order(self):
        samples = []
        for i in range(self.low, self.high):
            for j in range(self.low, self.high):
                samples.append(generate_addition(i,j))

        if self.few_shot is None:
            return [self.tokenizer.generate(t) for t in samples]
        # pack overlapping examples as few shot
        result = []
        for i in range(len(samples) - self.few_shot):
            text  = EXAMPLE_SEPARATOR.join(samples[i:i+self.few_shot])
            result.append(self.tokenizer.generate(text))

        return result


def generate_train_val_dataset(max_seq_len=None, few_shot=None, eval_only=False, eval_range=None):
    """Few shot settings:
        None -- zero-shot
        2 - 1-shot, so basically 1 demo + 1 query
        3 - 2-host, 2 demo + 1 query
    """
    # train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    # val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    if max_seq_len is None:
        max_seq_len = MAX_SEQ_LEN

    tokenizer = GPTTokenizer(max_seq_len, START_INDICATOR)
    if not eval_only:
        d1 = FixedLenAdditionDataset(max_seq_len, num_examples=NUM_EXAMPLES, tokenizer=tokenizer, few_shot=few_shot) # noqa
        d2 = FixedLenAdditionDataset(max_seq_len, low=0, high=LOW, in_order=True, tokenizer=tokenizer, few_shot=few_shot)
        dataset = ConcatDataset([d1, d2])
    else:
        dataset = None

    if eval_range:
        low, high = eval_range
    else:
        low, high = 1000, 9999

    val_dataset = FixedLenAdditionDataset(max_seq_len, num_examples=200, low=low, high=high, tokenizer=tokenizer, few_shot=few_shot)

    return dataset, val_dataset
