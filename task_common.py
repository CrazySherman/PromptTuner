import numpy as np
import tiktoken
import torch

def compute_acc(input, tgt, num_prompt):
    """There are 2 types of eval acc, 1) auto-regressive 2) teacher enforced
        Here we skip 2) cuz it doesn't provide much value

    Args:
        input: [b, n] integer numpy arr from decoded model output
        tgt: [b,n] integer numpy arr from `tgt` in an eval sample
        num_prompt: an integer, here assuming all samples in the batch have the same prompt len
    """
    # align input and target, here decoded input is the same as `src`, see GPTTokenizer specs
    tgt = tgt[:,num_prompt-1:-1]
    input = input[:,num_prompt:]

    # auto-regressive acc
    acc1 = (np.sum(tgt == input) / input.size)
    # teacher enforcing acc
    # # acc3, fully match
    acc3 = 1- np.any(tgt != input, axis=1)
    acc3 = acc3.astype(float).mean()
    return acc1, acc3


class GPTTokenizer:
    """Train/Eval Example:

        src:    p1  p2  p3... pn    x     y    z
        tgt:    -1  -1  -1 ... x     y     z   [END]  -- should be exact same size as input
            here, "-1" tells cross entropy loss function to ignore prompt indices (ignore_index=-1)

        2 APIs:

            - `generate(text)`  -> generate GPT model input {'src': array, 'tgt': array, 'num_prompt': int}
            - `reverse(tensor)` -> de-tokenize the list and return text
    """
    def __init__(self, max_seq_len, start_indictor):
        """
        Args:
            start_indicator: where does prompt end
        """
        self.enc = tiktoken.get_encoding("gpt2")
        self.end_token = "<|endoftext|>" # idx 50256
        self.max_seq_len = max_seq_len
        self.start_indicator = start_indictor
        self._show_example = False

    def encode(self, text):
        ids = self.enc.encode(text, allowed_special={self.end_token})
        return np.array(ids, dtype=int)

    def padding(self, ids):
        assert len(ids) <= self.max_seq_len, len(ids)
        return np.concatenate((ids, [50256] * (self.max_seq_len - len(ids))), dtype=int)

    def reverse(self, ids):
        if isinstance(ids, (np.ndarray, torch.Tensor)):
            ids = ids.tolist()

        return self.enc.decode(ids)


    def generate(self, text: str):
        idx = text.rfind(self.start_indicator) # last occurance of "start" indicator
        prompt = self.encode(text[:idx+1])
        target = self.encode(text[idx+1:])

        example = {
            "src": self.padding(np.concatenate((prompt,target))),
            "tgt": self.padding(np.concatenate((-1 * np.ones(len(prompt) - 1, dtype=int), target))),
            "num_prompt": len(prompt),
        }
        if not self._show_example:
            print('[Example] ', text)
            print('[Example] prompt tokens: ', prompt)
            print('[Example] target tokens: ', target)
            print('[Example] full example: ', example)
            print('[Example] reverse tokenize: ', self.reverse(example["src"]))
            self._show_example = True

        return example
