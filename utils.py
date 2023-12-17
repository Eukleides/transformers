from glob import glob
import pickle
import tiktoken
import torch
import random

RAW_DATA_FOLDER = 'data\\training\\raw_data\\'
SAVED_MODELS_FOLDER = 'data\\training\\models\\'

def set_seed(seed=0):
    if seed is None:
        return
    torch.manual_seed(seed)
    random.seed(seed)
class Config():
    def __init__(self, model_name='gpt2'):
        self.RAW_DATA_FOLDER = RAW_DATA_FOLDER
        self.SAVED_MODELS_FOLDER = SAVED_MODELS_FOLDER

        if model_name.upper()=='GPT2':
            self.model_name = 'GPT2'
            self.N = 12 # number of transformer blocks
            self.h = 12 # number of heads
            self.d_model = 768 # token embedding size
            self.seq_length = 1024 # context size
            self.vocab_size = 50257 # number of tokens in vocabulary
            self.encoding = 'r50k_base' # tiktoken tokeniser to use
        else:
            raise TypeError(f'Unrecognised model {model_name}')

def get_files(directory, filter='*.txt'):
    return glob(f'{directory}/{filter}')

def raw_file_to_tokens(fpath, encoding='r50k_base', max_token_length=100000):
    f = open(fpath, "r", encoding="utf8")
    txt = ''
    for line in f:
        line_txt = " ".join(line.split())
        txt += ' ' + line_txt

    txt = ' '.join(txt.split())
    enc = tiktoken.get_encoding(encoding)
    txt_tokens = enc.encode(txt)

    tokens=[]

    current_tokens = []
    count = 0
    for t in txt_tokens:
        current_tokens.append(t)
        count += 1

        if count>=max_token_length:
            tokens.append(current_tokens)
            current_tokens = []
            count = 0

    if len(current_tokens)>0:
        tokens.append(current_tokens)

    return tokens

def get_token_fname(cfg, id=''):
    if id == '':
        return f'data\\training\\{cfg.model_name}_tokens.pk'
    else:
        return f'data\\training\\{cfg.model_name}_{id}.pk'
def raw_data_to_tokens(cfg, save_file='tokens.pk', max_token_length=100000):
    tokens = []
    encoding = cfg.encoding
    for f in get_files(f'{RAW_DATA_FOLDER}'):
        f_tokens = raw_file_to_tokens(f, encoding=encoding, max_token_length=max_token_length)
        tokens.extend(f_tokens)

    with open(get_token_fname(cfg), 'wb') as fp:
        pickle.dump(tokens, fp)

def load_tokens(cfg, verbose=False):
    tokens = []
    with open(get_token_fname(cfg), 'rb') as fp:
        tokens = pickle.load(fp)

        if verbose:
            count = 0
            for i, tline in enumerate(tokens):
                n = len(tline)
                count += n
                print(f'group {i+1}, {n:,} tokens ')

            print(f'{len(tokens):,} token rows.')
            print(f'{count:,} tokens.')

    return tokens

def diagnostic():
    cfg = Config(model_name='gtp2')
    raw_data_to_tokens(cfg, max_token_length=100000)
    tokens = load_tokens()

    enc = tiktoken.get_encoding(cfg.encoding)
    f = open("data\\training\\temp.txt", "w", encoding="utf8")
    for r in tokens:
        txt = enc.decode(r)
        f.write(txt)
        f.write('\n')
    f.close()
