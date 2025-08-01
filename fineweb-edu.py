import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

local_dir = "fineweb-edu"
remote_name= "sample-10BT"
shard_size = int(1e8) # each shard is 100M tokens

FOLDER = "./" # Output Folder in Kaggle

DATA_CACHE_DIR = os.path.join(FOLDER, local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

#download the dataset
fw = load_dataset("HuggingFaceFw/fineweb-edu", name=remote_name, split="train")

#init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc.special_tokens['<|endoftext|>'] # end of text token

def tokenize(doc):
    # tokenizes a document and returns a numpy array of uint16 tokens
    tokens = [eot] # eot token delimits (separates) all documents
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert(0<=tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16" #
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

nprocs = max(1, os.cpu_count()//2) # In Kaggle you can use all 4 CPUs there is no problem

with mp.Pool(nprocs) as pool:
    shard_index = 0

    # preallocate buffer for the current shard
    all_tokens_np = np.empty((shard_size,),dtype=np.uint16)
    token_count=0
    progress_bar = None

    for tokens in pool.imap(tokenize, fw, chunksize=16):

        # check if there is enough space in the current shard for new tokens
        if token_count + len(tokens) < shard_size:
            all_tokens_np[token_count: token_count+len(tokens)] = tokens
            token_count+=len(tokens)
            #update progress bar
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            # split the tokens into whatever fits in this shard; the rest go to the next
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count: token_count + remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            shard_index+=1
            progress_bar = None
            all_tokens_np[0:len(tokens) - remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder


# write any remaining tokens as the last shard
if token_count != 0:
    split = "val" if shard_index == 0 else "train"
    filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
    write_datafile(filename, all_tokens_np[:token_count])


