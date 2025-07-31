import math
import tiktoken
import torch
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
import os
import time
import numpy as np
from my_gpt import GPT, GPTConfig

B = 4
T = 1024
total_batch_size = 524288  # batch_size GPT (num_tokens)
max_lr = 6e-4
min_lr = max_lr * 0.1
# 10B tokens // 524288 tokens (batch_size) = 19073 steps
# 19073 * 15 seconds per step (2 T4 Kaggle GPU) = 286,095 seconds
# 79.5 hours
# We only have 30 hours weekly on kaggle.
#
# So I will use 20 hours which correspond to 4800 steps
# This will process a total of about 2.5B tokens.
total_steps = 4800

# Original GPT-2 Paper has 715 steps for warmup
# ratio = 19073 / 715 = 26.68
# new_warmup_steps = 4800 / 26.68 = 180

warmup_steps = 180

shard_size = 1e8

model_path = "/kaggle/input/models/log"  # change path to checkpoint-paths

assert torch.cuda.is_available()
ddp = int(os.environ.get('RANK', -1)) != -1

if ddp and torch.cuda.device_count() > 1:  # if we have more than one GPU
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ["RANK"])  # Global rank (across multi node)
    ddp_local_rank = int(os.environ["LOCAL_RANK"])  # Local Rank on the current node
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # flag to identify the master process, usually cuda: 0

else:
    ddp_rank = 0
    ddp_local_rank = 0  # just in case of multiple nodes. (Clusters) but we only have one, with 2 GPUs.
    ddp_world_size = 1
    master_process = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

torch.manual_seed(1337)  # removes CPU side randomness (e.g tensor inits)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)  # removes GPU side randomness


def load_model():
    if os.path.exists(model_path):
        models = [f for f in os.listdir(model_path) if f.endswith('.pt') and 'model_' in f]
        if not models:
            model = GPT(GPTConfig())
            model = torch.compile(model)  # speed-up quirk. Better Compilation.
            print("No model checkpoint files found.")
            return model, None

        latest_model_name = sorted(models)[-1]  # assumes the naming convention model_XXXXX.pt for all.
        checkpoint = torch.load(os.path.join(model_path, latest_model_name), weights_only=False)
        step = checkpoint['step']
        config = checkpoint['config']
        model_weights = checkpoint['model']
        val_loss = checkpoint['val_loss']

        # TODO: Make sure to save the Raw Model next time Instead of the Compiled One
        model = GPT(config)
        model = torch.compile(model)  # speed-up quirk. Better Compilation.
        model.load_state_dict(model_weights)

        print(f"loaded previous checkpoint. Step: {step}, val_loss: {val_loss}  ")
        return model, step
    else:
        model = GPT(GPTConfig())
        model = torch.compile(model)  # speed-up quirk. Better Compilation.
        print(f"No Previous Checkpoint found  ")
        return model, None


model, checkpoint_step = load_model()

# model.eval()
model.to(device)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model  # for the optimizer below

enc = tiktoken.get_encoding("gpt2")


def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    def __init__(self, B, T, rank, num_proc, split, step=0):
        self.B = B
        self.T = T
        self.rank = rank
        self.num_proc = num_proc
        assert split in {'train', 'val'}

        # get the shards
        data_root = "/kaggle/input/fineweb-edu/edu_fineweb10b"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0

        if master_process:
            print(f" Shards: {len(shards)} for split {split}")

        self.reset(step)

    def reset(self, step=0):
        # state, init the shard depending on check_point step.
        self.current_shard = int((step * total_batch_size) // (shard_size))
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.rank * self.B * self.T

        print(f"Current shard: {self.current_shard}")

    def next_batch(self):
        # Get the whole line of tokens (B * T)

        buf = self.tokens[self.current_position: self.current_position + (self.B * self.T) + 1]

        # Separate the line into batches
        x = buf[:-1].view(self.B, self.T)  # inputs
        y = buf[1:].view(self.B, self.T)  # targets

        # set start index of next batch
        self.current_position += (self.num_proc * self.B * self.T)

        # if the end index of next batch is beyond num of tokens then we just reset
        if self.current_position + (self.B * self.T) + 1 > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)  # next shard
            self.tokens = load_tokens(self.shards[self.current_shard])  # load next shard
            self.current_position = self.rank * self.B * self.T  # Reset Pos in next shard

        return x, y


# Scheduler

def get_lr(step):
    # Warmup
    step += 1  # step must start at 1
    if step < warmup_steps:
        lr = max_lr * (step / warmup_steps)

    # Cosine Decay
    else:
        rem_step_ratio = ((step - warmup_steps) / (total_steps - warmup_steps))
        inter_lr = max_lr * (math.cos(rem_step_ratio * math.pi * 0.5))
        lr = max(min_lr, inter_lr)
    return lr


optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=max_lr, betas=(0.9, 0.95), device=device)

assert total_batch_size % (B * T * ddp_world_size) == 0
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

if master_process:
    print(f"Total batch size {total_batch_size}")
    print(f"accumulation steps {grad_accum_steps}")

print(f"I am GPU: ", ddp_rank)
train_data = DataLoaderLite(B, T, ddp_rank, ddp_world_size, split="train", step=checkpoint_step)
val_data = DataLoaderLite(B, T, ddp_rank, ddp_world_size, split="val")

# makes matrix multiplications use tf32 or bfloat16
# Both not available on both kaggle GPUs
# torch.set_float32_matmul_precision('high')


# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)

scaler = torch.amp.GradScaler("cuda")

step = checkpoint_step if checkpoint_step is not None else 0

while step < total_steps:
    start = time.time()
    last_step = (step == total_steps - 1)
    # eval loop
    if step % 100 == 0 or last_step:
        model.eval()
        val_data.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_data.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.float16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")

            if step > 0 and (step % 900 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)

    # Text Generation loop (visualization of improvement)
    if step > 0 and step % 100 == 0:
        model.eval()
        num_return_sequences = 1
        max_length = 32
        tokens = enc.encode("Hello I am a language Model")
        tokens = torch.tensor(tokens, dtype=torch.long)  # shape (T,)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # shape (B,T)
        xgen = tokens.to(device)  # (B,T)
        sample_rng = torch.Generator(
            device=device)  # set a local random number generator to get varying responses for each gpu.
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            logits, _ = model(xgen)  # (B, T, vocab_size)
            logits = logits[:, -1, :]  # (B, vocab_size)
            # Get the probs
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)  # (B, 50)
            # get the next token from the top 50 tokens
            ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # (B, 1), returns an index
            # gather the corresponding indices in vocab_size
            xcol = torch.gather(topk_indices, dim=-1, index=ix)  # (B, 1)
            # append the new token
            xgen = torch.cat((xgen, xcol), dim=1)
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded} ")

    # train loop
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        X, Y = train_data.next_batch()
        X, Y = X.to(device), Y.to(device)
        with torch.autocast(device_type=device, dtype=torch.float16):
            logits, loss = model(X, Y)

        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (
                        micro_step == grad_accum_steps - 1)  # prevents ddp syncing (with other models) with each grad accum step. (Sync only with last step)

        # Scales the losses to prevent underflow, calls backward() creating scaled gradients.
        # Without this scaling we would get more often underflown gradients.
        scaler.scale(loss).backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    # Unscales the gradients of optimizer's assigned params in-place
    scaler.unscale_(optimizer)

    # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    lr = get_lr(step)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # With now presence of gradients. They need to be scaled back down as they are too big.
    # scaler.step scales them down.
    # (If we explicitly unscaled them like above then step will not unscale them again)
    # If some gradients are inf or NaN, then step makes sure
    # the parameters are not updated with the gradients.(avoid corruption)
    scaler.step(optimizer)  # updates Parameters

    # Updates the scaler for next iteration.
    scaler.update()

    torch.cuda.synchronize()  # Forces CPU to wait for GPU to finish.

    end = time.time()
    duration = (end - start) * 1000  # in milliseconds
    tok_per_sec = (train_data.B * train_data.T * grad_accum_steps * ddp_world_size) / (
                end - start)  # tokens processed per sec
    if master_process:
        print(
            f"step: {step} loss: {loss_accum.item():.4f} lr: {lr:.3e} norm:{norm:.4f} Duration: {duration:.4f} tok/sec: {tok_per_sec:.4f}")

    step += 1

if ddp:
    destroy_process_group()
