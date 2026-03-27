"""
Here are some important points that help me understand the training objective:
(Note: these notes taught me a lot https://arxiv.org/pdf/2506.02070 and I borrow their framing/notation)

- If using Gaussian probability paths (which appears to be pretty standard for most SOTA models), you can:
    1) define the conditional probability path p_t(x|z) which at t=0 is Gaussian noise p_0(x|z) = N(0, I_d) 
       and at t=1 is a dirac delta centered at a specific data point z (e.g. an image), p_1(x|z) = delta(x-z).
       We define it by p_t(x|z) = N(alpha_t z, beta_t^2 I_d) where alpha_t and beta_t are continuously differentiable
       w.r.t. t (also, I_d is the d-dimensional identity matrix since we are assuming x and z are in R^d). alpha_t
       and beta_t are defined such that alpha_0 = beta_1 = 0 and alpha_1 = beta_0 = 1 (so at t=0 we get a standard 
       Gaussian as our initial noise distribution and at t=1, we get a Gaussian with mean z and variance 0 which is
       a dirac delta centered at z, just as I specified above) 
    2) derive an analytical expression for the conditional vector field u_t^{target}(x|z) that defines our ODE.
       In particular, if we have the ODE d/dt(X_t) = u_t^{target}(x|z) along with an initial condition X_0 ~ p_0(.|z)
       (that is, our initial condition is a random variable sampled from our initial noise distribution and thus X_t
       itself is now nondeterminstic for all t), the whole point is to find the special target conditional vector field
       such that X_t ~ p_t(.|z) for all t in [0,1] --- that is, we are essentially finding the conditional vector field
       that induces an ODE such that its trajectories are distributed according to the conditional probability path. The
       reason why we are doing this is because later we'll have a training objective to learn a marginal vector field
       (which we are modeling with a neural network) by minimizing a loss function involving a target marginal vector field
       that we would like our neural network to approximate and it can be shown that the marginal loss function is equivalent
       to a conditional version of that loss function up to an additive constant so that gradients (w.r.t. model parameters)
       are the same, which will be important because we cannot get an analytical expression for the marginal vector field but
       we CAN get one for the conditional vector field! To derive an analytical expression for the conditional Gaussian vector
       field, consider trajectories of the form   F_t(x|z) = alpha_t z + beta_t x   and notice that:
                    F_t(X_0|z) ~ p_t(.|z) = N(alpha_t z, beta_t^2 I_d)   if   X_0 ~ p_0(.|z) = N(0, I_d)
       so these tracjectories are distributed according to our conditional probability path. Next, you can plug F_t(x|z) into
       our ODE (i.e. replace X_t with F_t(x|z)) and then a change of variables and some basic algebra can let you solve for 
       what the conditional vector field needs to be in order for our ODE to have the trajectories we constructed as solutions.
       The conditional vector field is:
            u_t^{target}(x|z) = (A_t - alpha_t * (B_t/beta_t)) * z + (B_t/beta_t) * x 
       where A_t = d/dt(alpha_t) and B_t = d/dt(beta_t) are the time derivatives of alpha_t and beta_t

- Using the continuity equation (special case of Fokker-Planck), you can show that the ODE whose trajectories are distributed
according to the marginal probability path is induced by a marginal vector field defined by 
    u_t^{target}(x) = int_{R^d}[u_t^{target}(x|z) * (p_t(x|z) * p_{data}(z) / p_t(x)) dz] 
(note: the marginal probability path is defined by p_t(x) = int_{R^d}[p_t(x|z) p_{data}(z) dz] --- p_t(x) is intractable to compute
in practice because of the high-dimensional integral and such, but we can stll sample from p_t(x) since we can effectively sample 
from p_{data} by simply randomly selecting an image from our large dataset of images and in the case of Gaussian probability paths
we can of course sample from our conditional probability path since we have a closed-form expression for it. Also, the important
quality of the marginal probability path is that p_0(x) is our noise distribution, but p_1(x) = p_{data} is our data distribution,
which can be shown using basic properties of the dirac delta that we get from our conditional probability path --- the fact that
p_1(x) = p_{data} is what we really care about because if we sample an initial condition X_0 for our ODE from p_0(x) = N(0, I_d)
for example, then we can just numerically integrate e.g. with forward euler to transform X_0 into X_1 such that X_1 ~ p_{data}
approximately. Thus, if we can compute the marginal vector field, we can do forward euler and effectively sample novel points
from p_{data} that don't already appear in our dataset --- this is the whole goal of generative modeling).

- Although the marginal vector field is not directly useful to us as we cannot compute it in practice (the integral is intractable
to compute, same issue we have for computing the marginal probability path), it is useful for defining our training objective because
training a model that approximates the target marginal vector field is what we genuinely care about (once we know the marginal vector
field, we have everything we need to do numerical integration to approximately sample novel points from p_{data}). Using this theory,
we can define the marginal flow-matching loss:
    L_{FM}(theta) = E_{t~Unif([0,1]), x~p_t}[|| u_t^{theta}(x) - u_t^{target}(x) ||_2^2]
                  = E_{t~Unif([0,1]), z~p_{data}, x~p_t(.|z)}[|| u_t^{theta}(x) - u_t^{target}(x) ||_2^2]
where in practice we approximate this expected value by sampling random times uniformly from [0,1] and we sample random X_t trajectory 
values when we sample x from the marginal probability path (which we sample from using our dataset as approximately sampling z from p_{data}
and then sampling x from p_t(.|z) which we can do in the Gaussian case as we have an analytical expression for the conditional probability path).

- You can show that the conditional flow-matching loss L_{CFM} is equal to the marginal flow-matching loss L_{FM} up to an additive constant, i.e.
L_{FM} = L_{CFM} + C where C does not depend on the model parameters theta, so it suffices to optimize the conditional flow-matching loss which we
can actually do in the Gaussian case because we can get a closed-form expression for the conditional vector field in this case (as discussed above).
The conditional flow-matching loss is given by:
    L_{CFM}(theta) = E_{t~Unif([0,1]), x~p_t}[|| u_t^{theta}(x) - u_t^{target}(x|z) ||_2^2]
                   = E_{t~Unif([0,1]), z~p_{data}, x~p_t(.|z)}[|| u_t^{theta}(x) - u_t^{target}(x|z) ||_2^2]
where we just replaced u_t^{target}(x) with u_t^{target}(x|z) from the marginal flow-matching loss.

- For Gaussian probability paths, we can plug in our closed-form expression for the conditional vector field into the CFM loss and plug in our
expression for the conditional probability path (which shows up as part of our expectation is over x ~ p_t(.|z) = N(alpha_t z, beta_t^2 I_d)):
    L_{CFM}(theta) = E_{t~Unif([0,1]), z~p_{data}, x~p_t(.|z)}[|| u_t^{theta}(x) - u_t^{target}(x|z) ||_2^2]
                   = E_{t~Unif([0,1]), z~p_{data}, x~N(alpha_t z, beta_t^2 I_d)}[|| u_t^{theta}(x) - u_t^{target}(x|z) ||_2^2]
                   = E_{t~Unif([0,1]), z~p_{data}, x~N(alpha_t z, beta_t^2 I_d)}[|| u_t^{theta}(x) - ((A_t - alpha_t * (B_t/beta_t)) * z + (B_t/beta_t) * x)  ||_2^2]
                   = E_{t~Unif([0,1]), z~p_{data}, epsilon~N(0, I_d)}[|| u_t^{theta}(alpha_t * z + beta_t * epsilon) - (A_t * z + B_t * epsilon) ||_2^2]

again, A_t = d/dt(alpha_t), B_t = d/dt(beta_t). If we choose linear noise schedulers, then we have alpha_t = t and beta_t = 1-t so that 
A_t = 1 and B_t = -1 for all t in [0,1]. With these choices, our CFM loss simplifies even further:
    L_{CFM}(theta) = E_{t~Unif([0,1]), z~p_{data}, epsilon~N(0, I_d)}[|| u_t^{theta}(t*z + (1-t)*epsilon) - (z - epsilon) ||_2^2]
where our model's input is a simple convex combination between some data point z and standard Gaussian noise epsilon and we are trying to optimize to make
our model's output effectively predict the difference between the true data point z and the noise epsilon.

- That's great, this training objective would let us sample novel points from p_{data}, but what if we specifically want to generate a picture
of a dog rather than just any arbitrary picture? We'll use classifier-free guidance (CFG).

- CFG is really just a heuristic (can somewhat motivate it using Gaussian paths and relating the vector field to the score function and doing some Bayes stuff)
Basically, in practice people observed that doing inference with the so-called classifier-free guided vector field
    tilde{u}_t(x|y) = (1-w)*u_t^{target}(x) + w*u_t^{target}(x|y)
empirically gives higher-quality images for some w > 1 (e.g. w=4) where we can view u_t^{target}(x) as u_t^{target}(x|null) where null can be thought of as the
"empty" class label --- this null class label lets us train just a single neural network instead of a separate one for u_t^{target}(x) and u_t^{target}(x|y).

- When training our model with CFG, all that changes is that instead of sampling z ~ p_{data} we now treat our data distribution as a joint distribution over data
points z and class labels y, so we sample (z, y) ~ p_{data} where in practice we replace y with the null class with probability eta > 0 to effectively make it
equivalent to sampling from a joint over all data points and extended class labels (i.e. including the null class label).

- The w > 1 weight from CFG doesn't come into play at all during training, it is used during inference where we simply drop in the classifier-free guided vector field
tilde{u}_t(x|y) = (1-w)*u_t^{target}(x|null) + w*u_t^{target}(x|y) into our ODE and numerically integrate it. Note how this technically doubles the number of function
evaluations that we have to perform during each step of inference (though I've never seen anyone explicitly point this out, I feel like it is an important detail to be
aware of --- better image quality in the same number of numerical integration steps isn't really a fair comparison if we double the number of function evaluations). Claude
suggested some arxiv paper that claims that it could be good to use the CFG vector field for the first 30-50% of numerical integration steps and then just fall back on
to w=1 (when w=1, the null class term goes to 0 and you only do one function evaluation each step instead of two) for the remainder of the steps to reduce the number of
function evaluations.

FLOW MATCHING w/ CFG TRAINING RECIPE (Gaussian probability paths with linear noise schedules):

Start with: 
    - dataset of labeled images (z, y) ~ p_{data}
    - neural network u_t^{theta}

For each training step:
    1. Sample a batch (z, y) of data from your dataloader where z has shape (B, d) and y has shape (B,) where B is the batch size, z is in R^d and y is a scalar class label
    2. Sample a batch of times t where t has shape (B,) and all entries of t are i.i.d. sampled from Unif([0,1])
    3. Sample a batch of noises epsilon where epsilon has shape (B, d) where all entries (index by batch) are i.i.d. sampled from N(0, I_d)
    4. Set x = alpha_t * z + beta_t * epsilon, e.g. alpha_t = t, beta_t = (1-t) --- let A_t := d/dt(alpha_t) and B_t := d/dt(beta_t)
    5. With probability eta, change a given batch's class label to the null class label
    6. Compute loss (1/B) * || u_t^{theta}(x|y) - (A_t * z + B_t * epsilon) ||_2^2    <-- batch approximation of expected value
    7. Take gradient and update model parameters with optimizer of choice

Can do gradient accumulation easily if desired


I'm working on a small scale so I'm probably not going to bother with sharded optimizer states, fp16 (V100s), etc., will just get a basic
proof of concept just to learn continuous diffusion basics and then I'll move on to other projects
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import argparse
import os
from model import UNet, UNetConfig
import math
from datetime import datetime
import time

def sample(model, batch_size, num_steps, class_labels, device, cfg_weight=4, null_class_label=10):

    model.eval()
    with torch.no_grad():

        # TODO use forward euler 

        null_class_labels = null_class_label * torch.ones(batch_size).to(device) # shape (batch_size,)

        u = (1-cfg_weight) * model(x=x, y=null_class_labels, t=t) + cfg_weight * model(x=x, y=class_labels, t=t)

    return images  # (batch_size, 3, 32, 32)

def print0(s="", **kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)

def write0(s, log_file):
    """
    `s` is the string to write to the log file
    `log_file` is the path to the log.txt file
    """
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        with open(log_file, 'a') as f:
            f.write(s)

def create_log_dir(parent_dir):
    ddp_rank = int(os.environ.get('RANK', 0))
    log_dir = None # initialize as None to avoid errors on nonzero ranks
    if ddp_rank == 0:
        timestamp = datetime.now().strftime("%m-%d-%Y-%Hh%Mm%Ss")
        log_dir = os.path.join(parent_dir, f"{timestamp}")
        os.makedirs(log_dir, exist_ok=True)
    return log_dir

effective_batch_size = 1024
grad_accum_steps = 1
training_steps = 100000
batch_size = 128
eta = 1/11
null_label = 10
val_loss_interval = 50
checkpoint_interval = 10000
image_sample_interval = 500
num_image_samples = 10

save_dir = "/mnt/data_r60_1/adv_robust_project/mnist-flow-model/experiments"

log_dir = create_log_dir(save_dir)

warmup_steps = 1000
max_lr = 3e-4
min_lr = max_lr / 10
lr_decay_steps = training_steps - warmup_steps
def get_lr(it):
    # 1) linear warmup for warmup_steps steps
    if it < warmup_steps:
        return max_lr * (it + 1) / (warmup_steps + 1)
    # 2) if it > lr_decay_steps, return min learning rate
    if it > lr_decay_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (lr_decay_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (max_lr - min_lr)
# TODO define/load config, define model

model_config = UNetConfig()
model = UNet(model_config)


ddp = int(os.environ.get('RANK', -1)) != -1

if ddp:
    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    world_size = dist.get_world_size()

    assert ddp_rank == dist.get_rank(), f"{ddp_rank=}, {dist.get_rank()=}"
    ##assert ddp_local_rank == dist.local_rank(), f"{ddp_local_rank=}, {dist.local_rank()=}"

    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    print(f"{ddp_rank=}, {ddp_local_rank=}, {world_size=}, {device=}")
else:
    ddp_rank = 0
    ddp_local_rank = 0
    world_size = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"{ddp_rank=}, {ddp_local_rank=}, {world_size=}, {device=}")


if world_size * batch_size * grad_accum_steps != effective_batch_size:
    raise ValueError(f"{effective_batch_size=}, {world_size=}, {batch_size=}, {grad_accum_steps=}, {world_size*batch_size*grad_accum_steps=}")

# Initialize log file
log_file = f"{log_dir}/log.txt"
if ddp_rank == 0:
    with open(log_file, 'w') as f:
        f.write("")

write0(f"Using {world_size} GPU(s)\n", log_file=log_file)
write0(f"GPU Type: {torch.cuda.get_device_name()}\n", log_file=log_file)

model = model.to(device)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

num_params = sum(p.numel() for p in model.parameters())
num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
write0(f"Model Parameters: {num_params:,}\nTrainable Model Parameters: {num_trainable_params:,}\n", log_file=log_file)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # maps [0,1] -> [-1,1] since noise is N(0, I_d)
])

# Set random seed for when we shuffle our data
generator = torch.Generator()
generator.manual_seed(42) 

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=42)
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True, generator=generator)

val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
val_sampler = DistributedSampler(val_dataset, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=4, pin_memory=True)


### I'm just going to do everything in fp32 for now... can switch to fp16 on V100s after if I want
# # GradScaler for fp16 training (bf16/fp32 don't need it — bf16 has the same exponent range as fp32)
# scaler = torch.amp.GradScaler() if COMPUTE_DTYPE == torch.float16 else None
# if scaler is not None:
#     print0("GradScaler enabled for fp16 training")

##### predefining noise to sample from periodically so that we use the same initial noise for all samples
torch.manual_seed(42)
fixed_noise = torch.randn(num_image_samples, 3, 32, 32, device=device)
fixed_labels = torch.arange(num_image_samples, device=device) % 10  # one of each class, cycling if num_image_samples > 10


epoch = 0
train_iter = iter(train_loader)
for step in range(training_steps):

    if step % val_loss_interval == 0 or step == training_steps - 1:
        torch.cuda.synchronize()
        t0 = time.time()
        model.eval()
        with torch.no_grad():
            val_loss_accum = 0.0
            num_val_batches = 0
            for z_val, y_val in val_dataloader:
                B, C, H, W = z_val.shape
                z_val = z_val.to(device)
                y_val = y_val.to(device)
                # With probability eta, change class label to the null class label
                mask_val = torch.rand(B, device=device) < eta
                y_val[mask_val] = null_label
                t_val = torch.rand(B, device=device) # shape (B,), will need to view as (B, 1, 1, 1) later for broadcasting, consists of B iid Unif([0,1]) samples
                epsilon_val = torch.randn(B, C, H, W, device=device) # shape (B, 3, 32, 32) consisting of B iid N(0,1) samples
                
                x_val = t_val.view(B, 1, 1, 1) * z_val + (1-t_val.view(B, 1, 1, 1)) * epsilon_val # (B, C, H, W)
                target_val = z_val - epsilon_val

                val_loss = (model(x=x_val, t=t_val, y=y_val) - target_val).square().mean()

                val_loss_accum += val_loss.item()
                num_val_batches += 1
            avg_val_loss = val_loss_accum / num_val_batches
        model.train()
        torch.cuda.synchronize()
        t1 = time.time()
        write0(f"step {step} | val loss {avg_val_loss:.4f} | time {t1-t0:.4f}s\n", log_file=log_file)
    
    if False and ddp_rank == 0 and (step % image_sample_interval == 0 or step == training_steps - 1):
        
        model.eval()
        images = sample(model, ...) # TODO

        # undo image transformation!!!
        # save images somewhere

        model.train()

    if ddp_rank == 0 and (step % checkpoint_interval == 0 or step == training_steps - 1):
        torch.cuda.synchronize()
        t0 = time.time()
        checkpoint = {
            'step': step,
            'model': model.module.state_dict() if ddp else model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }
        checkpoint_path = os.path.join(log_dir, f'checkpoint_step{step}.pt')
        torch.save(checkpoint, checkpoint_path)
        torch.cuda.synchronize()
        t1 = time.time()
        write0(f" --- Checkpoint saved to {checkpoint_path} in {t1-t0:.4f}s\n", log_file=log_file)


    train_sampler.set_epoch(epoch) # ensures different shuffle each epoch

    torch.cuda.synchronize()
    t0 = time.time()
    # Get next batch, restart dataloader if epoch completed
    try:
        z, y = next(train_iter)
    except StopIteration:
        epoch += 1
        train_sampler.set_epoch(epoch) # reshuffles dataloader at start of new epoch
        train_iter = iter(train_loader)
        z, y = next(train_iter)

    B, C, H, W = z.shape
    #assert B == batch_size, f"{B=}, {batch_size=}"

    z = z.to(device) # (B, 3, 32, 32)
    y = y.to(device) # (B,)

    # With probability eta, change class label to the null class label
    mask = torch.rand(B, device=device) < eta
    y[mask] = null_label

    t = torch.rand(B, device=device) # shape (B,), will need to view as (B, 1, 1, 1) later for broadcasting, consists of B iid Unif([0,1]) samples
    epsilon = torch.randn(B, C, H, W, device=device) # shape (B, 3, 32, 32) consisting of B iid N(0,1) samples
    
    for micro_step in range(grad_accum_steps):

        if ddp: # only sync gradients on the last micro step
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

        # Using alpha_t = t and beta_t = 1-t
        x = t.view(B, 1, 1, 1) * z + (1-t.view(B, 1, 1, 1)) * epsilon # (B, C, H, W)
        target = z - epsilon

        # Effective batch size is per_gpu_batch_size * num_gpus * grad_accum_steps, when we call loss.backward() the local gradients
        # are averaged over all ranks, the gradient on the existing rank has per_gpu_batch_size * num_gpus in the denominator and then
        # averaging all of these local averages over all ranks will give us the correct final denominator in our effective batch expectation
        loss = (model(x=x, t=t, y=y) - target).square().mean() / grad_accum_steps # .mean() averages over all dimensions, including batches! 
        train_loss = loss.detach().item() * grad_accum_steps # for logging, approximation of training loss w/o communicating across gpus
        loss.backward()

        ##### LOAD DATA FOR NEXT ITERATION -- would probably be cleaner to put the times and noise and null label stuff all inside a dataloader class, but this is just a quick and dirty script so I'm not worrying about it
        try:
            z, y = next(train_iter)
        except StopIteration:
            epoch += 1
            train_sampler.set_epoch(epoch) # reshuffles dataloader at start of new epoch
            train_iter = iter(train_loader)
            z, y = next(train_iter)
        z = z.to(device) # (B, 3, 32, 32)
        y = y.to(device) # (B,)
        B, C, H, W = z.shape # if not using drop_last=True in dataloader, this can be necessary to update B to avoid an error for the last batch being smaller

        # With probability eta, change class label to the null class label
        mask = torch.rand(B, device=device) < eta
        y[mask] = null_label

        t = torch.rand(B, device=device) # shape (B,), will need to view as (B, 1, 1, 1) later for broadcasting, consists of B iid Unif([0,1]) samples
        epsilon = torch.randn(B, C, H, W, device=device) # shape (B, 3, 32, 32) consisting of B iid N(0,1) samples
        
    for param_group in optimizer.param_groups:
        param_group['lr'] = get_lr(step)
    optimizer.step()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t1 = time.time()

    write0(f"Step {step}:{' '*(8 - len(str(step)))}{(t1-t0)*1000:.0f}ms    train loss: {train_loss:.6f}    epoch: {epoch}\n", log_file=log_file)
    

        # TODO logging, checkpointing, DDP, grad accum, eval (val loss, FID, sample image grids w/ fixed noise inputs? have to remember to undo the transformation to visualize properly!!! )

if ddp:
    dist.destroy_process_group()