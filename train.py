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

"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

batch_size = 128
eta = 1/11
null_label = 10
# TODO define/load config, define model
device = "cuda" if torch.cuda.is_available() else "cpu" ### TODO add DDP


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # maps [0,1] -> [-1,1] since noise is N(0, I_d)
])

# Set random seed for when we shuffle our data
generator = torch.Generator()
generator.manual_seed(42) 

dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, generator=generator)


for z, y in train_loader:
    # z has shape (B, 3, 32, 32)
    # y has shape (B,)
    B, C, H, W = z.shape
    assert B == batch_size, f"{B=}, {batch_size=}"

    z = z.to(device)
    y = y.to(device)

    # With probability eta, change class label to the null class label
    mask = torch.rand(B, device=device) < eta
    y[mask] = null_label

    t = torch.rand(batch_size, 1, 1, 1, device=device) # shape (B, 1, 1, 1) for broadcasting, consists of B iid Unif([0,1]) samples
    epsilon = torch.randn(B, C, H, W, device=device) # shape (B, 3, 32, 32) consisting of B iid N(0,1) samples

    # Using alpha_t = t and beta_t = 1-t
    x = t * z + (1-t) * epsilon # (B, C, H, W)
    target = z - epsilon
    loss = (model(x=x, t=t, y=y) - target).square().mean()
    loss.backward()

    # TODO define optimizer to step
    # TODO logging, checkpointing, DDP, grad accum, eval (val loss, FID, sample image grids w/ fixed noise inputs? have to remember to undo the transformation to visualize properly!!! )