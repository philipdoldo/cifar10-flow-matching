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
of a dog rather than just any arbitrary picture? We'll use classifier-free guidance (CFG) TODO TODO TODO
TODO CFG!!!!!!!!

FLOW MATCHING TRAINING RECIPE (Gaussian probability paths with linear noise schedules): TODO CFG

"""