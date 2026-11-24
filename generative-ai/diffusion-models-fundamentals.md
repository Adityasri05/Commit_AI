# Diffusion Models: Forward Noise and Reverse Denoising

Diffusion models are generative models that learn to create samples by reversing a gradual corruption process. They are widely used for image, audio, and other structured generation tasks.

The central intuition is simple: adding noise repeatedly eventually destroys recognizable structure. A neural network learns how to reverse that process, starting from noise and gradually producing a sample.

## Forward process

Let x_0 be a clean training example. A forward process adds Gaussian noise over T steps according to a variance schedule beta_t:

\[
q(x_t \mid x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I)
\]

Define alpha_t = 1 - beta_t and alpha_bar_t as the product of alpha values up to t. A useful property lets us sample a noisy version directly:

\[
x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon,\quad \epsilon \sim \mathcal{N}(0,I)
\]

At small t, x_t retains much of the original signal. At large t, it approaches Gaussian noise.

## Learning the reverse process

A denoising network receives a noisy sample and the timestep, then predicts the noise that was added. A common simplified training objective is:

\[
L = \mathbb{E}_{x_0,t,\epsilon}\left[\|\epsilon - \epsilon_\theta(x_t,t)\|_2^2\right]
\]

The network is often a U-Net for image generation, with timestep embeddings and sometimes conditioning information such as text embeddings.

~~~python
import torch

def make_noisy_sample(x0, alpha_bar, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    # alpha_bar is selected for each example's sampled timestep.
    a = alpha_bar.view(-1, 1, 1, 1)
    xt = a.sqrt() * x0 + (1.0 - a).sqrt() * noise
    return xt, noise

# During training:
# t = torch.randint(0, num_steps, (x0.size(0),), device=x0.device)
# xt, target_noise = make_noisy_sample(x0, alpha_bar[t])
# predicted_noise = denoiser(xt, t, conditioning)
# loss = torch.nn.functional.mse_loss(predicted_noise, target_noise)
~~~

The example assumes image tensors shaped [batch, channels, height, width] and one alpha_bar value per batch item.

## Sampling

At inference, start from random Gaussian noise and apply a learned reverse update for a sequence of timesteps. The exact update depends on the parameterization and sampler. DDPM uses a stochastic reverse process; DDIM and other samplers can reduce the number of steps or change the sampling trajectory.

Fewer steps can improve speed, but aggressive reduction may lower fidelity or diversity. Sampler choice should be evaluated for the model and task rather than assumed to be interchangeable.

## Conditioning and classifier-free guidance

A conditional model can receive text, class labels, or another input. Classifier-free guidance combines conditional and unconditional predictions:

\[
\hat{\epsilon} = \epsilon_\theta(x_t,t,\varnothing) + s\left(\epsilon_\theta(x_t,t,c)-\epsilon_\theta(x_t,t,\varnothing)\right)
\]

Here c is the condition and s is the guidance scale. Higher guidance can improve condition alignment but may reduce diversity or create artifacts. It is a trade-off, not a universal quality knob.

## Design choices

- **Noise schedule:** Controls how quickly signal is destroyed.
- **Prediction target:** Models may predict noise, the clean sample, or a velocity parameterization.
- **Network:** U-Nets are common; transformer-based denoisers are also used.
- **Conditioning:** Text or class information must be encoded and injected consistently.
- **Sampler:** Determines the inference update sequence and number of network evaluations.

## Common pitfalls

- Mixing timestep conventions or alpha-bar indexing between training and sampling.
- Forgetting to normalize image values consistently.
- Using a guidance scale without checking prompt adherence and diversity.
- Assuming a low training loss guarantees good samples.
- Evaluating only a few cherry-picked generations.
- Ignoring licensing, training-data provenance, and safety risks for generated content.

## Mini-project

Train a tiny diffusion model on a small grayscale image dataset. Visualize samples at several forward timesteps, train a noise predictor, and generate a grid using two sampler settings. Track sample quality, diversity, inference steps, and runtime.

## References

- Ho, Jain, and Abbeel, *Denoising Diffusion Probabilistic Models* (2020): https://arxiv.org/abs/2006.11239
- Hugging Face Diffusers documentation: https://huggingface.co/docs/diffusers/
