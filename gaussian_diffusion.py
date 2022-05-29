import math

import tqdm
import tensorflow as tf
import tensorflow.keras.layers as l

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = tf.gather(a, t, axis=-1)
    return tf.reshape(out, shape=(b, *((1,) * (len(x_shape) - 1))))

def noise_like(shape, repeat=False):
    if repeat:
        return tf.tile(tf.random.normal(shape=(1, *shape[1:])), (shape[0], *(1,) * (len(shape) - 1)))
    else:
        return tf.random.normal(shape=shape)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = tf.linspace(tf.cast(0, tf.float64), tf.cast(timesteps, tf.float64), steps)
    alphas_cumprod = tf.math.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return tf.clip_by_value(betas, 0, 0.999)

class GaussianDiffusion():
    def __init__(
        self,
        *,
        image_size,
        channels = 3,
        timesteps = 1000,
        ):
        self.channels = channels
        self.image_size = image_size

        betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = tf.math.cumprod(alphas, axis=0)
        alphas_cumprod_prev = tf.concat([[1.], alphas_cumprod[:-1]], axis=0)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # helper function to convert to float32 and make non-Trainable
        make_buffer = lambda val: tf.Variable(initial_value=tf.cast(val, tf.float32), dtype=tf.float32, trainable=False)

        self.betas = make_buffer(betas)
        self.alphas_cumprod = make_buffer(alphas_cumprod)
        self.alphas_cumprod_prev = make_buffer(alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.sqrt_alphas_cumprod = make_buffer(tf.math.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = make_buffer(tf.math.sqrt(1. - alphas_cumprod))
        self.log_one_minus_alphas_cumprod = make_buffer(tf.math.log(1. - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = make_buffer(tf.math.sqrt(1. / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = make_buffer(tf.math.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_t1) + alpha_t / beta_t)

        self.posterior_variance = make_buffer(posterior_variance)

        # below log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        self.posterior_log_variance_clipped = make_buffer(tf.math.log(tf.clip_by_value(posterior_variance, clip_value_min=1e-20, clip_value_max=tf.float64.max)))
        self.posterior_mean_coef1 = make_buffer(betas * tf.math.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.posterior_mean_coef2 = make_buffer((1. - alphas_cumprod_prev) * tf.math.sqrt(alphas) / (1. - alphas_cumprod))

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_output = self.denoise_fn(x, t)

        if self.objective == 'pred_noise':
            x_start = self.predict_start_from_noise(x, t = t, noise = model_output)
        elif self.objective == 'pred_x0':
            x_start = model_output
        else:
            raise ValueError(f'unknown objective {self.objective}')

        if clip_denoised:
            x_start = tf.clip_by_value(x_start, -1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_ = x.shape
        model_mean, _, model_log_variance = self.p_mean_variance(x = x, t = t, clip_denoised = clip_denoised)
        noise = noise_like(x.shape, repeat_noise)
        # no noise when t == 0
        nonzero_mask = tf.reshape(1 - (t == 0).float(), (b, *((1,) * (len(x.shape) - 1))))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_loop(self, shape):
        b = shape[0]
        img = tf.random.normal(shape)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, tf.fill((b,), i))
            
        img = unnormalize_to_zero_to_one(img)
        return img

    def sample(self, batch_size = 16):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size))

    def interpolate(self, x1, x2, t = None, lam = 0.5):
        pass

    def q_sample(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
