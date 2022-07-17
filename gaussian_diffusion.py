import math

import tensorflow as tf
import tensorflow.keras.layers as l

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = tf.gather(a, t, axis=-1)
    return tf.reshape(out, shape=(b, *((1,) * (len(x_shape) - 1))))

def normal_kl(mean1, logvar1, mean2, logvar2):
    return 0.5 * (
        -1.0 
        + logvar2 
        - logvar1 
        + tf.math.exp(logvar1 - logvar2) 
        + ((mean1 - mean2) ** 2) * tf.math.exp(-logvar2)
    )

def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + tf.math.tanh(tf.math.sqrt(2.0 / math.pi) * (x + 0.044715 * tf.math.pow(x, 3))))

def discretized_guassian_log_likelihood(x, means, log_scales):
    centered_x = x - means
    inv_stdv = tf.math.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = tf.math.log(tf.clip_by_value(cdf_plus, 1e-12, cdf_plus.dtype.max))
    log_one_minus_cdf_min = tf.math.log(tf.clip_by_value((1.0 - cdf_min), 1e-12, cdf_min.dtype.max))
    cdf_delta = cdf_plus - cdf_min
    log_cdf_delta = tf.math.log(tf.clip_by_value(cdf_delta, 1e-12, cdf_delta.dtype.max))

    log_probs = tf.where(x < 0.999,
        log_cdf_plus,
        tf.where(x > 0.999, 
            log_one_minus_cdf_min,
            log_cdf_delta
            )
    )

    return log_probs

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
        image_size,
        betas,
        objective = 'pred_noise',
        model_var_type = 'fixed',
        channels = 3,
        timesteps = 1000,
        ):
        super().__init__()
        self.objective = objective
        self.model_var_type = model_var_type
        self.channels = channels
        self.image_size = image_size

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

    def interpolate(self, x1, x2, t = None, lam = 0.5):
        pass

    def q_sample(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_mean_variance(self, model_output, x, t, clip_denoised: bool):
        if self.model_var_type == 'learned_range':
            model_output, model_var_values = tf.split(model_output, 2, axis = 1)

            min_log = extract(self.posterior_log_variance_clipped, t, x.shape)
            max_log = extract(tf.math.log(self.betas), t, x.shape)

            frac = (model_var_values + 1) / 2

            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = tf.math.exp(model_log_variance)
            return model_output, model_variance, model_log_variance

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

    def vb_terms_bpd(self, model_output, x_start, x_t, t, clip_denoised = True):
        true_mean, _, true_log_variance_clipped = self.q_posterior(x_start, x_t, t)
        model_mean, _, model_log_variance_clipped = self.p_mean_variance(model_output, x_t, t, clip_denoised)
        kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance_clipped)
        kl = tf.math.reduce_mean(kl, axis=(1, 2, 3)) / math.log(2.0)

        decoder_nll = -discretized_guassian_log_likelihood(
            x_start, means = model_mean, log_scales = 0.5 * model_log_variance_clipped
        )
        decoder_nll = tf.math.reduce_mean(decoder_nll, axis=(1, 2, 3)) / math.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = tf.where((t == 0), decoder_nll, kl)

        return output
