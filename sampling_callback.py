
import os
import math
import tensorflow as tf
import tensorflow_addons as tfa

from gaussian_diffusion import GaussianDiffusion, cosine_beta_schedule, extract


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def noise_like(shape, repeat=False):
    if repeat:
        return tf.tile(tf.random.normal(shape=(1, *shape[1:])), (shape[0], *(1,) * (len(shape) - 1)))
    else:
        return tf.random.normal(shape=shape)

class SamplingCallback(GaussianDiffusion, tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir, batch_size, sample_classes, run_every = 1, timesteps = 1000, resampled_timesteps = None, **kwargs):
        betas = cosine_beta_schedule(timesteps)
        if resampled_timesteps:
            assert timesteps % resampled_timesteps == 0

            use_timesteps = set(range(0, timesteps, timesteps // resampled_timesteps))
            base_diffusion = GaussianDiffusion(timesteps = timesteps, betas = betas, **kwargs)
            last_alpha_cumprod = 1.0
            new_betas = []
            self.timestamp_map = []
            for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
                if i in use_timesteps:
                    new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                    last_alpha_cumprod = alpha_cumprod
                    self.timestamp_map.append(i)
            betas = tf.stack(new_betas)
        else:
            self.timestamp_map = list(range(timesteps))

        self.timestamp_map = tf.constant(self.timestamp_map)

        super().__init__(timesteps = timesteps, betas = betas, **kwargs)

        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size
        self.run_every = run_every
        self.sample_classes = sample_classes

    @tf.function
    def p_sample(self, x, c, t, clip_denoised=True, repeat_noise=False):
        b, *_ = x.shape
        model_output = self.model({'noisy': x, 'class': c, 'timestep': tf.gather(self.timestamp_map, t)})
        model_output = tf.cast(model_output, dtype=x.dtype)

        model_mean, _, model_log_variance = self.p_mean_variance(model_output, x = x, t = t, clip_denoised = clip_denoised)
        noise = noise_like(x.shape, repeat_noise)
        # no noise when t == 0
        nonzero_mask = tf.reshape(1 - tf.cast((t == 0), tf.float32), (b, *((1,) * (len(x.shape) - 1))))
        return model_mean + nonzero_mask * tf.math.exp((0.5 * model_log_variance)) * noise

    def p_sample_loop(self, c, shape):
        b = shape[0]
        img = tf.random.normal(shape)

        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample(img, c, tf.fill((b,), i))
            
        img = unnormalize_to_zero_to_one(img)
        return img

    def sample(self, c, batch_size = 16):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop(c, (batch_size, image_size, image_size, channels))

    def _get_optimizer(self):
        optimizer = self.model.optimizer
        if type(optimizer).__name__ in ["LossScaleOptimizer", "LossScaleOptimizerV1"]:
            optimizer = optimizer.inner_optimizer

        return optimizer

    def on_epoch_end(self, epoch, logs):
        epoch_one_indexed = epoch + 1
        if (epoch_one_indexed) % self.run_every != 0:
            return

        optimizer = self._get_optimizer()
        assert isinstance(optimizer, tfa.optimizers.MovingAverage), type(optimizer)

        optimizer.swap_weights()
        imgs = self.sample(self.sample_classes, self.batch_size)
        optimizer.swap_weights()

        os.makedirs(f'{self.checkpoint_dir}/samples/epoch_{epoch_one_indexed}')

        for i, img in enumerate(imgs):
            tf.keras.utils.save_img(f'{self.checkpoint_dir}/samples/epoch_{epoch_one_indexed}/{i}.png', img, data_format='channels_last')


