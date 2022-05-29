
import os
import tensorflow as tf

from gaussian_diffusion import GaussianDiffusion


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def noise_like(shape, repeat=False):
    if repeat:
        return tf.tile(tf.random.normal(shape=(1, *shape[1:])), (shape[0], *(1,) * (len(shape) - 1)))
    else:
        return tf.random.normal(shape=shape)

class SamplingCallback(GaussianDiffusion, tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir, **kwargs):
        super().__init__(**kwargs)

        self.checkpoint_dir = checkpoint_dir

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_output = self.model((x, t))

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

    @tf.function
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_ = x.shape
        model_mean, _, model_log_variance = self.p_mean_variance(x = x, t = t, clip_denoised = clip_denoised)
        noise = noise_like(x.shape, repeat_noise)
        # no noise when t == 0
        nonzero_mask = tf.reshape(1 - tf.cast((t == 0), tf.float32), (b, *((1,) * (len(x.shape) - 1))))
        return model_mean + nonzero_mask * tf.math.exp((0.5 * model_log_variance)) * noise

    def p_sample_loop(self, shape):
        b = shape[0]
        img = tf.random.normal(shape)

        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample(img, tf.fill((b,), i))
            
        img = unnormalize_to_zero_to_one(img)
        return img

    def sample(self, batch_size = 16):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size))

    def on_epoch_end(self, epoch, logs):
        imgs = self.sample()
        os.makedirs(f'{self.checkpoint_dir}/samples/epoch_{epoch}')

        for i, img in enumerate(imgs):
            tf.keras.utils.save_img(f'{self.checkpoint_dir}/samples/epoch_{epoch}/{i}.jpg', img, data_format='channels_first')

