
import os
import tensorflow as tf
import tensorflow_addons as tfa

from gaussian_diffusion import GaussianDiffusion


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

def noise_like(shape, repeat=False):
    if repeat:
        return tf.tile(tf.random.normal(shape=(1, *shape[1:])), (shape[0], *(1,) * (len(shape) - 1)))
    else:
        return tf.random.normal(shape=shape)

class SamplingCallback(GaussianDiffusion, tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir, batch_size, run_every = 1, **kwargs):
        super().__init__(**kwargs)

        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size
        self.run_every = run_every

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_output = self.model((x, t))
        model_output = tf.cast(model_output, dtype=x.dtype)

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
        imgs = self.sample(self.batch_size)
        optimizer.swap_weights()

        os.makedirs(f'{self.checkpoint_dir}/samples/epoch_{epoch_one_indexed}')

        for i, img in enumerate(imgs):
            tf.keras.utils.save_img(f'{self.checkpoint_dir}/samples/epoch_{epoch_one_indexed}/{i}.jpg', img, data_format='channels_first')


