from gaussian_diffusion import GaussianDiffusion
import tensorflow as tf


class DiffusionModel(GaussianDiffusion, tf.keras.Model):

    def __init__(
        self,
        image_size,
        betas,
        unet,
        objective = 'pred_noise',
        channels = 3,
        timesteps = 1000,
    ):
        super().__init__(image_size, betas, objective = objective, channels = channels, timesteps = timesteps)
        self.unet = unet

    def call(self, inputs, training = None, mask = None):
        return self.unet(inputs['noisy'], inputs['class'], inputs['timestep'])
