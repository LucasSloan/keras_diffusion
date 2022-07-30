from gaussian_diffusion import GaussianDiffusion
import tensorflow as tf


class DiffusionModel(GaussianDiffusion, tf.keras.Model):

    def __init__(
        self,
        image_size,
        betas,
        unet,
        objective = 'pred_noise',
        model_var_type = 'fixed',
        channels = 3,
        timesteps = 1000,
    ):
        super().__init__(image_size, betas, objective = objective, model_var_type = model_var_type, channels = channels, timesteps = timesteps)
        self.unet = unet
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')
        self.mse_tracker = tf.keras.metrics.Mean(name='mse')
        self.vlb_tracker = tf.keras.metrics.Mean(name='vlb')


    def call(self, inputs, training = None, mask = None):
        return self.unet(inputs['noisy'], inputs['class'], inputs['timestep'])

    def compute_loss(self, x = None, y = None, y_pred = None, sample_weight = None):
        y_pred = tf.cast(y_pred, y.dtype)
        if self.model_var_type == 'learned_range':
            model_output, model_var_values = tf.split(y_pred, 2, axis = -1)
            # Learn the variance using the variational bound, but don't let
            # it affect our mean prediction.
            frozen_out = tf.concat((tf.stop_gradient(model_output), model_var_values), axis = -1)
            vlb = self.vb_terms_bpd(frozen_out, x['original'], x['noisy'], x['timestep'])
            # multiply by the overall number of timesteps to estimate the overall VLB,
            # then divide by 1000 to avoid overwhelming the MSE mean loss
            vlb *= self.num_timesteps / 1000
            self.vlb_tracker.update_state(vlb)
            mse = tf.reduce_mean(tf.math.squared_difference(model_output, y))
            self.mse_tracker.update_state(mse)
            loss = vlb + mse
        else:
            loss = tf.reduce_mean(tf.math.squared_difference(y_pred, y))
        self.loss_tracker.update_state(loss)
        return loss

    @property
    def metrics(self):
        return [self.loss_tracker, self.vlb_tracker, self.mse_tracker]