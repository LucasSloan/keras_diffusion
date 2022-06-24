import math

import tensorflow as tf
import tensorflow_addons as tfa

from dataset import CifarDataset
from sampling_callback import SamplingCallback
from unet import Unet

tf.get_logger().setLevel('ERROR')

BATCH_SIZE = 32
flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint_dir", None, "Directory to load model state from to resume training.")
flags.DEFINE_bool("use_mixed_precision", False, "Whether to use float16 mixed precision training.")
flags.DEFINE_integer("image_size", 32, "The size of the image")
flags.DEFINE_integer("num_classes", 10, "The number of classes")
flags.DEFINE_integer("original_timesteps", 1000, "The number of timesteps the model was trained to use")
flags.DEFINE_integer("resampled_timesteps", None, "The number of timesteps to sample with (defaults to original_timesteps)")

if FLAGS.use_mixed_precision:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    noisy = tf.keras.Input(shape=(3, FLAGS.image_size, FLAGS.image_size), batch_size=BATCH_SIZE, name='noisy')
    cls = tf.keras.Input(shape=[], batch_size=BATCH_SIZE, name='class')
    timestep = tf.keras.Input(shape=[], batch_size=BATCH_SIZE, name='timestep')

    unet = Unet(dim=128, dropout=0.3, dim_mults=[1, 2, 2, 2], num_classes=FLAGS.num_classes)

    noise = unet(noisy, cls, timestep)

    model = tf.keras.Model(inputs=[noisy, cls, timestep], outputs=noise)

    if FLAGS.checkpoint_dir:
        checkpoint_dir = FLAGS.checkpoint_dir
        
        model.load_weights(checkpoint_dir + "/checkpoint.ckpt")

    sampling_callback = SamplingCallback(checkpoint_dir=checkpoint_dir, batch_size=BATCH_SIZE, run_every=100, image_size=FLAGS.image_size, timesteps=FLAGS.original_timesteps, resampled_timesteps=FLAGS.resampled_timesteps)
    sampling_callback.model = model

    c = tf.repeat(tf.range(0, FLAGS.num_classes, FLAGS.num_classes // 10), math.ceil(BATCH_SIZE / 10))[:BATCH_SIZE]
    imgs = sampling_callback.sample(c, BATCH_SIZE)

    for i, img in enumerate(imgs):
        tf.keras.utils.save_img(f'{FLAGS.checkpoint_dir}/samples/{i}.jpg', img, data_format='channels_first')
