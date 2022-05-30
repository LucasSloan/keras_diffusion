import time

import tensorflow as tf

from dataset import CifarDataset
from sampling_callback import SamplingCallback
from unet import Unet

BATCH_SIZE = 64

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint_dir", None, "Directory to load model state from to resume training.")
flags.DEFINE_string("experiment_name", None, "Name of the experiment being run.")
flags.DEFINE_bool("use_mixed_precision", False, "Whether to use float16 mixed precision training.")

if FLAGS.use_mixed_precision:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

dataset = CifarDataset(BATCH_SIZE).load()

noisy = tf.keras.Input(shape=(3, 32, 32), batch_size=BATCH_SIZE, name='noisy')
timestep = tf.keras.Input(shape=[], batch_size=BATCH_SIZE, name='timestep')

unet = Unet(dim=64)

noise = unet(noisy, timestep)

model = tf.keras.Model(inputs=[noisy, timestep], outputs=noise)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='mae')

if FLAGS.checkpoint_dir:
    checkpoint_dir = FLAGS.checkpoint_dir
    # print('attempting to load checkpoint from {}'.format(checkpoint_dir))
    
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)
elif FLAGS.experiment_name:
    checkpoint_dir = 'checkpoints/{}'.format(FLAGS.experiment_name)
else:
    checkpoint_dir = 'checkpoints/{}'.format(time.strftime("%m_%d_%y-%H_%M"))

checkpoint_path = checkpoint_dir + "/checkpoint.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                save_weights_only=True,
                                                verbose=1)
# tb_callback = tf.keras.callbacks.TensorBoard(checkpoint_dir, update_freq=1)
buar_callback = tf.keras.callbacks.experimental.BackupAndRestore(checkpoint_dir)
sampling_callback = SamplingCallback(checkpoint_dir=checkpoint_dir, batch_size=BATCH_SIZE, run_every=5, image_size=32)

model.fit(dataset, epochs=500, callbacks=[cp_callback, buar_callback, sampling_callback])