import time

import tensorflow as tf

from dataset import CifarDataset
from unet import Unet

BATCH_SIZE = 16

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

# policy = tf.keras.mixed_precision.Policy('mixed_float16')
# tf.keras.mixed_precision.set_global_policy(policy)

flags.DEFINE_string("checkpoint_dir", None, "Directory to load model state from to resume training.")
flags.DEFINE_string("experiment_name", None, "Name of the experiment being run.")

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

checkpoint_path = checkpoint_dir + "/cp-{epoch:04d}.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                save_weights_only=True,
                                                verbose=1)
# tb_callback = tf.keras.callbacks.TensorBoard(checkpoint_dir, update_freq=1)
buar_callback = tf.keras.callbacks.experimental.BackupAndRestore(checkpoint_dir)

model.fit(dataset, epochs=100, callbacks=[cp_callback, buar_callback])