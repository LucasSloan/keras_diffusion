import time

import tensorflow as tf
import tensorflow_addons as tfa

from dataset import CifarDataset
from sampling_callback import SamplingCallback
from unet import Unet

BATCH_SIZE = 128
EPOCHS = 1000
LEARNING_RATE = 1e-4

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("checkpoint_dir", None, "Directory to load model state from to resume training.")
flags.DEFINE_string("experiment_name", None, "Name of the experiment being run.")
flags.DEFINE_bool("use_mixed_precision", False, "Whether to use float16 mixed precision training.")

if FLAGS.use_mixed_precision:
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    dataset = CifarDataset(BATCH_SIZE)

    noisy = tf.keras.Input(shape=(3, dataset.image_size, dataset.image_size), batch_size=BATCH_SIZE, name='noisy')
    cls = tf.keras.Input(shape=[], batch_size=BATCH_SIZE, name='class')
    timestep = tf.keras.Input(shape=[], batch_size=BATCH_SIZE, name='timestep')

    unet = Unet(dim=128, dropout=0.3, dim_mults=[1, 2, 2, 2], num_classes=dataset.num_classes)

    noise = unet(noisy, cls, timestep)

    model = tf.keras.Model(inputs=[noisy, cls, timestep], outputs=noise)

    adam = tf.keras.optimizers.Adam(LEARNING_RATE)
    optimizer = tfa.optimizers.MovingAverage(adam, average_decay=0.9999)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(LEARNING_RATE, EPOCHS)
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    model.compile(optimizer=optimizer, loss='mae')

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
    cp_callback = tfa.callbacks.AverageModelCheckpoint(filepath=checkpoint_path,
                                                    update_weights=False,
                                                    save_weights_only=True,
                                                    verbose=1)
    # tb_callback = tf.keras.callbacks.TensorBoard(checkpoint_dir, update_freq=1)
    buar_callback = tf.keras.callbacks.experimental.BackupAndRestore(checkpoint_dir)
    sampling_callback = SamplingCallback(checkpoint_dir=checkpoint_dir, batch_size=BATCH_SIZE, run_every=100, image_size=dataset.image_size, num_classes=dataset.num_classes)

    model.fit(dataset.load(), epochs=EPOCHS, callbacks=[cp_callback, buar_callback, sampling_callback, lr_callback])
