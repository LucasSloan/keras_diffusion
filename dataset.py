import math
import os

import tensorflow as tf
from cifar10_classes import CIFAR10_CLASSES
from imagenet_classes import IMAGENET_CLASSES

from gaussian_diffusion import GaussianDiffusion, cosine_beta_schedule


class CifarDataset(GaussianDiffusion):
    def __init__(self, batch_size, timesteps = 1000, **kwargs):
        betas = cosine_beta_schedule(timesteps)
        super().__init__(image_size = 32, betas = betas, **kwargs)

        self.batch_size = batch_size
        self.num_classes = 10
        self.class_names = CIFAR10_CLASSES

    def get_sample_classes(self):
        return tf.repeat(tf.range(0, self.num_classes, self.num_classes // 10), math.ceil(self.batch_size / 10))[:self.batch_size]

    def parse_image(self, filename):
        image_string = tf.io.read_file(filename)
        image_decoded = tf.image.decode_png(image_string, channels=3)
        image_flipped = tf.image.random_flip_left_right(image_decoded)

        image_normalized = 2.0 * \
            tf.image.convert_image_dtype(image_flipped, tf.float32) - 1.0
        image_normalized.set_shape((self.image_size, self.image_size, 3))

        filename = tf.reshape(filename, [1])
        path_parts = tf.strings.split(filename, os.sep)
        dir = path_parts.values[-2]
        int_label = tf.strings.to_number(dir, out_type=tf.int32)

        return {'image': image_normalized, 'class': int_label}

    def gen_samples(self, data):
        noise = tf.random.normal([self.batch_size, self.image_size, self.image_size, 3])
        t = tf.random.uniform([self.batch_size], maxval=self.num_timesteps, dtype=tf.int32)

        noisy = self.q_sample(x_start=data['image'], t=t, noise = noise)

        return {'original': data['image'], 'noisy': noisy, 'class': data['class'], 'timestep': t}, noise


    def load(self):
        image_files_dataset = tf.data.Dataset.list_files(
            "/mnt/Bulk Storage/cifar10/32x32/*/*")
        image_files_dataset.shuffle(100000, reshuffle_each_iteration=True)
        dataset = image_files_dataset.map(self.parse_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.map(self.gen_samples)
        dataset = dataset.prefetch(self.batch_size)
        return dataset


image_feature_description = {
    'image_raw': tf.io.FixedLenFeature([], tf.string),
    'label': tf.io.FixedLenFeature([], tf.int64),
}

class ImagenetDataset(GaussianDiffusion):
    def __init__(self, batch_size, timesteps = 1000, **kwargs):
        betas = cosine_beta_schedule(timesteps)
        super().__init__(image_size = 64, betas = betas, **kwargs)

        self.batch_size = batch_size
        self.num_classes = 1000
        self.class_names = IMAGENET_CLASSES

    def get_sample_classes(self):
        classes = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 281, 812, 786, 996, 386, 847]
        assert self.batch_size % len(classes) == 0
        return tf.repeat(classes, self.batch_size // len(classes))

    @tf.function()
    def parse_image(self, tfrecord):
        proto = tf.io.parse_single_example(tfrecord, image_feature_description)
        image_decoded = tf.image.decode_jpeg(proto['image_raw'], channels=3)
        image_flipped = tf.image.random_flip_left_right(image_decoded)

        image_normalized = 2.0 * \
            tf.image.convert_image_dtype(image_flipped, tf.float32) - 1.0
        image_normalized.set_shape((self.image_size, self.image_size, 3))

        raw_label = proto['label'] - 1

        return {'image': image_normalized, 'class': raw_label}

    @tf.function()
    def gen_samples(self, data):
        noise = tf.random.normal([self.batch_size, self.image_size, self.image_size, 3])
        t = tf.random.uniform([self.batch_size], maxval=self.num_timesteps, dtype=tf.int32)

        noisy = self.q_sample(x_start=data['image'], t=t, noise = noise)

        return {'original': data['image'], 'noisy': noisy, 'class': data['class'], 'timestep': t}, noise

    def load(self):
        files = tf.data.Dataset.list_files("/mnt/Bulk Storage/imagenet/tfrecords/64x64/*")
        dataset = files.interleave(lambda f: tf.data.TFRecordDataset(f), cycle_length=tf.data.AUTOTUNE, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self.parse_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.map(self.gen_samples, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
