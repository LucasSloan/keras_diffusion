import os

import tensorflow as tf

from gaussian_diffusion import GaussianDiffusion, cosine_beta_schedule


class CifarDataset(GaussianDiffusion):
    def __init__(self, batch_size, timesteps = 1000, **kwargs):
        betas = cosine_beta_schedule(timesteps)
        super().__init__(image_size = 32, betas = betas, **kwargs)

        self.batch_size = batch_size
        self.num_classes = 10

    def parse_image(self, filename):
        image_string = tf.io.read_file(filename)
        image_decoded = tf.image.decode_png(image_string, channels=3)
        image_flipped = tf.image.random_flip_left_right(image_decoded)

        image_normalized = 2.0 * \
            tf.image.convert_image_dtype(image_flipped, tf.float32) - 1.0
        image_nchw = tf.transpose(image_normalized, [2, 0, 1])
        image_nchw.set_shape((3, self.image_size, self.image_size))

        filename = tf.reshape(filename, [1])
        path_parts = tf.strings.split(filename, os.sep)
        dir = path_parts.values[-2]
        int_label = tf.strings.to_number(dir, out_type=tf.int32)

        return {'image': image_nchw, 'class': int_label}

    def gen_samples(self, data):
        noise = tf.random.normal([self.batch_size, 3, self.image_size, self.image_size])
        t = tf.random.uniform([self.batch_size], maxval=self.num_timesteps, dtype=tf.int32)

        noisy = self.q_sample(x_start=data['image'], t=t, noise = noise)

        return {'noisy': noisy, 'class': data['class'], 'timestep': t}, noise


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

    @tf.function()
    def parse_image(self, tfrecord):
        proto = tf.io.parse_single_example(tfrecord, image_feature_description)
        image_decoded = tf.image.decode_jpeg(proto['image_raw'], channels=3)
        image_flipped = tf.image.random_flip_left_right(image_decoded)

        image_normalized = 2.0 * \
            tf.image.convert_image_dtype(image_flipped, tf.float32) - 1.0
        image_nchw = tf.transpose(image_normalized, [2, 0, 1])
        image_nchw.set_shape((3, self.image_size, self.image_size))

        raw_label = proto['label'] - 1

        return {'image': image_nchw, 'class': raw_label}

    @tf.function()
    def gen_samples(self, data):
        noise = tf.random.normal([self.batch_size, 3, self.image_size, self.image_size])
        t = tf.random.uniform([self.batch_size], maxval=self.num_timesteps, dtype=tf.int32)

        noisy = self.q_sample(x_start=data['image'], t=t, noise = noise)

        return {'noisy': noisy, 'class': data['class'], 'timestep': t}, noise

    def load(self):
        files = tf.data.Dataset.list_files("/mnt/Bulk Storage/imagenet/tfrecords/64x64/*")
        dataset = files.interleave(lambda f: tf.data.TFRecordDataset(f), cycle_length=tf.data.AUTOTUNE, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(self.parse_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.map(self.gen_samples, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
