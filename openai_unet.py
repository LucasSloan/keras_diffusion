from abc import abstractmethod

import math

import tensorflow as tf
import tensorflow.keras.layers as l
import tensorflow_addons as tfa

from einops import rearrange


def exists(x):
    return x is not None

class Identity(l.Layer):
    def call(self, x):
        return x

class TimestepBlock(l.Layer):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def call(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(tf.keras.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def call(self, x, emb):
        for layer in self.layers:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class Upsample(l.Layer):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """
    def __init__(self, channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.upsample = l.UpSampling2D(size=(2, 2), interpolation='nearest', data_format = 'channels_last')
        self.use_conv = use_conv
        if use_conv:
            self.conv = l.Conv2D(out_channels, 3, padding = 'same', data_format = 'channels_last')

    def call(self, x):
        x = self.upsample(x)
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(l.Layer):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    """
    def __init__(self, channels, use_conv, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        stride = 2

        if use_conv:
            self.op = l.Conv2D(out_channels, 3, strides = stride, padding = 'same', data_format = 'channels_last')
        else:
            assert self.channels == self.out_channels
            self.op = l.AveragePooling2D(pool_size = stride, strides = stride, data_format = 'channels_last')

    def call(self, x):
        return self.op(x)


class SinusoidalPosEmb(l.Layer):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def call(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = tf.math.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
        x = tf.cast(x, dtype=tf.float32)
        emb = x[:, None] * emb[None, :]
        emb = tf.concat([tf.math.sin(emb), tf.math.cos(emb)], axis=-1)
        return emb

class ResnetBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels = None,
        use_conv = False,
        up = False,
        down = False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv

        self.norm_in = tfa.layers.GroupNormalization(8, axis=-1)
        self.act_in = l.Activation('swish')
        self.conv_in = l.Conv2D(self.out_channels, 3, padding = 'same', data_format = 'channels_last')

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False)
            self.x_upd = Upsample(channels, False)
        elif down:
            self.h_upd = Downsample(channels, False)
            self.x_upd = Downsample(channels, False)

        self.emb_layers = tf.keras.Sequential([
                l.Activation('swish'),
                l.Dense(self.out_channels * 2),
            ])

        self.norm_out = tfa.layers.GroupNormalization(8, axis=-1)
        self.act_out = l.Activation('swish')
        self.drop_out = l.Dropout(dropout)
        # initialize to all zeroes, so at initializations, this resblock acts just as an identity, "shrinking" the depth of the network
        self.conv_out = l.Conv2D(self.out_channels, 3, padding = 'same', data_format = 'channels_last', kernel_initializer='zeros', bias_initializer='zeros')

        if self.out_channels == channels:
            self.skip_connection = Identity()
        elif use_conv:
            self.skip_connection = l.Conv2D(self.out_channels, 3, padding = 'same', data_format = 'channels_last')
        else:
            self.skip_connection = l.Conv2D(self.out_channels, 1, padding = 'valid', data_format = 'channels_last')

    def call(self, x, emb):
        if self.updown:
            h = self.norm_in(x)
            h = self.act_in(h)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = self.conv_in(h)
        else:
            h = self.norm_in(x)
            h = self.act_in(h)
            h = self.conv_in(h)

        emb_out = self.emb_layers(emb)
        emb_out = rearrange(emb_out, 'b c -> b 1 1 c')
        scale, shift = tf.split(emb_out, 2, axis = -1)
        h = self.norm_out(h) * (1 + scale) + shift

        h = self.act_out(h)
        h = self.drop_out(h)
        h = self.conv_out(h)

        return self.skip_connection(x) + h

class AttentionBlock(l.Layer):
    def __init__(
        self,
        channels,
        num_heads = 1,
        num_head_channels = -1,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0
            self.num_heads = channels // num_head_channels

        self.norm = tfa.layers.GroupNormalization(8, axis=-1)
        self.qkv = l.Conv1D(channels * 3, 1, data_format = 'channels_last')
        self.attention = QKVAttention(self.num_heads)

        # initialize to all zeroes, so at initializations, this resblock acts just as an identity, "shrinking" the depth of the network
        self.proj_out = l.Conv1D(channels, 1, data_format = 'channels_last', kernel_initializer='zeros', bias_initializer='zeros')

    def build(self, input_shape):
        self.h = input_shape[1]
        self.w = input_shape[2]

    def call(self, x):
        x = rearrange(x, 'b h w c -> b (h w) c')
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return rearrange((x + h), 'b (h w) c -> b h w c', h = self.h, w = self.w)

class QKVAttention(l.Layer):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def build(self, input_shape):
        self.bs = input_shape[0]
        self.length = input_shape[1]
        self.width = input_shape[2]

    def call(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x T x (3 * H * C)] tensor of Qs, Ks, and Vs.
        :return: an [N x T x (H * C)] tensor after attention.
        """
        ch = self.width // (3 * self.n_heads)
        q, k, v = tf.split(qkv, 3, axis = -1)
        scale = ch ** -0.25
        weight = tf.einsum(
            'btc, bsc -> bts',
            rearrange((q * scale), 'b l (h c) -> (b h) l c', b = self.bs, l = self.length, h = self.n_heads),
            rearrange((k * scale), 'b l (h c) -> (b h) l c', b = self.bs, l = self.length, h = self.n_heads),
        ) # More stable with fp16 than dividing afterwards
        weight = tf.nn.softmax(weight, axis = 1)
        a = tf.einsum(
            'bts, bsc -> btc', 
            weight,
            rearrange(v, 'b l (h c) -> (b h) l c', b = self.bs, l = self.length, h = self.n_heads)
        )
        return rearrange(a, '(b h) l c -> b l (h c)', b = self.bs, l = self.length, h = self.n_heads)

class Unet(l.Layer):
    def __init__(
        self,
        dim,
        dim_mults=(1, 2, 4, 8),
        num_res_blocks=3,
        attention_resolutions=(8, 16),
        channels = 3,
        learned_variance = False,
        dropout = 0.,
        num_classes = None,
        num_heads = 1,
        resblock_updown = False,
        conv_resample = True,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        
        self.init_conv = l.Conv2D(dim, 3, padding = 'same', data_format = 'channels_last')

        # time embeddings

        time_dim = dim * 4
        self.time_mlp = tf.keras.Sequential([
            SinusoidalPosEmb(dim),
            l.Dense(time_dim),
            l.Activation('swish'),
            l.Dense(time_dim)
        ])

        if num_classes:
            self.class_emb = l.Embedding(num_classes, time_dim)
        else:
            self.class_emb = None

        # layers

        self.downs = []
        self.ups = []

        input_block_chans = [dim]
        ch = dim
        ds = 1
        for level, mult in enumerate(dim_mults):
            for _ in range(num_res_blocks):
                layers = [
                    ResnetBlock(
                        ch, 
                        time_dim, 
                        dropout,
                        out_channels = mult * dim,
                    )
                ]
                ch = mult * dim
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, num_heads=num_heads))
                self.downs.append(TimestepEmbedSequential(layers))
                input_block_chans.append(ch)
            if level != len(dim_mults) - 1:
                self.downs.append(
                    TimestepEmbedSequential(
                        ResnetBlock(
                            ch, 
                            time_dim, 
                            dropout,
                            out_channels = ch,
                            down = True,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, out_channels=ch)
                    )
                )
                input_block_chans.append(ch)
                ds *= 2

        
        self.mid_block1 = ResnetBlock(
            ch,
            time_dim,
            dropout,
        )
        self.mid_attn = AttentionBlock(ch, num_heads=num_heads)
        self.mid_block2 = ResnetBlock(
            ch,
            time_dim,
            dropout,
        )

        for level, mult in list(enumerate(dim_mults))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResnetBlock(
                        ch + input_block_chans.pop(),
                        time_dim,
                        dropout,
                        out_channels = mult * dim,
                    )
                ]
                ch = dim * mult
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(ch, num_heads=num_heads)
                    )
                if level and i == num_res_blocks:
                    layers.append(
                        ResnetBlock(
                            ch,
                            time_dim,
                            dropout,
                            out_channels = ch,
                            up = True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, out_channels = ch)
                    )
                    ds //= 2
                self.ups.append(TimestepEmbedSequential(layers))

        self.out_dim = channels * (1 if not learned_variance else 2)

        self.final_conv = tf.keras.Sequential([
            tfa.layers.GroupNormalization(8, axis=-1),
            l.Activation('swish'),
            l.Conv2D(self.out_dim, 1, data_format = 'channels_last'),
        ])

    def call(self, x, c, time):
        t = self.time_mlp(time)

        t = t + self.class_emb(c)

        x = tf.transpose(x, [0, 2, 3, 1])
        x = self.init_conv(x)
        hs = [x]

        for block in self.downs:
            x = block(x, t)
            hs.append(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block in self.ups:
            h = hs.pop()
            x = tf.concat([x, h], axis = -1)
            x = block(x, t)

        x = self.final_conv(x)
        x = tf.transpose(x, [0, 3, 1, 2])
        return x

