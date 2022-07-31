from abc import abstractmethod

import math

import tensorflow as tf
import tensorflow.keras.layers as l
import tensorflow_addons as tfa

from einops import rearrange


def exists(x):
    return x is not None

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

class Residual(l.Layer):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def call(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

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

def Upsample(dim, use_conv = True):
    if use_conv:
        return l.Conv2DTranspose(dim, 4, strides = 2, padding = 'same', data_format = 'channels_last')
    else:
        return l.UpSampling2D(size=(2, 2), interpolation='nearest', data_format = 'channels_last')

def Downsample(dim, use_conv = True):
    if use_conv:
        return l.Conv2D(dim, 4, strides = 2, padding = 'same', data_format = 'channels_last')
    else:
        return l.AveragePooling2D(pool_size = 2, strides = 2, data_format = 'channels_last')

class PreNorm(l.Layer):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = l.LayerNormalization(axis = -1)

    def call(self, x):
        x = self.norm(x)
        return self.fn(x)

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
        up = False,
        down = False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels

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
            self.skip_connection = None
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

        if self.skip_connection:
            res = self.skip_connection(x)
        else:
            res = x
        return res + h

class Attention(l.Layer):
    def __init__(self, dim, heads = 1, dim_head = -1):
        super().__init__()
        if dim_head == -1:
            self.heads = heads
        else:
            assert dim % dim_head == 0
            self.num_heads = dim // dim_head
        self.scale = dim ** -0.25
        self.to_qkv = l.Conv2D(dim * 3, 1, use_bias = False, data_format = 'channels_last')
        # initialize to all zeroes, so at initialization, this resblock acts just as an identity, "shrinking" the depth of the network
        self.to_out = l.Conv2D(dim, 1, data_format = 'channels_last', kernel_initializer='zeros')

    def build(self, input_shape):
        self.h = input_shape[1]
        self.w = input_shape[2]

    def call(self, x):
        qkv = tf.split(self.to_qkv(x), 3, axis = -1)
        q, k, v = map(lambda t: rearrange(t, 'b x y (h c) -> b h c (x y)', h = self.heads), qkv)
        q = q * self.scale
        k = k * self.scale

        sim = tf.einsum('bhdi,bhdj->bhij', q, k)
        attn = tf.nn.softmax(sim, axis = -1)

        out = tf.einsum('bhij,bhdj->bhid', attn, v)
        out = rearrange(out, 'b h (x y) d -> b x y (h d)', x = self.h, y = self.w)
        return self.to_out(out)

class Unet(l.Layer):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        num_res_blocks=3,
        attention_resolutions=(8, 16),
        channels = 3,
        with_time_emb = True,
        resnet_block_groups = 8,
        learned_variance = False,
        dropout = 0.,
        num_classes = None,
        resblock_updown = False,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        
        self.init_conv = l.Conv2D(dim, 3, padding = 'same', data_format = 'channels_last')

        self.resnet_block_groups = resnet_block_groups

        # time embeddings

        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = tf.keras.Sequential([
                SinusoidalPosEmb(dim),
                l.Dense(time_dim),
                l.Activation(tf.keras.activations.gelu),
                l.Dense(time_dim)
            ])
        else:
            time_dim = None
            self.time_mlp = None

        if num_classes:
            self.class_emb = l.Embedding(num_classes, dim * 4)
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
                    layers.append(Residual(PreNorm(ch, Attention(ch, heads = 4))))
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
                        else Downsample(ch)
                    )
                )
                input_block_chans.append(ch)
                ds *= 2

        
        self.mid_block1 = ResnetBlock(
            ch,
            time_dim,
            dropout,
        )
        self.mid_attn = Residual(PreNorm(ch, Attention(ch, heads = 4)))
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
                        Residual(PreNorm(ch, Attention(ch, heads = 4)))
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
                        else Upsample(ch)
                    )
                    ds //= 2
                self.ups.append(TimestepEmbedSequential(layers))

        if out_dim:
            self.out_dim = out_dim
        else:
            self.out_dim = channels * (1 if not learned_variance else 2)

        self.final_conv = tf.keras.Sequential([
            l.Conv2D(self.out_dim, 1, data_format = 'channels_last'),
        ])

    def call(self, x, c, time):
        t = self.time_mlp(time) if self.time_mlp else None

        if self.class_emb:
            t = t + self.class_emb(c)

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
        return x

