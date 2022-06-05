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
        emb = tf.cast(emb, dtype=x.dtype)
        emb = x[:, None] * emb[None, :]
        emb = tf.concat([tf.math.sin(emb), tf.math.cos(emb)], axis=-1)
        return emb

def Upsample(dim):
    return l.Conv2DTranspose(dim, 4, strides = 2, padding = 'same', data_format = 'channels_first')

def Downsample(dim):
    return l.Conv2D(dim, 4, strides = 2, padding = 'same', data_format = 'channels_first')

class PreNorm(l.Layer):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = l.LayerNormalization(axis = 1)

    def call(self, x):
        x = self.norm(x)
        return self.fn(x)

class Block(l.Layer):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = l.Conv2D(dim_out, 3, padding = 'same', data_format = 'channels_first')
        self.norm = tfa.layers.GroupNormalization(groups, axis=-3)
        self.act = tf.keras.activations.swish

    def call(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(TimestepBlock):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8, dropout = 0.):
        super().__init__()
        if time_emb_dim:
            self.mlp = tf.keras.Sequential([
                l.Activation(tf.keras.activations.swish),
                l.Dense(dim_out * 2),
            ])
        else:
            self.mlp = None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.dropout = l.Dropout(dropout)
        self.block2 = Block(dim_out, dim_out, groups = groups)

        if dim != dim_out:
            self.res_conv = l.Conv2D(dim_out, 1, data_format = 'channels_first')
        else:
            self.res_conv = None
        
    def call(self, x, time_emb = None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = tf.split(time_emb, 2, axis = 1)

        h = self.block1(x, scale_shift = scale_shift)
        h = self.dropout(h)
        h = self.block2(h)

        if self.res_conv:
            res = self.res_conv(x)
        else:
            res = x

        return h + res

class LinearAttention(l.Layer):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = l.Conv2D(hidden_dim * 3, 1, use_bias = False, data_format = 'channels_first')

        self.to_out = tf.keras.Sequential([
            l.Conv2D(dim, 1, data_format = 'channels_first'),
            l.LayerNormalization(axis = 1),
        ])

    def build(self, input_shape):
        self.h = input_shape[2]
        self.w = input_shape[3]

    def call(self, x):
        h = tf.shape(x)[-2]
        w = tf.shape(x)[-1]
        qkv = tf.split(self.to_qkv(x), 3, axis = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = tf.nn.softmax(q, axis = -2)
        k = tf.nn.softmax(k, axis = -1)

        q = q * self.scale
        context = tf.einsum('bhdn,bhen->bhde', k, v)
        out = tf.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = self.h, y = self.w)
        return self.to_out(out)
        
class Attention(l.Layer):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = l.Conv2D(hidden_dim * 3, 1, use_bias = False, data_format = 'channels_first')
        self.to_out = l.Conv2D(dim, 1, data_format = 'channels_first')

    def build(self, input_shape):
        self.h = input_shape[2]
        self.w = input_shape[3]

    def call(self, x):
        qkv = tf.split(self.to_qkv(x), 3, axis = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)
        q = q * self.scale

        sim = tf.einsum('bhdi,bhdj->bhij', q, k)
        sim = sim - tf.stop_gradient(tf.math.reduce_max(sim, axis = -1, keepdims = True))
        attn = tf.nn.softmax(sim, axis = -1)

        out = tf.einsum('bhij,bhdj->bhid', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = self.h, y = self.w)
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
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        
        self.init_conv = l.Conv2D(dim, 3, padding = 'same', data_format = 'channels_first')

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

        # layers

        self.downs = []
        self.ups = []

        input_block_chans = [dim]
        ch = dim
        ds = 1
        for level, mult in enumerate(dim_mults):
            for _ in range(num_res_blocks):
                layers = [
                    ResnetBlock(ch, mult * dim, time_emb_dim=time_dim, groups=resnet_block_groups, dropout=dropout),
                ]
                ch = mult * dim
                if ds in attention_resolutions:
                    layers.append(Residual(PreNorm(ch, Attention(ch))))
                self.downs.append(TimestepEmbedSequential(layers))
                input_block_chans.append(ch)
            if level != len(dim_mults) - 1:
                self.downs.append(
                    TimestepEmbedSequential(Downsample(ch))
                )
                input_block_chans.append(ch)
                ds *= 2

        
        self.mid_block1 = ResnetBlock(ch, ch, time_emb_dim=time_dim, groups=resnet_block_groups, dropout=dropout)
        self.mid_attn = Residual(PreNorm(ch, Attention(ch)))
        self.mid_block2 = ResnetBlock(ch, ch, time_emb_dim=time_dim, groups=resnet_block_groups, dropout=dropout)

        for level, mult in list(enumerate(dim_mults))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResnetBlock(ch + input_block_chans.pop(), mult * dim, time_emb_dim=time_dim, groups=resnet_block_groups, dropout=dropout),
                ]
                ch = dim * mult
                if ds in attention_resolutions:
                    layers.append(
                        Residual(PreNorm(ch, Attention(ch)))
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch))
                    ds //= 2
                self.ups.append(TimestepEmbedSequential(layers))

        if out_dim:
            self.out_dim = out_dim
        else:
            self.out_dim = channels * (1 if not learned_variance else 2)

        self.final_conv = tf.keras.Sequential([
            ResnetBlock(dim, dim, groups=resnet_block_groups),
            l.Conv2D(self.out_dim, 1, data_format = 'channels_first'),
        ])

    def call(self, x, time):
        t = self.time_mlp(time) if self.time_mlp else None

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
            x = tf.concat([x, h], axis = 1)
            x = block(x, t)

        return self.final_conv(x)

