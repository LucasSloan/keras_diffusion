import math

import tensorflow as tf
import tensorflow.keras.layers as l
import tensorflow_addons as tfa

from einops import rearrange


def exists(x):
    return x is not None

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
        self.norm = l.LayerNormalization()

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

class ResnetBlock(l.Layer):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        if time_emb_dim:
            self.mlp = tf.keras.Sequential([
                l.Activation(tf.keras.activations.swish),
                l.Dense(dim_out * 2),
            ])
        else:
            self.mlp = None

        self.block1 = Block(dim, dim_out, groups = groups)
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
            l.LayerNormalization(),
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
        channels = 3,
        with_time_emb = True,
        resnet_block_groups = 8,
        learned_variance = False
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        
        if not init_dim:
            init_dim = dim // 3 * 2
        self.init_conv = l.Conv2D(init_dim, 7, padding = 'same', data_format = 'channels_first')

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

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
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append([
                ResnetBlock(dim_in, dim_out, time_emb_dim=time_dim, groups=resnet_block_groups),
                ResnetBlock(dim_out, dim_out, time_emb_dim=time_dim, groups=resnet_block_groups),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else None,
            ])

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, groups=resnet_block_groups)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim, groups=resnet_block_groups)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim=time_dim, groups=resnet_block_groups),
                ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim, groups=resnet_block_groups),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else None
            ])

        if out_dim:
            self.out_dim = out_dim
        else:
            self.out_dim = channels * (1 if not learned_variance else 2)

        self.final_conv = tf.keras.Sequential([
            ResnetBlock(dim, dim, groups=resnet_block_groups),
            l.Conv2D(self.out_dim, 1, data_format = 'channels_first'),
        ])

    def call(self, x, time):
        x = self.init_conv(x)

        t = self.time_mlp(time) if self.time_mlp else None

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            if downsample:
                x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = tf.concat([x, h.pop()], axis = 1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            if upsample:
                x = upsample(x)

        return self.final_conv(x)

