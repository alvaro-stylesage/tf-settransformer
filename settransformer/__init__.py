"""
An implementation of the Set Transformers framework paper in Tensorflow/Keras.

The paper can be found here: https://arxiv.org/abs/1810.00825
The official pytorch implementation can be at: https://github.com/juho-lee/set_transformer
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

DEFAULT_ACTIVATION_FN = "relu"

# Utility Functions --------------------------------------------------------------------------------

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_vars(layers={})
def CustomLayer(Layer):
    """
    A decorator for keeping track of custom layers
    """
    CustomLayer.layers[Layer.__name__] = Layer
    return Layer

def custom_layers():
    """
    Fetch custom layer instances
    """
    layers = CustomLayer.layers.copy()
    return layers

# Layer Utility Functions --------------------------------------------------------------------------

def spectral_norm(*args, **kwargs):
    import tensorflow_addons as tfa
    return tfa.layers.SpectralNormalization(*args, **kwargs)

def dense(dim, activation=None, use_spectral_norm=False):
    layer = keras.layers.Dense(dim, activation=activation)
    if use_spectral_norm:
        layer = spectral_norm(layer)
    return layer
    
# Layer Definitions --------------------------------------------------------------------------------

class MultiHeadAttention(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, use_spectral_norm=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        
        assert embed_dim % num_heads == 0, "Embed dim must be divisible by the number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_spectral_norm = use_spectral_norm
        
        self.fc_q = dense(embed_dim, None, use_spectral_norm)
        self.fc_k = dense(embed_dim, None, use_spectral_norm)
        self.fc_v = dense(embed_dim, None, use_spectral_norm)
        
        
    def call(self, q, v, k=None, training=None):
        """
        Compute multi-head attention in exactly the same manner
        as the official implementation.
        
        Reference: https://github.com/juho-lee/set_transformer/blob/master/modules.py#L20-L33
        """
        if k is None:
            k = v
        q, k, v = self.fc_q(q), self.fc_k(k), self.fc_v(v)
        num_batches = tf.shape(q)[0]
        
        # Divide for multi-head attention
        q_split = tf.concat(tf.split(q, self.num_heads, 2), 0)
        k_split = tf.concat(tf.split(k, self.num_heads, 2), 0)
        v_split = tf.concat(tf.split(v, self.num_heads, 2), 0)        
        
        # Compute attention
        att = tf.nn.softmax(tf.matmul(q_split, k_split, transpose_b=True)/np.sqrt(self.embed_dim), 2)
        out = tf.concat(tf.split(tf.matmul(att, v_split), self.num_heads, 0), 2)
        return out
        
    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "use_spectral_norm": self.use_spectral_norm
        })
        return config


@CustomLayer
class MultiHeadAttentionBlock(keras.layers.Layer):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 ff_dim=None,
                 ff_activation=DEFAULT_ACTIVATION_FN,
                 layernorm=True,
                 prelayernorm=False,
                 is_final=False,
                 use_keras_mha=True,
                 use_spectral_norm=False,
                 **kwargs):
        super(MultiHeadAttentionBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = embed_dim if ff_dim is None else ff_dim
        self.ff_activation = ff_activation
        self.layernorm = layernorm
        self.prelayernorm = prelayernorm
        self.is_final = is_final
        self.use_keras_mha = use_keras_mha
        self.use_spectral_norm = use_spectral_norm
        
        # Attention layer
        if use_keras_mha:
            self.att = keras.layers.MultiHeadAttention(key_dim=embed_dim, num_heads=num_heads)
        else:
            self.att = MultiHeadAttention(embed_dim, num_heads, use_spectral_norm)
        
        # Feed-forward layer
        ff_dim = embed_dim if ff_dim is None else ff_dim
        self.ffn = dense(ff_dim, ff_activation, use_spectral_norm)
            
        # Use layer normalization (yeah, this could be improved somehow...)
        if self.layernorm:
            self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
            if self.prelayernorm:
                self.layernorm3 = keras.layers.LayerNormalization(epsilon=1e-6)
                if self.is_final:
                    self.layernorm4 = keras.layers.LayerNormalization(epsilon=1e-6)
        
        
    def call_prenorm(self, x, y, training=None):
        x_norm = self.layernorm1(x)
        y_norm = x_norm if y is None else self.layernorm2(y)
        
        # Multi-head attention
        attn = x + self.att(x_norm, y_norm, y_norm)
        
        # ff-projection
        out = self.layernorm3(attn)
        out = attn + self.ffn(out)
        
        if self.is_final:
            out = self.layernorm4(out)
        
        return out
        
        
    def call(self, x, y=None, training=None):
        if self.layernorm and self.prelayernorm:
            return self.call_prenorm(x, y, training)
        
        if y is None:
            y = x
        
        # Multi-head attention
        attn = x + self.att(x, y, y)
        if self.layernorm:
            attn = self.layernorm1(attn)
        
        # ff-projection
        out = attn + self.ffn(attn)
        if self.layernorm:
            out = self.layernorm2(out)
        return out
    
    
    def get_config(self):
        config = super(MultiHeadAttentionBlock, self).get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "ff_activation": self.ff_activation,
            "layernorm": self.layernorm,
            "prelayernorm": self.prelayernorm,
            "is_final": self.is_final,
            "use_keras_mha": self.use_keras_mha,
            "use_spectral_norm": self.use_spectral_norm
        })
        return config

    
@CustomLayer
class SetAttentionBlock(MultiHeadAttentionBlock):
    def call(self, x, training=None):
        return super(SetAttentionBlock, self).call(x, training=training)
    

@CustomLayer
class InducedSetAttentionBlock(keras.layers.Layer):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 num_induce,
                 ff_dim=None,
                 ff_activation=DEFAULT_ACTIVATION_FN,
                 layernorm=True,
                 prelayernorm=False,
                 is_final=False,
                 use_keras_mha=True,
                 use_spectral_norm=False,
                 **kwargs):
        super(InducedSetAttentionBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_induce = num_induce
        self.mab1 = MultiHeadAttentionBlock(
            embed_dim, num_heads, ff_dim, ff_activation, layernorm,
            prelayernorm, False, use_keras_mha, use_spectral_norm)
        self.mab2 = MultiHeadAttentionBlock(
            embed_dim, num_heads, ff_dim, ff_activation, layernorm,
            prelayernorm, is_final, use_keras_mha, use_spectral_norm)
        
    def build(self, input_shape):
        self.inducing_points = self.add_weight(
            shape=(1, self.num_induce, self.embed_dim),
            initializer="glorot_uniform", # xavier_uniform from pytorch implementation
            trainable=True)
        
    def call(self, x):
        batch_size = tf.shape(x)[0]
        i = tf.tile(self.inducing_points, (batch_size, 1, 1))
        h = self.mab1(i, x)
        return self.mab2(x, h)
    
    def get_config(self):
        config = super(InducedSetAttentionBlock, self).get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.mab2.num_heads,
            "num_induce": self.num_induce,
            "ff_dim": self.mab2.ff_dim,
            "ff_activation": self.mab2.ff_activation,
            "layernorm": self.mab2.layernorm,
            "prelayernorm": self.mab2.prelayernorm,
            "is_final": self.mab2.is_final,
            "use_keras_mha": self.mab2.use_keras_mha,
            "use_spectral_norm": self.mab2.use_spectral_norm
        })
        return config
    

@CustomLayer
class ConditionedSetAttentionBlock(keras.layers.Layer):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 num_anchors,
                 ff_dim=None,
                 ff_activation=DEFAULT_ACTIVATION_FN,
                 mlp_dim=None,
                 mlp_activation=DEFAULT_ACTIVATION_FN,
                 layernorm=True,
                 prelayernorm=False,
                 is_final=False,
                 use_keras_mha=True,
                 use_spectral_norm=False,
                 **kwargs):
        super(ConditionedSetAttentionBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_anchors = num_anchors
        self.mlp_dim = mlp_dim if mlp_dim is not None else 2*embed_dim
        self.mlp_activation = mlp_activation
        self.mab1 = MultiHeadAttentionBlock(
            embed_dim, num_heads, ff_dim, ff_activation, layernorm,
            prelayernorm, False, use_keras_mha, use_spectral_norm)
        self.mab2 = MultiHeadAttentionBlock(
            embed_dim, num_heads, ff_dim, ff_activation, layernorm,
            prelayernorm, is_final, use_keras_mha, use_spectral_norm)
        self.anchor_predict = keras.models.Sequential([
            keras.layers.Dense(
                self.mlp_dim,
                input_shape=(embed_dim,),
                activation=mlp_activation),
            dense(num_anchors*embed_dim, None, use_spectral_norm),
            keras.layers.Reshape((num_anchors, embed_dim))
        ])
        
    def call(self, x, condition, training=None):
        anchor_points = self.anchor_predict(condition)
        h = self.mab1(anchor_points, x)
        return self.mab2(x, h)
    
    def get_config(self):
        config = super(ConditionedSetAttentionBlock, self).get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.mab2.num_heads,
            "num_anchors": self.num_anchors,
            "ff_dim": self.mab2.ff_dim,
            "ff_activation": self.mab2.ff_activation,
            "mlp_dim": self.mlp_dim,
            "mlp_activation": self.mlp_activation,
            "layernorm": self.mab2.layernorm,
            "prelayernorm": self.mab2.prelayernorm,
            "is_final": self.mab2.is_final,
            "use_keras_mha": self.mab2.use_keras_mha,
            "use_spectral_norm": self.mab2.use_spectral_norm
        })
        return config


# Pooling Methods ----------------------------------------------------------------------------------

@CustomLayer
class PoolingByMultiHeadAttention(keras.layers.Layer):
    def __init__(self,
                 num_seeds,
                 embed_dim,
                 num_heads,
                 ff_dim=None,
                 ff_activation=DEFAULT_ACTIVATION_FN,
                 layernorm=True,
                 prelayernorm=False,
                 is_final=False,
                 use_keras_mha=True,
                 use_spectral_norm=False,
                 **kwargs):
        super(PoolingByMultiHeadAttention, self).__init__(**kwargs)
        self.num_seeds = num_seeds
        self.embed_dim = embed_dim
        self.mab = MultiHeadAttentionBlock(
            embed_dim, num_heads, ff_dim, ff_activation, layernorm,
            prelayernorm, is_final, use_keras_mha, use_spectral_norm)

        
    def build(self, input_shape):
        self.seed_vectors = self.add_weight(
            shape=(1, self.num_seeds, self.embed_dim),
            initializer="random_normal",
            trainable=True)
        
        
    def call(self, z):
        batch_size = tf.shape(z)[0]
        seeds = tf.tile(self.seed_vectors, (batch_size, 1, 1))
        return self.mab(seeds, z)
    
    
    def get_config(self):
        config = super(PoolingByMultiHeadAttention, self).get_config()
        config.update({
            "num_seeds": self.num_seeds,
            "embed_dim": self.embed_dim,
            "num_heads": self.mab.num_heads,
            "ff_dim": self.mab.ff_dim,
            "ff_activation": self.mab.ff_activation,
            "layernorm": self.mab.layernorm,
            "prelayernorm": self.mab.prelayernorm,
            "is_final": self.mab.is_final,
            "use_keras_mha": self.mab.use_keras_mha,
            "use_spectral_norm": self.mab.use_spectral_norm
        })
        return config
    

@CustomLayer
class InducedSetEncoder(PoolingByMultiHeadAttention):
    """
    Same as PMA, except resulting rows are summed together.
    """ 
    def call(self, x):
        out = super(InducedSetEncoder, self).call(x)
        return tf.reduce_sum(out, axis=1)
    
# Alias exports
MAB = MultiHeadAttentionBlock
SAB = SetAttentionBlock
ISAB = InducedSetAttentionBlock
PMA = PoolingByMultiHeadAttention
ISE = InducedSetEncoder
CSAB = ConditionedSetAttentionBlock