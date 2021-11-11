"""
An implementation of the Set Transformers framework paper in Tensorflow/Keras.

The paper can be found here: https://arxiv.org/abs/1810.00825
The official pytorch implementation can be at: https://github.com/juho-lee/set_transformer
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

DEFAULT_ACTIVATION_FN = "relu"

# Use Keras implementation of multi-head attention
_config = {
    "use_keras_mha": False
}

def config(*args):
    """
    Indicate whether or not to use Keras'
    implementation of multi-head attention
    """
    if len(args) == 0:
        return _config
    if len(args) > 2:
        raise Exception("Too many arguments. Should be: key, [value]")
    if len(args) == 2:
        _config[args[0]] = args[1]
    return _config[args[0]]
    

class MultiHeadAttentionBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, activation=DEFAULT_ACTIVATION_FN, layernorm=False):
        super(MultiHeadAttentionBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.layernorm = layernorm
        self.ffn = keras.layers.Dense(embed_dim, activation=activation)
        
        # Attention method
        if _config["use_keras_mha"]:
            self.att = keras.layers.MultiHeadAttention(key_dim=embed_dim, num_heads=num_heads)
        else:
            self.fc_q = keras.layers.Dense(embed_dim)
            self.fc_k = keras.layers.Dense(embed_dim)
            self.fc_v = keras.layers.Dense(embed_dim)
            self.att = self.compute_multihead_attention
            
        # Use layer normalization
        if self.layernorm:
            self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)

    def compute_output_shape(self, input_shape):
        return input_shape
    
    def compute_multihead_attention(self, q, v, k=None):
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
        
    def call(self, x, y):
        out = x + self.att(x, y, y) # query, value (optional key defaults to value)
        if self.layernorm:
            out = self.layernorm1(out)        
        out = out + self.ffn(out)
        if self.layernorm:
            out = self.layernorm2(out)
        return out


class SetAttentionBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, activation=DEFAULT_ACTIVATION_FN, layernorm=False):
        super(SetAttentionBlock, self).__init__()
        self.mab = MultiHeadAttentionBlock(embed_dim, num_heads, activation, layernorm)
        
    def compute_output_shape(self, *args):
        return self.mab.compute_output_shape(*args)
    
    def call(self, x):
        return self.mab(x, x)
    
    
class InducedSetAttentionBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, num_induce, activation=DEFAULT_ACTIVATION_FN, layernorm=False):
        super(InducedSetAttentionBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_induce = num_induce
        self.mab1 = MultiHeadAttentionBlock(embed_dim, num_heads, activation, layernorm)
        self.mab2 = MultiHeadAttentionBlock(embed_dim, num_heads, activation, layernorm)
        
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
    
    
# Pooling Methods ----------------------------------------------------------------------------------
    
class PoolingByMultiHeadAttention(keras.layers.Layer):
    def __init__(self, num_seeds, embed_dim, num_heads, activation=DEFAULT_ACTIVATION_FN, layernorm=False, **kwargs):
        super(PoolingByMultiHeadAttention, self).__init__(**kwargs)
        self.num_seeds = num_seeds
        self.embed_dim = embed_dim
        self.mab = MultiHeadAttentionBlock(embed_dim, num_heads, activation, layernorm)
        
        self.seed_vectors = self.add_weight(
            shape=(1, self.num_seeds, self.embed_dim),
            initializer="random_normal",
            trainable=True)
        
    def build(self, input_shape):
        self.seed_vectors = self.add_weight(
            shape=(1, self.num_seeds, self.embed_dim),
            initializer="random_normal",
            trainable=True)
        
    def call(self, z):
        batch_size = tf.shape(z)[0]
        seeds = tf.tile(self.seed_vectors, (batch_size, 1, 1))
        return self.mab(seeds, z)
    
    
class InducedSetEncoder(PoolingByMultiHeadAttention):
    """
    Same as PMA, except resulting rows are summed together.
    """
    def __init__(self, *args, **kwargs):
        super(InducedSetEncoder, self).__init__(*args, **kwargs)
        
    def call(self, x):
        out = super(InducedSetEncoder, self).call(x)
        return tf.reduce_sum(out, axis=1)
    
# Alias exports
MAB = MultiHeadAttentionBlock
SAB = SetAttentionBlock
ISAB = InducedSetAttentionBlock
PMA = PoolingByMultiHeadAttention
ISE = InducedSetEncoder