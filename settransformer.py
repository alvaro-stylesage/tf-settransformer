"""
An implementation of the Set Transformers framework paper in Tensorflow/Keras.

The paper can be found here: https://arxiv.org/abs/1810.00825
The official pytorch implementation can be at: https://github.com/juho-lee/set_transformer
"""

import tensorflow as tf
import tensorflow.keras as keras

class MultiHeadAttentionBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ffn_activation="relu"):
        super(MultiHeadAttentionBlock, self).__init__()
        self.att = keras.layers.MultiHeadAttention(key_dim=embed_dim, num_heads=num_heads)
        self.ffn = keras.layers.Dense(embed_dim, activation=ffn_activation)
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)

    def compute_output_shape(self, input_shape):
        return input_shape
        
    def call(self, x, y):
        attn_output = self.att(x, y, y) # query, value (optional key defaults to value)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)


class SetAttentionBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(SetAttentionBlock, self).__init__()
        self.mab = MultiHeadAttentionBlock(embed_dim, num_heads)
        
    def compute_output_shape(self, *args):
        return self.mab.compute_output_shape(*args)
    
    def call(self, x):
        return self.mab(x, x)
    
    
class InducedSetAttentionBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, num_induce):
        super(InducedSetAttentionBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_induce = num_induce
        self.mab1 = MultiHeadAttentionBlock(embed_dim, num_heads)
        self.mab2 = MultiHeadAttentionBlock(embed_dim, num_heads)
        
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
    
    
class PoolingByMultiHeadAttention(keras.layers.Layer):
    def __init__(self, num_seeds, embed_dim, num_heads, **kwargs):
        super(PoolingByMultiHeadAttention, self).__init__(**kwargs)
        self.num_seeds = num_seeds
        self.embed_dim = embed_dim
        self.mab = MultiHeadAttentionBlock(embed_dim, num_heads)
        
    def build(self, input_shape):
        self.seed_vectors = self.add_weight(
            shape=(1, self.num_seeds, self.embed_dim),
            initializer="random_normal",
            trainable=True)
        
    def call(self, z):
        batch_size = tf.shape(z)[0]
        seeds = tf.tile(self.seed_vectors, (batch_size, 1, 1))
        return self.mab(seeds, z)
    
    
# Alias exports
MAB = MultiHeadAttentionBlock
SAB = SetAttentionBlock
ISAB = InducedSetAttentionBlock
PMA = PoolingByMultiHeadAttention