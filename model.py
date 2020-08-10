import tensorflow as tf

INITIALIZER = tf.keras.initializers.RandomNormal(stddev=0.01)


def relative_mask(q_len, m_len):
    """相对位置掩码，当前位置左侧为1、右侧为0"""
    mask = tf.sequence_mask(tf.range(1, q_len + 1), q_len, dtype=tf.float32)
    mask = tf.pad(mask, [[0, 0], [m_len, 0]], constant_values=1)
    return mask


def positional_embedding(k_len, d_model):
    """绝对位置编码"""
    inv_freq = 1. / (10000 ** (tf.range(0, d_model, 2.0) / d_model))
    pos_seq = tf.range(k_len - 1, -1, -1.0)
    sinusoid_inp = tf.einsum('i,j->ij', pos_seq, inv_freq)
    pos_emb = tf.concat([tf.sin(sinusoid_inp), tf.cos(sinusoid_inp)], -1)
    return pos_emb[None, :, :]


def point_wise_feed_forward_network(d_model, d_ff):
    """前馈网络"""
    return tf.keras.Sequential([
        tf.keras.layers.Dense(d_ff, activation='relu',
                              kernel_initializer=INITIALIZER, name='ffn1'),
        tf.keras.layers.Dense(d_model, kernel_initializer=INITIALIZER, name='ffn2')
    ])


class RelMultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, dropout_rate):
        super(RelMultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_model = d_model
        self.d_depth = self.d_model // self.num_heads

        self.w_head = tf.keras.layers.Dense(
            3 * d_model, use_bias=False, kernel_initializer=INITIALIZER)
        self.r_head = tf.keras.layers.Dense(
            d_model, use_bias=False, kernel_initializer=INITIALIZER)

        self.dense = tf.keras.layers.Dense(
            d_model, use_bias=False, kernel_initializer=INITIALIZER)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    @staticmethod
    def relative_shift(x):
        """行元素左移，结合mask实现相对位置编码。移动步数为n-i（n为行总数，i为行号，首行为1）。
        """
        x_size = tf.shape(x)
        x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [1, 0]])
        x = tf.reshape(x, (x_size[0], x_size[1], x_size[3] + 1, x_size[2]))
        x = tf.slice(x, [0, 0, 1, 0], [-1, -1, -1, -1])
        x = tf.reshape(x, x_size)
        return x

    def call(self, inputs, pos_emb, r_w_bias, r_r_bias, mems, training, **kwargs):
        """
        inputs: shape=(batch_size, q_len, d_model)
        pos_emb: shape=(1, k_len, d_model)
        u: shape=(num_heads, d_depth)
        v: shape=(num_heads, d_depth)
        mems: shape=(batch_size, m_len, d_model)
        attn_mask: shape=(m_len + q_len, q_len)
        """
        batch_size = tf.shape(inputs)[0]
        q_len = tf.shape(inputs)[1]
        # 拼接缓存
        if mems is None:
            cat = inputs
        else:
            cat = tf.concat((mems, inputs), axis=1)
        cat = self.dropout1(cat, training=training)
        # 拼接后的上下文长度，k_len = m_len + q_len
        k_len = tf.shape(cat)[1]
        m_len = k_len - q_len
        # shape=(1, k_len, d_model)
        pos_emb = pos_emb[:, -k_len:]
        pos_emb = self.dropout2(pos_emb, training=training)

        w_heads = tf.reshape(self.w_head(cat), (
            batch_size, k_len, 3 * self.num_heads, self.d_depth))
        w_head_q, w_head_k, w_head_v = tf.split(w_heads, 3, axis=2)
        # shape=(batch_size, q_len, num_heads, d_depth)
        w_head_q = w_head_q[:, -q_len:]

        # shape=(batch_size, num_heads, q_len, k_len)
        ac = tf.einsum('bqnd,bknd->bnqk', w_head_q + r_w_bias, w_head_k)
        r_head_k = tf.reshape(self.r_head(pos_emb), (k_len, self.num_heads, self.d_depth))
        bd = tf.einsum('bqnd,knd->bnqk', w_head_q + r_r_bias, r_head_k)
        bd = self.relative_shift(bd)

        attn_mask = relative_mask(q_len, m_len)
        # shape=(batch_size, num_heads, q_len, k_len)
        attn_score = (ac + bd) / (self.d_depth ** 0.5)
        attn_score = attn_score * attn_mask - 1e30 * (1. - attn_mask)
        attn_score = tf.nn.softmax(attn_score, axis=-1)

        attn_vec = tf.einsum('bnqk,bknd->bqnd', attn_score, w_head_v)
        attn_vec = tf.reshape(attn_vec, (batch_size, q_len, self.d_model))

        attn_out = self.dense(attn_vec)
        return attn_out


class TransformerLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, d_ff, num_heads, dropout_rate):
        super(TransformerLayer, self).__init__()

        self.rel_multihead_attn = RelMultiHeadAttention(
            d_model=d_model, num_heads=num_heads, dropout_rate=dropout_rate)
        # feed forward network
        self.ffn = point_wise_feed_forward_network(d_model, d_ff)
        # layer normalization
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # dropout
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, pos_emb, r_w_bias, r_r_bias, mems, training, **kwargs):
        attn_out = self.rel_multihead_attn(inputs=inputs, pos_emb=pos_emb,
                                           r_w_bias=r_w_bias, r_r_bias=r_r_bias,
                                           mems=mems, training=training)
        attn_out = self.dropout1(attn_out, training=training)
        out1 = self.layer_norm1(inputs + attn_out)

        ffn_out = self.ffn(out1, training=training)
        ffn_out = self.dropout2(ffn_out, training=training)
        out2 = self.layer_norm2(out1 + ffn_out)
        return out2


class TransformerXL(tf.keras.Model):

    def __init__(self, n_vocab, d_embed, d_model, d_ff, q_len, m_len, num_heads,
                 n_layer, dropout_rate, untie_rel_bias):
        super(TransformerXL, self).__init__()
        self.d_embed = d_embed
        self.d_model = d_model

        self.q_len = q_len
        self.m_len = m_len
        self.n_layer = n_layer
        self.untie_rel_bias = untie_rel_bias

        # word embedding
        self.embedding = tf.Variable(INITIALIZER((n_vocab, d_embed)), name='embedding')
        # word embedding size to model size
        self.projection = tf.Variable(INITIALIZER((d_embed, d_model)), name='projection')
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)

        self.pos_emb = positional_embedding(q_len + m_len, d_model)

        shape = (2, n_layer if untie_rel_bias else 1, num_heads, d_model // num_heads)
        self.rw_bias = tf.Variable(INITIALIZER(shape), name='rw_bias')
        self.logit_bias = tf.Variable(tf.zeros((n_vocab,)), name='logit_bias')

        self.multihead_layers = []
        for i in range(n_layer):
            layer = TransformerLayer(d_model=d_model, d_ff=d_ff, num_heads=num_heads,
                                     dropout_rate=dropout_rate)
            self.multihead_layers.append(layer)

    def cache_mems(self, cur_out, pre_mem):
        if self.m_len is None or self.m_len <= 0:
            return None
        if pre_mem is None:
            new_mem = cur_out
        else:
            new_mem = tf.concat((pre_mem, cur_out), axis=1)
        return tf.stop_gradient(new_mem[:, -self.m_len:])

    def call(self, inputs, mems=None, training=False, **kwargs):
        new_mems = []
        x = tf.nn.embedding_lookup(self.embedding, inputs)
        x = tf.matmul(x, self.projection)

        if mems is None:
            mems = [None] * self.n_layer

        for i in range(self.n_layer):
            new_mems.append(self.cache_mems(x, mems[i]))
            j = i if self.untie_rel_bias else 0
            x = self.multihead_layers[i](inputs=x,
                                         pos_emb=self.pos_emb,
                                         r_w_bias=self.rw_bias[0][j],
                                         r_r_bias=self.rw_bias[1][j],
                                         mems=mems[i],
                                         training=training)

        x = self.dropout1(x, training=training)
        # share embedding parameters with inputs
        # shape=(batch_size, seq_len, d_embed)
        # tf.einsum('bik,jk->bij', x, self.projection)
        x = tf.matmul(x, self.projection, transpose_b=True)
        # shape=(batch_size, seq_len, n_vocab)
        x = tf.matmul(x, self.embedding, transpose_b=True) + self.logit_bias

        return x, new_mems


if __name__ == '__main__':
    n_vocab = 1000
    d_embed = 128
    d_model = 128
    d_ff = 512
    q_len = 16
    m_len = 32
    num_heads = 8
    n_layer = 6
    dropout_rate = 6
    batch_size = 8
    mem_transformer = TransformerXL(n_vocab=n_vocab,
                                    d_embed=d_embed,
                                    d_model=d_model,
                                    d_ff=d_ff,
                                    q_len=q_len,
                                    m_len=m_len,
                                    num_heads=num_heads,
                                    n_layer=n_layer,
                                    dropout_rate=dropout_rate,
                                    untie_rel_bias=True)
    inputs = tf.reshape(tf.range(batch_size * q_len), shape=(batch_size, q_len))
    output1, mems1 = mem_transformer(inputs, training=False)
    mem_transformer.mems = mems1
    output2, mems2 = mem_transformer(inputs, training=False)
    print(output1[0][0])
    print(output2[0][0])
