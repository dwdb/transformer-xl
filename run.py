import numpy as np
import tensorflow as tf
import time
from data_utils import ChineseCorpus, Vocab
from model import TransformerXL
import sys

# 'Description: Transformer-XL Simplified Version.'
# vocabulary file
VOCAB_FILE = 'data/poetry/vocab.pkl'
# dataset path
DATA_PATH = 'data/poetry'
# path for training and valid output, such as save model
OUTPUT_PATH = 'output/'
# tensorboard summary
SUMMARY_PATH = 'summary/'
BATCH_SIZE = 64
# target length, or sequence length
SEQ_LEN = 50
# memory length
MEM_LEN = 50
# word embeeding size
EMBEDDING_SIZE = 410
# multihead attetion hidden size
HIDDEN_SIZE = 410
# feed forward network hidden size
FFN_SIZE = 2100
# number of heads of multiheads
NUM_HEADS = 10
# number of layers of multihead attention
N_LAYER = 16
DROPOUT_RATE = 0.1
# wheather the bias of each layer of relative multihead attention is different or not
UNTIE_REL_BIAS = True
# training steps
STEPS = 200000
# warmup steps in the begging of training
WARMUP_STEPS = 0
# initial learning rate
LEARNING_RATE = 0.0001
# minimal learning rate
MIN_LEARNING_RATE = 0.004
# clips values of multiple tensors by the ratio of the sum of their norms
CLIP_NORM = 0.25
# number of steps between show information during training
VERBOSE_STEP = 100
# number of steps between save model
SAVE_STEP = 2000
# number of steps between verify model
VALID_STEP = 500
EARLY_STOPPING_TIMES = 5


class CosineDecayWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, init_lr, steps, warmup_steps, min_lr):
        super(CosineDecayWarmup, self).__init__()

        self.init_lr = init_lr
        self.warmup_steps = warmup_steps
        self.cosine_decay = tf.keras.experimental.CosineDecay(
            init_lr, steps - warmup_steps, min_lr)

    def __call__(self, step):
        linear_increase = self.init_lr * tf.cast(step, tf.float32) / (
                tf.cast(self.warmup_steps, tf.float32) + 1e-5)
        cosine_decay = self.cosine_decay(step)
        return tf.cond(pred=step <= self.warmup_steps,
                       true_fn=lambda: linear_increase,
                       false_fn=lambda: cosine_decay)

    def get_config(self):
        return {
            'warmup_steps': self.warmup_steps,
            'init_lr': self.init_lr
        }


def model_fn():
    model = TransformerXL(n_vocab=corpus.vocab.size, d_embed=EMBEDDING_SIZE,
                          d_model=HIDDEN_SIZE, d_ff=FFN_SIZE, q_len=SEQ_LEN,
                          m_len=MEM_LEN,
                          num_heads=NUM_HEADS, n_layer=N_LAYER, dropout_rate=DROPOUT_RATE,
                          untie_rel_bias=UNTIE_REL_BIAS)

    return model


corpus = ChineseCorpus(path=DATA_PATH, vocab=Vocab(VOCAB_FILE))
model = model_fn()


def logits_to_symbols(logits):
    indices = np.argmax(logits, axis=-1)
    if np.ndim(indices) <= 1:
        indices = [indices]
    return corpus.vocab.get_symbols(indices, join=True)


def loss_function(labels, logits):
    """损失函数"""
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        labels, logits, from_logits=True)
    loss = tf.reduce_mean(loss)
    return loss


@tf.function
def train_step(inputs, labels, optimizer, mems):
    """训练一个batch"""
    with tf.GradientTape() as tape:
        logits, new_mems = model(inputs, mems=mems, training=True)
        loss = loss_function(labels, logits)

    gradients = tape.gradient(loss, model.trainable_variables)
    clipped, gnorm = tf.clip_by_global_norm(gradients, CLIP_NORM)
    optimizer.apply_gradients(zip(clipped, model.trainable_variables))

    return loss, logits, new_mems


def train():
    """模型训练"""
    ckpt = tf.train.Checkpoint(model=model)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, OUTPUT_PATH, max_to_keep=3, checkpoint_name='xl-ckpt')
    writer = tf.summary.create_file_writer(SUMMARY_PATH)

    # 暖启衰减学习率
    learning_rate = CosineDecayWarmup(LEARNING_RATE, STEPS, WARMUP_STEPS,
                                      MIN_LEARNING_RATE)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    # create corpus dataset
    train_dataset = corpus.get_dataset('train', batch_size=BATCH_SIZE, seq_len=SEQ_LEN)
    mems = None

    train_loss = tf.keras.metrics.Mean(name='train_loss')

    old_time = time.time()
    for step, batch in enumerate(train_dataset):
        loss, logits, mems = train_step(
            batch['inputs'], batch['labels'], optimizer=optimizer, mems=mems)
        train_loss(loss)

        if step % VERBOSE_STEP == 0:
            print('{} step: {} | loss: {:.4f} | lr: {} | {:.2f} step/s'.format(
                time.strftime("%Y-%m-%d %H:%M:%S"),
                step,
                train_loss.result(),
                learning_rate(step),
                VERBOSE_STEP / (time.time() - old_time)))
            old_time = time.time()

            with writer.as_default():
                tf.summary.scalar('train_loss', train_loss.result(), step=step)
            train_loss.reset_states()

            inps = corpus.vocab.get_symbols(batch['inputs'], join=True)[:3]
            outs = logits_to_symbols(logits)[:5]
            print(inps, '\n', outs, '\n', sep='')

        if step % SAVE_STEP == 0:
            print('saving checkpoint for epoch {} at {}'.format(
                step, ckpt_manager.save()))

        # if step % VALID_STEP == 0:
        #     loss = evaluate()
        #     print(f'====\nvalidation average loss: {loss:.3f}\n====')
        #     with writer.as_default():
        #         tf.summary.scalar('valid_loss', loss, step=step)

        if step >= STEPS:
            print(f'reach max step of iteations {STEPS}, training completed.')
            break


def evaluate():
    """模型验证"""
    mems = [None] * model.n_layer
    total_loss, total_cnt = 0., 0

    valid_dataset = corpus.get_dataset('valid', batch_size=8, seq_len=SEQ_LEN)
    for batch in valid_dataset:
        inputs, labels = batch['inputs'], batch['labels']
        logits, mems = model(inputs, mems=mems, training=False)
        loss = loss_function(labels, logits)
        # statistic total loss
        cnt = np.prod(np.shape(labels))
        total_cnt += cnt
        total_loss += loss * cnt

    avg_loss = total_loss / total_cnt
    return avg_loss


def inference(sentence=None, tgt_len=50, mem_len=50, max_len=64):
    """推理，每次生成一个字符
    todo：每次生成一定长度的单词"""

    def fn(sentence):
        sentence = sentence[-rel_len:]
        x = corpus.vocab.get_indices(sentence)
        sequence = []
        mems = [None] * model.n_layer
        for i in range(max_len):
            x = tf.constant([x], dtype=tf.int32)
            output, mems = model(x, mems=mems, training=False)
            x = tf.argmax(output[:, -1], axis=-1).numpy()
            # early stop when the eos symbol has generated
            if x[0] == corpus.vocab.get_index(corpus.vocab.eos):
                break
            sequence.append(x[0])
        gene_sent = corpus.vocab.get_symbols(sequence, join=True)
        return sentence + gene_sent

    model = model_fn()
    # 相对位置编码的长度
    rel_len = model.q_len + model.m_len
    # memory length of inference
    model.m_len = mem_len
    model.q_len = tgt_len

    checkpoint_path = tf.train.latest_checkpoint(OUTPUT_PATH)
    print('restoring model from {}'.format(checkpoint_path))
    tf.train.Checkpoint(model=model).restore(checkpoint_path)

    if sentence:
        return fn(sentence)

    while True:
        # init inputs
        while not sentence:
            print('seed text >>> ', end='')
            sentence = input()
        print('>> ' + fn(sentence))
        sentence = ''


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        raise ValueError('Missing running method!')
    if sys.argv[1] == 'train':
        train()
    if sys.argv[1] == 'inference':
        inference()
