
import numpy as np
import tensorflow as tf


# Input

def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    Input = tf.placeholder(tf.int32, [None, None], name = 'input')
    Targets = tf.placeholder(tf.int32, [None, None], name = 'targets')
    LearningRate = tf.placeholder(tf.float32, name = 'learning_rate')

    return Input, Targets, LearningRate

# Build RNN Cell and Initialize

def get_init_cell(batch_size, rnn_size, keep_prob):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :param keep_prob: keep probability
    :return: Tuple (cell, initialize state)
    """
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell([lstm])
    initial_state = tf.identity(cell.zero_state(batch_size = batch_size, dtype = tf.float32),
                                name = 'initial_state')
    return cell, initial_state


# Word Embedding
# Apply embedding to `input_data` using TensorFlow.  Return the embedded sequence.

def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """
    embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim), -1, 1))
    embed_input = tf.nn.embedding_lookup(embedding, input_data)
    return embed_input


# Build RNN

def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    final_state = tf.identity(state, name='final_state')
    return outputs, final_state


# Build the Neural Network

def build_nn(cell, input_data, vocab_size, embed_dim):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :param embed_dim: Number of embedding dimensions
    :return: Tuple (Logits, FinalState)
    """
    embed_input = get_embed(input_data, vocab_size, embed_dim)
    outputs, final_state = build_rnn(cell, embed_input)
    logits = tf.contrib.layers.fully_connected(activation_fn=None,
                                               num_outputs=vocab_size,
                                               inputs = outputs)
    return logits, final_state


# Batches

def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    characters_per_batch = seq_length * batch_size
    n_batches = len(int_text)//characters_per_batch

    # Keep only enough characters to make full batches
    int_text = int_text[:n_batches * characters_per_batch]
    int_text = np.array(int_text)
    # Reshape into n_seqs rows
    int_text = int_text.reshape((batch_size, -1))
    batches = []
    for n in range(0, int_text.shape[1], seq_length):
            # The features
            x = int_text[:, n:n+seq_length]
            # The targets, shifted by one
            y = np.zeros_like(x)
            if (n == int_text.shape[1] - seq_length):
                y[:, :-1], y[:, -1] = x[:, 1:], x[:, -1] + 1
                y[-1, -1] = 0
            else:
                y[:, :-1], y[:, -1] = x[:, 1:], x[:, -1] + 1

            batch = [x, y]
            batches.append(batch)
    return np.array(batches)
