
import preprocess
import build_network
import tensorflow as tf

from tensorflow.contrib import seq2seq


# Neural Network Training
# Hyperparameters

# Build the Graph
# Build the graph using the neural network you implemented.

def train(params, vars):

    int_text = vars['int_text']
    vocab_to_int = vars['vocab_to_int']
    int_to_vocab = vars['int_to_vocab']
    token_dict = vars['token_dict']

    num_epochs = params['num_epochs']
    batch_size = params['batch_size']
    rnn_size = params['rnn_size']
    embed_dim = params['embed_dim']
    seq_length = params['seq_length']
    learning_rate = params['learning_rate']
    keep_prob = params['keep_prob']
    show_every_n_batches = params['show_every_n_batches']
    save_dir = params['save_dir']

    train_graph = tf.Graph()
    with train_graph.as_default():
        vocab_size = len(int_to_vocab)
        input_text, targets, lr = build_network.get_inputs()
        input_data_shape = tf.shape(input_text)
        cell, initial_state = build_network.get_init_cell(input_data_shape[0], rnn_size, keep_prob)
        logits, final_state = build_network.build_nn(cell, input_text, vocab_size, embed_dim)

        # Probabilities for generating words
        probs = tf.nn.softmax(logits, name='probs')

        # Loss function
        cost = seq2seq.sequence_loss(
            logits,
            targets,
            tf.ones([input_data_shape[0], input_data_shape[1]]))

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)


    # Train
    # Train the neural network on the preprocessed data.

    batches = build_network.get_batches(int_text, batch_size, seq_length)

    with tf.Session(graph=train_graph) as sess:
        sess.run(tf.global_variables_initializer())

        for epoch_i in range(num_epochs):
            state = sess.run(initial_state, {input_text: batches[0][0]})

            for batch_i, (x, y) in enumerate(batches):
                feed = {
                    input_text: x,
                    targets: y,
                    initial_state: state,
                    lr: learning_rate}
                train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

                # Show every <show_every_n_batches> batches
                if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                    print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                        epoch_i,
                        batch_i,
                        len(batches),
                        train_loss))

        # Save Model
        saver = tf.train.Saver()
        saver.save(sess, save_dir)
        print('Model Trained and Saved')