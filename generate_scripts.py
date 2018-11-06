
import preprocess
import numpy as np
import tensorflow as tf

# Implement Generate Functions
# Get Tensors


def get_tensors(loaded_graph):
    """
    Get input, initial state, final state, and probabilities tensor from <loaded_graph>
    :param loaded_graph: TensorFlow graph loaded from file
    :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
    """
    InputTensor = tf.Graph.get_tensor_by_name(loaded_graph, name='input:0')
    InitialStateTensor = tf.Graph.get_tensor_by_name(loaded_graph, name='initial_state:0')
    FinalStateTensor = tf.Graph.get_tensor_by_name(loaded_graph, name='final_state:0')
    ProbsTensor = tf.Graph.get_tensor_by_name(loaded_graph, name='probs:0')
    return InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor


# Choose Word

def pick_word(probabilities, int_to_vocab):
    """
    Pick the next word in the generated text
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :return: String of the predicted word
    """
    return int_to_vocab[np.argmax(probabilities)]


# Generate TV Script

def generate(gen_length, prime_word, data_path):
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # Load saved model
        loader = tf.train.import_meta_graph(data_path + '.meta')
        loader.restore(sess, data_path)

        # Get Tensors from loaded model
        input_text, initial_state, final_state, probs = get_tensors(loaded_graph)

        # Sentences generation setup
        gen_sentences = [prime_word + ':']
        prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

        # Generate sentences
        for n in range(gen_length):
            # Dynamic Input
            dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
            dyn_seq_length = len(dyn_input[0])

            # Get Prediction
            probabilities, prev_state = sess.run(
                [probs, final_state],
                {input_text: dyn_input, initial_state: prev_state})

            pred_word = pick_word(probabilities[dyn_seq_length-1], int_to_vocab)

            gen_sentences.append(pred_word)

        # Remove tokens
        tv_script = ' '.join(gen_sentences)
        for key, token in token_dict.items():
            ending = ' ' if key in ['\n', '(', '"'] else ''
            tv_script = tv_script.replace(' ' + token.lower(), key)
        tv_script = tv_script.replace('\n ', '\n')
        tv_script = tv_script.replace('( ', '(')

        print(tv_script)