
# - Set `num_epochs` to the number of epochs.
# - Set `batch_size` to the batch size.
# - Set `rnn_size` to the size of the RNNs.
# - Set `embed_dim` to the size of the embedding.
# - Set `seq_length` to the length of sequence.
# - Set `learning_rate` to the learning rate.
# - Set `show_every_n_batches` to the number of batches the neural network should print progress.
from preprocess import preprocess
from generate_scripts import generate
from train_network import train

data_dir = './data/simpsons/moes_tavern_lines.txt'
prep = preprocess(data_dir)
int_text, vocab_to_int, int_to_vocab, token_dict = prep.preprocess_and_save_data()

variables = {}
variables['int_text'] = int_text
variables['vocab_to_int'] = vocab_to_int
variables['int_to_vocab'] = int_to_vocab
variables['token_dict'] = token_dict

params = {
    'num_epochs':200,
    'batch_size':128,
    'rnn_size':256,
    'embed_dim':25,
    'seq_length':25,
    'learning_rate':0.003,
    'keep_prob':0.8,
    'show_every_n_batches':50,
    'save_dir':'./save'
    }
# Set `gen_length` to the length of TV script you want to generate.
gen_length = 200
# homer_simpson, moe_szyslak, or Barney_Gumble
prime_word = 'moe_szyslak'

train(params, variables)

generate(gen_length, prime_word, data_dir)