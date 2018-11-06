import os
import pickle

class preprocess:

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path


    def token_lookup(self):
        """
        Generate a dict to turn punctuation into a token.
        :return: Tokenize dictionary where the key is the punctuation and the value is the token
        """
        token_dict = {}
        token_dict['.'] = '||Period||'
        token_dict[','] = '||Comma||'
        token_dict['"'] = '||Quotation_Mark||'
        token_dict[';'] = '||Semicolon||'
        token_dict['?'] = '||Question_Mark||'
        token_dict['!'] = '||Exclamation_Mark||'
        token_dict['('] = '||Left_Parentheses||'
        token_dict[')'] = '||Right_Parentheses||'
        token_dict['--'] = '||Dash||'
        token_dict['\n'] = '||Return||'

        return token_dict

    def create_lookup_tables(self, text):
        """
        Create lookup tables for vocabulary
        :param text: The text of tv scripts split into words
        :return: A tuple of dicts (vocab_to_int, int_to_vocab)
        """
        vocab_to_int = {word: i for i, word in enumerate(set(text), 0)}
        int_to_vocab = {i: word for i, word in enumerate(set(text), 0)}

        return vocab_to_int, int_to_vocab

    def load_data(self):
        """
        Load Dataset from File
        """
        input_file = os.path.join(self.dataset_path)
        with open(input_file, "r") as f:
            data = f.read()

        return data

    def preprocess_and_save_data(self):
        """
        Preprocess Text Data
        """
        text = self.load_data()

        # Ignore notice, since we don't use it for analysing the data
        text = text[81:]

        token_dict = self.token_lookup()
        for key, token in token_dict.items():
            text = text.replace(key, ' {} '.format(token))

        text = text.lower()
        text = text.split()

        vocab_to_int, int_to_vocab = self.create_lookup_tables(text)
        int_text = [vocab_to_int[word] for word in text]
        pickle.dump((int_text, vocab_to_int, int_to_vocab, token_dict), open('preprocess.p', 'wb'))

        return pickle.load(open('preprocess.p', mode='rb'))

    def load_preprocess(self):
        """
        Load the Preprocessed Training data and return them in batches of <batch_size> or less
        """
        return pickle.load(open('preprocess.p', mode='rb'))

    def save_params(self, params):
        """
        Save parameters to file
        """
        pickle.dump(params, open('params.p', 'wb'))

    def load_params(self):
        """
        Load parameters from file
        """
        return pickle.load(open('params.p', mode='rb'))