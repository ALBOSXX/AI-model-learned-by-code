

import numpy as np
from data import raw_test_texts


def text_to_ids(text, mapping):
    return [mapping.get(word, mapping["<UNK>"]) for word in text.lower().split()]


def to_one_hot(id_list, vocab_size):

    one_hot = np.zeros((len(id_list), vocab_size))
    for i, idx in enumerate(id_list):
        one_hot[i, idx] = 1.0
    return one_hot


if __name__ == "__main__":
    unique_words = set()
    for text in raw_test_texts:
        for word in text.lower().split():
            unique_words.add(word)


    word_list = ["<UNK>"] + list(unique_words)


    word_to_id = {word: i for i, word in enumerate(word_list)}
    id_to_word = {i: word for i, word in enumerate(word_list)}

    vocab_size = len(word_to_id)
    print(f"{vocab_size}")


    dataset_ids = [text_to_ids(text, word_to_id) for text in raw_test_texts]



    input_matrix = to_one_hot(dataset_ids[0], vocab_size)

    hidden_size = 16


    W1 = np.random.randn(vocab_size, hidden_size) * 0.01

    hidden_layer_output = np.dot(input_matrix, W1)

    print("Result of passing through the first layer:\n", hidden_layer_output)
    print("The size of the resulting matrix:", hidden_layer_output.shape)

    W2 = np.random.randn(hidden_size, vocab_size) * 0.01

    w2_layer_output = np.dot(hidden_layer_output, W2)

    print("Result of passing through the second layer:\n", w2_layer_output)
    print("The size of the resulting matrix:", w2_layer_output.shape)




