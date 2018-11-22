import sys
from perceptron_pos_tagger import Perceptron_POS_Tagger
from data_structures import Sentence
from sparse_vector import Vector


def read_in_gold_data(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [[tup.split('_') for tup in line.split()] for line in lines]
        sents = [Sentence(line) for line in lines]

    return sents


def read_in_plain_data(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.split() for line in lines]
        sents = [Sentence(line) for line in lines]

    return sents


def output_auto_data(auto_data, plain_data, filename):
    ''' According to the data structure you used for "auto_data",
        write code here to output your auto tagged data into a file,
        using the same format as the provided gold data (i.e. word_pos word_pos ...).
    '''
    with open(filename, 'a') as the_file:
        for i in range(len(auto_data)):
            tagged_sent = ''
            for tup in zip(plain_data[i].snt, auto_data[i]):
                tagged_sent += (tup[0] + '_' + tup[1] + ' ')
            tagged_sent = (tagged_sent.strip()) + '\n'
            the_file.write(tagged_sent)


if __name__ == '__main__':

    # Run python train_test_tagger.py train/ptb_02-21.tagged dev/ptb_22.tagged dev/ptb_22.snt test/ptb_23.snt to train & test your tagger
    train_file = sys.argv[1]
    gold_dev_file = sys.argv[2]
    plain_dev_file = sys.argv[3]
    test_file = sys.argv[4]

    # Read in data
    train_data = read_in_gold_data(train_file)
    gold_dev_data = read_in_gold_data(gold_dev_file)
    plain_dev_data = read_in_plain_data(plain_dev_file)
    test_data = read_in_plain_data(test_file)

    # Train your tagger
    my_tagger = Perceptron_POS_Tagger()
    alpha, average_alpha = my_tagger.train(train_data, gold_dev_data, plain_dev_data, 1)

    # Apply your tagger on dev & test data
    auto_dev_data = []
    for sentence in plain_dev_data:
        auto = my_tagger.tag(sentence, average_alpha)
        auto_dev_data.append(auto)

    auto_test_data = []
    for sentence in test_data:
        auto = my_tagger.tag(sentence, average_alpha)
        auto_test_data.append(auto)

    # auto_dev_data = my_tagger.tag(plain_dev_data)
    # auto_test_data = my_tagger.tag(test_data)

    # Outpur your auto tagged data
    output_auto_data(auto_dev_data, plain_dev_data, 'iter_3_train_all_auto_dev.txt')
    output_auto_data(auto_test_data, test_data, 'iter_3_train_all_auto_test.txt')
