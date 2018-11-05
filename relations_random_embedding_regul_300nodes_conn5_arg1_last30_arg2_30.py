import tensorflow as tf
import numpy as np
import codecs
import json


def main(train_x_list, train_y_list, dev_x_list, dev_y_list, test_x_list, test_y_list, vocab_size):
    word_embed_size = 300
    hidden_size = 300
    output_size = 21
    batch_size = 32
    max_sentence_len = len(train_x_list[0])
    print("max sentence len:", max_sentence_len)

    # 1. build a feed-forward neural network architecture
    #       (symbolic programming)
    print('\nbuilding the neural network ...')

    embeddings = tf.Variable(tf.random_uniform((vocab_size, word_embed_size), -1, 1))

    W1 = tf.Variable(tf.random_uniform((word_embed_size * max_sentence_len, hidden_size), -1, 1))
    b1 = tf.Variable(tf.zeros((1, hidden_size)))
    W2 = tf.Variable(tf.random_uniform((hidden_size, output_size), -1, 1))
    b2 = tf.Variable(tf.zeros((1, output_size)))

    x = tf.placeholder(tf.int32, (None, max_sentence_len))
    embed = tf.reshape(
        tf.nn.embedding_lookup(embeddings, x),
        [-1, word_embed_size * max_sentence_len])
    # logit = tf.matmul(tf.nn.relu(tf.matmul(embed, W1) + b1), W2) + b2
    logit = tf.matmul(tf.nn.relu(tf.matmul(embed, W1) + b1), W2) + b2

    # predicted = tf.nn.softmax(tf.matmul(tf.nn.relu(tf.matmul(embed, W1) + b1), W2) + b2)
    predicted = tf.nn.softmax(logit)
    # 2. run forward computation
    #       (run a session)
    # print('\nforward computation ...\n')
    #
    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    #
    # print(sess.run([predicted], feed_dict = {x:test_x_list}))

    # 3. train the model
    print('\ntraining ...')
    #   3.1 define cross-entropy loss (symbolic)

    gold_y = tf.placeholder(tf.float32, [None, output_size])
    gold = tf.argmax(gold_y, axis=1)
    pred = tf.argmax(logit, axis=1)
    acc  = tf.reduce_mean(tf.cast(tf.equal(gold, pred), tf.float64))
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = gold_y,logits = logit))
    # cross_entropy = - tf.reduce_sum(gold_y * tf.log(predicted), axis = 1)

    # 3.2 define optimizer (symbolic)
    regularizers = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2)
    cross_entropy = tf.reduce_mean(cross_entropy + 0.01 * regularizers)

    learning_rate = 0.0001
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_step = optimizer.minimize(cross_entropy)

    # 3.3 execute training (run session)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    dev_loss_list = []
    for iteration in range(150):
        for i in range(int(len(train_x_list) / batch_size)):
            sess.run(
                train_step,
                feed_dict = {
                    x:train_x_list[i * batch_size:(i + 1) * batch_size],
                    gold_y:train_y_list[i * batch_size:(i + 1) * batch_size]})

        if iteration % 1 == 0:
            train_loss_value, train_acc, train_pred = sess.run([cross_entropy, acc, pred],
                feed_dict = {x:train_x_list, gold_y:train_y_list})
            dev_loss_value, dev_acc, dev_pred = sess.run([cross_entropy, acc, pred],
                feed_dict = {x:dev_x_list, gold_y:dev_y_list})
            print('iter:', iteration, 'loss on train:', train_loss_value, 'train acc:', train_acc, ', loss on dev:', dev_loss_value, "dev acc:", dev_acc, "PRED:", dev_pred)
            #print("PRED:", dev_pred)
            dev_loss_list.append(dev_loss_value)


        if len(dev_loss_list) >= 3 and dev_loss_list[len(dev_loss_list)-1]>dev_loss_list[len(dev_loss_list)-2] and dev_loss_list[len(dev_loss_list)-2]>dev_loss_list[len(dev_loss_list)-3]:
            test_loss_value, test_acc, test_pred = sess.run([cross_entropy, acc, pred],
                feed_dict = {x:test_x_list, gold_y:test_y_list})
            print('stop training \n')
            print('loss on test:', test_loss_value, "ACC:", test_acc)
            break

    # test trained model
    print('\ntesting trained model ...\n')
    results = sess.run([predicted], feed_dict = {x:test_x_list})
    print(results)
    return results[0]

def vocab_sense(rows):
    sense_list = []
    sense = {}
    vocab_list = []
    vocab_set = set()
    vocab = {}

    i = 0
    for x in rows:
        t = [0] * 21
        if x['Sense'][0] not in sense_list:
            sense_list.append(x['Sense'][0])
            t[i] = 1
            sense[x['Sense'][0]] = t
            i += 1
        for word in (x['Arg1']['RawText'] + ' ' + x['Connective']['RawText'] + ' ' + x['Arg2']['RawText']).split():
            if word not in vocab_set:
                vocab_set.add(word)
    vocab = {word: i+1 for i, word in enumerate(vocab_set)}
    vocab['UNK'] = 0
    print("len of vocab:",len(vocab))
    print("len of sense:",len(sense))
    return (vocab, sense_list, sense)

def row2vec(row,vocab):
    vec = []
    arg1 = row['Arg1']['RawText'].lower().split()
    arg2 = row['Arg2']['RawText'].lower().split()
    conn = row['Connective']['RawText'].lower().split()
    if len(conn) < 5:
        vec += [0] * (5 - len(conn))
    for word in conn[:5]:
        if word in vocab:
            vec.append(vocab[word])
        else:
            vec += [0]
    if len(arg1) < 30:
        vec += [0] * (30 - len(arg1))
    for word in arg1[-30:]:
        if word in vocab:
            vec.append(vocab[word])
        else:
            vec += [0]
    if len(arg2) < 30:
        vec += [0] * (30 - len(arg2))
    for word in arg2[:30]:
        if word in vocab:
            vec.append(vocab[word])
        else:
            vec += [0]
    return vec


if __name__ == '__main__':
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # read in data
    train_file = codecs.open('train/relations.json',encoding='utf8')
    rows_train = [json.loads(x) for x in train_file]
    dev_file = codecs.open('dev/relations.json',encoding='utf8')
    rows_dev = [json.loads(x) for x in dev_file]
    test_file = codecs.open('test/relations.json',encoding='utf8')
    rows_test = [json.loads(x) for x in test_file]

    vocab,sense_list,sense = vocab_sense(rows_train)

    train_x_list = [row2vec(row,vocab) for row in rows_train]
    train_y_list = [sense[row['Sense'][0]] for row in rows_train]

    dev_x_list = [row2vec(row,vocab) for row in rows_dev]
    dev_y_list = [sense[row['Sense'][0]] for row in rows_dev]

    test_x_list = [row2vec(row,vocab) for row in rows_test]
    test_y_list = [sense[row['Sense'][0]] for row in rows_test]

    # padding short sentences
    # max_len = max([len(row) for row in train_x_list])
    max_len = 30
    train_x_list = np.array(
        [np.array(row + [0] * (max_len - len(row))) if len(row) < max_len else np.array(row[:max_len])
        for row in train_x_list])
    dev_x_list = np.array(
        [np.array(row + [0] * (max_len - len(row))) if len(row) < max_len else np.array(row[:max_len])
        for row in dev_x_list])
    test_x_list = np.array(
        [np.array(row + [0] * (max_len - len(row))) if len(row) < max_len else np.array(row[:max_len])
        for row in test_x_list])

    results = main(train_x_list, train_y_list, dev_x_list, dev_y_list, test_x_list, test_y_list, len(vocab))

    predicted_y_list = []
    for tup in results:
        predicted_y_list.append(sense[sense_list[tup.argmax(axis=0)]])

    # writing the output file in json format
    ind = 0
    for row in rows_test:
        dict = {}
        offset1 = []
        for token in row['Arg1']['TokenList']:
            offset1.append(token[2])
        dict['Arg1'] = {'TokenList': offset1}
        offset2 = []
        for token in row['Arg2']['TokenList']:
            offset1.append(token[2])
        dict['Arg2'] = {'TokenList': offset2}
        dict['Connective'] = row['Connective']
        dict['DocID'] = row['DocID']
        dict['Sense'] = [sense_list[predicted_y_list[ind].index(1)]]
        dict['Type'] = row['Type']
        with open('output_random_embedding_regul_300nodes_conn5_arg1_last30_arg2_30.json', 'a') as outfile:
            json.dump(dict, outfile)
            outfile.write('\n')
        ind += 1


    # calculating accuracy on test
    acc = sum([1 if tup[0] == tup[1] else 0 for tup in zip(test_y_list, predicted_y_list)]) / len(test_y_list)
    print('accuracy:', acc)
