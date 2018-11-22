from sparse_vector import Vector
from data_structures import Sentence

class Perceptron_POS_Tagger(object):
    tag_list = ['IN', 'DT', 'NNP', 'CD', 'NN', '``', "''", 'POS', '-LRB-', 'VBN', 'NNS', 'VBP', ',', 'CC', '-RRB-', 'VBD', 'RB', 'TO', '.', 'VBZ', 'NNPS', 'PRP', 'PRP$', 'VB', 'JJ', 'MD', 'VBG', 'RBR', ':', 'WP', 'WDT', 'JJR', 'PDT', 'RBS', 'WRB', 'JJS', '$', 'RP', 'FW', 'EX', 'SYM', '#', 'LS', 'UH', 'WP$']

    def __init__(self):
        ''' Modify if necessary.
        '''

    def tag(self, sentence, alpha):
        ''' Implement the Viterbi decoding algorithm here.
        '''
        z = []
        b_matrix = []
        score_matrix = []
        score_start = []
        start_word = sentence.snt[0]
        for tag in Perceptron_POS_Tagger.tag_list:
            start_feature = Vector(sentence.start_word_feature(start_word, tag))
            score_start.append(alpha.dot(start_feature))
        score_matrix.append(score_start)
        b_matrix.append([0] * len(Perceptron_POS_Tagger.tag_list))
        if len(sentence.snt) >= 4:
            for i in range(1, len(sentence.snt)-2):
                score = []
                word = sentence.snt[i]
                b = []
                for j in range(len(Perceptron_POS_Tagger.tag_list)):
                    score_max = -10000
                    score_max_index = 0
                    for k in range(len(Perceptron_POS_Tagger.tag_list)):
                        feature = Vector(sentence.middle_word_feature(word, Perceptron_POS_Tagger.tag_list[j], Perceptron_POS_Tagger.tag_list[k]))
                        if (score_matrix[i-1][k] + alpha.dot(feature)) > score_max:
                            score_max = score_matrix[i-1][k] + alpha.dot(feature)
                            score_max_index = k
                    score.append(score_max)
                    b.append(score_max_index)
                b_matrix.append(b)
                score_matrix.append(score)
        if len(sentence.snt) >= 3:
            end_word = sentence.snt[-2]
            b = []
            score = []
            for j in range(len(Perceptron_POS_Tagger.tag_list)):
                score_max = -10000
                score_max_index = 0
                for k in range(len(Perceptron_POS_Tagger.tag_list)):
                    feature = Vector(sentence.end_word_feature(end_word, Perceptron_POS_Tagger.tag_list[j], Perceptron_POS_Tagger.tag_list[k]))

                    if (score_matrix[len(sentence.snt)-3][k] + alpha.dot(feature)) > score_max:
                        score_max = score_matrix[len(sentence.snt)-3][k] + alpha.dot(feature)
                        score_max_index = k
                score.append(score_max)
                b.append(score_max_index)
            score_matrix.append(score)
            b_matrix.append(b)

        end_tag_index = score_matrix[-1].index(max(score_matrix[-1]))

        z_index = [end_tag_index]
        z.append(Perceptron_POS_Tagger.tag_list[z_index[-1]])
        for i in range(len(sentence.snt)-2):
            z_index.append(b_matrix[len(sentence.snt)-i-2][z_index[-1]])
            z.append(Perceptron_POS_Tagger.tag_list[z_index[-1]])
        z = list(reversed(z))
        z.append(sentence.snt[-1])
        return z

    def train(self, train_data, gold_dev_data, plain_dev_data, iterations):
        ''' Implement the Perceptron training algorithm here.
        '''
        alpha = Vector({})
        alpha_sum = Vector({})
        print("length of training data:", len(train_data), "\nstart training...")
        for t in range(iterations):
            print("\niter:", t, "\n")
            for i in range(len(train_data)):
                z = self.tag(Sentence(self.raw_sentence(train_data[i].snt)), alpha)
                curr_tags = Sentence(self.comb_word_tag(self.raw_sentence(train_data[i].snt), z))
                local_features_list, global_feature_z = curr_tags.features()
                true_local_features, true_global = train_data[i].features()
                if z != self.true_tag(train_data[i].snt):
                    alpha = Vector.__iadd__(alpha, Vector.__sub__(true_global, global_feature_z))
                alpha_sum = Vector.__iadd__(alpha_sum, alpha)
                if i == 499:
                    print("training size:", i + 1, "acc on dev with regular alpha:", self.acc_dev(gold_dev_data, plain_dev_data, alpha))
                    print("acc with avg alpha:", self.acc_dev(gold_dev_data, plain_dev_data, self.average_alpha(alpha_sum, t, train_data, i)))
                if i == 999:
                    print("training size:", i + 1, "acc on dev with regular alpha:", self.acc_dev(gold_dev_data, plain_dev_data, alpha))
                    print("acc with avg alpha:", self.acc_dev(gold_dev_data, plain_dev_data, self.average_alpha(alpha_sum, t, train_data, i)))
                if i == 9999:
                    print("training size:", i + 1, "acc on dev with regular alpha:", self.acc_dev(gold_dev_data, plain_dev_data, alpha))
                    print("acc with avg alpha:", self.acc_dev(gold_dev_data, plain_dev_data, self.average_alpha(alpha_sum, t, train_data, i)))
                if i == 24999:
                    print("training size:", i + 1, "acc on dev with regular alpha:", self.acc_dev(gold_dev_data, plain_dev_data, alpha))
                    print("acc with avg alpha:", self.acc_dev(gold_dev_data, plain_dev_data, self.average_alpha(alpha_sum, t, train_data, i)))
                if i == len(train_data) - 1:
                    final_avg_alpha = self.average_alpha(alpha_sum, t, train_data, i)
                    print("training size:", i + 1, "acc on dev with regular alpha:", self.acc_dev(gold_dev_data, plain_dev_data, alpha))
                    print("acc with avg alpha:", self.acc_dev(gold_dev_data, plain_dev_data, final_avg_alpha))
                    print("\n")
            # acc_dev = self.acc_dev(gold_dev_data, plain_dev_data, alpha)
            # print("iter", t, ": acc on dev with regular alpha:", acc_dev)

        # print("calculate acc with average alpha...")
        # average_alpha = Vector.__rmul__(average_alpha, 1 / (len(train_data) * iterations))
        # acc_avg = self.acc_dev(gold_dev_data, plain_dev_data, average_alpha)
        # print("acc on dev with average alpha:", acc_avg)
        return (alpha, final_avg_alpha)

    def true_tag(self, gold_sentence):
        tags = [item[1] for item in gold_sentence]
        return tags

    def comb_word_tag(self, sentence, tags):
        return [tup for tup in zip(sentence, tags)]

    def raw_sentence(self, gold_sentence):
        raw = []
        for item in gold_sentence:
            raw.append(item[0])
        return raw

    def acc_dev(self, gold_dev_data, plain_dev_data, alpha):
        correct = 0.0
        total = 0.0
        for k in range(len(plain_dev_data)):
            pred_tags = self.tag(plain_dev_data[k], alpha)
            correct += sum(true_tag == pred_tag for true_tag, pred_tag in zip(self.true_tag(gold_dev_data[k].snt), pred_tags))
            total += len(gold_dev_data[k].snt)
        return correct / total

    def average_alpha(self, alpha_sum, iter, train_data, i):
        avg_alpha = Vector.__rmul__(alpha_sum, 1 / (len(train_data) * iter + i + 1))
        return avg_alpha
