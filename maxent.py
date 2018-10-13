# -*- mode: Python; coding: utf-8 -*-

from classifier import Classifier
import numpy as np
import math
import scipy.misc
from random import shuffle, seed
import matplotlib.pyplot as plt



class MaxEnt(Classifier):

    def get_model(self): return None

    def set_model(self, model): pass

    model = property(get_model, set_model)

    def train(self, instances, dev_instances=None, learning_rate=0.0001):
        """Construct a statistical model from labeled instances."""
        self.train_sgd(instances, dev_instances, learning_rate, 30)

    def feature_labels(self,instances):
        self.labels = []
        self.dict = {}
        dict1 = {}

        stop_words = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]

        for instance in instances:
            if instance.label not in self.labels:
                self.labels.append(instance.label)

            if len(instance.features()) == 2:
                word = instance.features()[0] + instance.features()[1]
                if word not in self.dict:
                    self.dict[word] = 1
            else:
                for word in instance.features():
                    word = word.lower()
                    if word not in dict1:
                        dict1[word] = 1
                    else:
                        dict1[word] += 1
        for key in dict1:
            if dict1[key] > 10 and key not in stop_words:
                self.dict[key] = dict1[key]

        self.dict["Bias"] = 1
        if len(dict1) > 1:
            print("features shown:", len(dict1))
        print("dict size:", len(self.dict))
        print("labels size:", len(self.labels))

        self.lamda = np.zeros((len(self.labels),len(self.dict)))

    def featurization(self,instance):
        feature_vector = np.zeros((1,len(self.dict)))

        if len(instance.features()) == 2:
            word = instance.features()[0] + instance.features()[1]
            if word in self.dict:
                feature_vector[0][list(self.dict).index(word)] = 1
        else:
            for word in instance.features():
                if word in self.dict:
                    feature_vector[0][list(self.dict).index(word)] = 1

        feature_vector[0][len(self.dict) - 1] = 1
        return feature_vector[0]

    def feature_matrix(self,label,instance):
        f = np.zeros((len(self.labels),len(self.dict)))
        if len(instance.feature_vector) == 0:
            instance.feature_vector = self.featurization(instance)
        f[self.labels.index(label)] = instance.feature_vector
        return f

    def posterior(self,label,instance):
        unnormalized_score1 = np.dot(self.lamda.flatten(),self.feature_matrix(label,instance).flatten())
        unnormalized_score2 = []
        for y in self.labels:
            unnormalized_score2.append(np.dot(self.lamda.flatten(),self.feature_matrix(y,instance).flatten()))
        return math.exp(unnormalized_score1 - scipy.misc.logsumexp(unnormalized_score2))

    def negative_loglikelihood(self,minibatch):
        loglikelihood = 0
        for instance in minibatch:
            loglikelihood += np.dot(self.lamda.flatten(),self.feature_matrix(instance.label,instance).flatten())

            unnormalized_score2 = []
            for y in self.labels:
                unnormalized_score2.append(np.dot(self.lamda.flatten(),self.feature_matrix(y,instance).flatten()))

            loglikelihood -= scipy.misc.logsumexp(unnormalized_score2)
        return -loglikelihood

    def chop_up(self,training_set,batch_size):
        minibatches = []
        for i in range(0,len(training_set)//batch_size):
            minibatches.append(training_set[i*batch_size:(i+1)*batch_size])
        if len(training_set)%batch_size != 0:
            minibatches.append(training_set[len(training_set)//batch_size * batch_size:])
        return minibatches

    def compute_gradient(self,minibatch):
        gradient = np.zeros((len(self.labels),len(self.dict)))
        for instance in minibatch:
            gradient += self.feature_matrix(instance.label,instance)
            for y in self.labels:
                gradient -= (self.posterior(y,instance) * self.feature_matrix(y,instance))
        return gradient

    def train_sgd(self, train_instances, dev_instances, learning_rate, batch_size):
        """Train MaxEnt model with Mini-batch Stochastic Gradient
        """
        self.feature_labels(train_instances)
        print("batch_size:", batch_size)
        iter = 0
        lamda_list = [0]
        dev_loss_list = [0]
        x_array = []
        y_array = []
        while iter < 10:
            iter += 1
            minibatches = self.chop_up(train_instances,batch_size)
            x = 0

            for minibatch in minibatches:
                delta_lamda = self.compute_gradient(minibatch)
                self.lamda += (delta_lamda * learning_rate)


            #     if iter == 5:
            #         x += len(minibatch)
            #         x_array.append(x)
            #         correct_number = 0
            #         for instance in dev_instances:
            #             if self.classify(instance) == instance.label:
            #                correct_number += 1
            #         y_array.append(correct_number / len(dev_instances))
            # if iter == 5:
            #     plt.plot(x_array,y_array)
            #     plt.ylabel("accuracy")
            #     plt.xlabel("batch size: 30, number of datapoints used to compute gradient")
            #     plt.show()



            lamda_list.append(self.lamda)
            dev_loss = self.negative_loglikelihood(dev_instances)
            dev_loss_list.append(dev_loss)
            acc_number = 0
            for instance in dev_instances:
                if self.classify(instance) == instance.label:
                   acc_number += 1

            print("iter", iter, ", train loss:", self.negative_loglikelihood(train_instances),
                ", dev loss", dev_loss, "dev acc:", acc_number/len(dev_instances))
            if iter > 4 and dev_loss_list[iter] > dev_loss_list[iter-1] and dev_loss_list[iter-1] > dev_loss_list[iter-2]:
                self.lamda = lamda_list[iter-2]
                break
            shuffle(train_instances)


    def classify(self, instance):
        p = 0
        for i in range(0,len(self.labels)):
            if self.posterior(self.labels[i],instance) > p:
                p = self.posterior(self.labels[i],instance)
                result = self.labels[i]
        return result
