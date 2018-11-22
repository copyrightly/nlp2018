from sparse_vector import Vector

class Sentence(object):
    suffix2_list = ['fy', 'ly', 'al', 'ed', 'es']
    suffix3_list = ['ise', 'ize', 'ful', 'ess', 'ism', 'ist', 'ish', 'old', 'lty', 'ant', 'ing']
    suffix4_list = ['able', 'ible', 'hood', 'ment', 'like', 'tion', 'logy']

    def __init__(self, snt):
        ''' Modify if necessary.
        '''
        self.snt = snt


    def features(self):
        ''' Implement your features here.
        '''
        local_features_list = []
        global_feature = Vector({})
        if len(self.snt[0]) == 2:
            start_word, start_suffix, start_tag = self.word_suffix_tag(self.snt[0])
            start_feature = self.start_word_feature(start_word, start_tag)
            local_features_list.append(start_feature)
            if len(self.snt) >= 4:
                for i in range(1, len(self.snt)-2):
                    item = self.snt[i]
                    item_prev = self.snt[i-1]
                    word, suffix, tag = self.word_suffix_tag(item)
                    word_prev, suffix_prev, tag_prev = self.word_suffix_tag(item_prev)
                    local_feature = self.middle_word_feature(word, tag, tag_prev)
                    local_features_list.append(local_feature)
            if len(self.snt) >= 3:
                end_word, end_suffix, end_tag = self.word_suffix_tag(self.snt[-2])
                word_prev, suffix_prev, tag_prev = self.word_suffix_tag(self.snt[-3])
                end_feature = self.end_word_feature(end_word, end_tag, tag_prev)
                local_features_list.append(end_feature)
            for feature in local_features_list:
                feature = Vector(feature)
                global_feature = Vector.__iadd__(global_feature, feature)
        return (local_features_list, global_feature)

    def start_word_feature(self, word, tag):
        start_feature = {}
        start_suffix = self.get_suffix(word)
        start_feature['w0=' + word + ' pos0=' + tag] = 1
        start_feature['suffix=' + start_suffix] = 1
        start_feature['START pos=' + tag] = 1
        return start_feature

    def middle_word_feature(self, word, tag, tag_prev):
        local_feature = {}
        suffix = self.get_suffix(word)
        local_feature['w0=' + word + ' pos0=' + tag] = 1
        local_feature['suffix=' + suffix] = 1
        local_feature['pos-1=' + tag_prev + ' pos0=' + tag] = 1
        return local_feature

    def end_word_feature(self, word, tag, tag_prev):
        end_feature = {}
        end_suffix = self.get_suffix(word)
        end_feature['w0=' + word + ' pos0=' + tag] = 1
        end_feature['suffix=' + end_suffix] = 1
        end_feature['pos-1=' + tag_prev + ' pos0=' + tag] = 1
        end_feature['END pos=' + tag] = 1
        return end_feature

    def get_suffix(self, word):
        suffix = ''
        if len(word) > 1:
            if word[-2:] in Sentence.suffix2_list:
                suffix = word[-2:]
            elif word[-3:] in Sentence.suffix3_list:
                suffix = word[-3:]
            elif word[-4:] in Sentence.suffix4_list:
                suffix = word[-4:]
        return suffix

    def word_suffix_tag(self, item):
        word = item[0]
        tag = item[1]
        suffix = self.get_suffix(word)
        return (word, suffix, tag)
