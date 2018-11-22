import codecs


class Vector:
    def __init__(self, dic):
        self.v = dic

    def __iadd__(self, other):
        for key in other.v:
            if key in self.v:
                self.v[key] += other.v[key]
            else:
                self.v[key] = other.v[key]
        return self

    def __isub__(self, other):
        for key in other.v:
            if key in self.v:
                self.v[key] -= other.v[key]
            else:
                self.v[key] = -other.v[key]
        return self

    # def __add__(self, other):
    #     result = Vector({})
    #     for key in other.v:
    #         if key in self.v:
    #            if self.v[key] == -other.v[key]:
    #                continue
    #            result.v[key] = self.v[key] + other.v[key]
    #         else:
    #            result.v[key] = other.v[key]
    #
    #     for key in self.v:
    #         if key not in other.v:
    #             result.v[key] = self.v[key]
    #
    #     return result

    def __sub__(self, other):
        result = Vector({})
        for key in other.v:
            if key in self.v:
               if self.v[key] == other.v[key]:
                   continue
               result.v[key] = self.v[key] - other.v[key]
            else:
               result.v[key] = -other.v[key]

        for key in self.v:
            if key not in other.v:
                result.v[key] = self.v[key]

        return result

    def __rmul__(self, other):
        result = Vector({})
        for key in self.v:
            result.v[key] = self.v[key] * other
        return result

    def dot(self, other):
        sum_ = 0
        for key in other.v:
            if key in self.v:
                sum_ += self.v[key] * other.v[key]
        return sum_

    def __str__(self):
        tmp = ''
        for key in self.v:
            tmp += key + '\t' + str(self.v[key])
        return tmp

    def element_wise_square(self):
        result = Vector({})
        for key in self.v:
            result.v[key] = self.v[key] * self.v[key]
        return result

    def save(self, filename):
        with codecs.open(filename, 'w', 'utf-8') as f:
            for key in self.v:
                f.write(key + '\t' + str(self.v[key]) + '\n')

    def load(self, filename):
        with codecs.open(filename, 'r', 'utf-8') as f:
            lines = [line.strip().split('\t') for line in f.readlines()]
            self.v = {line[0]:float(line[-1]) for line in lines}


if __name__ == '__main__':
    ''' Use cases. '''
    feature_vector = Vector({})
    weight_vector = Vector({})

    # uni-gram features
    feature_vector.v['w0=this pos0=DT'] = 1
    feature_vector.v['w0=cat pos0=NOUN'] = 1
    feature_vector.v['w0=is pos0=VBZ'] = 1
    feature_vector.v['w0=black pos0=ADJ'] = 1

    # bi-gram features
    feature_vector.v['w-1_w0=this_cat pos0=NOUN'] = 1
    feature_vector.v['w-1_w0=cat_is pos0=VBZ'] = 1
    feature_vector.v['w-1_w0=is_black pos0=ADJ'] = 1

    # tri-gram features
    feature_vector.v['w-2_w-1_w0=this_cat_is pos0=VBZ'] = 1
    feature_vector.v['w-2_w-1_w0=cat_is_black pos0=ADJ'] = 1

    score = weight_vector.dot(feature_vector)
    print(score)
