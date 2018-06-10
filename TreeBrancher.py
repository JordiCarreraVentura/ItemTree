# -*- encoding: utf-8 -*-
from __future__ import division

from collections import (
    Counter,
    defaultdict as deft
)


class TagTree:
    def __init__(self, min_size=1, max_freq=1.0, sorted=True, format='yx'):
        self.min_size = min_size
        self.max_freq = max_freq
        self.sorted = sorted
        self.format = format
    
    def __call__(self, X):
        clusters = [(True, [], X)]
        while True:
            if not [status for status, _, _ in clusters if status]:
                break
            _clusters = []
            for cluster in clusters:
                _clusters += self.__split(cluster)
            clusters = _clusters
        return self.__to_out(X, clusters)
    
    def __split(self, cluster):
        status, history, X = cluster
        if not status:
            return [cluster]
        elif len(X) <= self.min_size:
            return [(False, history, X)]
        features_to_deduct = set(history)
        F, I = self.__count(X, features_to_deduct)
        return self.__divide(cluster, F, I)
    
    def __count(self, X, features_to_deduct):
        F = Counter()
        I = deft(list)
        for i, x in enumerate(X):
            all_features = set(x)
            new_features = all_features - features_to_deduct
            for f in new_features:
                F[f] += 1
                I[f].append(i)
        return F, I
    
    def __divide(self, cluster, F, I):
        status, history, X = cluster
        max_freq = self.__get_max_freq(X)
        best_feat = None
        for feat, freq in F.most_common():
            if freq >= max_freq:
                continue
            elif len(I[feat]) < self.min_size:
                break
            best_feat = feat
            break
        if best_feat:
            a, b = self.__a_or_b(history, X, I, best_feat)
            return [a, b]
        else:
            return [(False, history, X)]
    
    def __a_or_b(self, history, X, I, best_feat):
        aX = []
        taken = set([])
        for i in I[best_feat]:
            aX.append(X[i])
            taken.add(i)
        not_taken = set(range(len(X))) - taken
        bX = [X[i] for i in not_taken]
        a = (True, history + [best_feat], aX)
        b = (True, history + [None], bX)
        return a, b
    
    def __get_max_freq(self, X):
        if isinstance(self.max_freq, float):
            return len(X) * self.max_freq
        else:
            return self.max_freq
    
    def __to_out(self, X, clusters):
        position_by_element = {tuple(x): i for i, x in enumerate(X)}
        out = self.__flatten_and_add_initial_positions(
            clusters, position_by_element
        )
        if not self.sorted:
            out.sort(key=lambda x: x[1])
        if self.format == 'x':
            return [e for _, _, e in out]
        elif self.format == 'y':
            return [h for h, _, _ in out]
        elif self.format == 'xy':
            return [(e, h) for h, _, e in out]
        elif self.format == 'yx':
            return [(h, e) for h, _, e in out]
        else:
            exit('FATAL: unrecognized argument for \'format\' parameter.')
    
    def __flatten_and_add_initial_positions(self, clusters, position_by_element):
        out = []
        for cl in clusters:
            _, history, elements = cl
            for e in elements:
                out.append((tuple(history), position_by_element[tuple(e)], e))
        return out
    


if __name__ == '__main__':
    
    r = round
    import random
    from random import shuffle
    tokenized = [
        [1, 2, 3] + [random.randint(4, 1001) for i in range(7)],
        [1, 2, 3] + [random.randint(4, 1001) for i in range(7)],
        [1, 2, 3] + [random.randint(4, 1001) for i in range(7)],
        [1, 2, 3] + [random.randint(4, 1001) for i in range(7)],
        [1, 2, 3] + [random.randint(4, 1001) for i in range(7)],
        [1] + [random.randint(4, 1001) for i in range(9)],
        [1] + [random.randint(4, 1001) for i in range(9)],
        [1] + [random.randint(4, 1001) for i in range(9)],
        [1] + [random.randint(4, 1001) for i in range(9)],
        [2] + [random.randint(4, 1001) for i in range(9)],
        [2] + [random.randint(4, 1001) for i in range(9)],
        [2] + [random.randint(4, 1001) for i in range(9)],
        [2] + [random.randint(4, 1001) for i in range(9)],
        [2] + [random.randint(4, 1001) for i in range(9)],
        [3] + [random.randint(4, 1001) for i in range(9)],
        [3] + [random.randint(4, 1001) for i in range(9)],
        [3] + [random.randint(4, 1001) for i in range(9)],
        [3] + [random.randint(4, 1001) for i in range(9)],
        [3] + [random.randint(4, 1001) for i in range(9)],
        [2, 3] + [random.randint(4, 1001) for i in range(8)],
        [2, 3] + [random.randint(4, 1001) for i in range(8)],
        [2, 3] + [random.randint(4, 1001) for i in range(8)],
        [2, 3] + [random.randint(4, 1001) for i in range(8)],
        [2, 3] + [random.randint(4, 1001) for i in range(8)],
        [1, 3] + [random.randint(4, 1001) for i in range(8)],
        [1, 3] + [random.randint(4, 1001) for i in range(8)],
        [1, 3] + [random.randint(4, 1001) for i in range(8)],
        [1, 3] + [random.randint(4, 1001) for i in range(8)],
        [1, 3] + [random.randint(4, 1001) for i in range(8)],
        [random.randint(4, 1001) for i in range(10)],
        [random.randint(4, 1001) for i in range(10)],
        [random.randint(4, 1001) for i in range(10)],
        [random.randint(4, 1001) for i in range(10)],
        [random.randint(4, 1001) for i in range(10)],
        [random.randint(4, 1001) for i in range(10)],
        [random.randint(4, 1001) for i in range(10)],
        [random.randint(4, 1001) for i in range(10)],
        [random.randint(4, 1001) for i in range(10)],
        [random.randint(4, 1001) for i in range(10)]
    ]
    
#     tokenized = [
#         'a b c'.split(),
#         'a b'.split(),
#         'a'.split(),
#         'a'.split(),
#         'b c'.split(),
#         'b'.split(),
#         'a b'.split(),
#         'a'.split(),
#         'a c'.split(),
#         'b a'.split(),
#     ]
    
#     tokenized = shuffle_many(tokenized)
    counts = Counter()
    for x in tokenized:
        print x
        counts.update(x)
    print counts.most_common(50)
    
    
    tt1 = TagTree(
        max_freq=1.0,
        min_size=2,
        sorted=True,      # sorted or original ordering (the latter only makes sense with
                          # also asking for the Y in the output, see below).
        format='xy'       # xy, yx, x, y
    )
    
    tt2 = TagTree(
        max_freq=1.0,
        min_size=2,
        sorted=False,     # sorted or original ordering (the latter only makes sense with
                          # also asking for the Y in the output, see below).
        format='y'        # xy, yx, x, y
    )
    
    tt3 = TagTree(
        max_freq=1.0,
        min_size=2,
        sorted=True,      # sorted or original ordering (the latter only makes sense with
                          # also asking for the Y in the output, see below).
        format='yx'        # xy, yx, x, y
    )
    
    
    clusters1 = tt1(tokenized)
    clusters2 = tt2(tokenized)
    clusters3 = tt3(tokenized)

    print '========='
    for x in clusters1:
        print x
    print '========='
    for x in clusters2:
        print x
    print '========='
    for x in clusters3:
        print x
    
