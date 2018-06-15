# -*- encoding: utf-8 -*-
from __future__ import division

import argparse
import re
import sys

from collections import (
    Counter,
    defaultdict as deft
)


HELP_FREQ = 'Maximum frequency of any item to be used as a node in the tree.'
HELP_PROB = 'Maximum probability of any item to be used as a node in the tree.'
HELP_SIZE = 'Minimum size for a tree branch to be returned.'
HELP_FORMAT = """Output format:
\'xy\' = input strings + item trees;
\'yx\' = item trees + input strings;
\'x\' = input strings;
\'y\' = item trees"""
HELP_SORT = 'Disable automatic sorting by item tree.'
HELP_PREPROC = 'Data preprocessing: word_normal | whitespace | non_alpha | non_alnum'  # sent_tokenize, word_normal, word_tokenize
NON_ALPHA = re.compile(u'[A-Za-z]+')
NON_ALNUM = re.compile(u'[A-Za-z0-9]+')
WORD_NORMAL = re.compile(u'[A-Za-z0-9,\.;:\(\)\-\'\"]+')
WORD_SEPARATOR = re.compile(u'[\.;:\(\)\-\'\"]')

DEFAULT_PROB = 0.5
DEFAULT_FREQ = 2


class ItemTree:
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



def config_display(args):
    return '<ItemTree input_file="%s" <max_freq=%d max_prob=%f min_size=%d> <format=%s nosort=%s preproc=%s>>\n' % (
        args.input_file, args.max_freq, args.max_prob,
        args.min_size, args.format, args.nosort, args.preproc
    )

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file')
    parser.add_argument(
        '-f', '--max_freq',
        type=int, default=2,
        help=HELP_FREQ
    )
    parser.add_argument(
        '-p', '--max_prob',
        type=float, default=0.5,
        help=HELP_PROB
    )
    parser.add_argument(
        '-s', '--min_size',
        type=int, default=2,
        help=HELP_SIZE
    )
    parser.add_argument(
        '--format',
        default='xy',
        help=HELP_FORMAT
    )
    parser.add_argument(
        '--nosort',
        action='store_true',
        default=False,
        help=HELP_SORT
    )
    parser.add_argument(
        '--preproc',
        default='non_alnum',
        help=HELP_PREPROC
    )
    args = parser.parse_args()
    if args.max_prob != DEFAULT_PROB \
    and args.max_freq == DEFAULT_FREQ:
        return args, args.max_prob
    else:
        return args, args.max_freq
        


def re_tokenize(words):
    tokens = []
    token = ''
    for w in words:
        for char in w:
            if WORD_SEPARATOR.match(char):
                if token:
                    tokens.append(token)
                tokens.append(char)
                token = ''
            elif char == ' ' and token:
                tokens.append(token)
                token = ''
            else:
                token += char
        if token:        
            tokens.append(token)
            token = ''
    return tokens
    

def preproc(args):
    lines = []
    with open(args.input_file, 'rb') as rd:
        for l in rd:
            lines.append(l.decode('utf-8').strip())

    #   word_normal | whitespace | non_alpha | non_alnum'  # sent_tokenize, word_normal, word_tokenize
    if args.preproc == 'whitespace':
        return [line.split() for line in lines]
    elif args.preproc == 'non_alpha':
        return [NON_ALPHA.findall(line) for line in lines]
    elif args.preproc == 'non_alnum':
        return [NON_ALNUM.findall(line) for line in lines]
    elif args.preproc == 'word_normal':
        return [
            [w.lower() for w in re_tokenize(WORD_NORMAL.findall(line))]
            for line in lines
        ]
    return lines


if __name__ == '__main__':

    args, max_freq = get_args()
    
    if args.input_file:
        
        sys.stderr.write(config_display(args))
        
        tokenized = preproc(args)
        
        it = ItemTree(
            max_freq=max_freq,
            min_size=args.min_size,
            sorted=not args.nosort,
            format=args.format
        )

        clusters = it(tokenized)

        for x in clusters:
            print x

    else:
        exit('NotImplementedException')
        rd = sys.stdin
    


