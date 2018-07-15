# -*- encoding: utf-8 -*-
from __future__ import division

import argparse
import csv
import random
import re
import sys
from tqdm import tqdm

from collections import (
    Counter,
    defaultdict as deft
)

from copy import deepcopy as cp


HELP_FREQ = 'Maximum frequency of any item to be used as a node in the tree.'
HELP_MIN_FREQ = 'Minimum frequency of any item to be used as a node in the tree.'
HELP_PROB = 'Maximum probability of any item to be used as a node in the tree.'
HELP_MIN_PROB = 'Minimum probability of any item to be used as a node in the tree.'
HELP_SIZE = 'Minimum size for a tree branch to be returned.'
HELP_MAXSIZE = 'Maximum size for a tree branch to be returned.'
HELP_FORMAT = """Output format:
\'xy\' = input strings + item trees;
\'yx\' = item trees + input strings;
\'x\' = input strings;
\'y\' = item trees"""
HELP_SORT = 'Disable automatic sorting by item tree.'
HELP_PREPROC = 'Data preprocessing: word_normal | whitespace | non_alpha | non_alnum'  # sent_tokenize, word_normal, word_tokenize
HELP_ITEMTEXT = 'Return tree items as text instead of text/None tuples.'

NON_ALPHA = re.compile(u'[A-Za-z]+')
NON_ALNUM = re.compile(u'[A-Za-z0-9]+')
WORD_NORMAL = re.compile(u'[A-Za-z0-9,\.;:\(\)\-\'\"]+')
WORD_SEPARATOR = re.compile(u'[\.;:\(\)\-\'\"]')

DEFAULT_PROB = 0.5
DEFAULT_FREQ = 1000
DEFAULT_MINPROB = 0.0
DEFAULT_MINFREQ = 2


class ItemTree:
    def __init__(
        self,
        min_size=1,
        max_size=1,
        max_freq=0.5,
        min_freq=2,
        sorted=True,
        format='yx',
        itemtext=False
    ):
        self.min_size = min_size
        self.max_size = max_size
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.sorted = sorted
        self.format = format
        self.itemtext = itemtext
        self.cross_cluster_penalty = dict([])
        self.is_feature = deft(bool)
    
    def __call__(self, rX, X):
        clusters = [(True, [], X)]
        while True:
            if not [status for status, _, _ in clusters if status]:
                break
            _clusters = []
            self.__update_cross_cluster_penalties(clusters)
            for cluster in tqdm(clusters):
                _clusters += self.__split(cluster)
            clusters = _clusters
        return self.__to_out(rX, X, clusters)
    
    def __update_cross_cluster_penalties(self, clusters):
        if self.cross_cluster_penalty.keys() \
        and random.random() >= 0.1:
            return
        n = 100
        V = set([])
        penalties = deft(set)
        if len(clusters) > n:
            _clusters = random.sample(clusters, n)
        else:
            _clusters = clusters
            n = len(clusters)
        for i, (_, _, X) in enumerate(_clusters):
            for x in X:
                for w in x:
                    penalties[w].add(i)
        self.cross_cluster_penalty = {
            feat: len(feat_clusters) / n
            for feat, feat_clusters in penalties.items()
        }
    
    def __init_features(self, history, X, F):
        if history:
            return
        max_freq = self.__get_max_freq(X)
        min_freq = self.__get_min_freq(X)
        for feat, freq in F.most_common():
            if freq >= min_freq and freq <= max_freq:
                self.is_feature[feat] = True
    
    def __split(self, cluster):
        status, history, X = cluster
        if not status:
            return [cluster]
        elif len(X) <= self.min_size:
            return [(False, history, X)]
        features_to_deduct = set(history)
        over_max_size = True
        while over_max_size:
            F, I = self.__count(X, features_to_deduct)
            self.__init_features(history, X, F)
            bf, clusters = self.__divide(cluster, F, I)
            if not bf:
                over_max_size = False
            elif len(clusters[0]) > self.max_size \
            or len(clusters[1]) > self.max_size:
                features_to_deduct.add(bf)
            else:
                over_max_size = False
        return clusters
    
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
        best_feat = None
        feat_ranks = []
        half = int(len(X) / 2)
        for feat, freq in F.most_common():
            if not self.is_feature[feat]:
                continue
            elif len(I[feat]) < self.min_size:
                break
            specif = len(X) - len(I[feat])
            if specif < half:
                _feat_rank = half - specif
            else:
                _feat_rank = specif - half
            try:
                decimal_rank = _feat_rank / 100
                cross_cluster_penalty = (1 - self.cross_cluster_penalty[feat])
                feat_rank = (
                    decimal_rank * cross_cluster_penalty,
                    feat
                )
            except KeyError:
                feat_rank = (
                    _feat_rank / 100, feat
                )
            feat_ranks.append(feat_rank)
        feat_ranks.sort()
        feat_ranks = [f for r, f in feat_ranks if r >= 0]
        if feat_ranks:
            best_feat = feat_ranks[0]
            a, b = self.__a_or_b(history, X, I, best_feat)
            return best_feat, [a, b]
        else:
            return None, [(False, history, X)]
    
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
    
    def __get_min_freq(self, X):
        if isinstance(self.min_freq, float):
            return len(X) * self.min_freq
        else:
            return self.min_freq
    
    def __to_out(self, rX, X, clusters):
        positions_by_element = deft(set)
        for i, x in enumerate(X):
            positions_by_element[tuple(x)].add(i)
        original_by_position = {i: x for i, x in enumerate(rX)}
        out = self.__flatten_and_add_initial_positions(
            clusters, positions_by_element, original_by_position
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
    
    def __flatten_and_add_initial_positions(
        self, clusters, positions_by_element, original_by_position
    ):
        out = dict([])
        c = 0
        for cl in clusters:
            _, history, elements = cl
            y = self.__make_itemtext(history)
            for e in elements:
                for i in positions_by_element[tuple(e)]:
                    result = (y, i, original_by_position[i])
                    try:
                        out[result]
                    except Exception:
                        out[result] = c
                        c += 1
        return [x for x, y in sorted(out.items(), key=lambda (a, b): b)]

    def __make_itemtext(self, history):
        if not [node for node in history if node]:
            if self.itemtext:
                return '*'
            else:
                return (None, )
        if self.itemtext:
            y = []
            for node in history:
                if not node:
                    y.append('*')
                else:
                    y.append(node)
            return '/'.join(y)
        else:
            return tuple(history)


def config_display(args):
    return '<ItemTree input_file=%s <max_freq=%d max_prob=%f min_freq=%d min_prob=%f min_size=%d max_size=%d> <output=%s nosort=%s preproc=%s itemtext=%s>>\n' % (
        args.input_file,
        args.max_freq, args.max_prob, args.min_freq, args.min_prob,
        args.min_size, args.max_size,
        args.output, args.nosort, args.preproc,
        args.itemtext
    )


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', nargs='?')
    parser.add_argument('output_file', nargs='?')
    parser.add_argument(
        '--max_freq',
        type=int, default=1000,
        help=HELP_FREQ
    )
    parser.add_argument(
        '--min_freq',
        type=int, default=2,
        help=HELP_MIN_FREQ
    )
    parser.add_argument(
        '--max_prob',
        type=float, default=0.5,
        help=HELP_PROB
    )
    parser.add_argument(
        '--min_prob',
        type=float, default=0.0,
        help=HELP_MIN_PROB
    )
    parser.add_argument(
        '--min_size',
        type=int, default=2,
        help=HELP_SIZE
    )
    parser.add_argument(
        '--max_size',
        type=int, default=50,
        help=HELP_MAXSIZE
    )
    parser.add_argument(
        '--output',
        default='xy',
        help=HELP_FORMAT
    )
    parser.add_argument(
        '--itemtext',
        action='store_true',
        #default=False,
        help=HELP_ITEMTEXT
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
        maxim = args.max_prob
    else:
        maxim = args.max_freq

    if args.min_prob != DEFAULT_MINPROB \
    and args.min_freq == DEFAULT_MINFREQ:
        minim = args.min_prob
    else:
        minim = args.min_freq

    if not args.output_file or sys.stdin:
        args.itemtext = True

    return args, maxim, minim
        


def re_tokenize(words):
    tokens = []
    token = ''
    for w in words:
        for char in w:
            if WORD_SEPARATOR.match(char):
                if token:
                    tokens.append(token)
                #tokens.append(char)
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

    if args.input_file:
        lines = []
        with open(args.input_file, 'rb') as rd:
            for l in rd:
                lines.append(l.decode('utf-8').strip())
    else:
        lines = [l.decode('utf-8').strip() for l in sys.stdin]

    #   word_normal | whitespace | non_alpha | non_alnum'  # sent_tokenize, word_normal, word_tokenize
    if args.preproc == 'whitespace':
        return lines, [line.split() for line in lines]
    elif args.preproc == 'non_alpha':
        return lines, [NON_ALPHA.findall(line) for line in lines]
    elif args.preproc == 'non_alnum':
        return lines, [NON_ALNUM.findall(line) for line in lines]
    elif args.preproc == 'word_normal':
        return lines, [
            [w.lower() for w in re_tokenize(WORD_NORMAL.findall(line))]
            for line in lines
        ]
    return lines, lines


def write(args, clusters):

    def save(clusters, out):
        wrt = open(out, 'wb')
        wrtr = csv.writer(wrt, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        for row in clusters:
            if isinstance(row, basestring):
                row = [row]
            wrtr.writerow([field.encode('utf-8') for field in row])
        wrt.close()
    
    def display(clusters):
        for cl in clusters:
            if isinstance(cl, basestring):
                cl_txt = [cl.encode('utf-8')]
            else:
                cl_txt = [cl[0].encode('utf-8'), cl[1].encode('utf-8')]
            sys.stdout.write('%s\n' % "\t".join(cl_txt))

    if args.output_file:
        save(clusters, args.output_file)
    else:
        display(clusters)


if __name__ == '__main__':

    args, maxim, minim = get_args()

    sys.stderr.write(config_display(args))
    
    raw, tokenized = preproc(args)

    it = ItemTree(
        max_freq=maxim,
        min_freq=minim,
        min_size=args.min_size,
        max_size=args.max_size,
        sorted=not args.nosort,
        format=args.output,
        itemtext=args.itemtext
    )

    clusters = it(raw, tokenized)

    write(args, clusters)
