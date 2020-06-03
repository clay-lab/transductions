# adapted from generator.py from Bob Frank
from nltk import PCFG, Tree
from nltk import nonterminals, Nonterminal, Production
import itertools
import sys
import random
import csv


def generate(grammar, start=None, depth=None):
    """
    Generates an iterator of all sentences from a CFG.

    :param grammar: The Grammar used to generate sentences.
    :param start: The Nonterminal from which to start generate sentences.
    :param depth: The maximal depth of the generated tree.
    :param n: The maximum number of sentences to return.
    :return: An iterator of lists of terminal tokens.
    """
    if not start:
        start = grammar.start()
    if depth is None:
        depth = sys.maxsize
    items = [start]
    tree = _generate(grammar,items, depth)
    sentence = ' '.join(tree[0].leaves())
    # condition for length of sentence. Sentence must be less than 160 (about 25-30 words)
    if len(sentence) < 15:   
        tree = tree
    else:
        tree[0] = generate(grammar)
    return tree[0]


def _generate(grammar,items,depth=None):
    if depth > 0:
        result = []
        for i in items:
            p = random.random()
            total_rule_prob = 0.
            if isinstance(i, Nonterminal):
                for prod in grammar.productions(lhs=i):
                    total_rule_prob += prod.prob()
                    if p < total_rule_prob:
                        expansion = _generate(grammar, prod.rhs(), depth - 1)
                        result += [Tree(i, expansion)]
                        break
            else:
                result += [i]  
                break         
        return result

#function will write 10 sentences into test_file.csv
def create_file (filename, grammar, ex_generator, n=1000):
    with open("negation.val", mode='w') as output_file:
        output_writer = csv.writer(output_file, delimiter=',', lineterminator ='\n', quotechar="/")
#        output_writer.writerow({'SRC', 'TRANSFORM', 'TRG'})
        output_writer.writerow(['source', 'transform', 'target'])
        for _ in range(n):
            # src, trans, targ = ex_generator(grammar)
            src, trans, targ = ex_generator(grammar)
            # output_writer.writerow([src + ' ' + trans, targ])
            output_writer.writerow([src, trans,  targ])

#this function will generate sentences without writing them into a file

# def demo(N=5):
#     for _ in range(N):
#         sent = generate(grammars)
#         print(" ".join(sent.leaves()))

# if __name__ == "__main__":
#     demo()