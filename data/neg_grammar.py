# code adapted from Bob Frank's grammars.py
from nltk import PCFG, Tree
from nltk import nonterminals, Nonterminal, Production

import random
from generator import generate
from generator import create_file

# Create some nonterminals
# S, # Sentence
# S2, # Sentence 2: follows and Advp
# NP, # Noun Phrase
# MP, # Modal Phrase
# AdvP, # Adverb Phrase
# VPTr, # Verb Phrase with a Transitive Verb
# RelP, # Relative Phrase
# NPTr, # Noun Phrase following a VPTr 
# VP, # Verb Phrase: occurs at the end of a RelP
# Det, # Determiner
# N, # Noun
# PN, # Proper Noun
# Pron, # Pronoun
# M, # Modal
# VInTr, # Intransitive Verb
# VTr, # Transitive Verb
# NTr, # Noun Following a VPTr
# PlDet, # Plural Determiner
# PlNTr, # Plural Noun following a VPTr
# RP, # Relative Pronoun
# V, # Verb that occurs at the end of an AdvP
# Adv, # Adverb
S, NP, MP, VP, AdvP, VPTr, RelP, NPTr, Det, N, PN, Pron, M, VInTr, VTr, NTr, PlDet,PlNTr, RP, NTand, Adv = nonterminals('S, NP, MP, VP, AdvP, VPTr, RelP, NPTr, Det, N, PN, Pron, M, VInTr, VTr, NTr, PlDet,PlNTr, RP, NTand, Adv')

neg_grammar = PCFG.fromstring("""
    S -> NP MP [0.6] | AdvP S [0.2] | S AdvP [0.2]
    NP -> Det N [0.2] | PN [0.7] | Pron [0.1]
    MP -> M VP [1.0]
    VP -> VInTr [0.2] | VPTr [0.4] | VPTr RelP [0.4] 
    AdvP -> Adv NP MP [1.0] 
    VPTr -> VTr NPTr  [1.0]
    RelP -> RP NP M VTr [1.0] 
    NPTr -> Det NTr [0.5] | PlDet PlNTr [0.5] 
    Det -> 'the' [0.5] | 'a' [0.5]
    N -> 'student' [0.3] | 'professor' [0.3] | 'wizard' [0.2] | 'witch' [0.2]
    PN -> 'Harry' [0.05] | 'Ginny' [0.05] | 'Hermione' [0.05] | 'Ron' [0.05] | 'Fred' [0.05] | 'George' [0.05] | 'Petunia' [0.05] | 'Vernon' [0.05] | 'Lily' [0.05] | 'Hagrid' [0.05] | 'James' [0.05] | 'Neville' [0.05] | 'Snape' [0.05] | 'Dobby' [0.05] | 'McGonagall' [0.05] | 'Lupin' [0.05] | 'Draco' [0.05] | 'Voldemort' [0.05] | 'Sirius' [0.05] | 'Albus' [0.05]
    Pron -> 'he' [0.5] | 'she' [0.5]
    M -> 'can' [0.2] | 'may' [0.2] | 'must' [0.3] | 'should' [0.3] 
    VInTr -> 'hiccup' [0.1] | 'party'[0.1] | 'wiggle' [0.1] | 'laugh' [0.1] | 'smile' [0.1] | 'giggle' [0.1] | 'jump' [0.1] | 'run' [0.1] | 'walk' [0.1] | 'swim' [0.1]
    VTr -> 'prepare' [0.1] | 'make' [0.1] | 'eat' [0.1] | 'sprinkle' [0.1] | 'arrange' [0.1] | 'chew' [0.1] | 'gobble' [0.1] | 'assemble' [0.1] | 'create' [0.1] | 'hide' [0.1]
    NTr -> 'cookie' [0.1] | 'cake' [0.1] | 'chocolate' [0.1] | 'pancake' [0.1] | 'souffle' [0.1] | 'eclaire' [0.1] | 'croissant' [0.1] | 'strudel' [0.1] | 'baklava' [0.1] | 'doughnut' [0.1]
    PlDet -> 'the' [0.8] | 'many' [0.2] 
    PlNTr -> 'cookies' [0.1] | 'cakes' [0.1] | 'chocolates' [0.1] | 'pancakes' [0.1] | 'souffles' [0.1] | 'eclaires' [0.1] | 'croissants' [0.1] | 'strudels' [0.1] | 'baklava' [0.1] | 'doughnuts' [0.1]
    RP -> 'that' [0.6] | 'which' [0.4] 
    NTand -> 'and' [1.0]
    Adv -> 'because' [0.5] | 'since' [0.5]   
""")

# generate positive and negative sentences
# return source, neg, and target to be used in create file
def negation(grammar):
    pos_tree = generate(grammar)
    pos = ' '.join(pos_tree.leaves())
    neg_tree = negate(pos_tree)
    neg = ' '.join(neg_tree.leaves())
    source = pos
    coin = random.randint(0,1)
    if coin:
        target = pos
    else:
        target = neg
    transform = 'pos' if coin else 'neg'
    return source, transform, target        

# function to negate the tree
# 3 cases: S -> NP MP, AdvP S, S AdvP
def negate(t):
    symbol = t[0].label().symbol()
    # case 1: S -> NP MP, insert a 'not' after the modal verb
    if symbol == 'NP':
        modal = t[1,0]
        modal = modal[-1]
        modal = modal + ' not'
        t[1,0] = modal
    # case 2: S -> S AdvP, extract the MP and negate it
    elif symbol == 'S':
        MP = t[0]
        negate(MP)
    # case 3: AdvP S, recurse until reaching the MP-
    else:
        negate(t[1])
    return t

# uncomment the line below to run this file alongside generator.py
create_file("test_file", neg_grammar, negation)
