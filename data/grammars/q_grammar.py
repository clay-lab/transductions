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
S, NP_Ms, NP_Mp, NP_O, N, VP_Ms, VP_Mp, RC_S, RC_P, Det, Ns, Np, VInTr, VTr, Aux_P, Aux_S,Prep, Rel = nonterminals('S, NP_Ms, NP_Mp, NP_O, N, VP_Ms, VP_Mp, RC_S, RC_P, Det, Ns, Np, VInTr, VTr, Aux_P, Aux_S,Prep, Rel')

q_grammar = PCFG.fromstring("""
    S -> NP_Ms VP_Ms [0.5] | NP_Mp VP_Mp [0.5]
    NP_Ms -> Det Ns [.33] | Det Ns Prep Det N [.33] | Det Ns RC_S [.34]
    NP_Mp -> Det Np [.33] | Det Np Prep Det N [.33] | Det Np RC_P [.34]
    NP_O -> Det Ns [.17] | Det Np [.17] | Det Ns Prep Det N [.17] | Det Np Prep Det N [.17] | Det Ns RC_S [.16] | Det Np RC_P [.16]
    N -> Ns [0.5] | Np [0.5]
    VP_Ms -> Aux_S VInTr [0.5] | Aux_S VTr NP_O [0.5]
    VP_Mp -> Aux_P VInTr [0.5] | Aux_P VTr NP_O [0.5]
    RC_S -> Aux_S VInTr [0.3] | Rel Det Ns Aux_S VTr [0.2] | Rel Det Np Aux_P VTr [0.2] | Rel Aux_S VTr Det N [0.3]
    RC_P -> Rel Aux_P VInTr [0.25] | Rel Det Ns Aux_S VTr [0.25] | Rel Det Np Aux_P VTr [0.25] | Rel Aux_P VTr Det N [0.25]
    Det -> 'the' [.16] | 'some' [.16] | 'my' [.17] | 'your' [.17] | 'our' [.17] | 'her' [.17]
    Ns -> 'newt' [0.1] | 'orangutan' [0.1] | 'peacock' [0.1] | 'quail' [0.1] | 'raven' [0.1] | 'salamander' [0.1] | 'tyrannosaurus' [0.1] | 'unicorn' [0.05] | 'vulture' [0.05] | 'walrus' [0.05] | 'xylophone' [0.05] | 'yak' [0.05] | 'zebra' [0.05]
    Np -> 'newts' [0.1] | 'orangutans' [0.1] | 'peacocks' [0.1] | 'quails' [0.1] | 'ravens' [0.1] | 'salamanders' [0.1] | 'tyrannosauruses' [0.1] | 'unicorns' [0.05] | 'vultures' [0.05] | 'walruses' [0.05] | 'xylophones' [0.05] | 'yaks' [0.05] | 'zebras' [0.05]
    VInTr -> 'giggle' [.11] | 'smile' [.11] | 'sleep' [.11] | 'swim' [.11] | 'wait' [.11] | 'move' [.11] | 'change' [.11] | 'read' [.11] | 'eat' [.12]
    VTr -> 'entertain' [.11] | 'amuse' [.11] | 'high_five' [.11] | 'applaud' [.11] | 'confuse' [.11] | 'admire' [.11] | 'accept' [.11] | 'remember' [.11] | 'comfort' [.12]
    Aux_P -> 'do' [0.5] | 'dont' [0.5] 
    Aux_S -> 'does' [0.5] | 'doesnt' [0.5]
    Prep -> 'around' [0.125] | 'near' [0.125] | 'with' [0.125] | 'upon' [0.125] | 'by' [0.125] | 'behind' [0.125] | 'above' [0.125] | 'below' [0.125]
    Rel -> 'who' [0.5] | 'that' [0.5]
    """)

# generate positive and negative sentences
# return source, neg, and target to be used in create file
def question(grammar):
    q_tree = generate(grammar)
    decl = ' '.join(q_tree.leaves())
    d_tree = quest(q_tree)
    que = ' '.join(d_tree.leaves())
    target = que + "?"
    source = decl + "."
    coin = random.randint(0,1)
    if coin:
        target = que + "?"
    else:
        target = decl + "."
    transform = 'quest' if coin else 'decl'
    return source, transform, target        

# function to transform decl to quest
def quest(t):
    first = t[0,0]
    first = first[-1]
    aux = t[1,0]
    aux = aux[-1]
    t[0,0] = (aux + " " + first)
    del t[1,0]
    return t

# uncomment the line below to run this file alongside generator.py
create_file("test_file", q_grammar, question)

# def demo(N=5):
#     for _ in range(N):
#         sent = generate(grammars)
#         print(" ".join(sent.leaves()))

# if __name__ == "__main__":
#     demo()