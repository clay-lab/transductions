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
S, NP_M_sg, NP_M_pl, NP_O_sg, NP_O_pl, VP_M_sg, VP_M_pl, RC_sg, RC_pl, Det, N_sg, N_pl, V_intrans_sg, V_trans_sg, V_intrans_pl, V_trans_pl, Prep, Rel = nonterminals('S, NP_M_sg, NP_M_pl, NP_O_sg, NP_O_pl, VP_M_sg, VP_M_pl, RC_sg, RC_pl, Det, N_sg, N_pl, V_intrans_sg, V_trans_sg, V_intrans_pl, V_trans_pl, Prep, Rel')

t_grammar = PCFG.fromstring("""
    S -> NP_M_sg VP_M_sg [0.5] | NP_M_pl VP_M_pl [0.5]
    NP_M_sg -> Det N_sg [0.3] | Det N_sg Prep Det N_sg [0.2] | Det N_sg Prep Det N_pl [0.2] | Det N_sg RC_sg [0.3]
    NP_M_pl -> Det N_pl [0.3] | Det N_pl Prep Det N_sg [0.2] | Det N_pl Prep Det N_pl [0.2] | Det N_pl RC_pl [0.3]
    NP_O_sg -> Det N_sg [0.3] | Det N_sg Prep Det N_sg [0.2] | Det N_sg Prep Det N_pl [0.2] | Det N_sg RC_sg [0.3]
    NP_O_pl -> Det N_pl [0.3] | Det N_pl Prep Det N_sg [0.2] | Det N_pl Prep Det N_pl [0.2] | Det N_pl RC_pl [0.3]
    VP_M_sg -> V_intrans_sg [0.5] | V_trans_sg NP_O_sg [0.25] | V_trans_sg NP_O_pl [0.25]
    VP_M_pl -> V_intrans_pl [0.5] | V_trans_pl NP_O_sg [0.25] | V_trans_pl NP_O_pl [0.25]
    RC_sg -> Rel V_intrans_sg [0.4] | Rel Det N_sg V_trans_sg [0.15] | Rel Det N_pl V_trans_pl [0.15] | Rel V_trans_sg Det N_sg [0.15] | Rel V_trans_sg Det N_pl [0.15]
    RC_pl -> Rel V_intrans_pl [0.4] | Rel Det N_sg V_trans_sg [0.15] | Rel Det N_pl V_trans_pl [0.15] | Rel V_trans_pl Det N_sg [0.15] | Rel V_trans_pl Det N_pl [0.15]
    Det -> 'the'[.16] | 'some' [.16] | 'my' [0.17] | 'your' [0.17] | 'our' [0.17] | 'her' [0.17] 
    N_sg -> 'newt' [.07] | 'orangutan' [.07] | 'peacock' [.07] | 'quail' [.07] | 'raven' [.08] | 'salamander' [.08] | 'tyrannosaurus' [.08] | 'unicorn' [.08] | 'vulture' [.08] | 'walrus' [.08] | 'xylophone' [.08] | 'yak' [.08] | 'zebra' [.08]
    N_pl -> 'newts' [.07] | 'orangutans' [.07] | 'peacocks' [.07] | 'quails' [.07] | 'ravens' [.08] | 'salamanders' [.08] | 'tyrannosauruses' [.08] | 'unicorns' [.08] | 'vultures' [.08] | 'walruses' [.08] | 'xylophones' [.08] | 'yaks' [.08] | 'zebras' [.08]
    V_intrans_sg -> 'giggles' [.11] | 'smiles' [.11] | 'sleeps' [.11] | 'swims' [.11] | 'waits' [.11] | 'moves' [.11] | 'changes' [.11] | 'reads' [.11] | 'eats' [.12]
    V_trans_sg -> 'entertains' [.11] | 'amuses' [.11] | 'high_fives' [.11] | 'applauds' [.11] | 'confuses' [.11] | 'admires' [.11] | 'accepts' [.11] | 'remembers' [.11] | 'comforts' [.12]
    V_intrans_pl -> 'giggle'  [.11] | 'smile' [.11] | 'sleep' [.11] | 'swim' [.11]| 'wait'   [.11] | 'move' [.11] | 'change' [.11] | 'read' [.11] | 'eat' [.12]
    V_trans_pl -> 'entertain' [.11] | 'amuse' [.11] | 'high_five' [.11] | 'applaud' [.11] | 'confuse' [.11] | 'admire' [.11] | 'accept' [.11] | 'remember' [.11] | 'comfort' [.12]
    Prep -> 'around' [.125] | 'near' [.125] | 'with' [.125] | 'upon' [.125] | 'by' [.125] | 'behind' [.125] | 'above' [.125] | 'below' [.125]
    Rel -> 'who' [0.5] | 'that' [0.5]
    """)

# generate positive and negative sentences
# return source, neg, and target to be used in create file
def tense(grammar):
    pres_tree = generate(grammar)
    pres = ' '.join(pres_tree.leaves())
    past_tree = inflect_to_past(pres_tree)
    past = ' '.join(past_tree.leaves())
    target = past + "."
    source = pres + "."
    coin = random.randint(0,1)
    if coin:
        target = pres + "."
    else:
        target = past + "."
    transform = 'PRESENT' if coin else 'PAST'
    return source, transform, target        

# function to transform decl to quest
def inflect_to_past(t):
    leaf = t[1,0]
    verb = leaf[-1]
    # catch the irregulars
    if verb == 'swims' or verb == "swim": 
        verb = "swam"
    elif verb == "sleeps" or verb == "sleep":
        verb = "slept"
    elif verb == "eats" or verb == "eat":
        verb = "ate"
    elif verb == "reads" or verb ==  "read":
        verb = "read"   
    elif verb[-1] == 's' and verb[-2] == 'e':
        verb = verb[0:-1] + 'd' 
    elif verb[-1] == 's' and verb[-2] != 'e':
        verb = verb[0:-1] + 'ed' 
    elif verb[-1] == 'e':
        verb = verb + 'd'
    else:
        verb = verb + 'ed'
    t[1,0] = verb
    return t

# uncomment the line below to run this file alongside generator.py
create_file("test_file", t_grammar, tense)

# def demo(N=5):
#     for _ in range(N):
#         sent = generate(t_grammar)
#         print(" ".join(sent.leaves()))

# if __name__ == "__main__":
#     # demo()
#     inflect_to_past(generate(t_grammar))