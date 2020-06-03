from nltk import PCFG, Tree
from nltk import nonterminals, Nonterminal, Production

import random
from generator import generate

# Create some nonterminals
S, NP, VP, AdjP, NP_pron, N, V, P, Det, Adj, Pron, PLex, NP_PP, NPSg, NPPl, NSg, NPl, Vsg, Vpl, VPSg, VPPl, PronSg, PronPl, DetSg, DetPl, NPobj, PronObj, PPSg, PPPl, NPObjSg, NPObjPl = nonterminals('S, NP, VP, AdjP, NP_pron, N, V, P, Det, Adj, Pron, PLex, NP_PP, NPSg, NPPl, NSg, NPl, Vsg, Vpl, VPSg, VPPl, PronSg, PronPl, DetSg, DetPl, NPobj, PronObj, PPSg, PPPl, NPObjSg, NPObjPl')

pcfg_agreement_pp = PCFG.fromstring("""
    S -> PP NPSg VSg [0.1] | PP NPPl VPl [0.1]
    S -> NPSg VSg [0.4]
    S -> NPPl VPl [0.4]    
    VSg -> 'laughs' [0.4] | 'dances' [0.2] | 'hopes' [0.15] | 'burps' [0.1] | 'coughs' [0.1] | 'dies' [0.05]
    VPl -> 'laugh' [0.4] | 'dance' [0.2] | 'hope' [0.15] | 'burp' [0.1] | 'cough' [0.1] | 'die' [0.05]
    P -> 'near' [0.7] | 'with' [0.3]
    PP -> P NPObj [1.0]
    NPObj -> PronObj [0.2] | DetSg NSg [0.2] | DetSg AdjP NSg [0.1] | DetSg NSg PP [0.1] | DetPl NPl [0.2] | DetPl AdjP NPl [0.1] | DetPl NPl PP [0.1] 
    NPSg -> PronSg [0.2] | DetSg NSg [0.4] | DetSg AdjP NSg [0.2] | DetSg NSg PP [0.2]
    NPPl -> PronPl [0.2] | DetPl NPl [0.4] | DetPl AdjP NPl [0.2] | DetPl NPl PP [0.2]
    DetSg -> 'the' [0.5] | 'a' [0.5]
    DetPl -> 'the' [0.8] | 'most' [0.2]
    NSg -> 'zebra' [0.4] | 'badger' [0.2] | 'chicken' [0.15] | 'dog' [0.1] | 'robin' [0.1] | 'frog' [0.05]
    NPl -> 'zebras' [0.4] | 'badgers' [0.2] | 'chickens' [0.15] | 'dogs' [0.1] | 'robins' [0.1] | 'frogs' [0.05]
    AdjP -> Adj [0.7] | Adj AdjP [0.3]
    Adj -> 'gentle' [0.4] | 'humble' [0.2] | 'clever' [0.15]  | 'jocular' [0.1] | 'kindly' [0.1] | 'lovely' [0.05]
    PronSg -> 'he' [0.5] |  'she' [0.5] 
    PronPl -> 'they' [1.0] 
    PronObj -> 'him' [.33] | 'her' [.33] | 'them' [.34]
""")


pcfg_agreement_pp_ambig = PCFG.fromstring("""
    S -> PPSg NPSg VSg [0.1] | PPPl NPPl VPl [0.1]
    S -> NPSg VSg [0.4]
    S -> NPPl VPl [0.4]    
    VSg -> 'laughs' [0.4] | 'dances' [0.2] | 'hopes' [0.15] | 'burps' [0.1] | 'coughs' [0.1] | 'dies' [0.05]
    VPl -> 'laugh' [0.4] | 'dance' [0.2] | 'hope' [0.15] | 'burp' [0.1] | 'cough' [0.1] | 'die' [0.05]
    P -> 'near' [0.7] | 'with' [0.3]
    PP -> P NPObj [1.0]
    PPSg -> P NPObjSg [1.0]
    PPPl -> P NPObjPl [1.0]
    NPObj -> PronObj [0.2] | DetSg NSg [0.2] | DetSg AdjP NSg [0.1] | DetSg NSg PP [0.1] | DetPl NPl [0.2] | DetPl AdjP NPl [0.1] | DetPl NPl PP [0.1] 
    NPObjSg -> 'him' [0.1] | 'her' [0.1] | DetSg NSg [0.4] | DetSg AdjP NSg [0.2] | DetSg NSg PP [0.2] 
    NPObjPl ->  'them' [0.2] | DetPl NPl [0.4] | DetPl AdjP NPl [0.2] | DetPl NPl PP [0.2] 
    NPSg -> PronSg [0.2] | DetSg NSg [0.4] | DetSg AdjP NSg [0.2] | DetSg NSg PP [0.2]
    NPPl -> PronPl [0.2] | DetPl NPl [0.4] | DetPl AdjP NPl [0.2] | DetPl NPl PP [0.2]
    DetSg -> 'the' [0.5] | 'a' [0.5]
    DetPl -> 'the' [0.8] | 'most' [0.2]
    NSg -> 'zebra' [0.4] | 'badger' [0.2] | 'chicken' [0.15] | 'dog' [0.1] | 'robin' [0.1] | 'frog' [0.05]
    NPl -> 'zebras' [0.4] | 'badgers' [0.2] | 'chickens' [0.15] | 'dogs' [0.1] | 'robins' [0.1] | 'frogs' [0.05]
    AdjP -> Adj [0.7] | Adj AdjP [0.3]
    Adj -> 'gentle' [0.4] | 'humble' [0.2] | 'clever' [0.15]  | 'jocular' [0.1] | 'kindly' [0.1] | 'lovely' [0.05]
    PronSg -> 'he' [0.5] |  'she' [0.5] 
    PronPl -> 'they' [1.0] 
    PronObj -> 'him' [.33] | 'her' [.33] | 'them' [.34]
""")


pcfg_agreement_pp_unambig = PCFG.fromstring("""
    S -> PPPl NPSg VSg [0.5] | PPSg NPPl VPl [0.5]
    VSg -> 'laughs' [0.4] | 'dances' [0.2] | 'hopes' [0.15] | 'burps' [0.1] | 'coughs' [0.1] | 'dies' [0.05]
    VPl -> 'laugh' [0.4] | 'dance' [0.2] | 'hope' [0.15] | 'burp' [0.1] | 'cough' [0.1] | 'die' [0.05]
    P -> 'near' [0.7] | 'with' [0.3]
    PP -> P NPObj [1.0]
    PPSg -> P NPObjSg [1.0]
    PPPl -> P NPObjPl [1.0]
    NPObj -> PronObj [0.2] | DetSg NSg [0.2] | DetSg AdjP NSg [0.1] | DetSg NSg PP [0.1] | DetPl NPl [0.2] | DetPl AdjP NPl [0.1] | DetPl NPl PP [0.1] 
    NPObjSg -> 'him' [0.1] | 'her' [0.1] | DetSg NSg [0.4] | DetSg AdjP NSg [0.2] | DetSg NSg PP [0.2] 
    NPObjPl ->  'them' [0.2] | DetPl NPl [0.4] | DetPl AdjP NPl [0.2] | DetPl NPl PP [0.2] 
    NPSg -> PronSg [0.2] | DetSg NSg [0.4] | DetSg AdjP NSg [0.2] | DetSg NSg PP [0.2]
    NPPl -> PronPl [0.2] | DetPl NPl [0.4] | DetPl AdjP NPl [0.2] | DetPl NPl PP [0.2]
    DetSg -> 'the' [0.5] | 'a' [0.5]
    DetPl -> 'the' [0.8] | 'most' [0.2]
    NSg -> 'zebra' [0.4] | 'badger' [0.2] | 'chicken' [0.15] | 'dog' [0.1] | 'robin' [0.1] | 'frog' [0.05]
    NPl -> 'zebras' [0.4] | 'badgers' [0.2] | 'chickens' [0.15] | 'dogs' [0.1] | 'robins' [0.1] | 'frogs' [0.05]
    AdjP -> Adj [0.7] | Adj AdjP [0.3]
    Adj -> 'gentle' [0.4] | 'humble' [0.2] | 'clever' [0.15]  | 'jocular' [0.1] | 'kindly' [0.1] | 'lovely' [0.05]
    PronSg -> 'he' [0.5] |  'she' [0.5] 
    PronPl -> 'they' [1.0] 
    PronObj -> 'him' [.33] | 'her' [.33] | 'them' [.34]
""")

def gen_reinflection_example(grammar):
    pres_tree = generate(grammar)
    pres = " ".join(pres_tree.leaves())
    past_tree = inflect_to_past(pres_tree)
    past = " ".join(past_tree.leaves())
    source = past
    coin = random.randint(0,1)
    if coin:
        target = past
    else:
        target = pres
    transform = 'past' if coin else 'pres'
    return source, transform, target

def gen_pres_reinflection_example(grammar):
    pres_tree = generate(grammar)
    pres = " ".join(pres_tree.leaves())
    past_tree = inflect_to_past(pres_tree)
    past = " ".join(past_tree.leaves())
    source = past
    coin = 0
    if coin:
        target = past
    else:
        target = pres
    transform = 'past' if coin else 'pres'
    return source, transform, target

def inflect_to_past(t):
    pp_init = False
    if t[1].label().symbol()[0] == 'V':
        verb = t[1,0]
    else:
        verb = t[2,0]
        pp_init = True
    if verb[-1] == 's':
        verb = verb[0:-1]
    if verb[-1] == 'e':
        verb = verb + 'd'
    else:
        verb = verb + 'ed'
    if pp_init:
        t[2,0] = verb
    else:
        t[1,0] = verb
    return t

    

