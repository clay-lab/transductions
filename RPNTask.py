"""
Stole some code from StackNNs /formalisms/trees.py
"""

from nltk import CFG, PCFG
import nltk.tree

import numpy.random


def get_terminals(grammar):
    return [x for production_rule in grammar.productions() for x in production_rule.rhs() if (not isinstance(x, nltk.grammar.Nonterminal))]

def generic_tree_swap(tree, trans_children, ops_tags):
    if not isinstance(tree, nltk.tree.Tree):
        # tree is a leaf
        return tree
    else:
        #cls = type(tree)
        cls = nltk.tree.Tree
        children = [generic_tree_swap(child, trans_children, ops_tags=ops_tags) for child in tree]
        if tree.label() in ops_tags:
            children = trans_children(children)
        #return cls(tree.label(), children, prob=-1)
        return cls(tree.label(), children)

def infix2polish(tree, ops_tags):
    def trans_children(children):
        return [children[1], children[0]] + children[2:]
    
    return generic_tree_swap(tree, trans_children, ops_tags=ops_tags)
    
def polish2infix(tree, ops_tags):
    return infix2polish(tree, ops_tags=ops_tags)
 
def polish2reversePolish(tree, ops_tags):
    def trans_children(children):
        return children[1:] + [children[0]]
    
    return generic_tree_swap(tree, trans_children, ops_tags=ops_tags)

def reversePolish2polish(tree, ops_tags):
    def trans_children(children):
        return [children[-1]] + children[:-1]
    
    return generic_tree_swap(tree, trans_children, ops_tags=ops_tags)


#def annotate_grammar(grammar, )

# polish notation not infix
pre_arithmetic_grammar = PCFG.fromstring("""
START ->  OPERATOR EXPR EXPR [1.0]
EXPR -> OPERAND [0.6] | OPERATOR EXPR EXPR [0.4]
OPERATOR -> '+' [0.25] | '-' [0.25] | '*' [0.25] | '/' [0.25]
OPERAND -> VARIABLE [0.5] | NUMBER [0.5]
VARIABLE -> 'a' [0.25] | 'b' [0.25] | 'x' [0.25] | 'y' [0.25]
NUMBER -> '2' [0.25] | '3' [0.25] | '4' [0.25] | '5' [0.25]
""")




def tree2index_form(tree):
    """
    some classes in models.py take a tree in this form
    but we should get rid of them and then we won't need this
    horrible function
    """
    subtree_layers = [[tree]]
    index_layers = [[[0]]]
    
    while not all(piece.height() <= 2 for piece in subtree_layers[-1]):
        layer_subtrees = subtree_layers[-1]
        layer_indices = index_layers[-1]
        layer_index = 0
        next_layer_subtrees = []
        next_layer_indices = []
        for piece in layer_subtrees:
            next_layer_subtrees += list(piece) if piece.height() > 2 else [piece]
            next_layer_indices.append(list(range(layer_index, layer_index+len(piece))))
            layer_index += len(piece) 
        subtree_layers.append(next_layer_subtrees)
        index_layers.append(next_layer_indices)
        
    return list(reversed(index_layers))