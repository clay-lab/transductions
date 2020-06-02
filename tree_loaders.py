import nltk
from torchtext.data import Field, TabularDataset, Example, RawField, Dataset
from numpy.random import default_rng

class TreeField(RawField):
    def __init__(self, tree_transformation_fun=None, read_heads=False, brackets="[]", 
                 chomsky_normalize=False, collapse_unary=True, inner_label="NULL", is_target=False):
        # all this does is set is_target
        # we don't use the preprocess and postprocess pipelines because those expect string inputs
        super(TreeField, self).__init__(self, is_target=is_target)
        
        self.tree_transformation_fun = tree_transformation_fun
        self.read_heads = read_heads
        self.brackets = brackets
        self.chomsky_normalize = chomsky_normalize
        self.collapse_unary = collapse_unary
        
        self.inner_label = inner_label
        
        #self._cached_preprocessed = None
    
    def preprocess(self, x):
        if isinstance(x, nltk.tree.Tree):
            tree = x
        else:
            if not self.read_heads:
                x = x.replace(self.brackets[0], "{} {} ".format(self.brackets[0], self.inner_label))

            tree = nltk.Tree.fromstring(x, brackets=self.brackets)
        
        if self.tree_transformation_fun is not None:
            tree = self.tree_transformation_fun(tree)
        
        if self.chomsky_normalize:
            tree.chomsky_normal_form()
            
        if self.collapse_unary:
            tree.collapse_unary(True, True)
            
        #self._cached_preprocessed = tree
        return tree
    
class TreeSequenceField(Field):
    def __init__(self, tree_field, inner_order=None, **field_kwargs):
        super(TreeSequenceField, self).__init__(self, **field_kwargs)
        
        self.tree_field = tree_field
        self.inner_order = inner_order
        
    def tree2seq(self, tree):
        if isinstance(tree, nltk.tree.Tree):
            # non-terminal
            if len(tree) == 1:
                # unary production
                return self.tree2seq(tree[0])
            else:
                # not unary
                seq = sum((self.tree2seq(child) for child in tree), [])
                inner_production = tree.label()
                if self.inner_order == "pre":
                    seq.insert(0, inner_production)
                elif self.inner_order == "post":
                    seq += [inner_production]
                elif self.inner_order == "infix":
                    seq.insert(1, inner_production)
                elif self.inner_order == "repeated_infix":
                    assert False, "Not yet implemented"
                elif (not self.inner_order):
                    pass
                else:
                    assert False, f"In tree2seq, inner={self.inner_order} not understood"

                return seq
        else:
            # leaf
            return [tree]
        
    def preprocess(self, x):
        tree = self.tree_field.preprocess(x)
        seqstr = " ".join(self.tree2seq(tree))
        return super(TreeSequenceField, self).preprocess(seqstr)
    
"""
we want to be able to encode which kind of trasnformation was used as a separate column
encode in the grammar itself?
first production tells us which transformation to apply
the transformation can then read 
"""

class PCFGDataset(Dataset):
    def __init__(self, grammar, n, fields, min_length=0, max_length=100, seed=None, **kwargs):
        self.min_length = min_length
        self.max_length = max_length
        self.seed = seed
        self.rng = default_rng() if self.seed is None else default_rng(seed)
        examples = [Example.fromlist([self.draw_from_PCFG(grammar, rng=self.rng, min_length=min_length, max_length=max_length)], fields) for _ in range(n)]
        super(PCFGDataset, self).__init__(examples, fields, **kwargs)
    
    @classmethod
    def generate_trees(cls, grammar, n, min_length=0, max_length=100, seed=None):
        rng = default_rng() if self.seed is None else default_rng(seed)
        return [cls.draw_from_PCFG(grammar, rng=rng, min_length=min_length, max_length=max_length) for _ in range(n)]        

    @staticmethod
    def draw_from_PCFG(grammar, rng=default_rng(), min_length=0, max_length=100):
        def helper(grammar, start, rng, max_leaves):
            if start in grammar._lhs_index:
                # start is a non-terminal
                derivations = grammar._lhs_index[start]
                derivation = rng.choice(derivations, p=[rule.prob() for rule in derivations])
                
                children = []
                num_leaves = 0
                for child in derivation._rhs:
                    # we've already accounted for 
                    subtree, subtree_leaves = helper(grammar, child, rng, max_leaves - num_leaves)
                    num_leaves += subtree_leaves
                    if (subtree is None) or (num_leaves > max_leaves):
                        return None, num_leaves
                    else:
                        children.append(subtree)
                        
                return nltk.Tree(str(start), children), num_leaves            
            else:
                # start is a terminal
                return start, 1
        
        assert 0 <= min_length <= max_length
        tree = None
        num_leaves = -1
        while (tree is None) or (num_leaves < min_length):
            tree, num_leaves = helper(grammar, grammar.start(), rng, max_length)
        
        return tree

        
        
        