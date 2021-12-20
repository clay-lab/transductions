# test_metrics.py
#
# Unit tests for metrics.

import os
import sys
import unittest

import torch
from torch.nn import functional as F

# Library imports
# Since we are outside the main tree (/core/...), we
# need to insert the root back into the syspath
DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, "..")
sys.path.append(ROOT)

from core.metrics.base_metric import (  # noqa: E402
    FullSequenceModAccuracy,
    HierarchicalAccuracy,
    LinearAError,
    LinearBError,
    LinearCError,
    NegAccuracy,
)


class TestMetrics(unittest.TestCase):
    # class TestMetrics:

    def test_neg_acc(self):
        """
        Negative accuracy checks to see if target contains a NEG token,
        then checks to see if the prediction has a NEG token in the same
        position.

        total: # of sequences in batch with NEG token in target
        correct: # of predictions with NEG in same position as in target
        """

        # TOKENS
        NEG = 0
        SOS = 1
        EOS = 2
        PAD = 3

        ALICE = 4
        MARY = 5

        SEE = 6
        HEAR = 7

        # TEST ALL CORRECT
        target = torch.tensor(
            [
                [SOS, ALICE, SEE, MARY, EOS, PAD],  # No NEG
                [SOS, ALICE, NEG, HEAR, ALICE, EOS],  # NEG @ 2
                [SOS, MARY, NEG, SEE, MARY, EOS],  # NEG @ 2
                [SOS, MARY, HEAR, ALICE, EOS, PAD],
            ]  # No NEG
        )
        prediction = F.one_hot(target).permute(0, 2, 1)

        neg_acc = NegAccuracy(neg_token_id=NEG)
        neg_acc(prediction, target)

        assert neg_acc.total == 2.0
        assert neg_acc.correct == 2.0

        # TEST HALF CORRECT
        target = torch.tensor(
            [
                [SOS, ALICE, SEE, MARY, EOS, PAD],  # No NEG
                [SOS, ALICE, NEG, HEAR, ALICE, EOS],  # NEG @ 2
                [SOS, MARY, NEG, SEE, MARY, EOS],  # NEG @ 2
                [SOS, MARY, HEAR, ALICE, EOS, PAD],
            ]  # No NEG
        )
        prediction = torch.tensor(
            [
                [SOS, ALICE, SEE, MARY, EOS, PAD],  #
                [SOS, NEG, ALICE, HEAR, ALICE, EOS],  # WRONG POSITION
                [SOS, MARY, NEG, SEE, MARY, EOS],  # CORRECT
                [SOS, MARY, HEAR, ALICE, EOS, PAD],
            ]  #
        )
        prediction = F.one_hot(prediction).permute(0, 2, 1)

        neg_acc.reset()
        neg_acc(prediction, target)

        assert neg_acc.total == 2.0
        assert neg_acc.correct == 1.0

    def test_linearA(self):
        """
        Given a target of the form
          Alice near Mary sees herself -> See ( Alice , Alice ) & near ( Alice , Mary )
        Count the number of outputs which mix up the antecedent of 'herself', interpreting
        the VP as 'Mary sees herself', resulting in
          See ( Mary  , Mary  ) & near ( ............. )

        total = # target sequences of the form "<sos> verb ( X , X ) & p ( X , Y )"
        correct = # predictions of the form "<sos> verb (Y, Y) ...."
        """

        # TOKENS
        AND = 0
        SOS = 1
        EOS = 2
        PAD = 3

        L_PAREN = 4
        R_PAREN = 5
        COMMA = 6

        ALICE = 7
        MARY = 8

        SEE = 9

        NEAR = 11

        target = torch.tensor(
            [
                [
                    SOS,
                    SEE,
                    L_PAREN,
                    ALICE,
                    COMMA,
                    ALICE,
                    R_PAREN,
                    EOS,
                    PAD,
                    PAD,
                    PAD,
                    PAD,
                    PAD,
                    PAD,
                    PAD,
                ],
                [
                    SOS,
                    SEE,
                    L_PAREN,
                    ALICE,
                    COMMA,
                    ALICE,
                    R_PAREN,
                    AND,
                    NEAR,
                    L_PAREN,
                    ALICE,
                    COMMA,
                    MARY,
                    R_PAREN,
                    EOS,
                ],
                [
                    SOS,
                    SEE,
                    L_PAREN,
                    ALICE,
                    COMMA,
                    ALICE,
                    R_PAREN,
                    AND,
                    NEAR,
                    L_PAREN,
                    ALICE,
                    COMMA,
                    MARY,
                    R_PAREN,
                    EOS,
                ],
            ]
        )
        prediction = torch.tensor(
            [
                [
                    SOS,
                    SEE,
                    L_PAREN,
                    ALICE,
                    COMMA,
                    ALICE,
                    R_PAREN,
                    EOS,
                    PAD,
                    PAD,
                    PAD,
                    PAD,
                    PAD,
                    PAD,
                    PAD,
                ],  # Doesn't count
                [
                    SOS,
                    SEE,
                    L_PAREN,
                    ALICE,
                    COMMA,
                    ALICE,
                    R_PAREN,
                    AND,
                    NEAR,
                    L_PAREN,
                    ALICE,
                    COMMA,
                    MARY,
                    R_PAREN,
                    EOS,
                ],  # Same as target
                [
                    SOS,
                    SEE,
                    L_PAREN,
                    MARY,
                    COMMA,
                    MARY,
                    R_PAREN,
                    AND,
                    NEAR,
                    L_PAREN,
                    ALICE,
                    COMMA,
                    MARY,
                    R_PAREN,
                    EOS,
                ],  # Linear A error
            ]
        )
        prediction = F.one_hot(prediction).permute(0, 2, 1)

        lin_a_err = LinearAError(pad_token_id=PAD, and_token_id=AND)
        lin_a_err(prediction, target)

        assert lin_a_err.total == 2.0
        assert lin_a_err.correct == 1.0

    def test_linearB(self):
        """
        Given a target of the form
          victor beside leo knows himself -> <sos> know ( victor , victor ) & beside ( victor , leo )
        Test if the output is of form:
          know ( victor , leo ) & beside ( .... )

        total = # target sequences of the form "<sos> verb ( X , X ) & p ( X , Y )"
        correct = # predictions of the form "<sos> verb (X, Y) ...."
        """

        # TOKENS
        AND = 0
        SOS = 1
        EOS = 2
        PAD = 3

        L_PAREN = 4
        R_PAREN = 5
        COMMA = 6

        ALICE = 7
        MARY = 8

        SEE = 9

        NEAR = 11

        target = torch.tensor(
            [
                [
                    SOS,
                    SEE,
                    L_PAREN,
                    ALICE,
                    COMMA,
                    ALICE,
                    R_PAREN,
                    EOS,
                    PAD,
                    PAD,
                    PAD,
                    PAD,
                    PAD,
                    PAD,
                    PAD,
                ],
                [
                    SOS,
                    SEE,
                    L_PAREN,
                    ALICE,
                    COMMA,
                    ALICE,
                    R_PAREN,
                    AND,
                    NEAR,
                    L_PAREN,
                    ALICE,
                    COMMA,
                    MARY,
                    R_PAREN,
                    EOS,
                ],
                [
                    SOS,
                    SEE,
                    L_PAREN,
                    ALICE,
                    COMMA,
                    ALICE,
                    R_PAREN,
                    AND,
                    NEAR,
                    L_PAREN,
                    ALICE,
                    COMMA,
                    MARY,
                    R_PAREN,
                    EOS,
                ],
            ]
        )
        prediction = torch.tensor(
            [
                [
                    SOS,
                    SEE,
                    L_PAREN,
                    ALICE,
                    COMMA,
                    ALICE,
                    R_PAREN,
                    EOS,
                    PAD,
                    PAD,
                    PAD,
                    PAD,
                    PAD,
                    PAD,
                    PAD,
                ],  # Doesn't count
                [
                    SOS,
                    SEE,
                    L_PAREN,
                    ALICE,
                    COMMA,
                    ALICE,
                    R_PAREN,
                    AND,
                    NEAR,
                    L_PAREN,
                    ALICE,
                    COMMA,
                    MARY,
                    R_PAREN,
                    EOS,
                ],  # Same as target
                [
                    SOS,
                    SEE,
                    L_PAREN,
                    ALICE,
                    COMMA,
                    MARY,
                    R_PAREN,
                    AND,
                    NEAR,
                    L_PAREN,
                    ALICE,
                    COMMA,
                    MARY,
                    R_PAREN,
                    EOS,
                ],  # Linear B error
            ]
        )
        prediction = F.one_hot(prediction).permute(0, 2, 1)

        lin_b_err = LinearBError(pad_token_id=PAD, and_token_id=AND)
        lin_b_err(prediction, target)

        assert lin_b_err.total == 2.0
        assert lin_b_err.correct == 1.0

    def test_linearC(self):
        """
        Given a target of the form
          victor beside leo knows himself -> <sos> know ( victor , victor ) & beside ( victor , leo )

        Test if the output is of form:
          know ( leo , victor ) & beside ( .... )

        i.e., reflexives are structural but subj-verb is linear

        total = # target sequences of the form "<sos> verb ( X , X ) & p ( X , Y )"
        correct = # predictions of the form "<sos> verb (Y, X) ...."
        """

        # TOKENS
        AND = 0
        SOS = 1
        EOS = 2
        PAD = 3

        L_PAREN = 4
        R_PAREN = 5
        COMMA = 6

        ALICE = 7
        MARY = 8

        SEE = 9

        NEAR = 11

        target = torch.tensor(
            [
                [
                    SOS,
                    SEE,
                    L_PAREN,
                    ALICE,
                    COMMA,
                    ALICE,
                    R_PAREN,
                    EOS,
                    PAD,
                    PAD,
                    PAD,
                    PAD,
                    PAD,
                    PAD,
                    PAD,
                ],
                [
                    SOS,
                    SEE,
                    L_PAREN,
                    ALICE,
                    COMMA,
                    ALICE,
                    R_PAREN,
                    AND,
                    NEAR,
                    L_PAREN,
                    ALICE,
                    COMMA,
                    MARY,
                    R_PAREN,
                    EOS,
                ],
                [
                    SOS,
                    SEE,
                    L_PAREN,
                    ALICE,
                    COMMA,
                    ALICE,
                    R_PAREN,
                    AND,
                    NEAR,
                    L_PAREN,
                    ALICE,
                    COMMA,
                    MARY,
                    R_PAREN,
                    EOS,
                ],
            ]
        )
        prediction = torch.tensor(
            [
                [
                    SOS,
                    SEE,
                    L_PAREN,
                    ALICE,
                    COMMA,
                    ALICE,
                    R_PAREN,
                    EOS,
                    PAD,
                    PAD,
                    PAD,
                    PAD,
                    PAD,
                    PAD,
                    PAD,
                ],  # Doesn't count
                [
                    SOS,
                    SEE,
                    L_PAREN,
                    ALICE,
                    COMMA,
                    ALICE,
                    R_PAREN,
                    AND,
                    NEAR,
                    L_PAREN,
                    ALICE,
                    COMMA,
                    MARY,
                    R_PAREN,
                    EOS,
                ],  # Same as target
                [
                    SOS,
                    SEE,
                    L_PAREN,
                    MARY,
                    COMMA,
                    ALICE,
                    R_PAREN,
                    AND,
                    NEAR,
                    L_PAREN,
                    ALICE,
                    COMMA,
                    MARY,
                    R_PAREN,
                    EOS,
                ],  # Linear C error
            ]
        )
        prediction = F.one_hot(prediction).permute(0, 2, 1)

        lin_c_err = LinearCError(pad_token_id=PAD, and_token_id=AND)
        lin_c_err(prediction, target)

        assert lin_c_err.total == 2.0
        assert lin_c_err.correct == 1.0

    def test_hierarchical(self):
        """
        Given a target of the form
          victor beside leo knows himself -> <sos> know ( victor , victor ) & beside ( victor , leo )

        Test if the output is of form:
          know ( leo , victor ) & beside ( .... )

        i.e., reflexives are structural but subj-verb is linear

        total = # target sequences of the form "<sos> verb ( X , X ) & p ( X , Y )"
        correct = # predictions of the form "<sos> verb (Y, X) ...."
        """

        # TOKENS
        AND = 0
        SOS = 1
        EOS = 2
        PAD = 3

        L_PAREN = 4
        R_PAREN = 5
        COMMA = 6

        ALICE = 7
        MARY = 8

        SEE = 9

        NEAR = 11

        target = torch.tensor(
            [
                [
                    SOS,
                    SEE,
                    L_PAREN,
                    ALICE,
                    COMMA,
                    ALICE,
                    R_PAREN,
                    EOS,
                    PAD,
                    PAD,
                    PAD,
                    PAD,
                    PAD,
                    PAD,
                    PAD,
                ],
                [
                    SOS,
                    SEE,
                    L_PAREN,
                    ALICE,
                    COMMA,
                    ALICE,
                    R_PAREN,
                    AND,
                    NEAR,
                    L_PAREN,
                    ALICE,
                    COMMA,
                    MARY,
                    R_PAREN,
                    EOS,
                ],
                [
                    SOS,
                    SEE,
                    L_PAREN,
                    ALICE,
                    COMMA,
                    ALICE,
                    R_PAREN,
                    AND,
                    NEAR,
                    L_PAREN,
                    ALICE,
                    COMMA,
                    MARY,
                    R_PAREN,
                    EOS,
                ],
            ]
        )
        prediction = torch.tensor(
            [
                [
                    SOS,
                    SEE,
                    L_PAREN,
                    ALICE,
                    COMMA,
                    ALICE,
                    R_PAREN,
                    EOS,
                    PAD,
                    PAD,
                    PAD,
                    PAD,
                    PAD,
                    PAD,
                    PAD,
                ],  # Doesn't count
                [
                    SOS,
                    SEE,
                    L_PAREN,
                    ALICE,
                    COMMA,
                    ALICE,
                    R_PAREN,
                    AND,
                    NEAR,
                    L_PAREN,
                    ALICE,
                    COMMA,
                    MARY,
                    R_PAREN,
                    EOS,
                ],  # Same as target
                [
                    SOS,
                    SEE,
                    L_PAREN,
                    MARY,
                    COMMA,
                    ALICE,
                    R_PAREN,
                    AND,
                    NEAR,
                    L_PAREN,
                    ALICE,
                    COMMA,
                    MARY,
                    R_PAREN,
                    EOS,
                ],  # Linear C error
            ]
        )
        prediction = F.one_hot(prediction).permute(0, 2, 1)

        hier = HierarchicalAccuracy(pad_token_id=PAD, and_token_id=AND)
        hier(prediction, target)

        assert hier.total == 2.0
        assert hier.correct == 1.0

    def test_fullseqmodacc(self):
        # TOKENS
        SOS, EOS = 0, 1
        SEE, HEAR = 3, 4
        ALICE, MARY = 5, 6

        target = torch.tensor(
            [
                [SOS, ALICE, SEE, ALICE, EOS],
                [SOS, ALICE, SEE, ALICE, EOS],
                [SOS, ALICE, SEE, ALICE, EOS],
                [SOS, ALICE, SEE, ALICE, EOS],
            ]
        )
        prediction = torch.tensor(
            [
                [SOS, ALICE, SEE, ALICE, EOS],  # right
                [SOS, MARY, SEE, ALICE, EOS],  # wrong
                [SOS, ALICE, HEAR, ALICE, EOS],  # right if {SEE, HEAR}
                [SOS, MARY, HEAR, ALICE, EOS],  # wrong
            ]
        )
        prediction = F.one_hot(prediction).permute(0, 2, 1)

        acc = FullSequenceModAccuracy([[SEE, HEAR]])
        acc(prediction, target)

        assert acc.total == 4.0
        assert acc.correct == 2.0


if __name__ == "__main__":
    unittest.main()
