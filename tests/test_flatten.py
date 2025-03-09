import numpy as np
from efprob import Space, Channel, SpaceAtom, State

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from BayesianSurgeryNet import BayesianSurgeryNet

def test_flatten_vars():
    # 1) Original joint distribution for A,B,C,D
    omega = [
        # A=0, B=0, C=0, D=0 -> index=0
        0.05,
        # A=0, B=0, C=0, D=1 -> index=1
        0.01,
        # A=0, B=0, C=1, D=0 -> index=2
        0.02,
        # A=0, B=0, C=1, D=1 -> index=3
        0.0,
        # A=0, B=1, C=0, D=0 -> index=4
        0.02,
        # A=0, B=1, C=0, D=1 -> index=5
        0.05,
        # A=0, B=1, C=1, D=0 -> index=6
        0.01,
        # A=0, B=1, C=1, D=1 -> index=7
        0.01,

        # A=1, B=0, C=0, D=0 -> index=8
        0.06,
        # A=1, B=0, C=0, D=1 -> index=9
        0.03,
        # A=1, B=0, C=1, D=0 -> index=10
        0.0,
        # A=1, B=0, C=1, D=1 -> index=11
        0.02,
        # A=1, B=1, C=0, D=0 -> index=12
        0.05,
        # A=1, B=1, C=0, D=1 -> index=13
        0.03,
        # A=1, B=1, C=1, D=0 -> index=14
        0.01,
        # A=1, B=1, C=1, D=1 -> index=15
        0.03
    ]   
    vars = ["A","B","C","D"]

    # 2) Construct the BayesianSurgeryNet (or whatever class holds _flatten_vars)
    bsn = BayesianSurgeryNet(omega, vars)

    # 3) Call _flatten_vars to flatten (B,C) => "CUT_FLATTEN"
    flattened_omega, new_vars_list = bsn._flatten_vars(["B","C"])

    # 4) Check that new_vars_list is as expected
    #    Suppose we want "CUT_FLATTEN" first, followed by "A","D"
    assert new_vars_list == ["CUT_FLATTEN", "A", "D"]

    print(flattened_omega)