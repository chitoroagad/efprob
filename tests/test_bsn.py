import numpy as np
from efprob import Space, Channel, SpaceAtom, State

from src.BayesianSurgeryNet import BayesianSurgeryNet

# We use the data from the smoking example
omega = [
    0.5,   # S=0, T=0, C=0
    0.1,   # S=0, T=0, C=1
    0.01,  # S=0, T=1, C=0
    0.02,  # S=0, T=1, C=1
    0.1,   # S=1, T=0, C=0
    0.05,  # S=1, T=0, C=1
    0.02,  # S=1, T=1, C=0
    0.2    # S=1, T=1, C=1
]
vars = ['S', 'T', 'C']

bsn = BayesianSurgeryNet(omega, vars)

def test_comb_disint():
    # Given
    expected_f = np.array([
        [[0.525, 0.21],
        [0.105, 0.42]],
        [[0.24666667, 0.03363636],
        [0.12333333, 0.33636364]]
    ])

    expected_g = np.array([
        [0.95238095, 0.40540541],
        [0.04761905, 0.59459459]
    ])

    # When
    # Disintegrate on S and observe T
    f, g = bsn._comb_disint(Space(bsn.sp[0]), [1, 0, 0], Space(bsn.sp[1]), [0, 0, 1])

    # Then
    assert(g.array.shape == expected_g.shape)
    assert(f.array.shape == expected_f.shape)
    
    assert np.allclose(f.array, expected_f)
    assert np.allclose(g.array, expected_g)

def test_comb_compose():
    # Given
    expected_f = np.array([
        [[0.525, 0.21],
        [0.105, 0.42]],
        [[0.24666667, 0.03363636],
        [0.12333333, 0.33636364]]
    ])

    expected_g = np.array([
        [0.95238095, 0.40540541],
        [0.04761905, 0.59459459]
    ])

    expected_omega_cut = np.array([
        [[0.36746032, 0.10873016],
        [0.00580087, 0.01800866]],
        [[0.15641892, 0.04628378],
        [0.07243243, 0.22486486]]
    ])

    # When
    composed_state = bsn._comb_compose(
        Channel(expected_f, Space(bsn.sp[1]), Space(bsn.sp[0], bsn.sp[1])),
        Channel(expected_g, Space(bsn.sp[0]), Space(bsn.sp[1])),
        Space(bsn.sp[0]),
        Space(bsn.sp[1]),
        Space(bsn.sp[2]),
    )

    # Then
    assert composed_state.array.shape == expected_omega_cut.shape
    assert np.allclose(composed_state.array, expected_omega_cut)

def test_cut_and_compute():
    # Given
    expected_results = np.array([
        [[0.74652237, 0.4577027 ],
        [0.25347763, 0.5422973 ]]
    ])

    # When
    result = bsn.cut_and_compute('S', 'C')

    # Then
    assert np.allclose(result.array, expected_results)

def test_reorder():
    # Given
    permutation = ['S', 'C', 'T']
    new_omega = [
        0.5,   # S=0, T=0, C=0
        0.01,  # S=0, T=1, C=0
        0.1,   # S=0, T=0, C=1
        0.02,  # S=0, T=1, C=1
        0.1,   # S=1, T=0, C=0
        0.02,  # S=1, T=1, C=0
        0.05,  # S=1, T=0, C=1
        0.2    # S=1, T=1, C=1
    ]

    expected_state = State(new_omega, Space(*[SpaceAtom(var, [0, 1]) for var in permutation]))

    # Then
    state = bsn._reorder_states(permutation)

    # Expect
    assert state == expected_state

def test_reorder2():
        # Given
    permutation = ['T', 'S', 'C']
    new_omega = [
        0.5,   # S=0, T=0, C=0
        0.1,   # S=0, T=0, C=1
        0.1,   # S=1, T=0, C=0
        0.05,  # S=1, T=0, C=1
        0.01,  # S=0, T=1, C=0
        0.02,  # S=0, T=1, C=1
        0.02,  # S=1, T=1, C=0
        0.2    # S=1, T=1, C=1
    ]

    expected_state = State(new_omega, Space(*[SpaceAtom(var, [0, 1]) for var in permutation]))

    # Then
    state = bsn._reorder_states(permutation)

    # Expect
    assert state == expected_state

