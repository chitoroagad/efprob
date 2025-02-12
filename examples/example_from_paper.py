from sympy import Matrix, eye, pprint
from sympy.physics.quantum import TensorProduct

# composition is '*'


# monoidal product
def T(*args):
    if len(args) == 0:
        return Matrix([[1]])
    elif len(args) == 1:
        return args[0]
    else:
        return TensorProduct(*args)


# swaps
def swapMN(m, n):
    return Matrix(
        [
            [1 if (i // m == j % n and i % m == j // n) else 0 for i in range(m * n)]
            for j in range(n * m)
        ]
    )


# compact closed structure
def cupN(n):
    return Matrix([1 if j // n == j % n else 0 for j in range(n * n)])


def capN(n):
    return cupN(n).transpose()


# CDU structure
def copyN(n):
    return Matrix(
        [[1 if n * i + i == j else 0 for i in range(n)] for j in range(n * n)]
    )


def discardN(n):
    return Matrix([[1 for i in range(n)]])


def uniformN(n):
    return Matrix([[1.0 / n] for i in range(n)])


# specialisation to bits
copy = copyN(2)
discard = discardN(2)
uniform = uniformN(2)
cup = cupN(2)
cap = capN(2)
swap = swapMN(2, 2)
i = eye(2)
cut = uniform * discard


# (comb) disintegration functions
def disint(p):
    # print(f"PSHAPE: {p.shape}")
    in_dim = p.shape[0] // 2
    pA = T(eye(in_dim), discard) * p
    pAinv = pA.copy()
    for j in range(len(pAinv)):
        pAinv[j] = 1 / pAinv[j]
    adjust = (copyN(in_dim) * pAinv).transpose()
    return (pA, T(adjust, i) * T(eye(in_dim), p))


def comb_disint(p):
    pAB, pC_AB = disint(p)
    pA, g = disint(pAB)

    # print("\nPA:")
    # pprint(pA)
    # print("\ncopy PA:")
    # pprint(copy * pA)
    # print("\nT(copy PA, i):")
    # pprint(T(copy * pA, i))
    # print(f"P(C_AB).shape: {pC_AB.shape}")
    # print(f"T(i, P(C_AB)).shape: {T(i, pC_AB).shape}")
    #
    # print("\nT(copy*PA, i) * T(i, pC_AB):")
    f = T(i, pC_AB) * T(copy * pA, i)

    pprint(f)
    return (f, g)


def comb_compose(f, g):
    m = T(i, i, swap * f) * T(i, copy) * T(i, g) * copy
    return T(i, i, i, cap) * T(m, i) * cup


#### START OF DEMO ####

# Scenario 1: the scientist is right
omega1 = Matrix(
    [
        0.5,  # S = 0, T = 0, C = 0
        0.1,  # S = 0, T = 0, C = 1
        0.01,  # S = 0, T = 1, C = 0
        0.02,  # S = 0, T = 1, C = 1
        0.1,  # S = 1, T = 0, C = 0
        0.05,  # S = 1, T = 0, C = 1
        0.02,  # S = 1, T = 1, C = 0
        0.2,  # S = 1, T = 1, C = 1
    ]
)

# Scenario 2: the tobacco company is right
omega2 = Matrix(
    [
        0.14,  # S = 0, T = 0, C = 0
        0.05,  # S = 0, T = 0, C = 1
        0.16,  # S = 0, T = 1, C = 0
        0.05,  # S = 0, T = 1, C = 1
        0.1,  # S = 1, T = 0, C = 0
        0.21,  # S = 1, T = 0, C = 1
        0.1,  # S = 1, T = 1, C = 0
        0.19,  # S = 1, T = 1, C = 1
    ]
)

# Scenario 3: the data shows something totally unexpected
omega3 = Matrix(
    [
        0.3,  # S = 0, T = 0, C = 0
        0.05,  # S = 0, T = 0, C = 1
        0.2,  # S = 0, T = 1, C = 0
        0.05,  # S = 0, T = 1, C = 1
        0.05,  # S = 1, T = 0, C = 0
        0.05,  # S = 1, T = 0, C = 1
        0.25,  # S = 1, T = 1, C = 0
        0.05,  # S = 1, T = 1, C = 1
    ]
)


omega = omega1
# omega = omega2
# omega = omega3

print("\nomega =\n")
pprint(disint(T(i, discard, i) * omega)[1])

print("\nc =\n")
pprint(disint(T(i, discard, i) * omega)[1])

f, g = comb_disint(omega)
print("\nf =\n")
pprint(f)
print("\ng =\n")
pprint(g)

omega_cut = comb_compose(T(cut, i) * f, g)
print("\nomega' =\n")
pprint(omega_cut)

print("\nc' =\n")
pprint(disint(T(i, discard, i) * omega_cut)[1])

print()
