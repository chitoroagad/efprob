from efprob import (
    Space, SpaceAtom, State
)

omega1 = [
    0.5,   # S = 0, T = 0, C = 0
    0.1,   # S = 0, T = 0, C = 1
    0.01,  # S = 0, T = 1, C = 0
    0.02,  # S = 0, T = 1, C = 1
    0.1,   # S = 1, T = 0, C = 0
    0.05,  # S = 1, T = 0, C = 1
    0.02,  # S = 1, T = 1, C = 0
    0.2    # S = 1, T = 1, C = 1
]

S = SpaceAtom("S", [0, 1])
T = SpaceAtom("T", [0, 1])
C = SpaceAtom("C", [0, 1])

sample_space = Space(S, T, C)
omega_state = State(omega1, sample_space)

marginal_t = omega_state.MM(1, 0, 1)
print("Marginal T=")
print(marginal_t.array)

c = marginal_t.DM(0, 1)
print("p(C|S)=")
print(c.array)
