from efprob import *
from efprob.helpers import *
from efprob.builtins import *
import numpy as np

from efprob import *



def get_conditional_probability(state, conclusion_mask, condition_mask):
    sum_mask = mask_sum(conclusion_mask, condition_mask)
    marginal = state.MM(*sum_mask)
    sub_cond_mask = mask_restrict(sum_mask, condition_mask)
    return marginal.DM(*sub_cond_mask)

sp_S = SpaceAtom('S', [0, 1])
sp_T = SpaceAtom('T', [0, 1])
sp_C = SpaceAtom('C', [0, 1])

sp = Space(sp_S, sp_T, sp_C)


omega1 = [
    0.5,   # S=0, T=0, C=0
    0.1,   # S=0, T=0, C=1
    0.01,  # S=0, T=1, C=0
    0.02,  # S=0, T=1, C=1
    0.1,   # S=1, T=0, C=0
    0.05,  # S=1, T=0, C=1
    0.02,  # S=1, T=1, C=0
    0.2    # S=1, T=1, C=1
]

PSTC = State(omega1, sp)

PC_ST = get_conditional_probability(PSTC, [1, 1, 0], [0, 0, 1])
PS = PSTC.MM(1, 0, 0)
PT = PSTC.MM(0, 1, 0)
PC = PSTC.MM(0, 0, 1)
pST = PSTC.MM(1, 1, 0)

i = idn(PT.sp)
i2 = idn(PS.sp)
i3 = idn(PC.sp)
copy = copy2(PS.sp)
swap = swap(PS.sp, PC.sp)
cap = cap(PS.sp)
cup = cup(PS.sp)
discard = discard(PS.sp)
uniform = uniform_state(PS.sp)
cut = uniform * discard

PS_copy = copy * PS
y = PS_copy @ i
x = i2 @ PC_ST

f =  (i2 @ PC_ST) * (PS_copy @ i)
g = get_conditional_probability(PSTC, [1, 0, 0], [0, 1, 0])
print(g)
print(f)
def comb_compose(f, g):
    m = (i2 @ i @ (swap * f)) * (i @ copy) * (i2 @ g) * copy
    print(copy)
    print(swap * f)
    print(m)
    return (i2 @ i @ i3 @ cap) * (m @ i2) * cup

omega_cut = comb_compose((cut @ i2) * f, g)
result = get_conditional_probability(omega_cut, [1, 0, 0], [0, 0, 1])


# print(i.array)
# print(PC_ST.array)
# print(PT_CS.array)

# print(PS)
# print(PSTC.array)
# print(copy.array)
print(omega_cut)
print(result.array)
