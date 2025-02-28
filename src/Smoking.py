from efprob import *
from efprob.helpers import *
from efprob.builtins import *
import numpy as np

from efprob import *



def get_conditional_probability(state, condition_mask, conclusion_mask):
    sum_mask = mask_sum(conclusion_mask, condition_mask)
    marginal = state.MM(*sum_mask)
    sub_cond_mask = mask_restrict(sum_mask, conclusion_mask)
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

i_t = idn(PT.sp)
i_s = idn(PS.sp)
i_c = idn(PC.sp)
copy = copy2(PS.sp)
swap = swap(PS.sp, PC.sp)
cap = cap(PS.sp)
cup = cup(PS.sp)
discard = discard(PS.sp)
uniform = uniform_state(PS.sp)
cut = uniform * discard

PS_copy = copy * PS
y = PS_copy @ i_t
x = i_s @ PC_ST

f =  (i_s @ PC_ST) * (PS_copy @ i_t)
g = get_conditional_probability(PSTC, [1, 0, 0], [0, 1, 0])
print(g)
# print(f)
def comb_compose(f, g):
    m = (i_s @ i_t @ (swap * f)) * (i_s @ copy) * (i_s @ g) * copy
    print(copy)
    print(swap * f)
    print(m)
    return (i_s @ i_t @ i_c @ cap) * (m @ i_s) * cup

omega_cut = comb_compose((cut @ i_s) * f, g)
result = get_conditional_probability(omega_cut, [1, 0, 0], [0, 0, 1])


# print(i.array)
# print(PC_ST.array)
# print(PT_CS.array)

# print(PS)
# print(PSTC.array)
# print(copy.array)
# print(omega_cut)
print(result.array)
