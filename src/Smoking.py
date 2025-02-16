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
pST = PSTC.MM(1, 1, 0)

i = idn(PT.sp)
i2 = idn(PS.sp)
copy = copy2(PS.sp)


PS_copy = copy * PS
y = PS_copy @ i
x = i2 @ PC_ST

f =  (i2 @ PC_ST) * (PS_copy @ i)
# f = (PC_ST) * ((PS_copy) @ i)

# print(i.array)
# print(PC_ST.array)
# print(PT_CS.array)

# print(PS)
# print(PSTC.array)
# print(copy.array)
print(f.array)
