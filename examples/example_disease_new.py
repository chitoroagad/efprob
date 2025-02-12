from efprob import Channel, Space, SpaceAtom, flip, point_pred, yes_pred

# A Disease has a probability of 1% to occur.
# There is a Test for the disease with the following ‘sensitivity’:
# - If someone has the Disease, then the Test is 90% positive;
# - If someone does not have the Disease, there is still a 5% chance that the Test is positive.
# 1. Compute the probability that a test for an arbitrary person is positive.
# 2. Assume that the test turns out positive: what is the probability of having the disease?

disease_atom = SpaceAtom("disease", ["D", "~D"])
disease_dom = Space(disease_atom)
disease = flip(1 / 100, disease_dom)  # The Disease has probability 1%
test = Channel.fromstates(
    [flip(9 / 10), flip(1 / 20)], disease_dom
)  # The Test sensitivity depends on the presence of the Disease (so it is a channel).
# PS. The codomain of flip, as it is not explicitly described, is True, False.
# channel : f: X -> Y / f: X x Y -> [0,1] /  f(x,-): Y -> [0,1] probability measure

# To solve 1, we just compose the disease with the test and check the probability of True
general_outcome = test >> disease
# print(general_outcome)
# Alternatively, we could also just print the probability of true by an additional composition:
positive_outcome = general_outcome >= yes_pred
# domain general_outcome = [True, False]
# predicate (True,False) -> [0,1] / yes_pred(True)=1, yes_pred(False)=0

# Actually, yes_pred = point_pred ("True", bool_dom), where the second syntax is more general and can be used in a variety of situations.
print(
    "The probability that a test for an arbitrary person is positive is ",
    int(positive_outcome * 100),
    "%",
    sep="",
)

# To solve 2, we consider the predicate given by the test resulting positive (i.e. True) and update the state
positive_test = test << yes_pred  # now this is a predicate
print(positive_test)
disease_update = disease / positive_test
yes_disease_update = disease_update >= point_pred("D", disease_dom)
print(
    "The probability of having the disease once the test turns out positive is ",
    int(yes_disease_update * 100),
    "%",
    sep="",
)

# Additional questions:
# 1. Let us now consider a new text with the following sensitivity:
# - If someone has the Disease, then the Test is 99% positive;
# - If someone does not have the Disease, there is still a 1% chance that the Test is positive.
#   How does this change the probability?
# 2. Back to the previous sensitivity, what is the probability of having the disease once the test turns out negative?
