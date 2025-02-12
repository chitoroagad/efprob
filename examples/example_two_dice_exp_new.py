from efprob import Predicate, Space, uniform_state

# Assume you have a dice with uniform probability. What is the expectation of the sum of two throws?

dice = uniform_state(Space(None, [1, 2, 3, 4, 5, 6]))
twodice = dice @ dice
sum_rv = Predicate.fromfun(lambda x, y: x + y, twodice.cod)
print(f"pred: {sum_rv}")
print(twodice.expectation(sum_rv))

# What about n throws?


def sums_exp(n):
    ndice = dice**n
    sum_rv = Predicate.fromfun(lambda *xs: sum(xs), ndice.cod)
    return (ndice).expectation(sum_rv)


print("sum_exp(1)")
print(sums_exp(1))
print("sum_exp(4)")
print(sums_exp(4))
