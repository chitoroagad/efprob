from efprob import Space, binomial, chan_fromklmap, point_pred, range_sp, uniform_state

# We are looking at a pond and we wish to learn the number of fish.
# We catch twenty of them, mark them, and throw them back.
# Subsequently we catch another twenty, and find out that five of them are marked.
# What do we learn about the number of fish?
# Let’s assume the maximal number is 300, and we care only about tens.

# By assumption, the number of fish in the pond must be at least 20, so we start by defining the following.
#
fish_sp = Space("Fish number", [10 * i for i in range(2, 31)])
fish = uniform_state(fish_sp)
print("The probability distribution fish is a channel from", fish.dom, "to", fish.cod)
# In order not to complicate the calculations too much, we catch these 20 fish one by one, check if they
# are marked, and then throw them back, so that the probability of catching a marked fish remains the same.
# This is then described by binomials with N=20 and probability N/d=20/d, with d = the number of fish
N = 20
chan = chan_fromklmap(lambda d: binomial(N)(N / d), fish_sp, range_sp(N + 1))
# For the observation of 5 marked fish, we use a predicate:
marked_fish = point_pred(5, range_sp(N + 1))

# Finally, we compute the possible number of fish
fish_update = fish / (chan << marked_fish)
fish_update.plot()
print("\nThe expectation number of fish is", fish_update.expectation())
print(fish_update)
# assert np.isclose(fish_update.expectation(),
#                       116.491)
#
# Expected number after catching 10 marked
#
# assert np.isclose((fish /
#                     (chan << point_pred(10, range_sp(N+1)))).expectation(),
#                       47.481)

# posterior.plot()
