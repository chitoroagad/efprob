from efprob import Predicate, Space, flip

# If it rains, an umbrella sales person can earn €100 per day.
# If it is a fair weather day (s)he can lose €20 per day.
# What is the expected return if the probability of rain is 0.3?

rain = flip(0.3, Space(None, ["R", "~R"]))  # probability of rain
print(f"rain.cod: {rain.cod}, rain.dom: {rain.dom}")
umbrella_sales = Predicate(
    [100, -20], rain.cod
)  # this is basically a predicate but it has negative range as well
print("The expected return is €", int(rain >= umbrella_sales), sep="")

print(
    (rain.expectation(umbrella_sales)) == (rain >= umbrella_sales)
)  # these are two ways of doing the same thing

