from efprob import *

# Monty Hall Problem
# In a game show, the prize is a Car hidden behind one of three doors: Left (L), Middle (M), or Right (R). 
# The other two doors hide nothing.
# To win, the Player must choose the door hiding the Car.
# Initially, the Player has no information about the Car's location and chooses a door randomly, say Middle.
# 
# After the Player's choice, the Host -- who knows where the Car is -- opens one of the other two doors. 
# The opened door is not the Player's choice and does not hide the Car. 
# If the Player has chosen the door hiding the Car, the Host chooses randomly between the remaining doors.
#
# For example, assume the Player chooses Middle. Then
# - If the Car is behind Middle, the Host will open Left or Right with equal probability (50-50).
# - If the Car is behind Right, the Host will open Left.
#
# Suppose the Host opens the Left door (in the case where the Player initially chose Middle).
# The Player is then offered a choice: stick with the original choice (Middle) or switch to the other unopened door (Right).
# 
# Which strategy gives the Player a higher probability of winning: sticking with the original choice or switching?

###

# First task: describe the state of the system Car, Player, Host.
# This will have 3*3*3 = 27 outcomes, where some of them also have probability zero (Car=Left, Player=Right, Host=Left is an example).
# Important: the Host depends on both the Car and the Player!

car = uniform_state(Space(None,["L","M","R"])) # Car state
player = uniform_state(Space(None,["L","M","R"])) # Player state
car_and_player = car @ player # Joint state of Car and Player

# We now need to define the Host; this will be a channel, as it depends on the Car and the Player. 
noL = State([0,1/2,1/2],car.cod) 
# This state is uniform in M and R, and disregards L; similarly the ones below. They mimic what the Host does when the Player chooses the winning door.
noM = State([1/2,0,1/2],car.cod)
noR = State([1/2,1/2,0],car.cod)

host = chan_fromstates([noL,point_state("R",car.cod),point_state("M",car.cod),point_state("R",car.cod),noM,point_state("L",car.cod),point_state("M",car.cod),point_state("L",car.cod),noR],car_and_player.cod) 
# The order of car_and_player.cod is [L,L], [L,M], [L,R], [M,L], ...

# This has been lost in the new version for some reasons, but let us restore it.
def graph(chan):
    return idn(chan.dom) @ chan * copy(chan.dom)

host_car_player = graph(host) >> car_and_player # Finally, we have the whole state of the system.
# Important: The graph option here allows to get a state considering all three variables.
# If we had defined this to be host >> car_and_player, then the resulting state would've been over the host variable alone (it's actually the uniform probability)

# Second task: solve the problem by setting Player=M, Host=L
# These are predicates, as they are based on observations.
# We then need to update the system with both choices (conditional probability). 
# Then, by printing the resulting probability distribution of Car, we can see which choice is preferable.

player_choice = truth(car.cod) @ point_pred("M",player.cod) @ truth(car.cod) # predicate: the Player chooses M
host_choice=truth(car.cod) @ truth(car.cod) @ point_pred("L",car.cod) # predicate: the Host chooses L

observe_player_choice = host_car_player / player_choice # The system is updated with the Player's choice
observe_host_and_player_choices = observe_player_choice / host_choice # The system is updated with the Host's choice

final_monty = observe_host_and_player_choices.MM(1,0,0) # We marginalize to the Car variable

final_monty.plot() # We print the resulting probability

# Additional questions: 
# 1. How does final_monty change if we do not know the Player's choice?
# 2. Assume the Car distribution is not uniform, let's say L=R=1/3, M=2/3. How does this change final_monty?
# 3. Assume that whenever the Player chooses M, the Host chooses Left with probability 2/3. How does this change final_monty? How much should we increase the probability of the Host choosing Left to make final_monty uniform between Middle and Right?