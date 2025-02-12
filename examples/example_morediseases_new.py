from efprob import *

#Imagine we have the following diseases with these probability distributions:

lc=flip(0.01, Space(None,["LC","~LC"])) # Lung-cancer
tb=State([0.005,0.995],Space(None,["TB","~TB"])) # Tubercolosis
co=State([0.2,0.8],Space(None,["CO","~CO"])) # Cold
sf=State([0.1,0.9],Space(None,["SF","~SF"])) # Stomach-flu
ot=State([0.1,0.9],Space(None,["OT","~OT"])) # Other

# Let us consider them together as a state.
diseases=lc@tb@co@sf@ot

# We now want to define predicates that allow to consider when a certain disease occurs.
# First, a predicate that does nothing 
lcT=truth(lc.sp)
tbT=truth(tb.sp)
coT=truth(co.sp)
sfT=truth(sf.sp)
otT=truth(ot.sp)

# Then a predicate that checks the disease
lcY=point_pred("LC", lc.sp)
tbY=Predicate([1,0],tb.sp)
coY=Predicate([1,0],co.sp)
sfY=Predicate([1,0],sf.sp)
otY=Predicate([1,0],ot.sp)

# Now the predicates in diseases.dom
lc_yes=lcY@tbT@coT@sfT@otT 
tb_yes=lcT@tbY@coT@sfT@otT
co_yes=lcT@tbT@coY@sfT@otT
sf_yes=lcT@tbT@coT@sfY@otT
ot_yes=lcT@tbT@coT@sfT@otY 

# Let us now assume the following predicates, and how they are distributed based on the diseases. 
cough = 0.5 * co_yes | 0.3 * lc_yes | 0.7 * tb_yes | 0.01 * ot_yes 
fever = 0.3 * co_yes | 0.5 * sf_yes | 0.2 * tb_yes | 0.01 * ot_yes 
chest_pain = 0.4 * lc_yes | 0.5 * tb_yes | 0.01 * ot_yes 
short_breath = 0.4 * lc_yes | 0.5 * tb_yes | 0.01 * ot_yes

updated_diseases = diseases / (cough & fever & chest_pain & short_breath) # new state given by observing all the symptoms above

#What are the probabilities of the combinations lung-cancer and tuberculosis?
lc_and_tb_updated=updated_diseases.MM(1,1,0,0,0)
print(lc_and_tb_updated)

## Now plot doesn't work anymore, and I had to add my own flattening method...
# from flattening import *s
# flatten_state(lc_and_tb_updated).plot()