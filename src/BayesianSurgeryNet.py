import numpy as np

from typing import List, Tuple
from efprob import (
    Space,
    State,
    Channel,
    SpaceAtom,
    idn,
    cap,
    cup,
    swap,
    copy2,
    discard,
    uniform_state,
)

from efprob.helpers import mask_sum, mask_restrict

class BayesianSurgeryNet:
    def __init__(self, omega: List[float], vars: List[str]):
        """
        Parameters:
        omega   (List[float]):  The observation data
        vars    (List[str]):    The order of the variables

        For example, given
        omega = [
            0.5,   # S=0, T=0, C=0
            0.1,   # S=0, T=0, C=1
            0.01,  # S=0, T=1, C=0
            0.02,  # S=0, T=1, C=1
            0.1,   # S=1, T=0, C=0
            0.05,  # S=1, T=0, C=1
            0.02,  # S=1, T=1, C=0
            0.2    # S=1, T=1, C=1
        ], vars should be ['S', 'T', 'C']
        """
        # TODO: for now, we assume all variables are binary, change this?
        self.omega = omega
        self.vars = vars

        self.sp = Space(*[SpaceAtom(var, [0, 1]) for var in vars])
        self.state = State(omega, self.sp)

    def __repr__(self):
        return f"BayesianSurgeryNet({self.omega=}, {self.vars=})"

    def cut_and_compute(self, cut_var: str, observ_var: str) -> State:
        """
        Perform a causal intervention by cutting the influence to cut_var,
        and then compute the conditional probability of observ_var given cut_var.
        
        Parameters:
        cut_var     (str):  The variable to be cut (intervention)
        observ_var  (str):  The variable to observe the effect on
        
        Returns:
        State:  The conditional probability after intervention
        """
        # TODO: Only 1 cut variable for now
        # TODO: Only 1 observation variable for now

        # Ensure the variables are in the domain
        self._validate_vars(cut_var, observ_var)

        # According to our comb_disint invariant, we first have to reorder the dom
        # The ordering is [*cut_vars, *other_vars, *observ_vars]
        new_order = [cut_var] + [var for var in self.vars if var not in [cut_var, observ_var]] + [observ_var]
        state = self._reorder_states(new_order)

        # 1. Find indices for our variables
        cut_idx = new_order.index(cut_var)
        observ_idx = new_order.index(observ_var)

        # 2. Create masks for accessing different parts of the state
        cut_mask = [0 if var != cut_var else 1 for var in new_order]
        observ_mask = [0 if var != observ_var else 1 for var in new_order]

        # 3. Get the spaces for the variables
        cut_space = Space(self.sp[cut_idx])
        observ_space = Space(self.sp[observ_idx])
        irrelevant_space = Space(*[SpaceAtom(var, [0, 1]) for var in new_order if var not in [cut_var, observ_var]])

        # 4. Comb disintegration
        f, g = self._comb_disint(state, cut_space, cut_mask, observ_space, observ_mask)

        # 5. Comb compose
        composed_state = self._comb_compose(f, g, cut_space, irrelevant_space, observ_space)

        # 6. Get the conditional probability
        return self._get_conditional_probability(composed_state, cut_mask, observ_mask)

    def _comb_disint(
            self,
            state: State,
            cut_space: Space,
            cut_mask: List[int],
            observ_space: Space,
            observ_mask: List[int]
        ) -> Tuple[Channel, Channel]:
        """
        Perform comb disintegration on the state, given the cut and observation

        INVARIANT: The cut variable must be the first variable in the state

        Parameters:
        cut_space       (Space):        The space for the cut variable
        cut_mask        (List[0, 1]):   The mask for the cut variable
        observ_space    (Space):        The space for the observation variable
        observ_mask     (List[0, 1]):   The mask for the observation variable

        :return: The disintegrated states f and g
        """
        # TODO: This kind assume that the cut variable is the first variable (so we need to swap them first)

        # g describes the conditional probability of the irrelevant variable given the cut variable
        irrelevant_mask = [
            1 if cut_mask[i] == 0 and observ_mask[i] == 0 else 0
            for i in range(len(cut_mask))
        ]
        g = self._get_conditional_probability(state, cut_mask, irrelevant_mask)

        # f is the disintegration of the cut variable
        #   We need the cut marginal first and create a copy for it
        i_cut = idn(cut_space)
        i_observ = idn(observ_space)
        copy_chan = copy2(cut_space)

        cut_marginal = state.MM(*cut_mask)
        cut_copy = copy_chan * cut_marginal

        # create a mask that is the not(observ mask), mask flipped
        observ_complement_mask = [1 - mask for mask in observ_mask]
        observ_given_rest = self._get_conditional_probability(state, observ_complement_mask, observ_mask)

        f = (i_cut @ observ_given_rest) * (cut_copy @ i_observ)

        return f, g

    def _comb_compose(
            self,
            f: Channel,
            g: Channel,
            cut_space: Space,
            irrelevant_space: Space,
            observ_space: Space
        ) -> State:
        """
        Use the comb disintegrated morphisms to compose the 'cut' state

        Parameters:
        f   (Channel): The disintegrated state for the cut variable
        g   (Channel): The conditional probability for the observation variable given the cut variable

        Returns:
        The composed state
        """
        idn_cut = idn(cut_space)
        idn_irrelevant = idn(irrelevant_space)
        idn_observ = idn(observ_space)

        uniform_chan = uniform_state(cut_space)
        discard_chan = discard(cut_space)
        cut = uniform_chan * discard_chan
        swap_chan = swap(cut_space, observ_space)

        f = (cut @ idn_cut) * f
        m = (idn_cut @ idn_observ @ (swap_chan * f)) * (idn_cut @ copy2(cut_space)) * (idn_cut @ g) * copy2(cut_space)
        return (idn_cut @ idn_observ @ idn_irrelevant @ cap(cut_space)) * (m @ idn_cut) * cup(cut_space)

    def _reorder_states(self, new_order: List[str]) -> State:
        """
        Reorder the state variables to match the new variable order.
        
        Parameters:
        new_order   (List[str]): The new order of variables
        
        Returns:
        A new State with the variables in the new order
        """
        assert len(new_order) == len(self.vars), "New order must have the same number of variables"

        # Validate the variables
        self._validate_vars(*new_order)
        
        # If the orders are the same, return the original state
        if self.vars == new_order:
            return self.state
            
        # Create a working copy of current ordering
        current_order = list(self.vars)
        
        # Create a working copy of the state
        result_state = self.state
        
        # TODO: bubble sort for now, but just find the index and pad with identities in between, padding with identies will probably be annoying tho
        for target_idx, var in enumerate(new_order):
            current_idx = current_order.index(var)
            
            # Swap the variable into place
            while current_idx > target_idx:
                # Create swap channel to exchange adjacent variables
                idx = current_idx - 1
                swap_channel = swap(
                    Space(self.sp[idx]), 
                    Space(self.sp[current_idx])
                )

                # Create identity channels for all other spaces
                prefix_spaces = Space(*[self.sp[i] for i in range(0, idx)])
                suffix_spaces = Space(*[self.sp[i] for i in range(current_idx+1, len(self.vars))])
                
                # Pad with identity channels
                if current_idx > 0:
                    prefix_idn = idn(prefix_spaces)
                    swap_channel = prefix_idn @ swap_channel
                
                if current_idx < len(self.vars) - 1:
                    suffix_idn = idn(suffix_spaces)
                    swap_channel = swap_channel @ suffix_idn
                    
                # Apply the swap to the state
                result_state = swap_channel * result_state
                
                # Update our tracking of the current order
                current_order[idx], current_order[current_idx] = current_order[current_idx], current_order[idx]
                current_idx = idx
        
        return result_state

    def _validate_vars(self, *vars: List[str]):
        """
        Iterates through the variables and checks if they are in the domain
        
        Throws an `ValueError` if the var is not in the domain
        """
        for var in vars:
            if var not in self.vars:
                raise ValueError(f"Variable {var} not found in the domain")

    def _get_conditional_probability(self, state, condition_mask, conclusion_mask) -> State:
        """
        Helper function to get the conditional probability of a state
        Applies a mask to the state to get the conditional probability of the conclusion given the condition

        Parameters:
        state           (State):    The state to get the conditional probability from
        condition_mask  (List[int]): The mask for the condition
        conclusion_mask (List[int]): The mask for the conclusion

        Returns:
        State:  The conditional probability
        """
        sum_mask = mask_sum(conclusion_mask, condition_mask)
        marginal = state.MM(*sum_mask)
        sub_cond_mask = mask_restrict(sum_mask, conclusion_mask)
        return marginal.DM(*sub_cond_mask)
