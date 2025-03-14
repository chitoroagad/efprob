import numpy as np
import itertools

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
    # TODO: do we need to pass the spaces separately?
    # I get this is useful for passing in the domain for the variables, but it breaks our abstraction?
    def __init__(self, omega: List[float], vars: List[str], spaces: List[SpaceAtom]):
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
        self.sp = Space(*spaces)
        total_size = self.sp.size()
        if len(omega) != total_size:
            raise ValueError(f"Provided omega size {len(omega)} "
                            f"does not match space size {total_size}")
        self.state = State(omega, self.sp)

    def __repr__(self):
        return f"BayesianSurgeryNet({self.omega=}, {self.vars=})"

    # TODO: why are we passing obser_var? Is observ_var not just the remaining variables?
    def cut_and_compute(self, cut_var: str, trans_var: str, observ_var: str) -> State:
        """
        Perform a causal intervention by cutting the influence to cut_var,
        and then compute the conditional probability of observ_var given cut_var.
        
        Parameters:
        cut_var     (str):  The variable to be cut (intervention)
        trans_var   (str):  The bridge between cut and observ
        observ_var  (str):  The variable to observe the effect on
        
        Returns:
        State:  The conditional probability after intervention
        """
        # Ensure the variables are in the domain
        self._validate_vars(cut_var, trans_var, observ_var)

        # According to our comb_disint invariant, we first have to reorder the dom
        # The ordering is [*cut_vars, *other_vars, *observ_vars]
        new_order = [cut_var] + [trans_var] + [observ_var]
        state = self._reorder_states(new_order)


        # 1. Find indices for our variables
        cut_idx = new_order.index(cut_var)
        trans_idx = new_order.index(trans_var)
        observ_idx = new_order.index(observ_var)


        # 2. Create masks for accessing different parts of the state
        cut_mask = [0 if var != cut_var else 1 for var in new_order]
        trans_mask = [0 if var != trans_var else 1 for var in new_order]
        observ_mask = [0 if var != observ_var else 1 for var in new_order]


        # 3. Get the spaces for the variables
        cut_space = Space(self.sp[cut_idx])
        trans_space = Space(self.sp[trans_idx])
        observ_space = Space(self.sp[observ_idx])

        # 4. Comb disintegration
        f, g = self._comb_disint(state, cut_space, cut_mask, trans_space, trans_mask, observ_space, observ_mask)

        # 5. Comb compose
        composed_state = self._comb_compose(f, g, cut_space, observ_space, trans_space)


        # 6. Get the conditional probability
        return self._get_conditional_probability(composed_state, cut_mask, observ_mask)

    # TODO: do we really need all these params?
    def _comb_disint(
            self,
            state: State,
            cut_space: Space,
            cut_mask: List[int],
            trans_space: Space,
            trans_mask: List[int],
            observ_space: Space,
            observ_mask: List[int],
        ) -> Tuple[Channel, Channel]:
        """
        Perform comb disintegration on the state, given the cut and observation

        INVARIANT: The cut variable must be the first variable in the state

        Parameters:
        cut_space       (Space):        The space for the cut variable
        cut_mask        (List[0, 1]):   The mask for the cut variable
        trans_space       (Space):        The space for the trans variable
        trans_mask        (List[0, 1]):   The mask for the trans variable
        observ_space    (Space):        The space for the observation variable
        observ_mask     (List[0, 1]):   The mask for the observation variable

        :return: The disintegrated states f and g
        """
        # TODO: This kind assume that the cut variable is the first variable (so we need to swap them first)

        # g describes the conditional probability of the observ variable given the cut variable
        P_cut = state.MM(1, 0, 0)
        # print(trans_mask)
        summask = mask_sum(trans_mask, cut_mask)
        g = self._get_conditional_probability(state, cut_mask, trans_mask)

        # f is the disintegration of the cut variable
        #   We need the cut marginal first and create a copy for it
        i_cut = idn(cut_space)
        i_trans = idn(trans_space)
        cut_copy = copy2(cut_space)

        observ_given_rest = self._get_conditional_probability(state, summask, observ_mask)
        # print(observ_given_rest)
        f = (i_cut @ observ_given_rest) * ((cut_copy * P_cut) @ i_trans)
        # print(f)
        return f, g

    def _comb_compose(
            self,
            f: Channel,
            g: Channel,
            cut_space: Space,
            observ_space: Space,
            trans_space: Space,
        ) -> State:
        """
        Use the comb disintegrated morphisms to compose the 'cut' state

        Parameters:
        f   (Channel): The disintegrated state for the cut variable (trans -> cut @ observ)
        g   (Channel): The conditional probability for the observation variable given the cut variable (cut -> trans)

        Returns:
        The composed state
        """
        idn_cut = idn(cut_space)
        idn_observ = idn(observ_space)
        idn_trans = idn(trans_space)

        # Get the cut and swap channels
        swap_chan = swap(cut_space, observ_space)
        uniform_chan = uniform_state(cut_space)
        discard_chan = discard(cut_space)
        cut = uniform_chan * discard_chan

        # TODO: verify the identities used in this, they are the same, so passes regardless
        # TODO: is there a more visual (aesthetic) way to write this?
        m = (idn_cut @ idn_trans @ (swap_chan * (cut @ idn_observ) * f)) \
            * (idn_cut @ copy2(trans_space)) \
            * (idn_cut @ g) * copy2(cut_space)
        return (idn_cut @ idn_trans @ idn_observ @ cap(cut_space)) * (m @ idn_cut) * cup(cut_space)

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
        
        return result_state.as_state()

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


    def flatten_vars(self, new_name: str, var_list: List[str]) -> State:
        """
        Merge the binary variables in 'var_list' into one new dimension (of size 2^k).
        This new dimension is named 'new_name'. All remaining variables retain their original order.

        :param new_name:  The name for the new merged dimension
        :param var_list:  The list of variables to be merged
        :return:          A new State whose dimensionality is [new_dim, leftover_dims...].
                          The first dimension corresponds to 'new_name' with size=2^k.
        """
        self._validate_vars(*var_list)
    
        # If only one or no vars to merge, return the original state
        if len(var_list) < 2:
            return self.state
            
        # Get indices of variables to merge and keep
        merge_indices = [self.vars.index(v) for v in var_list]
        keep_indices = [i for i in range(len(self.vars)) if i not in merge_indices]
        
        # Create permutation for reordering (merged vars first, then kept vars)
        perm = merge_indices + keep_indices
        
        # Create the transpose operation to reorder dimensions
        reordered = np.transpose(self.state.array, perm)
        
        # Reshape to combine the first n dimensions
        n_merged = len(merge_indices)
        merged_shape = (2**n_merged,) + tuple(self.state.array.shape[i] for i in keep_indices)
        reshaped = reordered.reshape(merged_shape)
        
        # Create new space objects
        merged_combos = list(itertools.product([0, 1], repeat=n_merged))
        merged_atom = SpaceAtom(new_name, merged_combos)
        keep_atoms = [self.sp[i] for i in keep_indices]
        new_space = Space(merged_atom, *keep_atoms)
        
        return State(reshaped, new_space)

    # TODO: merge this into cut_and_compute
    # The interface should be something liFLATTEN_LIST_SIZEe cut_and_compute(cut_vars: str | List[str], obser_var: str | List[str])
    # Users shouldn't have to assign the merged var names, we should handle that
    def cut_multiple_vars(
        self,
        merge_vars: List[str], 
        new_name: str,
        cut_var: str,
        trans_var: str,
        observ_var: str
    ) -> State:
        """
        1) Merge the specified 'merge_vars' into one new dimension named 'new_name';
        2) Construct a new BayesianSurgeryNet from that merged State;
        3) Call cut_and_compute(cut_var, trans_var, observ_var) on the new net;
        4) Return the resulting conditional probability State.

        :param merge_vars: The list of variables to merge (e.g. ["A","D"]).
        :param new_name:   The name for the new merged dimension (e.g. "AD").
        :param cut_var:    In the merged network, which variable to cut/intervene on.
                           (Often the newly merged dimension, e.g. "AD", but it could be any.)
        :param trans_var:  Which variable to treat as the intermediate/bridge in cut_and_compute.
        :param observ_var: Which variable to observe in cut_and_compute.
        :return:           A State object representing P(observ_var | cut_var) after the intervention.
        """
        # 1 Flatten current net to combine merge_vars => new_name
        flattened_state = self.flatten_vars(new_name, merge_vars)
        # print("Flattened State:\n", flattened_state)

        # 2 Gather the dimension labels and space atoms from flattened_state
        new_vars_list = [at.label for at in flattened_state.sp]
        new_spaces_list = list(flattened_state.sp)
        
        # 3 Create a new net from the flattened state
        new_omega = flattened_state.array.flatten()
        merged_net = BayesianSurgeryNet(new_omega, new_vars_list, new_spaces_list)

        # 4 Perform the cut_and_compute in the merged net
        result_state = merged_net.cut_and_compute(cut_var, trans_var, observ_var)
        # print(f"Result after cut_and_compute({cut_var}, {trans_var}, {observ_var}):\n", result_state)

        return result_state


    

from efprob import SpaceAtom
from typing import List

# We'll define five variables (A,B,C,D,E), each binary [0,1]
vars5 = ["A","B","C","D","E"]
spaces5 = [SpaceAtom(v, [0,1]) for v in vars5]

omega5 = [0.03125]*32  # 32 elements, each 0.03125 => sums to 1.0

bsn5 = BayesianSurgeryNet(omega5, vars5, spaces5)


result_state = bsn5.cut_multiple_vars(
    merge_vars=["A","D","E"],  # merge these two binary variables
    new_name="ADE",         # call the merged dimension "ADE"
    cut_var="ADE",          # we want to cut on the newly merged "ADE"
    trans_var="B",         # "B" acts as the bridging variable
    observ_var="C"         # "C" is the observation
)

print("\nFinal result of cut_multiple_vars (ADE,B,C):")
print(result_state.array)


