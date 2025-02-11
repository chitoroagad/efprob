from typing import List, Dict


class BayesianNetwork:
    # Special node name for uniform distribution
    UNIFORM_NODE = "uniform"
    # TODO:
    #   do we really need this? Can we just update the cpd with the uniform distribution?
    #   one reason this might be useful is for plotting the graph, we can add some distinction to the uniform nodes i.e. color/name

    def __init__(self, graph: Dict[str, List[str]], cpds: Dict[str, Dict[str, float]]):
        """
        :param graph: A dictionary with nodes as keys and their parents as values
        :param cpds: A dictionary with nodes as keys and the distributions as values

        :return BayesianNetwork
        """
        self.graph = graph
        self.cpds = cpds

    def compute(self, node: str, evidence: Dict[str, bool]):
        """
        :param node: The node to compute
        :param evidence: A dictionary with nodes as keys and their values as values

        TODO: should we return a float or a channel?
        :return float: The probability of the node given the evidence
        """
        # TODO: Thus far, we assume that the evidence is boolean only, we should make it less restrictive
        pass

    def cut(self, node: str) -> BayesianNetwork:
        """
        :param node: The node to cut, i.e. substitute with an uniform distribution

        :return BayesianNetwork: A new BayesianNetwork with the node cut and otherwise identical
        """
        pass

    def plot(self) -> None:
        """
        :return None: Plot the BayesianNetwork (probably using networkx or something)
        """
        pass
