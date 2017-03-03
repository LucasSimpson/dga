
class Model:
    """Constructed from a genotype; the realization from a genotype."""

    gene_size = 1

    def __init__(self, genotype):
        self.genotype = genotype

    def evaluate(self, data_in):
        raise NotImplemented()