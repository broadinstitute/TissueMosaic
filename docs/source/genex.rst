Gene regression
===============
This set of classes/functions can be used to predict the gene expression profile
based on:

1. the cell-type and

2. cell-specific covariates (such as location in space and local micro-environment).

The intended use case is to test whether the semantic features
extracted by one the Self-Supervised Lerning (ssl) algorithm
contain biologically relevant information.

We treat each cell-type separately.
For each cell, the gene counts are modelled as a multinomial distribution:

.. math::
    c \sim \frac{N!}{n_1!n_2!\dots n_g!} p_1^{n_1} p_2^{n_2} \dots p_g^{n_g}

where :math:`N=\sum_{g=1}^G n_g` is the total number of counts in the cell
(sometimes referred as the total UMI count) and :math:`\sum_{g=1}^G p_g=1`
are the probabilities of measuring each gene.
When :math:`N` is large and :math:`p_i` are small the counts for each gene
can be approximated by a Poisson distribution with rate :math:`r_i = N p_i`.
Therefore the counts for cell :math:`n` and gene :math:`g` are modelled as:

.. math::
    c_{ng} \sim \text{Poi}( r_{ng} = N_n \, p_{ng})

To account for noise and the presence of L (cell-specific) covariates,
we model the probability as:

.. math::
    \log p_{ng} = \left( \beta_g^0 + \sum_l \beta_{lg} X_{nl} \right) 

where :math:`\beta_g^0` is a gene-specific intercepts, :math:`X_{nl}` are the cell covariates

We recap the dimension of the variable involved in the full model (for `K` different cell-types):

1. :math:`X_{nl}` is a fixed covariate matrix of shape :math:`N \times L` (i.e. cells by covariates)
2. :math:`N_n` is a fixed vector of shape :math:`N` with the (observed) total counts in a cell.
3. :math:`\beta_{kg}^0` is the intercepts of the regression of shape :math:`K \times G` (i.e. cell-types by genes)
4. :math:`\beta_{klg}` are the regression coefficients of shape :math:`K \times L \times G` (i.e. cell-types by covariates by genes)

Typical values are :math:`N \sim 10^3, G \sim 10^3, K\sim 10, L\sim 50`.
The goal of the inference is to determine :math:`\beta^0` and :math:`\beta`.
We enforce a penalty (either L1 or a L2) on the regression coefficients :math:`\beta` to encourage them to be small.
There is no prior on :math:`\beta^0`.
Overall the model has one hyper-parameter (the strength of the regularization on :math:`\beta`)
which is determined by cross-validation.

See `notebook3 <https://github.com/broadinstitute/TissueMosaic/blob/main/notebooks/notebook3.ipynb>`_ for an example.

.. automodule:: tissuemosaic.genex.gene_utils
   :members:

.. automodule:: tissuemosaic.genex.gene_visualization
   :members:

.. automodule:: tissuemosaic.genex.pyro_model
   :members:
