.. image:: https://travis-ci.org/musevlt/origin.svg?branch=master
  :target: https://travis-ci.org/musevlt/origin

.. image:: https://codecov.io/gh/musevlt/origin/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/musevlt/origin


ORIGIN is a software to perform blind detection of faint emitters in MUSE
datacubes.

The algorithm is tuned to efficiently detects faint spatial-spectral emission
signatures, while  allowing for a stable false detection rate over the data cube
and providing in the same time an automated and reliable estimation of the
purity.

The algorithm implements :

1. A nuisance removal part based on a continuum subtraction  combining
a Discrete Cosine Transform and an iterative Principal Component Analysis,

2. A detection part based on the local maxima of Generalized Likelihood
Ratio test  statistics obtained for a set of spatial-spectral profiles of
emission line emitters,

3. A purity estimation part, where the proportion of true emission lines
is estimated from the data itself:  the distribution of the local maxima in
the noise only configuration is estimated from that of the local minima.


Citation
--------
ORIGIN is presented in the following paper:
`Mary et al., A&A, 2020, in press <https://doi.org/10.1051/0004-6361/201937001>`_


Links
-----

- `Documentation <https://muse-origin.readthedocs.io/>`_
- `PyPI <https://pypi.org/project/muse-origin/>`_
- `Github <https://github.com/musevlt/origin>`_
