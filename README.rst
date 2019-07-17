.. image:: https://travis-ci.org/musevlt/origin.svg?branch=master
  :target: https://travis-ci.org/musevlt/origin

.. image:: https://codecov.io/gh/musevlt/origin/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/musevlt/origin

ORIGIN is a Python software for the blind detection of faint emission line
galaxies in MUSE datacubes. Several line detection algorithms exist but their
performances for the deepest MUSE exposures is hard to quantify, in particular
with respect to the actual purity of their detection results. This works
proposes an algorithm which is powerful for detecting faint spatial-spectral
emission signatures while allowing for a stable false detection rate over the
data cube and providing an automated and reliable estimation of the false
detections, or purity.

As for the detection part, the algorithm combines nuisance removal based on
iterative Principal Component Analysis and detection using a Generalized
Likelihood Ratio approach based on spatial-spectral profiles of emission line
emitters. The estimation of the resulting purity for emission lines is based on
endogenous training data: the statistics of the peaks is based on that of the
valley.

Citation
--------

ORIGIN was described briefly in `2017A&A...608A...1B
<https://ui.adsabs.harvard.edu/abs/2017A%26A...608A...1B/abstract>`_ and will be
presented in more details in a forthcoming publication ([Mary et al. in prep.]).

Links
-----

- `Documentation <https://muse-origin.readthedocs.io/>`_
- `PyPI <https://pypi.org/project/muse-origin/>`_ (coming soon)
- `Github <https://github.com/musevlt/origin>`_
