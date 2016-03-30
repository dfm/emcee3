emcee3
======

Seriously Kick-Ass MCMC
-----------------------

**emcee3** is an MIT licensed Python toolbox for MCMC sampling.
It is a backwards-incompatible extension of the popular `emcee
<https://github.com/dfm/emcee>`_ library to include more proposal
distributions and other real world niceties.
This documentation won't teach you too much about MCMC but there are a lot
of resources available for that (try `this one
<http://www.inference.phy.cam.ac.uk/mackay/itila/book.html>`_).
We also `published a paper <http://arxiv.org/abs/1202.3665>`_ explaining
the core **emcee** algorithm and implementation in detail.
**emcee** and **emcee3** have been used in `quite a few projects in the
astrophysical literature <testimonials>`_ and **emcee3** is being actively
developed on `GitHub <https://github.com/dfm/emcee3>`_.


Basic Usage
-----------

If you wanted to draw samples from a multidimensional Gaussian, you would do
something like:

.. code-block:: python

    import emcee3
    import numpy as np

    # Define the probabilistic model:
    def log_prior_function(x):
        if np.all(-10. < x) and np.all(x < 10.):
            return 0.0
        return -np.inf

    def log_likelihood_function(x):
        return -0.5 * np.sum(x ** 2)

    model = emcee3.SimpleModel(log_likelihood_function, log_prior_function)

    # Initialize:
    ndim, nwalkers = 10, 100
    ensemble = emcee3.Ensemble(model, np.random.randn(nwalkers, ndim))

    # Sample:
    sampler = emcee3.Sampler()
    sampler.run(ensemble, 1000)

A more complete example is available in the `quickstart documentation
<user/quickstart>`_.


User Guide
----------

.. toctree::
   :maxdepth: 2

   user/install
   user/modeling
   user/parallel
   user/porting

Tutorials
---------

.. toctree::
   :maxdepth: 2

   tutorials/line
   tutorials/mixture-models


API Documentation
-----------------

.. toctree::
   :maxdepth: 2

   api


Contributors
------------

.. include:: ../AUTHORS.rst


License & Attribution
---------------------

Copyright 2010-2016 Dan Foreman-Mackey and contributors.

emcee3 is free software made available under the MIT License. For details
see `LICENSE <license>`_.

If you make use of emcee in your work, please cite our paper
(`arXiv <http://arxiv.org/abs/1202.3665>`_,
`ADS <http://adsabs.harvard.edu/abs/2013PASP..125..306F>`_,
`BibTeX <http://adsabs.harvard.edu/cgi-bin/nph-bib_query?bibcode=2013PASP..125..306F&data_type=BIBTEX>`_)
and consider adding your paper to the :ref:`testimonials` list.


Changelog
---------

.. include:: ../HISTORY.rst
