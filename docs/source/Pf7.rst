Pf7
===

This page provides a curated list of functions and properties available in the ``malariagen_data`` API
for data on *Plasmodium falciparum* parasites from the Pf7 release.

To set up the API, use the following code::

    import malariagen_data
    pf7 = malariagen_data.Pf7()

All the functions below can then be accessed as methods on the ``pf7`` object. E.g., to call the
``sample_metadata()`` function, do::

    df_samples = pf7.sample_metadata()

For more information about the data and terms of use, please see the
`MalariaGEN website <https://www.malariagen.net/data>`_ or contact support@malariagen.net.

.. currentmodule:: malariagen_data.pf7.Pf7

Sample metadata access
----------------------
.. autosummary::
    :toctree: generated/

    sample_metadata

Variant data access
------------------
.. autosummary::
    :toctree: generated/

    variant_calls

Reference genome data access
----------------------------
.. autosummary::
    :toctree: generated/

    open_genome
    genome_sequence
    genome_features
