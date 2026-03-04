Pv4
===

This page provides a curated list of functions and properties available in the ``malariagen_data`` API
for data on *Plasmodium vivax* parasites from the Pv4 release.

To set up the API, use the following code::

    import malariagen_data
    pv4 = malariagen_data.Pv4()

All the functions below can then be accessed as methods on the ``pv4`` object. E.g., to call the
``sample_metadata()`` function, do::

    df_samples = pv4.sample_metadata()

For more information about the data and terms of use, please see the
`MalariaGEN website <https://www.malariagen.net/data>`_ or contact support@malariagen.net.

.. currentmodule:: malariagen_data.pv4.Pv4

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
