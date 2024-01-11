Amin1
=====

This page provides a curated list of functions and properties available in the ``malariagen_data`` API
for data on *Anopheles minimus* mosquitoes.

To set up the API, use the following code::

    import malariagen_data
    amin1 = malariagen_data.Amin1()

All the functions below can then be accessed as methods on the ``amin1`` object. E.g., to call the
``sample_metadata()`` function, do::

    df_samples = amin1.sample_metadata()

For more information about the data and terns of use, please see the
`MalariaGEN Vector Observatory Asia <https://www.malariagen.net/mosquito/vector-observatory-asia>`_
home page.

.. currentmodule:: malariagen_data.amin1.Amin1

Reference genome data access
----------------------------
.. autosummary::
    :toctree: generated/

    contigs
    genome_sequence
    genome_features

Sample metadata access
----------------------
.. autosummary::
    :toctree: generated/

    sample_metadata

SNP data access
---------------
.. autosummary::
    :toctree: generated/

    snp_calls
