MalariaGEN Python API documentation
===================================

**Useful links**: `Source Repository <https://github.com/malariagen/malariagen-data-python>`_ | `Issues & Ideas <https://github.com/malariagen/malariagen-data-python/issues>`_ | `Q&A Support <https://github.com/malariagen/malariagen-data-python/discussions>`_ | `Online Training <https://anopheles-genomic-surveillance.github.io/>`_

The ``malariagen_data`` Python package provides a library of functions (API) for accessing and analysing
data from the `Malaria Genomic Epidemiology Network (MalariaGEN) <https://www.malariagen.net/>`_.

API documentation
-----------------

.. grid::

   .. grid-item-card:: ``Ag3``
      :link: Ag3
      :link-type: doc

      *Anopheles gambiae* complex.

      .. image:: https://upload.wikimedia.org/wikipedia/commons/0/0a/AnophelesGambiaemosquito.jpg

   .. grid-item-card:: ``Af1``
      :link: Af1
      :link-type: doc

      *Anopheles funestus* subgroup.

      .. image:: https://upload.wikimedia.org/wikipedia/commons/thumb/f/ff/Anopheles_Funetus.jpg/640px-Anopheles_Funetus.jpg

   .. grid-item-card:: ``Amin1``
      :link: Amin1
      :link-type: doc

      *Anopheles minimus*.

      .. image:: https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/Anopheles_minimus_1.jpg/640px-Anopheles_minimus_1.jpg

   .. grid-item-card:: ``Adir1``
      :link: Adir1
      :link-type: doc

      *Anopheles dirus* complex.

      .. image:: https://phil.cdc.gov//PHIL_Images/8777/8777_lores.jpg

Documentation for the `Pf7 <https://malariagen.github.io/parasite-data/pf7/api.html>`_ (*Plasmodium falciparum*)
and `Pv4 <https://malariagen.github.io/parasite-data/pv4/api.html>`_ (*Plasmodium vivax*) APIs is also available,
currently hosted on a separate site.


Installation
------------

The ``malariagen_data`` package is available from the Python package index (PyPI) and can be installed
via pip::

   pip install malariagen_data

For accessing data in Google Cloud Storage (GCS) you will also need to authenticate with Google Cloud.

If you are using ``malariagen_data`` from within Google Colab, authentication will be automatically
initiated, please allow access when requested.

If you are using ``malariagen_data`` from any location other than Google Colab, you will need to `set up application
default credentials <https://cloud.google.com/docs/authentication/provide-credentials-adc>`_. Generally
the best way to do this will be to `install the Google Cloud CLI <https://cloud.google.com/sdk/docs/install>`_
and then run the following command::

   gcloud auth application-default login


Training
--------

If you would like to learn more about how to use ``malariagen_data`` to analyse data for genomic
surveillance of malaria vectors, please see the associated `online training course <https://anopheles-genomic-surveillance.github.io>`_.


About the data
--------------

This software package provides access to data from MalariaGEN. MalariaGEN is a network
of malaria researchers and control programmes using genomics to learn more about malaria
transmission and control in Africa and Asia.

MalariaGEN generates **genome variation data** from whole-genome sequencing (WGS) of malaria
parasites (*Plasmodium*) or malaria-transmitting mosquitoes (*Anopheles*). Parasite and mosquitoes
are generally sampled from natural infections and mosquito populations, and so these are data on
natural genetic variation.

Some data from MalariaGEN are subject to **terms of use** which may include an embargo on
public communication of any analysis results without permission from data owners. If you
have any questions about terms of use please email support@malariagen.net.

By default, this sofware package accesses data directly from the **MalariaGEN cloud data repository**
hosted in Google Cloud Storage in the US. Note that data access will be more efficient if your
computations are also run within the same region. Google Colab provides a convenient and free
service which you can use to explore data and run computations. If you have any questions about
other options for running computations please `open a discussion <https://github.com/malariagen/malariagen-data-python/discussions>`_.
