Using malariagen_data on Google Colab (TPU Runtime)
===================================================

Overview
--------

When using a Google Colab **v2-8 TPU runtime**, installing ``malariagen_data`` may fail due to a dependency conflict with a preinstalled system package.

Colab TPU runtimes ship with:

- ``blinker==1.4`` installed via distutils/system packages

During installation, ``dash`` → ``Flask`` requires:

- ``blinker>=1.6.2``

Because the preinstalled version is a distutils-installed package, ``pip`` cannot uninstall it, and installation fails with:

::

    error: uninstall-distutils-installed-package

    × Cannot uninstall blinker 1.4
    ╰─> It is a distutils installed project and thus we cannot accurately determine which files belong to it which would lead to only a partial uninstall.

This issue appears specific to the TPU runtime image.

Reproducing the Issue
---------------------

1. Open Google Colab
2. Select **Runtime → Change runtime type**
3. Choose **TPU**
4. Run:

::

    pip install malariagen_data

Installation fails due to the ``blinker`` conflict described above.

Recommended Workaround
----------------------

Step 1 — Install core package without dependencies:

::

    pip install malariagen_data --no-deps

Step 2 — Install required dependencies manually:

::

    pip install \
      "anjl>=1.2.0" \
      bed_reader \
      biopython \
      "dash<3.0.0" \
      "dash-cytoscape>=1.0.0" \
      distributed \
      gcsfs \
      "igv-notebook>=0.2.3" \
      "ipinfo!=4.4.1" \
      "ipyleaflet>=0.17.0" \
      "numcodecs<0.16" \
      "protopunica>=0.14.8.post2" \
      s3fs \
      statsmodels \
      yaspin \
      "zarr<3.0.0,>=2.11" \
      "bokeh<3.7.0" \
      "numpy<2.2" \
      xarray \
      scikit-allel

After installation, restart the runtime.

Cloud Data Access (GCS)
-----------------------

Most datasets are hosted on Google Cloud Storage.

If you see errors such as:

::

    403: Permission denied on storage.objects.get

Authenticate your Colab session:

::

    from google.colab import auth
    auth.authenticate_user()

You may also need to request access to certain datasets:
https://forms.gle/d1NV3aL3EoVQGSHYA

Troubleshooting
---------------

Check which version of ``blinker`` is installed:

::

    pip show blinker
    python -c "import blinker; print(blinker.__version__)"

If version ``1.4`` is installed under ``/usr/lib/python3/dist-packages``, this indicates the TPU system package.