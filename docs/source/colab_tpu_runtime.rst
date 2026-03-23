Google Colab Installation Guide
===============================
Prerequisites
-------------

Before installing the package, configure the runtime environment
correctly:

1. Open a new notebook in Google Colab.
2. Navigate to ``Runtime → Change runtime type``.
3. Set the runtime configuration as follows:

   - **Runtime type:** Python 3
   - **Hardware accelerator:** TPU
   - **TPU type:** v2-8

4. Click **Save** to apply the configuration.

Using the recommended TPU configuration ensures compatibility with
workflows that may require TPU-based computation.


Installation Procedure
----------------------

In a new notebook cell, install the package:

.. code-block:: bash

   !pip install malariagen_data

After installation completes, verify that the package is available:

.. code-block:: python

   import malariagen_data

If the import executes without errors, the installation was successful.

If dependency-related warnings or conflicts occur, follow one of the
resolution options described below.


Resolution Options
------------------

Resolution Option 1: Uninstall Panel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your notebook does not require ``panel``, uninstall it before
installing ``malariagen_data``.

.. code-block:: bash

   !pip uninstall -y panel
   !pip install malariagen_data

Verify installation:

.. code-block:: python

   import malariagen_data


Resolution Option 2: Install Compatible Panel Version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your workflow depends on ``panel``, install a compatible version:

.. code-block:: bash

   !pip install panel==1.7.0
   !pip install malariagen_data

Restart the runtime:

``Runtime → Restart runtime``

Then verify:

.. code-block:: python

   import malariagen_data


Resolution Option 3: Install Required Blinker Version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If a ``blinker`` version conflict occurs:

.. code-block:: bash

   !pip install blinker==1.9.0 --ignore-installed
   !pip install malariagen_data

Restart the runtime and verify:

.. code-block:: python

   import malariagen_data


Final Verification
------------------

After completing any of the procedures above:

- Ensure that ``malariagen_data`` installs without dependency errors.
- Confirm that ``import malariagen_data`` runs successfully.
- Restart the runtime whenever core dependencies are modified.
- Avoid mixing incompatible package versions within the same Colab session.
