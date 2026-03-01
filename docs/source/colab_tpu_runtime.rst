Google Colab Installation Guide
===============================

This guide provides step-by-step instructions to install malariagen_data in Google Colab using the TPU v2-8 runtime configuration. The procedure below ensures a stable and reproducible setup within Colab’s managed environment.

Prerequisites
-------------

Open a new notebook in Google Colab.

Navigate to:

Runtime → Change runtime type

Configure the runtime as follows:

Hardware accelerator: TPU

TPU type: v2-8

Click Save to apply the configuration.

Initial Installation
--------------------

In a new notebook cell, run::

`!pip install malariagen_data`

After installation completes, verify the setup::

`import malariagen_data`

If the import executes without errors, the installation is successful.

.. warning::

If dependency-related warnings or conflicts occur during installation,
follow one of the resolution procedures below.

Resolution Option 1: Uninstall Panel
------------------------------------

If your notebook does not require panel, uninstall it before installing malariagen_data.

Execute the following commands sequentially::

`!pip uninstall -y panel`
`!pip install malariagen_data`

Verify installation::

`import malariagen_data`

If no errors are raised, the setup is complete.

Resolution Option 2: Downgrade Panel (If Panel Is Required)
-----------------------------------------------------------

If panel is required in your environment, install a compatible version before installing malariagen_data.

Execute::

`!pip install "bokeh==3.6.3" "panel==1.7.0"`
`!pip install malariagen_data`

After installation, restart the runtime by selecting:

Runtime → Restart runtime

Once the runtime restarts, verify installation::

`import malariagen_data`

Successful execution confirms proper setup.

.. note::

Restarting the runtime ensures that the updated dependency versions are properly loaded.

Resolution Option 3: Install Required Blinker Version
-----------------------------------------------------

If installation warnings or errors are related to blinker, explicitly install the required version.

Run::

`!pip install blinker==1.9.0 --ignore-installed`
`!pip install malariagen_data`

Restart the runtime:

Runtime → Restart runtime

After restarting, re-run the installation to confirm warnings are resolved::

`!pip install malariagen_data`

Finally, verify::

`import malariagen_data`

If the import executes without errors, installation is successful.

.. warning::

Always restart the runtime after modifying core dependencies to prevent
version inconsistencies within the active session.

Final Verification
------------------

After completing any of the above procedures:

Ensure that malariagen_data installs without errors.

Confirm that import malariagen_data runs successfully.

Restart the runtime whenever core dependencies are modified.
