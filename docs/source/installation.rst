Installation Guide
==================

This guide provides instructions for installing and setting up the Analog Hawking Radiation Simulation Framework.

Prerequisites
-------------

The framework requires Python 3.8 or higher and the following dependencies:

* numpy
* scipy
* matplotlib
* h5py
* emcee

Installation Options
--------------------

There are several ways to install the framework depending on your needs.

Using pip (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~

The simplest way to install the framework is using pip:

.. code-block:: bash

   pip install analog-hawking-radiation

This will install the latest stable release and all required dependencies.

Installing from Source
~~~~~~~~~~~~~~~~~~~~~~

To install from source, first clone the repository:

.. code-block:: bash

   git clone https://github.com/Shannon-Labs/bayesian-analysis-hawking-radiation.git
   cd bayesian-analysis-hawking-radiation

Then install the package in development mode:

.. code-block:: bash

   pip install -e .

This will install the package and create links to the source code, allowing you to modify the code and see changes immediately.

Virtual Environment (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is recommended to install the framework in a virtual environment to avoid conflicts with other packages:

.. code-block:: bash

   python -m venv ahr_env
   source ahr_env/bin/activate  # On Windows: ahr_env\Scripts\activate
   pip install analog-hawking-radiation

To deactivate the virtual environment when finished:

.. code-block:: bash

   deactivate

Dependencies
------------

The framework depends on the following Python packages:

* **numpy**: Fundamental package for scientific computing
* **scipy**: Library for scientific and technical computing
* **matplotlib**: Plotting library for creating figures
* **h5py**: Interface to the HDF5 binary data format
* **emcee**: MCMC sampling toolkit

These dependencies will be automatically installed when using pip.

Development Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~

For development and testing, additional packages are recommended:

.. code-block:: bash

   pip install pytest pytest-cov

These are included in the ``dev`` extra:

.. code-block:: bash

   pip install analog-hawking-radiation[dev]

Verification
------------

To verify that the installation was successful, you can run a simple test:

.. code-block:: python

   import analog_hawking
   print(analog_hawking.__version__)

This should print the version number of the installed package.

Running Tests
-------------

To run the test suite and verify that all components are working correctly:

.. code-block:: bash

   pytest

This will run all unit tests and integration tests to ensure the framework is functioning properly.

Troubleshooting
---------------

If you encounter any issues during installation, please check the following:

1. **Python Version**: Ensure you are using Python 3.8 or higher
2. **Dependencies**: Make sure all required dependencies are installed
3. **Permissions**: You may need administrator privileges to install packages system-wide
4. **Virtual Environment**: Consider using a virtual environment to avoid conflicts

For further assistance, please open an issue on the GitHub repository.