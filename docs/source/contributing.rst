Contributing to the Documentation
=================================

This guide explains how to contribute to the documentation for the Analog Hawking Radiation Simulation Framework.

Getting Started
---------------

To contribute to the documentation, you'll need to:

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your changes
4. Make your changes
5. Submit a pull request

Documentation Style Guide
-------------------------

The documentation follows the `Sphinx documentation style guide <https://www.sphinx-doc.org/en/master/index.html>`_ and uses reStructuredText markup.

Writing Style
~~~~~~~~~~~~~

* Use clear, concise language
* Write in active voice when possible
* Use present tense
* Define technical terms when first introduced
* Include examples where appropriate

Code Examples
~~~~~~~~~~~~~

* Code examples should be complete and runnable when possible
* Use proper syntax highlighting with ``.. code-block:: python``
* Include comments to explain complex code sections
* Test code examples to ensure they work correctly

Mathematical Notation
~~~~~~~~~~~~~~~~~~~~~

* Use LaTeX-style math notation within ``:math:`` directives
* Define all variables and symbols
* Use consistent notation throughout the documentation
* Include units where appropriate

.. code-block:: rst

   The Hawking temperature is given by:
   
   .. math::
   
      T_H = \frac{\hbar\kappa}{2\pi k_B}
   
   where:
   - :math:`\hbar` is the reduced Planck constant
   - :math:`\kappa` is the surface gravity
   - :math:`k_B` is the Boltzmann constant

Building the Documentation
--------------------------

To build the documentation locally, you'll need to install the required dependencies:

.. code-block:: bash

   pip install sphinx

Then build the documentation:

.. code-block:: bash

   cd docs
   make html

The built documentation will be available in ``docs/build/html/index.html``.

Documentation Structure
-----------------------

The documentation is organized as follows:

* ``index.rst`` - Main documentation index
* ``overview.rst`` - High-level overview of the framework
* ``installation.rst`` - Installation instructions
* ``tutorials/`` - Step-by-step tutorials
* ``api/`` - API reference documentation
* ``contributing.rst`` - This guide
* ``license.rst`` - License information

Adding New Documentation
------------------------

To add new documentation:

1. Create a new ``.rst`` file in the appropriate directory
2. Add the file to the relevant ``toctree`` directive
3. Follow the style guidelines above
4. Build and test the documentation locally

Updating API Documentation
--------------------------

API documentation is automatically generated from docstrings in the source code. To update API documentation:

1. Ensure all functions and classes have proper docstrings
2. Follow the NumPy or Google docstring conventions
3. Include parameter descriptions, return values, and examples
4. Rebuild the documentation to see changes

Submitting Changes
------------------

To submit your documentation changes:

1. Commit your changes with a clear, descriptive commit message
2. Push to your fork
3. Create a pull request against the main repository
4. Include a description of the changes and why they're needed

Review Process
--------------

All documentation contributions are reviewed by the maintainers. The review process checks for:

* Technical accuracy
* Clarity and readability
* Consistency with existing documentation
* Proper formatting and style
* Completeness of examples

Reporting Issues
----------------

If you find issues with the existing documentation, please open an issue on GitHub with:

* A clear description of the problem
* The location of the issue in the documentation
* Suggestions for improvement
* Any relevant context or examples

Thank you for contributing to the documentation!