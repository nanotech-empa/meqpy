meqpy documentation
===================

**meqpy** is a modular Python framework for defining systems and solving
master equations for scanning tunneling microscopy (STM).

Installation
------------

.. code-block:: bash

   pip install git+https://github.com/nanotech-empa/meqpy.git

Quick start
-----------

.. code-block:: python

   import meqpy

   system = meqpy.System()

See the :doc:`tutorials <tutorials/index>` for worked examples; the notebooks
live in ``docs/source/tutorials/`` if you want to run them interactively.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   tutorials/index
   api
