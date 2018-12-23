.. image:: https://circleci.com/gh/keitakurita/torchtable.svg?style=svg
    :target: https://circleci.com/gh/keitakurita/torchtable

.. image:: https://readthedocs.org/projects/torchtable/badge/?version=master
    :target: https://torchtable.readthedocs.io/en/master/?badge=master
    :alt: Documentation Status

torchtable
++++++++++

Torchtable is a library for handling tabular datasets in PyTorch. It is heavily inspired by torchtext and uses a similar API but without some of the limitations (e.g. only one field per column).
Torchtable aims to be **simple to use** and **easily extensible**. 
It provides sensible defaults while allowing the user to define their own custom pipelines, putting all of this behind an intuitive interface.

Installation
============
Install via pip.

`$ pip install torchtable`

Documentation
=============
Documentation is a work in progress, but the current docs can be read `here <https://torchtable.readthedocs.io/en/master/>`_.
In addition, you can read the notebooks in the examples directory or dev_nb directory to learn more.

Usage
=====

Torchtable uses a declarative API similar to torchtext.
Here is an example of how you might handle an imaginary dataset where you are supposed to predict the price of some product.

.. code-block:: python

  >>> train = TabularDataset.from_csv('data/train.csv',
  ...    fields={'seller_id': CategoricalField(min_freq=3),
  ...            'timestamp': [DayofWeekField(), HourField()],
  ...            'price': NumericalField(fill_missing="median", is_target=True)
  ...    })
  ...

See the examples directory for more examples.

TODO
====
- Add more models
- Implement default field selection
- Implement text field/operations
- Implement swap noise
- Implement input/output validation
