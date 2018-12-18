.. image:: https://circleci.com/gh/keitakurita/torchtable.svg?style=svg
    :target: https://circleci.com/gh/keitakurita/torchtable

.. image:: https://readthedocs.org/projects/torchtable/badge/?version=master
    :target: https://torchtable.readthedocs.io/en/master/?badge=master
    :alt: Documentation Status

torchtable
++++++++++

**Caution: Work in Progress!**

Torchtable is a library for handling tabular datasets in PyTorch. It is heavily inspired by torchtext and uses a similar API but without some of the limitations (e.g. only one field per column).
Torchtable aims to be simple to use while being easily extensible. It provides sensible defaults while allowing the user to define their own custom pipelines.

Installation
============
Install by cloning and running `pip install` (will make the package available on PyPi once stable)

Documentation
=============
Documentation is a work in progress, but the current (incomplete) docs can be read `here <https://torchtable.readthedocs.io/en/master/>`_.

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

More examples set to come!
