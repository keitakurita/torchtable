torchtable
++++++++++

**Caution: Work in Progress!**

Torchtable is a library for handling tabular datasets in PyTorch. It is heavily inspired by torchtext and uses a similar API but without some of the limitations (e.g. only one field per column).
Torchtable aims to be simple to use while being easily extensible. It provides sensible defaults while allowing the user to define their own custom pipelines.

Installation
============
Making the repository installable is still a work in progress.


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
