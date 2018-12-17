all: test

test:
	pytest

export_nb:
	python export.py dev_nb/01_operators.ipynb
	python export.py dev_nb/02_fields.ipynb
	python export.py dev_nb/03_dataset.ipynb
	python export.py dev_nb/04_iterators.ipynb
