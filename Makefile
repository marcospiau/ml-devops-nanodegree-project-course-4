create_conda_env:
	conda env create --file=conda.yml

create_virtualenv_dev:
	test -d env || \
		virtualenv env && \
		source env/bin/activate && \
		python -m pip install -U pip && \
		pip install -r requirements-dev.txt
	python -m pip install -e .
	

clean_env:
	rm -rf env
