create_conda_env:
	conda env create --file=conda.yml

create_virtualenv:
	test -d env || \
		virtualenv env && \
		source env/bin/activate && \
		python3 -m pip install -U pip && \
		pip install -r requirements.txt

clean_env:
	rm -rf env
