create_conda_env:
	conda env create --file=conda.yml

create_virtualenv_dev:
	test -d env || \
		virtualenv env && \
		source env/bin/activate && \
		python -m pip install -U pip && \
		pip install -r requirements.txt
	# python -m pip install -e .
	
clean_env:
	rm -rf env

# Run same lint used on github action
lint:
	# stop the build if there are Python syntax errors or undefined names
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	# exit-zero treats all errors as warnings. The GitHub editor is 127 chars wide
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

test: lint
	pytest -vv .
