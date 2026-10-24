.PHONY: test demo lint

test:
	python3 -m unittest discover -s tests -v

demo:
	python3 examples/demo.py

lint:
	python3 -m compileall -q minirag tests examples
