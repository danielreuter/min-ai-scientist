.PHONY: check
check:
	ruff check --fix
	ruff format
	mypy scientist tests example

.PHONY: test
test:
	pytest