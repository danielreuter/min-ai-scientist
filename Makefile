.PHONY: check
check:
	ruff check --fix
	ruff format
	mypy reagency tests example

.PHONY: test
test:
	pytest