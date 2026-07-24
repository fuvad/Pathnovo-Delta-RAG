# Document Delta & Grounded Chat — Makefile
# Usage: make <target>

.PHONY: run chat eval test lint clean

# --- Server ---
run:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# --- Chat ---
chat:
	python -m src.chat.cli

# --- Evaluation ---
eval:
	python -m eval.run_eval

# --- Tests ---
test:
	pytest tests/ -v

# --- Lint ---
lint:
	python -m py_compile src/api/main.py
	python -m py_compile src/config/settings.py

# --- Clean generated outputs ---
clean:
	rmdir /s /q data\canonical 2>nul || true
	rmdir /s /q data\reports 2>nul || true
	rmdir /s /q data\outputs 2>nul || true
	rmdir /s /q logs 2>nul || true
	mkdir data\canonical
	mkdir data\reports
	mkdir data\outputs
	mkdir logs
