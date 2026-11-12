.PHONY: setup data train evaluate test api app clean

VENV_PY := .venv/bin/python
PYTHON := $(shell if [ -x $(VENV_PY) ]; then echo $(VENV_PY); else echo python3.11; fi)

setup:
	python3.11 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

data:
	$(PYTHON) -m data.generate_data
	$(PYTHON) -m src.data_prep
	$(PYTHON) -m src.features
	$(PYTHON) -m src.split

train:
	$(PYTHON) -m src.train

evaluate:
	$(PYTHON) -m src.evaluate
	$(PYTHON) -m src.explain

test:
	$(PYTHON) -m pytest -v

api:
	$(PYTHON) -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

app:
	$(PYTHON) -m streamlit run app/streamlit_app.py --server.port 8501

clean:
	rm -rf .pytest_cache __pycache__ src/__pycache__ tests/__pycache__ api/__pycache__

