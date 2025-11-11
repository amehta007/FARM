.PHONY: setup demo app test lint clean

setup:
	pip install -r requirements.txt
	python -m src.models.download_models

demo:
	python -m src.main process --video data/raw/sample.mp4 --config src/configs/default.yaml --save-annotated

app:
	streamlit run src/app.py

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

clean:
	rm -rf data/outputs/*
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

