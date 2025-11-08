# === CONFIGURATION ===
PYTHON := python3
MLFLOW_DIR := mlruns

# === HELP ===
help:
	@echo ""
	@echo "Available commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make test        - Run all unit tests"
	@echo "  make train       - Train the model (via main.py CLI)"
	@echo "  make predict     - Run inference (via main.py CLI)"
	@echo "  make clean       - Remove caches and temporary files"
	@echo "  make mlflow-ui   - Launch MLflow UI at http://127.0.0.1:5000"
	@echo ""

# === SETUP ===
install:
	@echo "Installing dependencies..."
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	@echo "Installation complete."

# === TESTING ===
test:
	@echo "Running all unit tests..."
	PYTHONPATH=. pytest -v --disable-warnings --maxfail=1
	@echo "All tests completed."

# === TRAINING ===
train:
	@echo "Starting model training via CLI..."
	$(PYTHON) main.py train --input data/house-prices-advanced-regression-techniques/train.csv
	@echo "Training complete. MLflow run saved in ./$(MLFLOW_DIR)"

# === PREDICTION ===
predict:
	@echo "Generating predictions via CLI..."
	$(PYTHON) main.py predict --input data/house-prices-advanced-regression-techniques/test.csv --output predictions.csv
	@echo "Predictions complete. Results saved to predictions.csv."

# === CLEANUP ===
clean:
	@echo "Cleaning temporary files..."
	rm -rf __pycache__ */__pycache__ .pytest_cache .mypy_cache
	@echo "Cleanup complete."

# === RUN MLflow UI ===
mlflow-ui:
	@echo "Launching MLflow UI at http://127.0.0.1:5000"
	mlflow ui --backend-store-uri file:./$(MLFLOW_DIR)
