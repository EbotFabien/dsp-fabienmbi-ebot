# === CONFIGURATION ===
PYTHON := python3
PROJECT_NAME := house_prices
MLFLOW_DIR := mlruns

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
	@echo "Starting model training..."
	$(PYTHON) -m $(PROJECT_NAME).train
	@echo "Training complete. MLflow run saved in ./$(MLFLOW_DIR)"

# === PREDICTION ===
predict:
	@echo "Generating predictions..."
	$(PYTHON) -m $(PROJECT_NAME).inference
	@echo "Predictions complete."

# === CLEANUP ===
clean:
	@echo "Cleaning temporary files..."
	rm -rf __pycache__ */__pycache__ .pytest_cache .mypy_cache
	@echo "Cleanup complete."

# === RUN MLflow UI ===
mlflow-ui:
	@echo "Launching MLflow UI at http://127.0.0.1:5000"
	mlflow ui --backend-store-uri file:./$(MLFLOW_DIR)
