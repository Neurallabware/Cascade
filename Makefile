.PHONY: train eval test clean help

## Train with default config
train:
	python src/train.py

## Evaluate with a checkpoint
eval:
	python src/eval.py ckpt_path=$(CKPT)

## Run all tests
test:
	python -m pytest tests/ -v

## Train with a specific experiment
experiment:
	python src/train.py experiment=$(EXP)

## Run debug mode (1 epoch, cpu)
debug:
	python src/train.py debug=default

## Clean generated files
clean:
	rm -rf logs/ outputs/ .hydra/
	find . -type d -name __pycache__ -exec rm -rf {} +

## Show help
help:
	@echo "Available targets:"
	@echo "  train       - Train with default config"
	@echo "  eval        - Evaluate (set CKPT=/path/to/checkpoint)"
	@echo "  test        - Run pytest suite"
	@echo "  experiment  - Train with experiment config (set EXP=name)"
	@echo "  debug       - Quick debug run (1 epoch, cpu)"
	@echo "  clean       - Remove logs, outputs, and caches"
