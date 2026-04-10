# Makefile for Ear Landmark Training Pipeline
# Usage: make [target]

.PHONY: all help train adult infant transfers clean

# Default configuration file
CONFIG ?= config.yaml

# Help target
help:
	@echo "Ear Landmark Training Pipeline"
	@echo ""
	@echo "Usage: make [target] [CONFIG=path/to/config.yaml]"
	@echo ""
	@echo "Targets:"
	@echo "  all        - Run full pipeline (default)"
	@echo "  adult      - Train adult model only"
	@echo "  infant     - Train infant model only (assumes checkpoints exist)"
	@echo "  transfers  - Run both transfer stages only"
	@echo "  resume     - Resume from infant training (skip adult)"
	@echo "  clean      - Remove all checkpoint files"
	@echo ""
	@echo "Examples:"
	@echo "  make                    # Run full pipeline"
	@echo "  make adult              # Train adult model only"
	@echo "  make resume             # Skip adult, run transfers + infant"
	@echo "  make CONFIG=exp1.yaml   # Use custom config"

# Run full pipeline (default)
all: train

train:
	python pipeline.py --config $(CONFIG)

# Train adult model only
adult:
	python pipeline.py --config $(CONFIG) --stages adult

# Train infant model only (assumes transfers are done)
infant:
	python pipeline.py --config $(CONFIG) --stages infant

# Run transfer stages only
transfers:
	python pipeline.py --config $(CONFIG) --stages transfer_55_22 transfer_22_23

# Resume from infant training (skip adult)
resume:
	python pipeline.py --config $(CONFIG) --skip adult

# Quick test (1 epoch each)
test:
	@echo "Creating test config..."
	@sed 's/num_epochs: 100/num_epochs: 1/g' $(CONFIG) > config_test.yaml
	python pipeline.py --config config_test.yaml
	@rm config_test.yaml

# Clean checkpoint files
clean:
	@echo "Removing checkpoint files..."
	rm -f *.pth
	@echo "Done!"

# Install dependencies
install:
	pip install torch torchvision pyyaml tqdm

# Verify config syntax
check-config:
	python -c "import yaml; yaml.safe_load(open('$(CONFIG)'))"
	@echo "Config file is valid!"
