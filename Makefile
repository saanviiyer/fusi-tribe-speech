PY ?= python
CLI = $(PY) -m fusi_tribe.cli

.PHONY: help install test fusi insilico tribe-full all clean

help:
	@echo "make install     install the package (+ TRIBE v2 and test deps)"
	@echo "make test        run the unit tests (no data needed)"
	@echo "make fusi        fUSI GLM + timing analysis          (~20 min, needs data/)"
	@echo "make insilico    stimuli + TRIBE audio-only + analysis (~1 h on an M-series Mac)"
	@echo "make tribe-full  TRIBE with the text pathway          (needs gated Llama-3.2-3B)"
	@echo "make all         fusi + insilico + report"

install:
	$(PY) -m pip install -e ".[tribe,dev]"

test:
	$(PY) -m pytest -q

fusi:
	$(CLI) fusi-glm
	$(CLI) fusi-timing

insilico:
	$(CLI) make-stimuli
	$(CLI) run-tribe --mode audio_only
	$(CLI) analyze-tribe --mode audio_only

tribe-full:
	$(CLI) run-tribe --mode full
	$(CLI) analyze-tribe --mode full

all: fusi insilico
	$(CLI) report

clean:
	rm -rf outputs
