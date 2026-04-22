VENV		= venv
SRC			= src
OUTPUT		= output
PYTHON		= $(VENV)/bin/python3
PIP			= $(VENV)/bin/pip
DATASET_TRAIN	= datasets/dataset_train.csv
DATASET_TEST	= datasets/dataset_test.csv
WEIGHTS		= $(OUTPUT)/weights.csv

# ─── Setup ────────────────────────────────────────────────────────────────────

all: $(VENV)

$(VENV): requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

output:
	mkdir -p $(OUTPUT)

# ─── Data Analysis ──────────────────────────────────────────────────

describe:
	$(PYTHON) $(SRC)/describe.py $(DATASET_TRAIN)

describe_all:
	$(PYTHON) $(SRC)/describe.py $(DATASET_TRAIN) --bonus

# ─── Data Visualization ────────────────────────────────────────────

histogram: output
	$(PYTHON) $(SRC)/histogram.py $(DATASET_TRAIN)

histogram_all: output
	$(PYTHON) $(SRC)/histogram.py $(DATASET_TRAIN) --all

scatter: output
	$(PYTHON) $(SRC)/scatter_plot.py $(DATASET_TRAIN)

scatter_all: output
	$(PYTHON) $(SRC)/scatter_plot.py $(DATASET_TRAIN) --all

pair: output
	$(PYTHON) $(SRC)/pair_plot.py $(DATASET_TRAIN)
	
visu: histogram scatter pair

# ─── Logistic Regression ───────────────────────────────────────────

train: output
	$(PYTHON) $(SRC)/logreg_train.py $(DATASET_TRAIN)

stochastic: output
	$(PYTHON) $(SRC)/logreg_train.py $(DATASET_TRAIN) --stochastic

mini-batch: output
	$(PYTHON) $(SRC)/logreg_train.py $(DATASET_TRAIN) --mini-batch

predict: output
	$(PYTHON) $(SRC)/logreg_predict.py $(DATASET_TEST) $(WEIGHTS)

# ─── Nettoyage ────────────────────────────────────────────────────────────────

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf $(OUTPUT)

fclean: clean
	rm -rf $(VENV)

re: fclean all

.PHONY: all describe histogram scatter pair visu train predict clean fclean re