VENV		= venv
PYTHON		= $(VENV)/bin/python3
PIP			= $(VENV)/bin/pip
DATASET_TRAIN	= datasets/dataset_train.csv
DATASET_TEST	= datasets/dataset_test.csv
WEIGHTS		= weights.csv

# ─── Setup ────────────────────────────────────────────────────────────────────

all: $(VENV)

$(VENV): requirements.txt
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Environnement prêt — lance: source venv/bin/activate"

# ─── Étape 1 : Data Analysis ──────────────────────────────────────────────────

describe:
	$(PYTHON) describe.py $(DATASET_TRAIN)

describe_all:
	$(PYTHON) describe.py $(DATASET_TRAIN) --bonus

# ─── Étape 2 : Data Visualization ────────────────────────────────────────────

histogram:
	$(PYTHON) histogram.py $(DATASET_TRAIN)

histogram_all:
	$(PYTHON) histogram.py $(DATASET_TRAIN) --all

scatter:
	$(PYTHON) scatter_plot.py $(DATASET_TRAIN)

scatter_all:
	$(PYTHON) scatter_plot.py $(DATASET_TRAIN) --all

pair:
	$(PYTHON) pair_plot.py $(DATASET_TRAIN)
	
visu: histogram scatter pair

# ─── Étape 3 : Logistic Regression ───────────────────────────────────────────

train:
	$(PYTHON) logreg_train.py $(DATASET_TRAIN)

stochastic:
	$(PYTHON) logreg_train.py $(DATASET_TRAIN) --stochastic

mini-batch:
	$(PYTHON) logreg_train.py $(DATASET_TRAIN) --mini-batch

predict:
	$(PYTHON) logreg_predict.py $(DATASET_TEST) $(WEIGHTS)

# ─── Nettoyage ────────────────────────────────────────────────────────────────

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete
	@echo "Cache Python supprimé"

fclean: clean
	rm -rf $(VENV)
	rm -f $(WEIGHTS)
	rm -f houses.csv
	@echo "Environnement virtuel supprimé"

re: fclean all

.PHONY: all describe histogram scatter pair visu train predict clean fclean re help