.PHONY: install data train train-tf benchmark eval anomaly serve drift test test-fast docker-up lint demo

PY := python

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest

test-fast:
	$(PY) -m pytest -m fast -q

lint:
	$(PY) -m ruff check pdm tests
	$(PY) -m black --check pdm tests
	$(PY) -m mypy pdm

data:
	$(PY) -m pdm.cli data download
	$(PY) -m pdm.cli data verify

train:
	$(PY) -m pdm.cli train

train-tf:
	$(PY) -m pdm.cli train-tf

benchmark:
	$(PY) -m pdm.cli benchmark

eval:
	$(PY) -m pdm.cli evaluate

anomaly:
	$(PY) -m pdm.cli anomaly

serve:
	$(PY) -m pdm.cli serve

drift:
	$(PY) -m pdm.cli drift check --production-subset FD002
	$(PY) -m pdm.cli drift check --production-subset FD003

docker-up:
	docker compose -f docker/docker-compose.yml up -d --build

# P10 deliverable #6: one command, clean checkout to a populated dashboard. Brings up the whole
# stack (API, MLflow, drift-monitor, node-exporter, Prometheus, Grafana), waits for the API to
# report healthy, replays real telemetry against it so the dashboards have data, then prints
# where to look -- no manual clicking anywhere in this sequence.
demo:
	docker compose -f docker/docker-compose.yml up -d --build
	@echo "Waiting for the API to become healthy..."
	@until curl -sf http://localhost:8000/health > /dev/null 2>&1; do sleep 2; done
	$(PY) -m pdm.cli load-test --base-url http://localhost:8000 --duration-seconds 120
	@echo ""
	@echo "Grafana:    http://localhost:3000  (anonymous viewer access enabled for this local demo)"
	@echo "Prometheus: http://localhost:9090"
	@echo "API:        http://localhost:8000/health"
