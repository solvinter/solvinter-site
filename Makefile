PY := .venv/bin/python3

.PHONY: validate build build-all

validate:
	$(PY) -c 'import glob,yaml; [yaml.safe_load(open(f)) for f in glob.glob("**/*.yaml", recursive=True)]; print("YAML OK")'

build:
	@test -n "$(BUILDING)" || (echo "Usage: make build BUILDING=atelje0_1" && exit 1)
	$(PY) -u tools/build_blender.py buildings/$(BUILDING).yaml --outdir outputs/$(BUILDING) --export-glb --render

build-all:
	@for f in buildings/*.yaml; do \
		b=$$(basename $$f .yaml); \
		echo "==> Building $$b"; \
		$(PY) -u tools/build_blender.py $$f --outdir outputs/$$b --export-glb --render; \
	done
