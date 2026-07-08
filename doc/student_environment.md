### Student Environment: Docker Compose and Dev Containers

This repository includes a ready-to-use containerized environment for running SUAVE and course materials with minimal setup. You can work locally with Docker Compose or in a Dev Container (VS Code/Cursor, GitHub Codespaces).

---

## What’s included

- Python base image with scientific stack (numpy, scipy, matplotlib, pandas, scikit-learn, plotly, jupyterlab, ipykernel, ipywidgets, seaborn, numba, black, pylint)
- Optional external tools (enabled by default):
  - SU2 (prebuilt tarball extracted to `/opt/su2`)
  - AVL (downloaded or built and installed to `/usr/local/bin/avl`)
  - Gmsh from apt
- Preconfigured `PYTHONPATH` for SUAVE trunk and student workspace
- Jupyter Lab server inside the container
- Convenience `Makefile` targets to build and run

---

## Quickstart (Docker Compose)

Prerequisites: Docker (and Docker Compose v2) installed.

- Build the image:

```bash
make build
```

- Start the container (Jupyter Lab on port 8888):

```bash
make up
```

- Open the Jupyter URL printed in the logs. By default, no token is set for local use. To view logs again:

```bash
make logs
```

- Open a shell inside the running container:

```bash
make bash
```

- Install SUAVE in editable mode inside the container (developer mode):

```bash
make dev-install
```

- Stop and remove the container and volumes:

```bash
make down
```

If `make` is unavailable, the equivalent commands are:

```bash
docker compose build
docker compose up -d
docker compose exec suave bash
docker compose down -v
```

---

## How it’s wired

- Service: `suave` defined in `docker-compose.yml`
- Image build context: `docker/` with `Dockerfile`
- Entrypoint: `/usr/local/bin/entrypoint.sh` via `tini` (for proper signal handling)
- Command: `jupyter` (starts Jupyter Lab by default)
- Ports: `8888:8888` (change with `PORT` env var)
- Volumes:
  - Project root mapped to `/workspace`
  - Named cache volume `suave-cache` mapped to `/root/.cache` (pip and tool caches persist across rebuilds)
- Working directory: `/workspace`

Jupyter starts with:

```bash
jupyter lab --ip=0.0.0.0 --port=$PORT --no-browser --ServerApp.token="$JUPYTER_TOKEN"
```

- To disable the token locally, leave `JUPYTER_TOKEN` empty (default). For shared networks, set a token (see Environment variables).

---

## Environment variables (compose)

Set via `docker-compose.yml` (can be overridden at runtime):

- `MPLBACKEND=Agg`: Safe default for headless environments
- `PYTHONUNBUFFERED=1`
- `PYTHONPATH=/workspace/trunk:/workspace/student_competition`
- `JUPYTER_TOKEN=`: Empty by default (no token). Set for security: e.g. `JUPYTER_TOKEN=your-secret`.
- `PORT=8888`: Host port mapping is `8888:8888`
- External tools:
  - `AVL_BIN=/usr/local/bin/avl`
  - `SU2_HOME=/opt/su2`
  - `SU2_RUN=/opt/su2/bin`
  - `SU2_BIN=/opt/su2/bin/SU2_CFD`
  - `OPENVSP_BIN=` (empty by default)
  - `GMSH_BIN=/usr/bin/gmsh`

Build-time args (in compose → `build.args`) control optional tools:

- `INSTALL_EXTERNAL_TOOLS=true`
- `SU2_DOWNLOAD_URL` and `SU2_SHA256`
- `AVL_DOWNLOAD_URL`

---

## Entrypoint behaviors

`docker/entrypoint.sh` performs:

- Detects headless mode and sets a safe matplotlib backend (`Agg`)
- Ensures student folders exist: `/workspace/student_competition/{results,plots}`
- Appends to `PYTHONPATH` for SUAVE trunk and student workspace
- Echoes a short environment summary for transparency
- Supports multiple commands:
  - `jupyter` (default): Jupyter Lab
  - `notebook`: classic Jupyter Notebook
  - `bash`: interactive shell
  - `python <file.py>`: run a Python script
  - `run <cmd>`: execute any command

Examples:

```bash
# Run a one-off script in a new container
docker compose run --rm suave python my_script.py

# Start Notebook instead of Lab
docker compose up -d
docker compose exec suave bash -lc 'entrypoint.sh notebook'
```

---

## Dev Containers and Codespaces

A separate Dev Container configuration is available in `.devcontainer/` for VS Code/Cursor and GitHub Codespaces.

- Base image: `mcr.microsoft.com/devcontainers/python:3.9`
- Dependencies from `.devcontainer/requirements.txt`
- Optional SU2/AVL via build args in `devcontainer.json`
- Post-create installs SUAVE in editable mode (`trunk/setup.py develop`)
- Exposes ports: 8888, 3000, 7860

Notes:

- The Dev Container uses Python 3.9 while Docker Compose uses Python 3.10. Functionally similar for most use cases, but aligning versions can reduce surprises.
- See `.devcontainer/INSTALLATION_NOTES.md` for brief specifics.

---

## Working directories and persistence

- Your local repository is mounted at `/workspace` inside the container, so files you edit in the container appear on your host.
- Pip caches and some tool caches persist in the named volume `suave-cache`.
- Student outputs are scaffolded under `/workspace/student_competition/`.

---

## Troubleshooting

- Can’t connect to Jupyter:
  - Check that port 8888 is available, or set `PORT` to a free port and re-run `make up`.
  - Try `make logs` to see the startup output.
- Missing external tools (SU2/AVL):
  - Confirm the build args are set in `docker-compose.yml`.
  - Rebuild the image: `make build`.
- Matplotlib errors in headless mode:
  - `Agg` backend is already enforced; verify custom code does not override it.

---

## Security note

For classroom or shared networks, set a token:

```bash
JUPYTER_TOKEN=choose-a-strong-token docker compose up -d
```

Then navigate to `http://localhost:8888/?token=choose-a-strong-token`.

---

## Command reference (Makefile)

- `make build` — Build local Docker image
- `make up` — Start Jupyter container (detached) and print initial logs
- `make logs` — Follow container logs
- `make bash` — Shell into the running container
- `make dev-install` — Install SUAVE in editable mode inside the container
- `make down` — Stop and remove container and volumes

---

## Ideas to make it even easier for students

- Publish prebuilt images to a registry (e.g., GHCR/Docker Hub) to avoid local builds:
  - Provide a one-liner: `docker run -it --rm -p 8888:8888 -v "$PWD":/workspace ghcr.io/<org>/suave-student:latest`
- Unify Python versions (3.10 vs 3.9) across Docker Compose and Dev Container; pin dependency versions for reproducibility.
- Add a `.env.example` and support `.env` in compose for easy customization (token, port, tool URLs) without editing YAML.
- Add Make targets:
  - `make pull` (pull prebuilt image), `make reset` (purge container + caches), `make notebook` (start classic Notebook), `make lite` (profile without external tools)
- Auto-print a direct Jupyter URL with a random token when the container starts; optionally open the browser automatically on macOS/Windows/Linux.
- Provide a minimal “smoke test” notebook/script that validates SUAVE import and optional tools (SU2/AVL) with green/red checks.
- Set Jupyter root to `/workspace/student_competition` to keep the focus on student files; optionally seed this folder with example notebooks and templates.
- Add a GitHub Codespaces badge and prebuilds to eliminate wait time in the cloud.
- Provide starter datasets and a `make data` target to download them into a mounted `data/` directory.
- Add a troubleshooting guide and short explainer videos linked from the docs.