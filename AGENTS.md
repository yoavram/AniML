# Repository Guidelines

## Project Structure & Module Organization

This repository is a workshop workspace. Key locations:

- Notebooks: `computervision.ipynb`, `bleaching_analysis.ipynb`
- Lesson handouts: `README.md`, `penguins.md`, `fish.md`, `coral_bleaching.md`, `Karoo.md`
- Scripts: `fish.py` (Python), `coral.R` (R)
- Data assets: `Fish.csv`, `global_bleaching_environmental.csv`
- Environment config: `pixi.toml`, `pixi.lock`

Keep notebooks as the primary teaching artifacts; scripts and data should stay small and easy to download.

## Build, Test, and Development Commands

This repo uses Pixi for the Python environment.

- `pixi install` — create/update the environment from `pixi.toml` and `pixi.lock`.
- `pixi shell` — enter the environment before running notebooks or scripts.
- `python <file.py>` — run a script once in the Pixi shell.

There are no Pixi tasks defined yet (see empty `[tasks]` in `pixi.toml`).

## Coding Style & Naming Conventions

- Python: 4‑space indentation, prefer clear variable names (`image_url`, `sample_paths`).
- R: 2‑space indentation is acceptable for small scripts.
- Notebooks: keep cells short and focused; favor Markdown explanations over long code blocks.
- Filenames: descriptive and lowercase where possible (e.g., `fish.md`, `coral_bleaching.md`).

## Testing Guidelines

No automated tests are configured. Validate changes by running the relevant notebook cells end‑to‑end (e.g., `computervision.ipynb`) and confirming outputs and plots render correctly.

## Commit & Pull Request Guidelines

Recent commits use short, lowercase messages without prefixes (examples: “vs code”, “Fish4Knowledge dataset”). Follow the same style and keep commits focused.

For pull requests, include:

- A brief summary of changes.
- Links to any issues or workshop session references, if applicable.
- Screenshots for notebook output changes when relevant.

## Environment & Dependency Notes

- Treat `pixi.toml` and `pixi.lock` as authoritative for Python dependencies.
- Avoid adding large datasets; prefer small samples or remote URLs.
