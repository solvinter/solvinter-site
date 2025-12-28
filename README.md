# solvinter-site

Terminal-first, git-tracked site + building specs for Solvinter Ghana (Mumford).

> Note: 3D-model of blackrocks

## Structure
- `specs/site/`      Site-level YAML specs
- `specs/buildings/` Building YAML specs
- `evidence/`        Source PDFs and documents
- `tools/`           Build scripts (Blender headless, export GLB, renders)
- `outputs/`         Generated files (ignored by git)

## First target
- b001: workshop_shell (envelope model driven by YAML)

## Next
- Add Blender headless builder to generate .blend/.glb and preview renders.
