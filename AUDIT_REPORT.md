# HistoScope technical audit

## Baseline (2026-11-02)

- The project directory is a source snapshot, not a Git checkout; no status, branch, or diff can be obtained.
- Python 3.14.8 is present, but none of the declared runtime or test packages are installed. `pip check` therefore reports no broken requirements but does not validate this project. Pytest and Streamlit commands cannot run in the global environment.
- `python -m compileall -q .` completed without errors.

## Findings before repair

| Severity | Finding | Consequence |
|---|---|---|
| High | Runtime environment is unprovisioned; package metadata permits broad, unpinned combinations and pyproject does not declare runtime dependencies. | A clean install may drift and local `pip check` is not reproducible. |
| High | Processing helpers mostly assume valid nonempty 2D/3D arrays and do not validate parameters or channel shapes. | Malformed arrays, non-finite data, invalid gamma/sigma and size-1 arrays can fail deep in OpenCV/SciPy or produce unclear errors. |
| Medium | `spatial_filters` sends a 1x1 image to medianBlur with a kernel >=3. | Tiny images are not reliably supported. |
| Medium | The hematoxylin channel is signed floating-point data but was passed directly to Streamlit's image renderer. | Some valid HED images could fail display validation instead of rendering the intended visualization. |
| Medium | Constant-image behavior and thresholds are not explicitly specified across enhancement, edges, segmentation and frequency transforms. | Some outputs can be misleading or fail despite valid input. |
| Medium | Empty nuclei measurements have no columns; counts are derived from the maximum label. | The empty CSV/table schema is inconsistent and count assumes contiguous labels. |
| Medium | ZIP path uses the uploaded filename stem and image conversion can rescale numeric outputs for display. | Export names need sanitization and categorical integer labels must remain data-preserving. |
| Medium | Streamlit reports full exception tracebacks to uploaders. | Internal diagnostics leak into the normal user experience. |
| Low | UI and README contain mojibake characters; README clone commands point to a placeholder repository. | The interface and setup instructions appear broken or unprofessional. |
| Low | Docker is installed, but Docker CLI configuration is inaccessible in this environment. | Docker build/run availability still needs direct verification. |

The algorithms are classical image processing. Tissue and nuclei segmentation remain heuristic and require explicit educational/research limitations.

## Repair and verification plan

Repair input/parameter validation, tiny/constant-image behavior, measurement/export integrity, user-facing error reporting and incorrect documentation. Add algorithmic regression tests. Install into a project-local virtual environment if package access is approved; then run pytest, compileall, dependency checks, Streamlit startup/HTTP checks and Docker build/run where available.
