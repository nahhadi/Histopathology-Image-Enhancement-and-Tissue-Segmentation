# HistoScope reproducible demonstration

This walkthrough uses the committed **synthetic H&E-like** sample. It is artificial, is not a patient slide, and its generated ellipse masks are not pathologist annotations.

## 1. Prepare and start

From the repository root in PowerShell, after following the README installation steps:

```powershell
python scripts\create_demo_image.py
streamlit run app.py
```

Open the local URL printed by Streamlit. In the sidebar, choose **Overview**. Expand **Add or change images**, upload `demo_data\synthetic_he_demo.png`, then select `01 - synthetic_he_demo.png` as the current image. The image is 512 x 384 RGB and should not be downsampled with the default maximum dimension of 1600.

The generator is deterministic: NumPy random seed `20261102`; image dimensions 512 x 384. It produces two pink tissue regions with slight spatially varying color and Gaussian noise, twenty purple ellipses with different sizes, touching pairs and a partial overlap.

## 2. Shared analysis settings

Use these defaults for the baseline audit-style run; parameters are in the collapsed sidebar groups.

| Control | Value |
|---|---:|
| Maximum analysis dimension | 1600 px |
| CLAHE clip limit | 2.0 |
| Gamma correction | 0.9 |
| Median kernel | 3 x 3 |
| Gaussian sigma | 1.0 px |
| Edge/frequency analysis input | CLAHE |
| Edge operator | All three |
| Adaptive edge percentile | 75 |
| Frequency filter | Gaussian, Low-pass |
| Normalized cutoff | 0.08 cycles/pixel radius |
| Butterworth order | 2 (used only if Butterworth is selected) |
| Minimum saturation floor | 0.05 |
| Morphological disk radius | 3 px |
| Minimum tissue component area | 80 pixels^2 |
| Minimum nuclei candidate area | 12 pixels^2 |
| Watershed peak separation | 5 px |

The widgets keep their app defaults after a fresh app session. Restore any changed slider or selectbox to these values before comparing results. Processing settings are included in the exported `analysis_report.json`.

## 3. Enhancement workflow

1. Open **Enhancement**.
2. Compare Original grayscale, Histogram equalization, CLAHE, Contrast stretching and Gamma correction.
3. Keep CLAHE clip at 2.0 and gamma at 0.9 for the recorded baseline.
4. Change gamma to 1.3 and back to 0.9 to observe the tonal response; this changes the display result but does not change the original RGB image.

## 4. Spatial filtering workflow

1. Open **Spatial Analysis**.
2. Set analysis input to **CLAHE**, median kernel to 3 and Gaussian sigma to 1.0.
3. Compare the median and Gaussian filtered grayscale images.
4. Review the Roberts, Prewitt and Sobel magnitude panels and their binary percentile maps. The edge threshold is image-specific and these maps are not segmentation ground truth.

## 5. Frequency-domain workflow

1. Open **Frequency Domain**.
2. Confirm Gaussian, Low-pass, cutoff 0.08 and Butterworth order 2.
3. Inspect the input grayscale, centered log-magnitude DFT spectrum and filtered output.
4. Optional experiment: select High-pass. The app displays the absolute, per-image scaled response; it is a visualization of high-frequency response, not a reconstructed intensity image.

## 6. Tissue segmentation workflow

1. Open **Tissue Segmentation**.
2. Confirm saturation floor 0.05, disk radius 3 and minimum component area 80.
3. Compare the source RGB, saturation channel, raw threshold, opening, closing, final mask and boundary overlay.
4. White means foreground. Record foreground percentages and the Otsu/effective thresholds shown under the panels.
5. The expected approach is HSV saturation with Otsu. The synthetic image has a pale background, so the border-color fallback should not be needed. Verify the displayed strategy reads **HSV saturation**; if it does not, first check that the correct sample was selected and that settings are at baseline.

## 7. Nuclei-candidate workflow

1. Open **Nuclei Analysis**.
2. Confirm minimum candidate area 12 and watershed peak separation 5.
3. Review the display-scaled hematoxylin channel, Boolean candidate mask, Euclidean distance transform, watershed markers, colored label map and overlay.
4. The expected object total in the standalone validation run is 20, but the UI's result is a heuristic candidate count. Touching and overlapping synthetic ellipses expose marker split/merge behavior; do not treat the total as validated nuclei accuracy.

## 8. Measurements

1. Open **Measurements**.
2. Review image-level tissue coverage and candidate count.
3. Inspect object area, perimeter, eccentricity, solidity, equivalent diameter, centroid and mean hematoxylin intensity. Units are analysis-image pixels (or pixels squared); no micron calibration is configured.
4. Download the measurements CSV. If no objects pass the thresholds, the page explains that the table is empty.

## 9. Report and exports

1. Choose **Reports and Exports** or use the header's **Reports & exports** button.
2. Review image metadata, settings, summary metrics and candidate measurements.
3. Download the measurements CSV and complete ZIP bundle.
4. In the ZIP, verify the image-named folder contains `original.png`, display outputs, `object_measurements.csv`, `summary_metrics.csv`, `analysis_report.json`, and `nuclei_labels.npy`. The label NPY file preserves integer label values.

## Recorded validation result

The separate audit script compares the union of predicted label pixels to the generator's known ellipse foreground. Its recorded result was Dice 0.99948, IoU 0.99895, precision 1.00000 and recall 0.99895. These scores are only for the generated scene. Four ellipses crossed multiple predicted labels and four predicted labels overlapped multiple ellipses, showing why pixel scores do not establish accurate object counts. Reproduce the full audit and contact sheets with:

```powershell
python scripts\validate_dip_algorithms.py
```

See [DIP_ALGORITHM_AUDIT.md](DIP_ALGORITHM_AUDIT.md) for setup, output interpretation, benchmark conditions and limitations.
