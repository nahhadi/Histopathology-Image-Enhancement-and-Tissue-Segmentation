# HistoScope DIP algorithm validation

**Run date:** 2026-11-02  
**Scope:** mathematical and numerical checks of the classical image-processing pipeline, synthetic H&E-like segmentation, a bundled unannotated IHC sample, UI display semantics, and runtime cost.  
**Environment:** Windows, Python 3.14.8, NumPy 2.5.4, OpenCV 4.14.0.94, Pillow 12.3.0, SciPy 1.18.1, scikit-image 0.26.0, pandas 2.3.3, Streamlit 1.66.1.

## Executive findings

The RGB, grayscale, enhancement, spatial-filter, edge, DFT, HSV, tissue-mask, stain-separation, threshold, distance-transform, watershed, and measurement paths were compared to direct library definitions and analytic properties. Dimensions and dtypes were consistent for tested non-empty RGB/grayscale inputs.

Two algorithm defects and one display defect were corrected:

1. **High-pass numerical noise:** inverse-FFT roundoff on constant images was percentile-stretched into a full-brightness response for all three filter families. Responses below a float32, input-magnitude-aware numerical floor now return a black display.
2. **Hole-size compatibility boundary:** scikit-image 0.26 `max_size` includes holes equal to the threshold, while older `area_threshold` excludes them. The compatibility wrapper now subtracts one for the new API so both supported API generations keep the former strict boundary.
3. **Scientific response visualization:** HED and distance values were within ranges where `to_uint8` preserved absolute values, making useful structure nearly black. A separate percentile display stretch now makes these intermediate images visible without changing their analysis arrays. The high-pass UI label now states that it is an absolute, display-scaled response.
4. **Tissue-mask visualization and hole filling:** Boolean masks now use explicit 0/255 display conversion, and the tissue path fills only holes up to the selected morphology disk's area. This prevents a large enclosed region from being filled into tissue. UI diagnostics expose Otsu/floor/effective thresholds, stage coverage, and border coverage.

No broad algorithm rewrite was warranted. Tissue and nuclei segmentation remain heuristics. The generated example attained high pixel overlap because its colors and backgrounds are deliberately simple; its instance-overlap diagnostics still show split/merge errors.

## Algorithm-by-algorithm checks

| Stage | Mathematical operation and implementation | Dtype, scale, shape and boundary observations |
|---|---|---|
| RGB channels | Channel planes are direct `R=rgb[:,:,0]`, `G=...1`, `B=...2` slices. | Input is RGB `uint8`; each plane is 2D `uint8` at the same H×W. Primary-color probes confirm channel order. |
| Grayscale | OpenCV RGB-to-gray implements the standard weighted luminance conversion (approximately `0.299R + 0.587G + 0.114B`) with integer output rounding. | Output is H×W `uint8`; primary probes yielded 76, 150, and 29. |
| Histogram | `numpy.histogram(gray, bins=256, range=(0,256))` counts values in 256 unit bins. | Integer counts sum to H×W; this is a count histogram, not a probability density. |
| Histogram equalization | OpenCV global histogram equalization maps intensity by the cumulative histogram/LUT. | H×W `uint8`; verified against `cv2.equalizeHist`. It can expand noise in sparse bins; no spatial locality. |
| CLAHE | OpenCV CLAHE clips tile histograms and interpolates neighboring tile mappings; configured as 8×8 tiles and user clip limit. | H×W `uint8`; verified against the same explicit OpenCV configuration. Small images still use the library's tile behavior. |
| Contrast stretch | Intensities between the 1st and 99th percentiles are linearly mapped to 0–255; values outside clip. | H×W `uint8`; constant images are returned unchanged. Percentile clipping is intentional and can clip rare tissue detail. |
| Gamma | `Iout = 255 × (Iin/255)^gamma`, implemented with scikit-image exposure conversion. | H×W `uint8`; gamma below 1 brightens mid-tones, above 1 darkens them. Parameter-response test passed. |
| Median filter | Sliding odd-sized neighborhood median using OpenCV. | H×W `uint8`; tested against OpenCV and for 1-pixel dimensions. Effective kernel is reduced to the largest valid odd size for tiny images. |
| Gaussian filter | Separable Gaussian convolution with sigma supplied in pixel units and OpenCV's default kernel support. | H×W `uint8`; tested against explicit `GaussianBlur`. Larger sigma smooths more. Library border extension applies at image edges. |
| Roberts, Prewitt, Sobel | scikit-image gradient operators compute local derivatives and magnitude. | Raw magnitudes are normalized by each image's maximum and mapped to `uint8`; thresholded maps are Boolean and thresholded by a percentile of positive magnitudes. Constant-image numerical residuals are treated as zero. Magnitudes are therefore display-normalized and not comparable across images. |
| DFT spectrum | 2D FFT, `fftshift`, then `log1p(abs(F))` for visualization. Frequency bins are `fftshift(fftfreq(N))` in cycles/pixel. | Spectrum display is H×W `uint8`, independently rescaled for visualization. Odd and even bin centers were checked against NumPy's authoritative frequency coordinates. |
| Ideal low/high pass | Radial transfer `Hlp = 1[D <= D0]`; high pass is `1-Hlp`. Inverse shift and inverse FFT follow filtering. | Low-pass reconstruction is rounded/clipped to source `uint8`. High-pass output is `abs(real(inverse))`, robustly percentile-scaled for display; sign and physical response amplitude are intentionally not preserved. |
| Butterworth | `Hlp = 1 / (1 + (D/D0)^(2n))`; high pass is the complement. | Same inverse/output conventions as ideal filtering. Order and cutoff are validated; all outputs are finite and shape preserving. |
| Gaussian frequency filter | `Hlp = exp(-D²/(2D0²))`; high pass is the complement. | Same inverse/output conventions. Low/high behavior was checked on constant and sinusoidal images. |
| HSV | scikit-image `rgb2hsv` on float RGB in [0,1]. | H×W×3 floating point values; saturation matched an independent `rgb2hsv` call exactly in the synthetic check. Hue is undefined for zero-saturation pixels by convention and is not used for tissue thresholding. |
| Tissue threshold and morphology | Otsu threshold on saturation with a configured lower floor; strict `S > threshold`; opening, closing, hole fill limited to the disk footprint's area, small-component removal, and inner-boundary detection. | Binary products remain H×W Boolean for calculations; only display copies are mapped to 0/255. Saturation is float. Coverage is reported after raw threshold, opening, closing and cleanup. Disk radius is in pixels. Saturation-only segmentation can still include colored backgrounds. |
| HED extraction | scikit-image RGB-to-HED stain deconvolution; hematoxylin is channel 0. | Raw HED is floating point and can have a narrow/signed display range. It is preserved for thresholding and measurement; only the shown/exported visualization is percentile-stretched. |
| H threshold and cleanup | Otsu on H values inside the tissue ROI, strict `H > threshold`, disk-1 opening, small-object removal, then filling holes below the configured area boundary. | Candidate mask is Boolean H×W. Tested equality to direct operations, including the strict hole-size boundary. Otsu is global within the ROI and does not adapt locally to stain gradients. |
| Distance transform | Euclidean distance transform of the cleaned candidate mask. | Float H×W, zero outside and distance to background inside. Diagnostic display is stretched separately. |
| Markers and watershed | Local maxima of the distance map with requested minimum spacing become integer markers; watershed floods `-distance` within the mask. | Marker and label arrays are `int32` H×W. Border peaks are permitted. Connected or touching objects can split or merge depending on maxima spacing and shape. |
| Region properties | `regionprops_table` measures label area, perimeter, eccentricity, solidity, equivalent diameter, centroid, and mean HED intensity. | Measurements correspond to nonzero labels; numeric values were finite and areas matched direct label-pixel counts. Border-truncated regions have incomplete geometry and should not be treated as full objects. |

## Synthetic H&E-like segmentation

The reproducible generator uses seed `20261102`, a 384×512 image, two pink tissue regions, uneven RGB staining, Gaussian background noise, 20 purple ellipses of different sizes, touching pairs, and a partial overlap. The ellipses form the exact synthetic binary ground truth. The validation script recomputes every requested intermediate and checks the implementation's watershed labels against a separately built marker image.

| Synthetic pixel comparison | Value |
|---|---:|
| Nucleus-mask Dice | 0.99948 |
| Nucleus-mask IoU | 0.99895 |
| Precision / recall | 1.00000 / 0.99895 |
| False-positive / false-negative pixels | 0 / 9 |
| Generated tissue-mask Dice / IoU | 0.99992 / 0.99983 |
| Generated nuclei / watershed regions | 20 / 20 |
| Truth ellipses crossing multiple predicted labels | 4 |
| Predicted labels overlapping multiple truth ellipses | 4 |

These Dice/IoU/precision/recall values apply only to this generated image. The high pixel score does not validate nuclei counting: four synthetic ellipses crossed more than one predicted label and four labels overlapped multiple ellipses. Equal object totals conceal those split/merge errors. No object-level accuracy claim is made.

### Bundled tissue sample

The package contains a real scikit-image **immunohistochemistry (IHC/DAB)** sample, not an H&E sample, with no annotation. The diagnostic shows saturation thresholding preferentially selecting strongly colored areas and not defining all pale tissue uniformly; HED/watershed also marks many structures. This demonstrates stain dependence and likely false positives/negatives but cannot quantify them without labels. No segmentation scores are reported for this image.

## Visual artifacts

- [Ten-stage synthetic H&E diagnostic](validation_artifacts/synthetic_he_diagnostic.png): original RGB, grayscale, CLAHE, HED, raw threshold, cleaned mask, distance transform, markers, labels, overlay.
- [Bundled IHC pipeline comparison](validation_artifacts/ihc_pipeline_comparison.png): IHC original, grayscale, saturation, tissue mask and heuristic label overlay.
- [Bundled IHC source image](validation_artifacts/ihc_sample_diagnostic.png).
- [Machine-readable results and timings](validation_artifacts/validation_results.json).
- [Reproducible validation script](scripts/validate_dip_algorithms.py).

## UI and parameter mapping

The UI displays the original RGB and grayscale, all enhancement variants, selected edge operator(s), DFT spectrum and output, tissue mask intermediates, HED display, candidate mask, and label overlay. The edge/frequency input selector is applied to the selected grayscale/enhanced image; CLAHE, gamma, median size, Gaussian sigma, edge operator/percentile, frequency kind/mode/cutoff/order and segmentation thresholds are passed to their corresponding functions. Regression tests verify that gamma, spatial kernel/sigma, edge percentile and frequency cutoff change the intended outputs. Batch failures are isolated per image.

High-pass images are absolute-value, per-image display-scaled responses; they are not signed filtered reconstructions. The caption now says so. HED display scaling does not modify raw HED values used by thresholding, means or labels.

## Performance

Measured on the synthetic pipeline, one run per square size, in the environment stated above. Times are wall-clock and depend on CPU and memory state.

| Working image | Full stages | HED + watershed + measurements | Peak process working-set increase sampled | Retained pipeline arrays |
|---|---:|---:|---:|---:|
| 256×256 | 151 ms | 84 ms | 25 MB | 4.62 MB |
| 512×512 | 554 ms | 265 ms | 106 MB | 18.50 MB |
| 1024×1024 | 2,444 ms | 1,157 ms | 452 MB | 74.00 MB |

Working set was sampled every 5 ms with the Windows process API, so very brief peaks can be missed. Retained array bytes are summed outputs, not total process memory. The HED/watershed/measurement path dominates at each size, with tissue morphology next. At 1024² the transient working-set increase is much larger than retained output arrays. The app's whole-analysis Streamlit cache retains up to eight parameter-keyed results, and any control change misses that combined cache and reruns unrelated stages; large images and repeated slider exploration can therefore consume substantial memory and repeat work. The app exposes analysis sizes up to 3072²; those largest settings were not benchmarked here. This remains a performance limitation rather than an accuracy correction.

## Corrective changes and added tests

- Added a numerical floor so high-pass display does not amplify inverse-FFT roundoff on constant images.
- Preserved the older strict hole-size threshold when using scikit-image 0.26's inclusive `max_size` API.
- Added robust display-only stretching for HED and other scientific response arrays; exported HED display uses the same conversion.
- Clarified that the UI high-pass result is an absolute, display-scaled response.
- Tissue-mask Boolean display and per-stage diagnostics are documented in the focused tissue-mask repair; the synthetic metrics above remain the same for the hole-free generated scene.
- Added regression tests for constant-image high-pass outputs, exact hole-threshold behavior, display scaling, and parameter responsiveness. The current suite has 25 passing tests.
- Added independent direct-reference checks for grayscale, histograms, equalization, CLAHE, contrast stretch, gamma, spatial filters, gradients, FFT bins/spectrum/transfer functions/inverse outputs, HSV, morphology, HED, thresholding, distance transforms, markers, watershed labels and measurements.

## Remaining limitations

- The saturation tissue mask is color-dependent and may omit pale tissue or include saturated artifacts/background.
- HED plus global Otsu thresholding is sensitive to stain chemistry and uneven illumination; this is not a general nuclei detector.
- Distance peaks and watershed can split one object or merge touching objects; area, perimeter, eccentricity and equivalent diameter inherit those mask errors.
- Segmentation is performed on the resized analysis copy, and there is no physical pixel calibration.
- IHC has no reference labels; synthetic results are not clinical validation. Annotated representative H&E cases are needed for meaningful external performance metrics.
- The cache and high-resolution memory profile deserve separate optimization if 1600–3072 pixel images are a primary deployment workload.
