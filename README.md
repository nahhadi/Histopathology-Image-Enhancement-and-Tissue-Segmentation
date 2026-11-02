# HistoScope 🔬

**Classical digital image processing for histopathology image exploration. No machine-learning model is used.**

HistoScope is a browser-based Streamlit app for exploring H&E-style histopathology images with conventional image-processing methods. It combines image enhancement, color analysis, spatial and frequency filtering, Roberts/Prewitt/Sobel edge detection, morphology-based tissue masking, and a heuristic hematoxylin-based candidate-object workflow with watershed separation. When run locally, processing occurs on your machine; when deployed to a hosted service, uploads are processed on that host. The app does not call a third-party AI service.

> **Research and education only.** HistoScope is not a medical device, does not diagnose disease, and must not be used to guide clinical decisions. Tissue and nuclei-candidate masks are heuristic and can fail when staining, focus, section quality, or background varies.

## Features

- **Image handling:** upload PNG, JPEG, BMP or TIFF; inspect one image while optionally building a batch summary for multiple uploads.
- **Image fundamentals:** RGB channels, grayscale conversion, image dimensions, intensity statistics and grayscale histogram.
- **Enhancement:** global histogram equalization, CLAHE, percentile contrast stretching and gamma correction.
- **Spatial filtering:** median and Gaussian smoothing with adjustable parameters.
- **Edge detection:** Roberts, Prewitt and Sobel gradient magnitude maps, plus adaptive thresholded edge maps.
- **Frequency domain:** centered DFT magnitude visualization and ideal, Butterworth or Gaussian low-/high-pass filtering.
- **Color analysis:** RGB channels and HSV hue, saturation and value components.
- **Morphology/tissue mask:** HSV saturation thresholding, opening, closing, dilation, erosion, hole filling, small-component removal and a red boundary overlay.
- **Candidate nuclei/object analysis:** hematoxylin stain-component thresholding, morphological cleanup, distance-transform markers and watershed labels. This is a conventional heuristic, not a trained or validated nuclei detector.
- **Quantification and export:** tissue coverage, image-level intensity statistics, candidate-object measurements (area, perimeter, eccentricity, solidity and equivalent diameter), CSV tables and a ZIP bundle of image outputs and a JSON settings report.
- **GitHub-friendly structure:** clear module boundaries, test suite, license and CI workflow.

## Quick start

Python 3.10 or newer is recommended.

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/histoscope.git
cd histoscope
```

Replace `YOUR-USERNAME` with your GitHub username after creating the repository.

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. Open it in your browser and upload an image.

## Run tests

```bash
python -m pytest -q
```

## Deploy from GitHub

**Streamlit Community Cloud:** push this repository to GitHub, create a new app in Streamlit Community Cloud, select this repository and branch, and set `app.py` as the main file. The cloud service will install packages from `requirements.txt`. Keep private or identifiable clinical images out of the repository.

**Docker:** from the repository root, build and run the container:

```bash
docker build -t histoscope .
docker run --rm -p 8501:8501 histoscope
```

Then open `http://localhost:8501`.

The tests use a small synthetic image to check code paths and output dimensions; they do not establish clinical validity or guarantee segmentation quality on real histology images.

## Suggested workflow

1. Upload one or more histopathology images.
2. Select an image and inspect its original color, grayscale image and histogram.
3. Compare histogram equalization, CLAHE, contrast stretching and gamma correction.
4. Adjust the input used for edge/frequency analysis and compare Roberts, Prewitt and Sobel.
5. Inspect the DFT spectrum and frequency-filtered result.
6. Inspect each intermediate tissue mask and the final boundary overlay.
7. Treat the hematoxylin-derived output as a *candidate* nuclei/object mask; adjust the thresholds and visually check for missed, merged or falsely identified objects.
8. Download the ZIP archive and CSV measurements. Measurements are in pixels because physical pixel calibration is not included.

## Method notes

### Tissue mask

The tissue mask uses HSV saturation with an Otsu-derived threshold and a user-set lower threshold floor, followed by morphological opening/closing, hole filling and removal of small connected components. This is a fast baseline for pale-background slides, but colored backgrounds, weak staining, folds and artifacts may confuse it.

### Candidate nuclei/object segmentation

The nuclei-candidate path uses a hematoxylin channel derived from an H&E stain deconvolution, Otsu thresholding within the estimated tissue region, morphological cleanup, a distance transform and watershed markers. This may split some touching components, but it may also merge, split, miss or falsely identify objects. Inspect the mask against the source image. A reliable segmentation evaluation requires annotated reference masks or another suitable ground truth.

### Frequency filtering

Frequency cutoff is normalized relative to the image dimensions. Low-pass filtering smooths high-frequency detail; high-pass display emphasizes rapid intensity changes and can also emphasize noise. The high-pass output is intensity-scaled for visualization and is not a reconstructed medical image.

### Batch summary

The batch summary runs the same tissue-mask and candidate-object heuristics over each uploaded file. Its candidate count is not a validated cell count. If working with sensitive clinical images, follow your institution's data-handling rules and do not commit them to GitHub.

## Project structure

```text
histoscope/
├── app.py                       # Streamlit user interface
├── histopathology/
│   ├── __init__.py
│   └── pipeline.py              # Classical image-processing functions
├── tests/
│   └── test_pipeline.py         # Small synthetic tests
├── .github/workflows/tests.yml  # Automated test workflow
├── .streamlit/config.toml       # App theme and upload limit
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Technology stack

- Python, NumPy and pandas
- Streamlit for the local browser interface
- OpenCV for several image operations and contrast-limited adaptive histogram equalization
- scikit-image for color transforms, thresholding, morphology, edge operators, region measurements and watershed
- SciPy for distance transforms and hole filling
- Pillow for image decoding and PNG export

No PyTorch, TensorFlow, pretrained model, neural network or ML classifier is used.

## Limitations and responsible use

- Algorithms are demonstrations of conventional digital image processing, not validated clinical software.
- The current pipeline analyzes a resized working copy when images exceed the selected maximum dimension. Reported areas and coordinates are in those analysis-image pixels.
- The app uses the first page of a multi-page TIFF.
- The current stain-separation and thresholding method assumes images broadly resemble H&E-stained sections; other stains may need a different approach.
- Object measurements depend on segmentation quality and are not meaningful if the mask is poor.
- There is no physical spatial calibration, clinical metadata handling, diagnosis, prognostic inference or patient-level conclusion.
- Do not upload or publish patient images unless you have the required permission and have followed your data governance rules. For a hosted deployment, assume uploaded images are sent to the hosting server for processing.

## License

MIT. See [LICENSE](LICENSE).
