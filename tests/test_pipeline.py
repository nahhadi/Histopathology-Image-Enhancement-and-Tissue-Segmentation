from io import BytesIO

import numpy as np
from PIL import Image

from histopathology.pipeline import (
    contrast_outputs,
    edge_outputs,
    frequency_filter,
    grayscale,
    load_image,
    nuclei_candidates,
    spatial_filters,
    tissue_mask,
    to_uint8,
)


def synthetic_histology_like_image(height=160, width=180):
    """Small deterministic pink field with dark-purple ellipse-like nuclei."""
    yy, xx = np.mgrid[:height, :width]
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = [246, 222, 226]
    tissue = ((xx - width / 2) / (width * 0.42)) ** 2 + ((yy - height / 2) / (height * 0.38)) ** 2 < 1
    image[tissue] = [218, 150, 174]
    for cy, cx in [(45, 45), (70, 65), (90, 100), (55, 120), (110, 135), (95, 70)]:
        nucleus = ((xx - cx) / 6) ** 2 + ((yy - cy) / 8) ** 2 < 1
        image[nucleus] = [80, 35, 105]
    return image


def test_load_image_and_downsample():
    image = synthetic_histology_like_image(300, 400)
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    rgb, metadata = load_image(buffer.getvalue(), max_dimension=200)
    assert rgb.dtype == np.uint8
    assert rgb.shape[2] == 3
    assert max(rgb.shape[:2]) <= 200
    assert metadata["downsampled"] is True
    assert metadata["original_width"] == 400


def test_contrast_and_spatial_filters_preserve_shape():
    rgb = synthetic_histology_like_image()
    gray = grayscale(rgb)
    contrast = contrast_outputs(gray, clahe_clip=2.0, gamma=0.9)
    spatial = spatial_filters(contrast["clahe"], median_kernel=3, gaussian_sigma=1.0)
    assert contrast["histogram_equalized"].shape == gray.shape
    assert contrast["clahe"].dtype == np.uint8
    assert spatial["median_filtered"].shape == gray.shape
    assert spatial["gaussian_filtered"].shape == gray.shape


def test_edge_detectors_produce_maps():
    gray = grayscale(synthetic_histology_like_image())
    maps = edge_outputs(gray, "All three", threshold_percentile=75)
    for name in ("roberts", "prewitt", "sobel"):
        assert maps[f"{name}_magnitude"].shape == gray.shape
        assert maps[f"{name}_edges"].shape == gray.shape
        assert maps[f"{name}_edges"].dtype == np.bool_


def test_frequency_filters_preserve_shape():
    gray = grayscale(synthetic_histology_like_image())
    for kind in ("Ideal", "Butterworth", "Gaussian"):
        for mode in ("Low-pass", "High-pass"):
            result = frequency_filter(gray, kind, mode, cutoff=0.08, butterworth_order=2)
            assert result["frequency_filtered"].shape == gray.shape
            assert result["dft_spectrum"].shape == gray.shape
            assert result["frequency_filtered"].dtype == np.uint8


def test_tissue_and_nuclei_analysis_return_structured_outputs():
    rgb = synthetic_histology_like_image()
    tissue = tissue_mask(rgb, minimum_saturation=0.03, morphology_radius=2, min_object_area=20)
    nuclei = nuclei_candidates(rgb, tissue["tissue_mask"], min_object_area=5, peak_distance=3)
    assert tissue["tissue_mask"].shape == rgb.shape[:2]
    assert 0.0 <= tissue["tissue_coverage"] <= 1.0
    assert nuclei["nuclei_labels"].shape == rgb.shape[:2]
    assert nuclei["nuclei_overlay"].shape == rgb.shape
    assert "object_id" in nuclei["nuclei_measurements"].columns or nuclei["nuclei_measurements"].empty


def test_to_uint8_handles_boolean_and_float_arrays():
    assert np.array_equal(to_uint8(np.array([[True, False]])), np.array([[255, 0]], dtype=np.uint8))
    normalized = to_uint8(np.array([[-1.0, 0.0, 1.0]], dtype=np.float32))
    assert normalized.dtype == np.uint8
    assert normalized.shape == (1, 3)


def test_export_bundle_contains_report_and_tables():
    import zipfile
    from histopathology.pipeline import build_export_zip, basic_metrics

    rgb = synthetic_histology_like_image(80, 100)
    gray = grayscale(rgb)
    tissue = tissue_mask(rgb, minimum_saturation=0.03, morphology_radius=1, min_object_area=5)
    nuclei = nuclei_candidates(rgb, tissue["tissue_mask"], min_object_area=4, peak_distance=2)
    contrast = contrast_outputs(gray)
    outputs = {**contrast, **tissue, **nuclei}
    metrics = basic_metrics(rgb, gray, tissue["tissue_mask"], nuclei["candidate_object_count"])
    bundle = build_export_zip("tiny_sample.png", rgb, outputs, {"format": "PNG"}, metrics, {"test": True})
    with zipfile.ZipFile(BytesIO(bundle)) as archive:
        names = set(archive.namelist())
    assert "tiny_sample/analysis_report.json" in names
    assert "tiny_sample/summary_metrics.csv" in names
    assert "tiny_sample/nuclei_labels.npy" in names
