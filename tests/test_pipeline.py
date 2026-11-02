from io import BytesIO
import json
import zipfile

import numpy as np
from PIL import Image
from skimage import morphology

from histopathology.pipeline import (
    contrast_outputs,
    display_mask,
    edge_outputs,
    frequency_filter,
    grayscale,
    load_image,
    nuclei_candidates,
    spatial_filters,
    scale_for_display,
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
    assert tissue["tissue_mask_raw_coverage_percent"] > 0
    assert tissue["tissue_mask_coverage_percent"] > 0
    assert np.isclose(tissue["tissue_mask_raw_coverage_percent"], 100 * tissue["tissue_mask_raw"].mean())
    assert np.isfinite(tissue["tissue_otsu_threshold"])
    assert tissue["tissue_threshold"] == max(tissue["tissue_otsu_threshold"], tissue["tissue_threshold_floor"])
    assert nuclei["nuclei_labels"].shape == rgb.shape[:2]
    assert nuclei["nuclei_overlay"].shape == rgb.shape
    assert "object_id" in nuclei["nuclei_measurements"].columns or nuclei["nuclei_measurements"].empty


def test_to_uint8_handles_boolean_and_float_arrays():
    assert np.array_equal(to_uint8(np.array([[True, False]])), np.array([[255, 0]], dtype=np.uint8))
    normalized = to_uint8(np.array([[-1.0, 0.0, 1.0]], dtype=np.float32))
    assert normalized.dtype == np.uint8
    assert normalized.shape == (1, 3)


def test_display_mask_maps_boolean_foreground_to_full_white_without_mutation():
    mask = np.array([[False, True, False], [True, True, False]], dtype=bool)
    original = mask.copy()
    displayed = display_mask(mask)
    assert displayed.dtype == np.uint8
    assert displayed.shape == mask.shape
    assert np.array_equal(displayed, np.array([[0, 255, 0], [255, 255, 0]], dtype=np.uint8))
    assert np.array_equal(mask, original)


def test_uniform_white_image_has_valid_empty_tissue_mask():
    rgb = np.full((37, 51, 3), 255, dtype=np.uint8)
    result = tissue_mask(rgb, minimum_saturation=0.05, morphology_radius=3, min_object_area=80)
    for key in ("tissue_mask_raw", "tissue_mask_opened", "tissue_mask_closed", "tissue_mask"):
        assert result[key].shape == rgb.shape[:2]
        assert result[key].dtype == np.bool_
        assert not result[key].any()
    assert result["tissue_mask_raw_coverage_percent"] == 0
    assert result["tissue_mask_opened_coverage_percent"] == 0
    assert result["tissue_mask_closed_coverage_percent"] == 0
    assert result["tissue_mask_coverage_percent"] == 0


def test_low_saturation_tissue_and_small_region_survive_adaptive_thresholding():
    rgb = np.full((64, 64, 3), 255, dtype=np.uint8)
    rgb[12:52, 12:52] = [245, 230, 232]  # Pale pink, low saturation tissue.
    result = tissue_mask(rgb, minimum_saturation=0.05, morphology_radius=2, min_object_area=20)
    assert result["tissue_mask"].any()
    assert not result["tissue_mask"][:8, :].any()
    assert result["tissue_mask_raw_coverage_percent"] > 0
    assert result["tissue_mask_coverage_percent"] > 0

    small = np.full((64, 64, 3), 255, dtype=np.uint8)
    small[25:39, 25:39] = [218, 150, 174]
    small_result = tissue_mask(small, minimum_saturation=0.05, morphology_radius=2, min_object_area=20)
    assert small_result["tissue_mask"].any()


def test_saturated_background_uses_otsu_border_color_fallback():
    rgb = np.full((64, 64, 3), [255, 0, 0], dtype=np.uint8)
    rgb[10:54, 10:54] = [218, 150, 174]
    result = tissue_mask(rgb, minimum_saturation=0.05, morphology_radius=3, min_object_area=80)
    assert result["tissue_mask_raw_border_coverage_percent"] == 100
    assert result["tissue_mask_raw_coverage_percent"] < 100
    assert result["tissue_segmentation_strategy"] == "Otsu RGB distance from image-border background"
    assert result["tissue_mask_morphology_input"].dtype == np.bool_
    assert result["tissue_mask_coverage_percent"] > 0
    expected = np.zeros(rgb.shape[:2], dtype=bool)
    expected[10:54, 10:54] = True
    intersection = np.count_nonzero(result["tissue_mask"] & expected)
    union = np.count_nonzero(result["tissue_mask"] | expected)
    assert intersection / union > 0.95
    assert result["tissue_hole_fill_area_limit_pixels"] == int(np.count_nonzero(morphology.disk(3)))


def test_display_scaling_stretches_signed_response_without_changing_source():
    response = np.linspace(-0.2, 0.3, 100, dtype=np.float32).reshape(10, 10)
    before = response.copy()
    display = scale_for_display(response)
    assert display.dtype == np.uint8 and display.shape == response.shape
    assert display.min() == 0 and display.max() == 255
    assert np.array_equal(response, before)


def test_export_bundle_contains_report_and_tables():
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


def _encode_pillow(image, format="PNG", **save_kwargs):
    stream = BytesIO()
    image.save(stream, format=format, **save_kwargs)
    return stream.getvalue()


def test_loader_normalizes_grayscale_rgba_palette_and_uint16():
    gray = np.arange(25, dtype=np.uint8).reshape(5, 5)
    rgba = np.zeros((5, 5, 4), dtype=np.uint8)
    rgba[:, :] = (10, 20, 30, 0)
    rgba[0, 1] = (10, 20, 30, 255)
    palette = Image.fromarray(np.array([[0, 1], [1, 0]], dtype=np.uint8), mode="P")
    palette.putpalette([255, 0, 0, 0, 255, 0] + [0] * 762)
    high_bit = np.linspace(0, 65535, 25, dtype=np.uint16).reshape(5, 5)

    outputs = [
        load_image(_encode_pillow(Image.fromarray(gray)))[0],
        load_image(_encode_pillow(Image.fromarray(rgba)))[0],
        load_image(_encode_pillow(palette))[0],
        load_image(_encode_pillow(Image.fromarray(high_bit)))[0],
    ]
    assert all(image.dtype == np.uint8 and image.shape[2] == 3 for image in outputs)
    assert [image.shape[:2] for image in outputs] == [(5, 5), (5, 5), (2, 2), (5, 5)]
    assert outputs[0][0, 0].tolist() == [0, 0, 0]
    assert outputs[1][0, 0].tolist() == [255, 255, 255]
    assert outputs[1][0, 1].tolist() == [10, 20, 30]
    assert outputs[2][0, 0].tolist() == [255, 0, 0]
    assert outputs[3].min() == 0 and outputs[3].max() == 255


def test_loader_preserves_constant_intensity_in_supported_grayscale_modes():
    binary_white = Image.fromarray(np.ones((3, 4), dtype=np.uint8) * 255).convert("1")
    constant_16_bit = Image.fromarray(np.full((3, 4), 32768, dtype=np.uint16))
    white_rgb, _ = load_image(_encode_pillow(binary_white))
    gray16_rgb, _ = load_image(_encode_pillow(constant_16_bit))
    assert np.all(white_rgb == 255)
    assert np.all(gray16_rgb == 128)


def test_loader_handles_multipage_tiff_and_invalid_uploads():
    first = Image.fromarray(np.full((4, 6), 40, dtype=np.uint8))
    second = Image.fromarray(np.full((4, 6), 200, dtype=np.uint8))
    data = _encode_pillow(first, format="TIFF", save_all=True, append_images=[second])
    rgb, metadata = load_image(data)
    assert rgb.shape == (4, 6, 3)
    assert np.all(rgb == 40)
    assert metadata["tiff_or_image_pages"] == 2
    for invalid in (b"", b"not an image"):
        try:
            load_image(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid uploads must raise ValueError")


def test_loader_applies_exif_orientation_and_handles_extreme_aspect_ratio():
    image = Image.new("RGB", (6, 3), (120, 40, 80))
    exif = Image.Exif()
    exif[274] = 6
    oriented, metadata = load_image(_encode_pillow(image, format="JPEG", exif=exif))
    assert oriented.shape == (6, 3, 3)
    assert metadata["original_width"] == 3 and metadata["original_height"] == 6

    narrow = Image.fromarray(np.full((1, 100), 80, dtype=np.uint8))
    resized, narrow_metadata = load_image(_encode_pillow(narrow), max_dimension=1)
    assert resized.shape == (1, 1, 3)
    assert narrow_metadata["analysis_width"] == 1


def test_loader_converts_cmyk_to_rgb():
    cmyk = Image.new("CMYK", (4, 3), (0, 255, 255, 0))
    rgb, _ = load_image(_encode_pillow(cmyk, format="TIFF"))
    assert rgb.shape == (3, 4, 3)
    assert rgb.dtype == np.uint8
    assert int(rgb[1, 1, 0]) > int(rgb[1, 1, 1])


def test_spatial_filters_and_processing_support_one_pixel_images():
    image = np.array([[127]], dtype=np.uint8)
    results = spatial_filters(image, median_kernel=7, gaussian_sigma=1)
    assert results["median_filtered"].shape == image.shape
    assert results["gaussian_filtered"].shape == image.shape
    assert np.isfinite(results["gaussian_filtered"]).all()
    assert frequency_filter(image)["frequency_filtered"].shape == image.shape


def test_uniform_image_outputs_are_finite_and_edge_free():
    gray = np.full((24, 31), 128, dtype=np.uint8)
    enhanced = contrast_outputs(gray)
    assert all(np.isfinite(value).all() for value in enhanced.values())
    edges = edge_outputs(gray, "All three")
    assert all(not edges[f"{name}_edges"].any() for name in ("roberts", "prewitt", "sobel"))
    for kind in ("Ideal", "Butterworth", "Gaussian"):
        for mode in ("Low-pass", "High-pass"):
            filtered = frequency_filter(gray, kind, mode)["frequency_filtered"]
            assert filtered.shape == gray.shape and np.isfinite(filtered).all()
            if mode == "High-pass":
                assert not filtered.any(), "A constant image contains no high-frequency response."


def test_frequency_filters_have_expected_frequency_response():
    rows, cols = 128, 128
    yy, xx = np.mgrid[:rows, :cols]
    high_frequency = (127.5 + 100 * np.cos(2 * np.pi * 24 * xx / cols)).astype(np.uint8)
    low = frequency_filter(high_frequency, "Gaussian", "Low-pass", cutoff=0.03)["frequency_filtered"]
    high = frequency_filter(high_frequency, "Gaussian", "High-pass", cutoff=0.03)["frequency_filtered"]
    assert float(low.std()) < float(high_frequency.std())
    assert high.max() > 0 and high.std() > 0
    assert np.array_equal(yy.shape, high.shape)


def test_processing_parameters_change_their_intended_outputs():
    yy, xx = np.mgrid[:96, :112]
    ramp = ((xx * 1.5 + yy * 0.7) % 256).astype(np.uint8)
    gamma_low = contrast_outputs(ramp, gamma=0.5)["gamma_corrected"]
    gamma_high = contrast_outputs(ramp, gamma=2.0)["gamma_corrected"]
    assert gamma_low.mean() > gamma_high.mean()
    assert not np.array_equal(spatial_filters(ramp, 3, 0.5)["median_filtered"], spatial_filters(ramp, 7, 0.5)["median_filtered"])
    assert not np.array_equal(spatial_filters(ramp, 3, 0.5)["gaussian_filtered"], spatial_filters(ramp, 3, 2.0)["gaussian_filtered"])
    edges_lo = edge_outputs(ramp, "Sobel", 50)["sobel_edges"]
    edges_hi = edge_outputs(ramp, "Sobel", 99)["sobel_edges"]
    assert edges_hi.sum() <= edges_lo.sum()
    low_cut = frequency_filter(ramp, "Gaussian", "Low-pass", cutoff=0.03)["frequency_filtered"]
    high_cut = frequency_filter(ramp, "Gaussian", "Low-pass", cutoff=0.2)["frequency_filtered"]
    assert not np.array_equal(low_cut, high_cut)


def test_empty_nuclei_result_has_stable_measurement_schema():
    rgb = np.full((20, 20, 3), 255, dtype=np.uint8)
    result = nuclei_candidates(rgb, np.zeros((20, 20), dtype=bool), min_object_area=1)
    assert result["candidate_object_count"] == 0
    assert result["nuclei_labels"].dtype.kind in "iu"
    assert list(result["nuclei_measurements"].columns) == [
        "object_id", "area_pixels", "perimeter_pixels", "eccentricity", "solidity",
        "equivalent_diameter_pixels", "centroid_row", "centroid_column",
        "mean_hematoxylin_intensity",
    ]


def test_small_hole_threshold_keeps_strict_legacy_boundary():
    from histopathology.pipeline import _fill_small_holes

    mask = np.ones((10, 20), dtype=bool)
    mask[2:4, 2:4] = False  # Four-pixel hole: strictly smaller than five.
    mask[2, 10:15] = False  # Five-pixel hole: exactly the threshold.
    filled = _fill_small_holes(mask, maximum_area=5)
    assert filled[2:4, 2:4].all()
    assert not filled[2, 10:15].any()


def test_invalid_shapes_and_parameters_fail_with_clear_value_errors():
    cases = [
        (lambda: grayscale(np.zeros((4, 4), dtype=np.uint8))),
        (lambda: contrast_outputs(np.zeros((0, 4), dtype=np.uint8))),
        (lambda: contrast_outputs(np.zeros((4, 4), dtype=np.uint8), gamma=0)),
        (lambda: spatial_filters(np.zeros((4, 4), dtype=np.uint8), gaussian_sigma=0)),
        (lambda: frequency_filter(np.zeros((4, 4), dtype=np.uint8), mode="middle")),
        (lambda: nuclei_candidates(np.zeros((4, 4, 3), dtype=np.uint8), np.zeros((3, 3)))),
    ]
    for operation in cases:
        try:
            operation()
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid inputs must raise ValueError")


def test_export_sanitizes_uploaded_names_and_preserves_label_array():
    from histopathology.pipeline import build_export_zip

    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    labels = np.array([[0, 300], [2, 0]], dtype=np.int32)
    bundle = build_export_zip("../../odd:name.tif", rgb, {"nuclei_labels": labels}, {}, {}, {})
    with zipfile.ZipFile(BytesIO(bundle)) as archive:
        names = archive.namelist()
        label_path = next(name for name in names if name.endswith("nuclei_labels.npy"))
        roundtrip = np.load(BytesIO(archive.read(label_path)))
        report_path = next(name for name in names if name.endswith("analysis_report.json"))
        json.loads(archive.read(report_path))
    assert all(".." not in name and ":" not in name for name in names)
    assert roundtrip.dtype == np.int32
    assert np.array_equal(roundtrip, labels)
