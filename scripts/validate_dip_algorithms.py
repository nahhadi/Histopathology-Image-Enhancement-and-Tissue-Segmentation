"""Reproducible independent checks and diagnostics for HistoScope DIP algorithms.

Run from the project root with:
    python scripts/validate_dip_algorithms.py

Creates diagnostic images and a JSON result under validation_artifacts/.
The IHC example is bundled with scikit-image and has no segmentation labels.
Only the programmatically generated synthetic image has pixel ground truth.
"""
from __future__ import annotations

import gc
import json
import threading
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from skimage import color, data, exposure, filters, measure, morphology, segmentation, util
from skimage.feature import peak_local_max

from histopathology.pipeline import (
    contrast_outputs,
    edge_outputs,
    frequency_filter,
    grayscale,
    nuclei_candidates,
    scale_for_display,
    spatial_filters,
    tissue_mask,
    to_uint8,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "validation_artifacts"


def synthetic_he_image(height: int = 384, width: int = 512, seed: int = 20261102):
    """Return a noisy, unevenly stained synthetic H&E-like RGB image and masks."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:height, :width]
    regions = [
        ((0.34 * width, 0.47 * height), (0.29 * width, 0.36 * height)),
        ((0.76 * width, 0.68 * height), (0.22 * width, 0.25 * height)),
    ]
    tissue_truth = np.zeros((height, width), dtype=bool)
    for (cx, cy), (rx, ry) in regions:
        tissue_truth |= ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1

    noise = rng.normal(0, 1.8, (height, width, 3))
    rgb = np.empty((height, width, 3), dtype=np.float32)
    background = np.stack((246 - 0.014 * xx, 242 - 0.009 * yy, 238 - 0.008 * xx), axis=2)
    tissue_base = np.stack((231 - 0.035 * xx, 166 + 0.018 * yy, 190 + 0.012 * xx), axis=2)
    rgb[:] = background
    rgb[tissue_truth] = tissue_base[tissue_truth]
    rgb += noise

    # Deliberate size variation, two touching pairs, and one partially
    # overlapping pair. Ellipses supply exact binary foreground ground truth.
    nuclei = [
        (92, 100, 8, 11), (125, 140, 12, 15), (167, 111, 7, 9),
        (180, 190, 17, 18), (203, 190, 16, 17),  # touching pair
        (251, 125, 10, 13), (285, 171, 7, 10), (132, 241, 14, 17),
        (193, 260, 8, 10), (240, 229, 13, 15), (333, 235, 9, 12),
        (367, 270, 11, 14), (398, 300, 16, 15), (417, 303, 13, 14),  # overlap
        (366, 328, 7, 9), (439, 263, 6, 8), (345, 294, 12, 16),
        (165, 314, 9, 12), (109, 188, 6, 8), (225, 300, 11, 13),
    ]
    nuclei_truth = np.zeros((height, width), dtype=bool)
    nucleus_instances = []
    for index, (cx, cy, rx, ry) in enumerate(nuclei):
        ellipse = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1
        ellipse &= tissue_truth
        nuclei_truth |= ellipse
        nucleus_instances.append(ellipse)
        # Purple stain varies slightly across objects and across each object.
        shade = 0.88 + 0.12 * (yy - cy) / max(ry, 1)
        purple = np.array((76 + index % 5 * 2, 34 + index % 4, 112 + index % 7 * 2), dtype=np.float32)
        rgb[ellipse] = (purple * shade[:, :, None])[ellipse]
    rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    return rgb, tissue_truth, nuclei_truth, nucleus_instances


def _time_call(function: Callable, *args, **kwargs):
    start = time.perf_counter()
    result = function(*args, **kwargs)
    return result, (time.perf_counter() - start) * 1000


def _working_set_bytes() -> int:
    """Read current Windows process working set without another dependency."""
    import ctypes
    from ctypes import wintypes

    class Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError()
    return int(counters.WorkingSetSize)


class WorkingSetSampler:
    def __init__(self, interval: float = 0.005):
        self.interval = interval
        self.baseline = _working_set_bytes()
        self.peak = self.baseline
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self.stop_event.is_set():
            self.peak = max(self.peak, _working_set_bytes())
            self.stop_event.wait(self.interval)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join()
        self.peak = max(self.peak, _working_set_bytes())

    @property
    def peak_delta_mb(self):
        return max(0, self.peak - self.baseline) / (1024 * 1024)


def run_stages(rgb: np.ndarray):
    result = {}
    timings = {}
    result["gray"], timings["grayscale_ms"] = _time_call(grayscale, rgb)
    result["contrast"], timings["enhancement_ms"] = _time_call(contrast_outputs, result["gray"])
    result["spatial"], timings["spatial_filters_ms"] = _time_call(spatial_filters, result["contrast"]["clahe"], 3, 1.0)
    result["edges"], timings["edge_operators_ms"] = _time_call(edge_outputs, result["contrast"]["clahe"], "All three", 75)
    result["frequency"], timings["dft_and_filter_ms"] = _time_call(frequency_filter, result["contrast"]["clahe"], "Gaussian", "Low-pass", 0.08, 2)
    result["hsv"] = color.rgb2hsv(util.img_as_float(rgb))
    result["tissue"], timings["tissue_segmentation_ms"] = _time_call(tissue_mask, rgb, 0.05, 3, 80)
    result["nuclei"], timings["hed_threshold_watershed_regions_ms"] = _time_call(
        nuclei_candidates, rgb, result["tissue"]["tissue_mask"], 12, 5
    )
    result["timings"] = timings
    return result


def validate_math(rgb: np.ndarray, tissue_truth: np.ndarray, nuclei_truth: np.ndarray, nucleus_instances):
    """Assert independent definitions, shapes, dtypes, and numerical properties."""
    h, w = rgb.shape[:2]
    assert rgb.shape == (h, w, 3) and rgb.dtype == np.uint8
    channel_probe = np.array([[[11, 22, 33], [44, 55, 66]]], dtype=np.uint8)
    assert np.array_equal(channel_probe[:, :, 0], [[11, 44]])
    assert np.array_equal(channel_probe[:, :, 1], [[22, 55]])
    assert np.array_equal(channel_probe[:, :, 2], [[33, 66]])
    gray = grayscale(rgb)
    assert gray.shape == (h, w) and gray.dtype == np.uint8
    # OpenCV uses the conventional integer approximation to 0.299R+0.587G+0.114B.
    color_probes = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
    assert np.array_equal(grayscale(color_probes), np.array([[76, 150, 29]], dtype=np.uint8))

    hist, edges = np.histogram(gray, bins=256, range=(0, 256))
    assert hist.sum() == h * w and edges.shape == (257,)
    # Independently build the CDF LUT: (CDF - first_nonzero_CDF) mapped to 0..255.
    first_nonzero = int(np.flatnonzero(hist)[0])
    cdf = hist.cumsum()
    cdf_min = int(cdf[first_nonzero])
    lut = np.rint((cdf - cdf_min) * (255.0 / (gray.size - cdf_min))).clip(0, 255).astype(np.uint8)
    lut[:first_nonzero] = 0
    contrast = contrast_outputs(gray, clahe_clip=2.0, gamma=0.9)
    assert np.array_equal(contrast["histogram_equalized"], cv2.equalizeHist(gray))
    assert np.array_equal(contrast["histogram_equalized"], lut[gray])
    expected_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    assert np.array_equal(contrast["clahe"], expected_clahe)
    low, high = np.percentile(gray, [1, 99])
    if high > low:
        expected_stretch = exposure.rescale_intensity(gray, in_range=(float(low), float(high)), out_range=(0, 255)).astype(np.uint8)
        assert np.array_equal(contrast["contrast_stretched"], expected_stretch)
    expected_gamma = util.img_as_ubyte(exposure.adjust_gamma(gray, gamma=0.9))
    assert np.array_equal(contrast["gamma_corrected"], expected_gamma)
    assert all(v.shape == (h, w) and v.dtype == np.uint8 for v in contrast.values())

    spatial = spatial_filters(gray, median_kernel=3, gaussian_sigma=1)
    assert np.array_equal(spatial["median_filtered"], cv2.medianBlur(gray, 3))
    assert np.array_equal(spatial["gaussian_filtered"], cv2.GaussianBlur(gray, (0, 0), 1, sigmaY=1))

    gray_float = util.img_as_float(gray)
    edge_functions = {"roberts": filters.roberts, "prewitt": filters.prewitt, "sobel": filters.sobel}
    edge_out = edge_outputs(gray, "All three", threshold_percentile=75)
    for name, function in edge_functions.items():
        direct = function(gray_float).astype(np.float32)
        peak = float(direct.max())
        expected_magnitude = np.zeros_like(direct) if peak <= np.finfo(np.float32).eps else np.clip(direct / peak * 255, 0, 255).astype(np.uint8)
        assert np.array_equal(edge_out[f"{name}_magnitude"], expected_magnitude)
        assert edge_out[f"{name}_edges"].dtype == np.bool_
        assert edge_out[f"{name}_edges"].shape == (h, w)
    blank_edges = edge_outputs(np.full((16, 19), 73, np.uint8))
    assert not any(blank_edges[f"{name}_edges"].any() for name in edge_functions)

    # fftshift(fftfreq) is the authoritative frequency-bin ordering for both
    # odd and even dimensions. This catches an off-center or normalized-bin bug.
    for length in (7, 8, 31, 32):
        observed = (np.arange(length) - length // 2) / length
        expected = np.fft.fftshift(np.fft.fftfreq(length))
        assert np.allclose(observed, expected)
    fft = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    spectrum = np.log1p(np.abs(fft))
    expected_spectrum = exposure.rescale_intensity(spectrum, out_range=(0, 255)).astype(np.uint8)
    fy = np.fft.fftshift(np.fft.fftfreq(h))
    fx = np.fft.fftshift(np.fft.fftfreq(w))
    radius = np.hypot(*np.meshgrid(fx, fy))
    d0 = 0.08
    outputs = {}
    for kind in ("Ideal", "Butterworth", "Gaussian"):
        if kind == "Ideal":
            low_transfer = (radius <= d0).astype(np.float32)
        elif kind == "Butterworth":
            low_transfer = 1.0 / (1.0 + (radius / d0) ** 4)
        else:
            low_transfer = np.exp(-(radius**2) / (2.0 * d0**2))
        for mode in ("Low-pass", "High-pass"):
            outputs[(kind, mode)] = frequency_filter(gray, kind, mode, cutoff=0.08, butterworth_order=2)
            value = outputs[(kind, mode)]
            assert value["dft_spectrum"].shape == value["frequency_filtered"].shape == (h, w)
            assert value["frequency_filtered"].dtype == np.uint8
            assert np.array_equal(value["dft_spectrum"], expected_spectrum)
            transfer = low_transfer if mode == "Low-pass" else 1.0 - low_transfer
            inverse = np.fft.ifft2(np.fft.ifftshift(fft * transfer)).real
            if mode == "Low-pass":
                expected_image = np.clip(np.rint(inverse), 0, 255).astype(np.uint8)
            else:
                response = np.abs(inverse)
                lo, hi = np.percentile(response, [1, 99.5])
                floor = np.finfo(np.float32).eps * max(1.0, float(np.max(gray)))
                if float(response.max(initial=0.0)) <= floor or hi <= lo:
                    expected_image = np.zeros(gray.shape, dtype=np.uint8)
                else:
                    expected_image = np.clip((response - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
            assert np.array_equal(value["frequency_filtered"], expected_image)
    # Constant input has only the DC coefficient. Low-pass retains it; ideal
    # high-pass removes it; smooth high-pass display is exactly black as well.
    constant = np.full((33, 40), 127, np.uint8)
    for kind in ("Ideal", "Butterworth", "Gaussian"):
        assert np.all(frequency_filter(constant, kind, "Low-pass")["frequency_filtered"] == 127)
        assert not frequency_filter(constant, kind, "High-pass")["frequency_filtered"].any()

    hsv = color.rgb2hsv(util.img_as_float(rgb))
    assert hsv.shape == (h, w, 3) and hsv.dtype.kind == "f"
    tissue = tissue_mask(rgb, 0.05, 3, 80)
    assert tissue["saturation"].shape == (h, w)
    assert tissue["tissue_mask"].dtype == np.bool_
    assert tissue["tissue_mask"].shape == (h, w)
    assert tissue["tissue_boundary"].shape == (h, w)
    assert 0 <= tissue["tissue_coverage"] <= 1
    assert np.array_equal(tissue["tissue_mask_opened"], morphology.opening(tissue["tissue_mask_raw"], morphology.disk(3)))
    assert np.array_equal(tissue["tissue_mask_closed"], morphology.closing(tissue["tissue_mask_opened"], morphology.disk(3)))
    expected_tissue = ndi.binary_fill_holes(tissue["tissue_mask_closed"])
    expected_tissue = morphology.remove_small_objects(expected_tissue, max_size=79)
    assert np.array_equal(tissue["tissue_mask"], expected_tissue)
    hsv_error = float(np.max(np.abs(tissue["saturation"] - hsv[:, :, 1])))

    hed = color.rgb2hed(util.img_as_float(rgb))
    nuclei = nuclei_candidates(rgb, tissue["tissue_mask"], min_object_area=12, peak_distance=5)
    assert np.allclose(nuclei["hematoxylin"], hed[:, :, 0])
    assert nuclei["nuclei_candidate_mask"].dtype == np.bool_
    assert nuclei["nuclei_labels"].dtype == np.int32
    assert nuclei["nuclei_labels"].shape == (h, w)
    assert nuclei["nuclei_overlay"].shape == (h, w, 3)
    candidate_raw = (nuclei["hematoxylin"] > nuclei["hematoxylin_threshold"]) & tissue["tissue_mask"]
    cleaned = morphology.opening(candidate_raw, morphology.disk(1))
    cleaned = morphology.remove_small_objects(cleaned, max_size=11)
    cleaned = morphology.remove_small_holes(cleaned, max_size=5)
    assert np.array_equal(cleaned, nuclei["nuclei_candidate_mask"])
    distance = ndi.distance_transform_edt(cleaned)
    assert distance.shape == (h, w) and distance.dtype.kind == "f"
    coords = peak_local_max(distance, labels=cleaned.astype(np.uint8), min_distance=5, exclude_border=False)
    assert len(coords) > 0
    markers = np.zeros((h, w), dtype=np.int32)
    for marker_id, coord in enumerate(coords, 1):
        markers[tuple(coord)] = marker_id
    assert int(markers.max()) == len(coords)
    expected_labels = segmentation.watershed(-distance, markers, mask=cleaned)
    assert np.array_equal(expected_labels, nuclei["nuclei_labels"])
    foreground_labels = np.unique(nuclei["nuclei_labels"])
    foreground_labels = foreground_labels[foreground_labels != 0]
    measurements = nuclei["nuclei_measurements"]
    assert set(measurements["object_id"].astype(int)) == set(foreground_labels.astype(int))
    assert np.array_equal(measurements["area_pixels"].to_numpy(), np.array([
        int(np.count_nonzero(nuclei["nuclei_labels"] == label)) for label in measurements["object_id"]
    ]))
    direct_props = measure.regionprops_table(
        nuclei["nuclei_labels"], intensity_image=nuclei["hematoxylin"],
        properties=("label", "area", "perimeter", "eccentricity", "solidity", "equivalent_diameter_area", "centroid", "mean_intensity"),
    )
    direct_frame = pd.DataFrame(direct_props).rename(columns={
        "label": "object_id", "area": "area_pixels", "perimeter": "perimeter_pixels",
        "equivalent_diameter_area": "equivalent_diameter_pixels", "centroid-0": "centroid_row",
        "centroid-1": "centroid_column", "mean_intensity": "mean_hematoxylin_intensity",
    })
    measured_sorted = measurements.sort_values("object_id").reset_index(drop=True)
    direct_sorted = direct_frame.sort_values("object_id").reset_index(drop=True)
    for column in measured_sorted.columns:
        assert np.allclose(measured_sorted[column], direct_sorted[column], rtol=1e-7, atol=1e-9)
    assert np.isfinite(measurements.select_dtypes(include=[np.number]).to_numpy()).all()

    # Synthetic annotation supports only foreground pixel metrics for this
    # generated scene; it is not a benchmark or clinical accuracy estimate.
    prediction = nuclei["nuclei_labels"] > 0
    tp = int(np.count_nonzero(prediction & nuclei_truth))
    fp = int(np.count_nonzero(prediction & ~nuclei_truth))
    fn = int(np.count_nonzero(~prediction & nuclei_truth))
    union = tp + fp + fn
    metrics = {
        "synthetic_pixel_dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
        "synthetic_pixel_iou": tp / union if union else 1.0,
        "synthetic_pixel_precision": tp / (tp + fp) if tp + fp else 1.0,
        "synthetic_pixel_recall": tp / (tp + fn) if tp + fn else 1.0,
        "synthetic_ground_truth_nuclei_count": int(len(nucleus_instances)),
        "synthetic_watershed_region_count": int(len(foreground_labels)),
        "synthetic_true_positive_pixels": tp,
        "synthetic_false_positive_pixels": fp,
        "synthetic_false_negative_pixels": fn,
        "hsv_saturation_max_abs_error": hsv_error,
    }
    gt_tissue_tp = int(np.count_nonzero(tissue["tissue_mask"] & tissue_truth))
    gt_tissue_fp = int(np.count_nonzero(tissue["tissue_mask"] & ~tissue_truth))
    gt_tissue_fn = int(np.count_nonzero(~tissue["tissue_mask"] & tissue_truth))
    metrics.update({
        "synthetic_tissue_dice": 2 * gt_tissue_tp / (2 * gt_tissue_tp + gt_tissue_fp + gt_tissue_fn),
        "synthetic_tissue_iou": gt_tissue_tp / (gt_tissue_tp + gt_tissue_fp + gt_tissue_fn),
        "synthetic_tissue_false_positive_pixels": gt_tissue_fp,
        "synthetic_tissue_false_negative_pixels": gt_tissue_fn,
    })
    truth_to_predicted = []
    for instance_mask in nucleus_instances:
        overlapping_labels = np.unique(nuclei["nuclei_labels"][instance_mask])
        overlapping_labels = overlapping_labels[overlapping_labels != 0]
        truth_to_predicted.append(set(int(label) for label in overlapping_labels))
    metrics.update({
        "synthetic_truth_nuclei_without_any_predicted_label": sum(not labels for labels in truth_to_predicted),
        "synthetic_truth_nuclei_spanning_multiple_predicted_labels": sum(len(labels) > 1 for labels in truth_to_predicted),
        "synthetic_predicted_labels_overlapping_multiple_truth_nuclei": sum(
            sum(label in labels for labels in truth_to_predicted) > 1 for label in foreground_labels
        ),
    })
    return {
        "gray": gray, "contrast": contrast, "spatial": spatial, "edges": edge_out,
        "spectrum": expected_spectrum, "tissue": tissue, "hed": hed[:, :, 0],
        "nuclei": nuclei, "raw_candidate": candidate_raw, "distance": distance,
        "markers": markers, "metrics": metrics,
    }


def _label_visual(labels: np.ndarray, rgb: np.ndarray):
    overlay = color.label2rgb(labels, image=util.img_as_float(rgb), bg_label=0, alpha=0.65)
    return Image.fromarray(np.clip(overlay * 255, 0, 255).astype(np.uint8))


def make_contact_sheet(rgb, values):
    h, w = rgb.shape[:2]
    tile_w = 256
    tile_h = max(144, round(tile_w * h / w))
    label_h = 30
    cols, rows = 5, 2
    sheet = Image.new("RGB", (cols * tile_w, rows * (tile_h + label_h)), "#f2f2f2")
    font = ImageFont.load_default()
    labels = [
        ("Original RGB", Image.fromarray(rgb)),
        ("Grayscale", Image.fromarray(values["gray"])),
        ("CLAHE", Image.fromarray(values["contrast"]["clahe"])),
        ("Hematoxylin (display scaled)", Image.fromarray(scale_for_display(values["hed"]))),
        ("Raw H threshold", Image.fromarray(values["raw_candidate"].astype(np.uint8) * 255)),
        ("Morphology cleaned", Image.fromarray(values["nuclei"]["nuclei_candidate_mask"].astype(np.uint8) * 255)),
        ("Distance transform", Image.fromarray(scale_for_display(values["distance"]))),
        ("Watershed markers", _label_visual(values["markers"], rgb)),
        ("Final watershed labels", _label_visual(values["nuclei"]["nuclei_labels"], rgb)),
        ("Final overlay", Image.fromarray(values["nuclei"]["nuclei_overlay"])),
    ]
    for index, (title, image) in enumerate(labels):
        x = (index % cols) * tile_w
        y = (index // cols) * (tile_h + label_h)
        sheet.paste(image.convert("RGB").resize((tile_w, tile_h), Image.Resampling.NEAREST if "mask" in title.lower() or "marker" in title.lower() or "label" in title.lower() else Image.Resampling.LANCZOS), (x, y + label_h))
        ImageDraw.Draw(sheet).text((x + 6, y + 7), title, fill="#111111", font=font)
    return sheet


def make_ihc_sheet(rgb):
    gray = grayscale(rgb)
    tissue = tissue_mask(rgb)
    nuclei = nuclei_candidates(rgb, tissue["tissue_mask"])
    hsv = color.rgb2hsv(util.img_as_float(rgb))
    entries = [
        ("Bundled IHC image (not H&E)", Image.fromarray(rgb)),
        ("Grayscale", Image.fromarray(gray)),
        ("HSV saturation", Image.fromarray(scale_for_display(hsv[:, :, 1]))),
        ("Saturation tissue mask", Image.fromarray(tissue["tissue_mask"].astype(np.uint8) * 255)),
        ("Heuristic labels over IHC", Image.fromarray(nuclei["nuclei_overlay"])),
    ]
    tile_w, tile_h, label_h = 240, 240, 34
    sheet = Image.new("RGB", (len(entries) * tile_w, tile_h + label_h), "#f2f2f2")
    draw = ImageDraw.Draw(sheet)
    for index, (title, image) in enumerate(entries):
        x = index * tile_w
        draw.text((x + 6, 8), title, fill="#111111", font=ImageFont.load_default())
        sheet.paste(image.convert("RGB").resize((tile_w, tile_h), Image.Resampling.LANCZOS), (x, label_h))
    return sheet


def run_performance_sizes(base_rgb):
    results = []
    for size in (256, 512, 1024):
        rgb = cv2.resize(base_rgb, (size, size), interpolation=cv2.INTER_AREA if size < base_rgb.shape[0] else cv2.INTER_LINEAR)
        gc.collect()
        timings = {}
        with WorkingSetSampler() as memory:
            start = time.perf_counter()
            outputs = run_stages(rgb)
            total_ms = (time.perf_counter() - start) * 1000
            array_bytes = rgb.nbytes
            for key in ("gray", "hsv"):
                array_bytes += outputs[key].nbytes
            array_bytes += sum(value.nbytes for value in outputs["contrast"].values())
            array_bytes += sum(value.nbytes for value in outputs["spatial"].values())
            array_bytes += sum(value.nbytes for value in outputs["edges"].values())
            array_bytes += sum(value.nbytes for value in outputs["frequency"].values())
            array_bytes += sum(value.nbytes for value in outputs["tissue"].values() if isinstance(value, np.ndarray))
            array_bytes += sum(value.nbytes for value in outputs["nuclei"].values() if isinstance(value, np.ndarray))
        timings.update(outputs["timings"])
        results.append({
            "size": f"{size}x{size}",
            "total_ms": round(total_ms, 2),
            "stage_ms": {key: round(value, 2) for key, value in timings.items()},
            "baseline_working_set_mb": round(memory.baseline / (1024 * 1024), 2),
            "peak_working_set_mb_sampled": round(memory.peak / (1024 * 1024), 2),
            "peak_working_set_delta_mb_sampled": round(memory.peak_delta_mb, 2),
            "retained_pipeline_array_mb": round(array_bytes / (1024 * 1024), 2),
        })
    return results


def main():
    if not hasattr(__import__("ctypes").windll, "psapi"):
        raise RuntimeError("This performance sampler is Windows-specific.")
    ARTIFACTS.mkdir(exist_ok=True)
    rgb, tissue_truth, nuclei_truth, nucleus_instances = synthetic_he_image()
    checked = validate_math(rgb, tissue_truth, nuclei_truth, nucleus_instances)
    make_contact_sheet(rgb, checked).save(ARTIFACTS / "synthetic_he_diagnostic.png")

    # This bundled color tissue sample is real histopathology imagery, but it
    # is immunohistochemistry (DAB) and has no annotation; no accuracy score.
    ihc = data.immunohistochemistry()
    Image.fromarray(ihc).save(ARTIFACTS / "ihc_sample_diagnostic.png")
    make_ihc_sheet(ihc).save(ARTIFACTS / "ihc_pipeline_comparison.png")

    performance = run_performance_sizes(rgb)
    report = {
        "validation_scope": "Independent mathematical checks on synthetic H&E-like image; descriptive run on bundled unannotated IHC image.",
        "synthetic_seed": 20261102,
        "synthetic_shape": list(rgb.shape),
        "synthetic_ground_truth_is_programmatic_only": True,
        "real_sample": "scikit-image bundled immunohistochemistry sample (DAB/IHC; no labels)",
        "synthetic_metrics": checked["metrics"],
        "performance": performance,
        "notes": [
            "Synthetic binary metrics compare the union of predicted watershed labels with the generated nucleus-pixel mask only.",
            "The generated mask includes known touching and overlapping nuclei; object-level matching is not performed.",
            "No ground-truth segmentation metrics are reported for the bundled IHC sample.",
            "Peak working-set deltas are sampled every 5 ms and can miss short-lived peaks.",
            "Retained pipeline array size is a byte sum, not total process memory.",
        ],
    }
    (ARTIFACTS / "validation_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Artifacts: {ARTIFACTS}")


if __name__ == "__main__":
    main()
