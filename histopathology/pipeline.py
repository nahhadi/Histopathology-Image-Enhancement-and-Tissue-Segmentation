"""Classical image-processing pipeline for histopathology exploration.

This module intentionally uses conventional image processing only. It does not
perform machine-learning inference or make medical diagnoses.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any
import json
import zipfile

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy import ndimage as ndi
from skimage import color, exposure, filters, measure, morphology, segmentation, util
from skimage.feature import peak_local_max


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _require_gray(image: np.ndarray) -> np.ndarray:
    """Return a non-empty 2D image or raise a clear input error."""
    arr = np.asarray(image)
    if arr.ndim != 2 or not arr.shape[0] or not arr.shape[1]:
        raise ValueError("Expected a non-empty two-dimensional grayscale image.")
    return arr


def _require_rgb(image: np.ndarray) -> np.ndarray:
    """Return a non-empty RGB image or raise a clear input error."""
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3 or not arr.shape[0] or not arr.shape[1]:
        raise ValueError("Expected a non-empty RGB image with exactly three channels.")
    return arr


def _scale_to_uint8(array: np.ndarray) -> np.ndarray:
    """Convert integer/float image arrays to uint8 using robust percentile scaling."""
    arr = np.asarray(array)
    if arr.dtype == np.uint8:
        return arr
    arr_float = arr.astype(np.float32, copy=False)
    finite = np.isfinite(arr_float)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    low, high = np.percentile(arr_float[finite], [1, 99])
    if high <= low:
        value = float(arr_float[finite][0])
        if arr.dtype == np.bool_:
            return np.full(arr.shape, 255 if value else 0, dtype=np.uint8)
        if np.issubdtype(arr.dtype, np.floating) and 0.0 <= value <= 1.0:
            return np.full(arr.shape, int(round(value * 255.0)), dtype=np.uint8)
        if arr.dtype == np.uint8:
            return np.full(arr.shape, int(round(value)), dtype=np.uint8)
        if np.issubdtype(arr.dtype, np.unsignedinteger):
            return np.full(arr.shape, int(round(value * 255.0 / np.iinfo(arr.dtype).max)), dtype=np.uint8)
        if np.issubdtype(arr.dtype, np.signedinteger) and 0.0 <= value <= 65535.0:
            return np.full(arr.shape, int(round(value * 255.0 / 65535.0)), dtype=np.uint8)
        return np.zeros(arr.shape, dtype=np.uint8)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(np.min(arr_float[finite]))
        high = float(np.max(arr_float[finite]))
    if high <= low:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr_float - low) * (255.0 / (high - low))
    return np.clip(np.nan_to_num(scaled, nan=0.0, posinf=255.0, neginf=0.0), 0, 255).astype(np.uint8)


def _remove_small_objects(mask: np.ndarray, minimum_area: int) -> np.ndarray:
    """Compatibility wrapper for scikit-image's min_size/max_size rename."""
    minimum_area = max(1, int(minimum_area))
    try:
        # scikit-image >= 0.26: remove components with area <= max_size.
        return morphology.remove_small_objects(mask.astype(bool), max_size=minimum_area - 1)
    except TypeError:
        # scikit-image < 0.26: remove components strictly smaller than min_size.
        return morphology.remove_small_objects(mask.astype(bool), min_size=minimum_area)


def _fill_small_holes(mask: np.ndarray, maximum_area: int) -> np.ndarray:
    """Compatibility wrapper for scikit-image's area_threshold/max_size rename."""
    maximum_area = max(1, int(maximum_area))
    try:
        # Prior area_threshold semantics fill strictly smaller holes. The new
        # max_size API fills <= max_size, so subtract one to preserve behavior.
        return morphology.remove_small_holes(mask.astype(bool), max_size=maximum_area - 1)
    except TypeError:
        return morphology.remove_small_holes(mask.astype(bool), area_threshold=maximum_area)


def load_image(image_bytes: bytes, max_dimension: int = 1600) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode an uploaded image as RGB uint8 and downsample large images safely.

    For multi-page TIFFs, the first page is used. Non-uint8 and grayscale images
    are normalized for visualization and classical processing.
    """
    if not image_bytes:
        raise ValueError("The uploaded file is empty.")
    max_dimension = max(1, int(max_dimension))
    try:
        with Image.open(BytesIO(image_bytes)) as source:
            file_format = source.format or "Unknown"
            page_count = getattr(source, "n_frames", 1)
            source.seek(0)
            image = ImageOps.exif_transpose(source.copy())
            original_width, original_height = image.size
            # Resize while still in Pillow so the full decoded pixel array is
            # never duplicated in NumPy for large uploads.
            scale = min(1.0, max_dimension / float(max(original_height, original_width)))
            if scale < 1.0:
                image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            raw = np.asarray(image)
            if raw.ndim == 2 and image.mode in ("1", "L", "I", "I;16", "I;16B", "I;16L", "F"):
                gray = _scale_to_uint8(raw)
                rgb = np.repeat(gray[:, :, None], 3, axis=2)
            elif raw.ndim == 3 and raw.shape[2] == 3 and raw.dtype == np.uint8:
                rgb = raw.copy()
            elif raw.ndim == 3 and raw.shape[2] == 4 and raw.dtype == np.uint8:
                rgba = image.convert("RGBA")
                background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                rgb = np.asarray(Image.alpha_composite(background, rgba).convert("RGB"), dtype=np.uint8).copy()
            else:
                # Convert palette/CMYK/high-bit-depth color and honor alpha by
                # compositing transparency onto white (slide background).
                if "transparency" in image.info or "A" in image.getbands():
                    rgba = image.convert("RGBA")
                    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                    image = Image.alpha_composite(background, rgba)
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    except Exception as exc:
        raise ValueError(f"Could not read this image: {exc}") from exc

    metadata = {
        "format": file_format,
        "tiff_or_image_pages": int(page_count),
        "original_width": int(original_width),
        "original_height": int(original_height),
        "analysis_width": int(rgb.shape[1]),
        "analysis_height": int(rgb.shape[0]),
        "downsampled": bool(scale < 1.0),
        "scale_factor": float(scale),
    }
    return np.ascontiguousarray(rgb), metadata


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert an image for display/export; preserves uint8 images."""
    arr = np.asarray(image)
    if arr.dtype == bool:
        return arr.astype(np.uint8) * 255
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.integer) and arr.size and arr.min() >= 0 and arr.max() <= 255:
        return arr.astype(np.uint8)
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    if arr.size and arr.min() >= 0.0 and arr.max() <= 1.0:
        arr = arr * 255.0
    elif arr.size and (arr.min() < 0.0 or arr.max() > 255.0):
        finite = arr[np.isfinite(arr)]
        if finite.size:
            low, high = np.percentile(finite, [1, 99])
            if high > low:
                arr = (arr - low) * (255.0 / (high - low))
            else:
                arr = np.zeros_like(arr)
    return np.clip(arr, 0, 255).astype(np.uint8)


def display_mask(mask: np.ndarray) -> np.ndarray:
    """Convert a Boolean mask to an 8-bit display image without mutating it."""
    arr = np.asarray(mask)
    if arr.dtype != np.bool_:
        raise ValueError("display_mask expects a Boolean mask.")
    return arr.astype(np.uint8) * np.uint8(255)


def scale_for_display(image: np.ndarray, lower_percentile: float = 1.0, upper_percentile: float = 99.0) -> np.ndarray:
    """Robustly stretch a numeric response for visualization without changing analysis data."""
    arr = np.asarray(image)
    if arr.dtype == np.uint8 or arr.dtype == bool:
        return to_uint8(arr)
    if not (0 <= lower_percentile < upper_percentile <= 100):
        raise ValueError("Display percentiles must satisfy 0 <= lower < upper <= 100.")
    values = arr.astype(np.float32, copy=False)
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    low, high = np.percentile(values[finite], [lower_percentile, upper_percentile])
    if high <= low:
        return to_uint8(values)
    scaled = (values - low) * (255.0 / (high - low))
    return np.clip(np.nan_to_num(scaled, nan=0.0, posinf=255.0, neginf=0.0), 0, 255).astype(np.uint8)


def grayscale(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(to_uint8(_require_rgb(rgb)), cv2.COLOR_RGB2GRAY)


def contrast_outputs(gray: np.ndarray, clahe_clip: float = 2.0, gamma: float = 0.9) -> dict[str, np.ndarray]:
    """Calculate global histogram equalization, CLAHE, percentile stretch and gamma correction."""
    gray = to_uint8(_require_gray(gray))
    if not np.isfinite(clahe_clip) or clahe_clip <= 0:
        raise ValueError("CLAHE clip limit must be a finite number greater than zero.")
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("Gamma must be a finite number greater than zero.")
    equalized = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(8, 8)).apply(gray)
    low, high = np.percentile(gray, [1, 99])
    if high <= low:
        stretched = gray.copy()
    else:
        stretched = exposure.rescale_intensity(gray, in_range=(float(low), float(high)), out_range=(0, 255)).astype(np.uint8)
    gamma_corrected = util.img_as_ubyte(exposure.adjust_gamma(gray, gamma=float(gamma)))
    return {
        "grayscale": gray,
        "histogram_equalized": equalized,
        "clahe": clahe,
        "contrast_stretched": stretched,
        "gamma_corrected": gamma_corrected,
    }


def spatial_filters(gray: np.ndarray, median_kernel: int = 3, gaussian_sigma: float = 1.0) -> dict[str, np.ndarray]:
    gray = to_uint8(_require_gray(gray))
    if not np.isfinite(gaussian_sigma) or gaussian_sigma <= 0:
        raise ValueError("Gaussian sigma must be a finite number greater than zero.")
    kernel = max(3, int(median_kernel))
    if kernel % 2 == 0:
        kernel += 1
    # OpenCV medianBlur requires a kernel no larger than the smallest image
    # dimension for non-uint8 inputs; use the largest valid odd kernel here.
    limit = min(gray.shape)
    if limit % 2 == 0:
        limit -= 1
    median = cv2.medianBlur(gray, min(kernel, max(1, limit)))
    gaussian = cv2.GaussianBlur(gray, (0, 0), sigmaX=float(gaussian_sigma), sigmaY=float(gaussian_sigma))
    return {"median_filtered": median, "gaussian_filtered": gaussian}


def edge_outputs(gray: np.ndarray, method: str = "All three", threshold_percentile: float = 75.0) -> dict[str, np.ndarray]:
    """Return Roberts, Prewitt and Sobel gradient magnitudes plus thresholded maps.

    Thresholds are adaptive per image and are intended for visualization, not as
    universally optimal segmentation parameters.
    """
    gray_float = util.img_as_float(to_uint8(_require_gray(gray)))
    all_methods = {
        "Roberts": filters.roberts(gray_float),
        "Prewitt": filters.prewitt(gray_float),
        "Sobel": filters.sobel(gray_float),
    }
    if method not in ("All three", *all_methods):
        raise ValueError("Edge method must be Roberts, Prewitt, Sobel, or All three.")
    selected = all_methods if method == "All three" else {method: all_methods[method]}
    output: dict[str, np.ndarray] = {}
    percentile = float(np.clip(threshold_percentile, 50.0, 99.5))
    for name, magnitude in selected.items():
        mag = np.asarray(magnitude, dtype=np.float32)
        max_value = float(np.max(mag)) if mag.size else 0.0
        # Floating point gradient operators can leave ~1e-17 residuals on a
        # constant image. Treat those as numerical zero, not a full-strength edge.
        norm = mag / max_value if max_value > np.finfo(np.float32).eps else np.zeros_like(mag)
        positive = norm[norm > 0]
        threshold = float(np.percentile(positive, percentile)) if positive.size else 1.0
        binary = norm >= threshold if positive.size else np.zeros(norm.shape, dtype=bool)
        safe_name = name.lower()
        output[f"{safe_name}_magnitude"] = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
        output[f"{safe_name}_edges"] = binary
    return output


def frequency_filter(gray: np.ndarray, filter_type: str = "Gaussian", mode: str = "Low-pass", cutoff: float = 0.08, butterworth_order: int = 2) -> dict[str, np.ndarray]:
    """Apply an ideal, Butterworth or Gaussian filter in the Fourier domain.

    cutoff is a normalized radial frequency (0, 0.5]. The output is clipped to
    the original uint8 intensity range rather than independently contrast-scaled.
    """
    image = to_uint8(_require_gray(gray)).astype(np.float32)
    if filter_type not in {"Ideal", "Butterworth", "Gaussian"}:
        raise ValueError("Filter type must be Ideal, Butterworth, or Gaussian.")
    if mode not in {"Low-pass", "High-pass"}:
        raise ValueError("Filter mode must be Low-pass or High-pass.")
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("Cutoff frequency must be a finite number greater than zero.")
    if int(butterworth_order) < 1:
        raise ValueError("Butterworth order must be at least one.")
    rows, cols = image.shape
    spectrum = np.fft.fftshift(np.fft.fft2(image))
    magnitude = np.log1p(np.abs(spectrum))
    yy = (np.arange(rows) - rows // 2) / max(rows, 1)
    xx = (np.arange(cols) - cols // 2) / max(cols, 1)
    xx_grid, yy_grid = np.meshgrid(xx, yy)
    radius = np.sqrt(xx_grid**2 + yy_grid**2)
    d0 = float(np.clip(cutoff, 0.005, 0.5))
    if filter_type == "Ideal":
        low_pass = (radius <= d0).astype(np.float32)
    elif filter_type == "Butterworth":
        order = max(1, int(butterworth_order))
        low_pass = 1.0 / (1.0 + (radius / d0) ** (2 * order))
    else:
        low_pass = np.exp(-(radius**2) / (2.0 * d0**2))
    transfer = low_pass if mode == "Low-pass" else (1.0 - low_pass)
    filtered = np.fft.ifft2(np.fft.ifftshift(spectrum * transfer)).real
    if mode == "High-pass":
        # A high-pass response is signed and centered near zero. Display its
        # absolute response with robust scaling instead of clipping negatives.
        response = np.abs(filtered)
        numerical_floor = np.finfo(np.float32).eps * max(1.0, float(np.max(np.abs(image))))
        lo, hi = np.percentile(response, [1, 99.5])
        if float(response.max(initial=0.0)) <= numerical_floor:
            filtered_uint8 = np.zeros_like(image, dtype=np.uint8)
        elif hi > lo:
            filtered_uint8 = np.clip((response - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
        else:
            filtered_uint8 = np.zeros_like(image, dtype=np.uint8)
    else:
        filtered_uint8 = np.clip(np.rint(filtered), 0, 255).astype(np.uint8)
    spectrum_display = exposure.rescale_intensity(magnitude, out_range=(0, 255)).astype(np.uint8)
    return {"dft_spectrum": spectrum_display, "frequency_filtered": filtered_uint8}


def tissue_mask(rgb: np.ndarray, minimum_saturation: float = 0.05, morphology_radius: int = 3, min_object_area: int = 80) -> dict[str, np.ndarray | float]:
    """Create a coarse tissue/background mask from HSV saturation and morphology."""
    if not np.isfinite(minimum_saturation):
        raise ValueError("Minimum saturation must be finite.")
    rgb_float = util.img_as_float(to_uint8(_require_rgb(rgb)))
    hsv = color.rgb2hsv(rgb_float)
    saturation = hsv[:, :, 1]
    finite_values = saturation[np.isfinite(saturation)]
    try:
        otsu = float(filters.threshold_otsu(finite_values)) if finite_values.size else 0.05
    except ValueError:
        otsu = 0.05
    threshold_floor = float(np.clip(minimum_saturation, 0.0, 0.95))
    threshold = max(otsu, threshold_floor)
    raw = saturation > threshold
    # Saturation alone cannot distinguish tissue from a uniformly saturated
    # slide background. When most of the perimeter is selected, use the
    # perimeter's median RGB color as an estimated background and Otsu on
    # color distance. This remains data-driven and is only a fallback; the
    # displayed raw mask always preserves the original saturation result.
    border = np.zeros(raw.shape, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    raw_border_coverage = (
        100.0 * float(np.count_nonzero(raw & border)) / int(np.count_nonzero(border))
        if border.any() else 0.0
    )
    morphology_input = raw
    strategy = "HSV saturation"
    border_color_threshold = float("nan")
    if raw_border_coverage > 50.0:
        border_rgb = np.median(rgb_float[border], axis=0)
        color_distance = np.linalg.norm(rgb_float - border_rgb, axis=2)
        try:
            border_color_threshold = float(filters.threshold_otsu(color_distance))
            border_candidate = color_distance > border_color_threshold
            if border_candidate.any() and not border_candidate.all():
                morphology_input = border_candidate
                strategy = "Otsu RGB distance from image-border background"
        except ValueError:
            pass
    radius = max(1, int(morphology_radius))
    selem = morphology.disk(radius)
    opened = morphology.opening(morphology_input, selem)
    closed = morphology.closing(opened, selem)
    # Restrict hole filling to the scale of the chosen structuring element.
    # Filling every enclosed hole can turn a saturated background ring into
    # an all-foreground image by filling the entire tissue-shaped interior.
    hole_area_limit = int(np.count_nonzero(selem))
    filled = _fill_small_holes(closed, hole_area_limit)
    cleaned = _remove_small_objects(filled.astype(bool), min_object_area)
    def foreground_percent(mask: np.ndarray) -> float:
        return 100.0 * float(np.count_nonzero(mask)) / mask.size if mask.size else 0.0

    coverage = foreground_percent(cleaned) / 100.0
    return {
        "saturation": saturation,
        "tissue_mask_raw": raw,
        "tissue_mask_morphology_input": morphology_input,
        "tissue_mask_dilated": morphology.dilation(raw, selem),
        "tissue_mask_eroded": morphology.erosion(raw, selem),
        "tissue_mask_opened": opened,
        "tissue_mask_closed": closed,
        "tissue_mask_filled": filled,
        "tissue_mask": cleaned,
        "tissue_boundary": segmentation.find_boundaries(cleaned, mode="inner"),
        "tissue_otsu_threshold": otsu,
        "tissue_threshold_floor": threshold_floor,
        "tissue_threshold": threshold,
        "tissue_segmentation_strategy": strategy,
        "tissue_border_color_threshold": border_color_threshold,
        "tissue_coverage": coverage,
        "tissue_mask_raw_coverage_percent": foreground_percent(raw),
        "tissue_mask_opened_coverage_percent": foreground_percent(opened),
        "tissue_mask_closed_coverage_percent": foreground_percent(closed),
        "tissue_mask_filled_coverage_percent": foreground_percent(filled),
        "tissue_mask_coverage_percent": foreground_percent(cleaned),
        "tissue_mask_raw_border_coverage_percent": raw_border_coverage,
        "tissue_hole_fill_area_limit_pixels": hole_area_limit,
    }


def nuclei_candidates(rgb: np.ndarray, tissue: np.ndarray | None = None, min_object_area: int = 12, peak_distance: int = 5) -> dict[str, Any]:
    """Create heuristic H&E hematoxylin-based candidate nuclei masks and watershed labels.

    This is a classical thresholding method, not a trained nuclei detector. Stain
    variation can lead to false positives or missed nuclei and must be inspected.
    """
    rgb_float = util.img_as_float(to_uint8(_require_rgb(rgb)))
    if tissue is not None and np.asarray(tissue).shape != rgb_float.shape[:2]:
        raise ValueError("Tissue mask dimensions must match the RGB image dimensions.")
    hed = color.rgb2hed(rgb_float)
    hematoxylin = hed[:, :, 0]
    roi = np.ones(hematoxylin.shape, dtype=bool) if tissue is None else np.asarray(tissue, dtype=bool)
    values = hematoxylin[roi]
    values = values[np.isfinite(values)]
    try:
        threshold = float(filters.threshold_otsu(values)) if values.size else float(np.max(hematoxylin))
    except ValueError:
        threshold = float(np.max(hematoxylin)) if hematoxylin.size else 0.0
    candidates = (hematoxylin > threshold) & roi
    candidates = morphology.opening(candidates, morphology.disk(1))
    candidates = _remove_small_objects(candidates, min_object_area)
    candidates = _fill_small_holes(candidates, max(1, int(min_object_area // 2)))

    distance = ndi.distance_transform_edt(candidates)
    coords = peak_local_max(
        distance,
        labels=candidates.astype(np.uint8),
        min_distance=max(1, int(peak_distance)),
        exclude_border=False,
    )
    markers = np.zeros(candidates.shape, dtype=np.int32)
    for idx, coord in enumerate(coords, start=1):
        markers[tuple(coord)] = idx
    if coords.size == 0 and candidates.any():
        markers, _ = ndi.label(candidates)
    labels = segmentation.watershed(-distance, markers, mask=candidates) if candidates.any() else np.zeros(candidates.shape, dtype=np.int32)
    labels = labels.astype(np.int32, copy=False)

    props = measure.regionprops_table(
        labels,
        intensity_image=hematoxylin,
        properties=("label", "area", "perimeter", "eccentricity", "solidity", "equivalent_diameter_area", "centroid", "mean_intensity"),
    )
    measurement_columns = [
        "object_id", "area_pixels", "perimeter_pixels", "eccentricity", "solidity",
        "equivalent_diameter_pixels", "centroid_row", "centroid_column",
        "mean_hematoxylin_intensity",
    ]
    measurements = pd.DataFrame(props)
    if not measurements.empty:
        measurements = measurements.rename(columns={
            "label": "object_id",
            "area": "area_pixels",
            "perimeter": "perimeter_pixels",
            "equivalent_diameter_area": "equivalent_diameter_pixels",
            "centroid-0": "centroid_row",
            "centroid-1": "centroid_column",
            "mean_intensity": "mean_hematoxylin_intensity",
        })
    measurements = measurements.reindex(columns=measurement_columns)
    overlay = color.label2rgb(labels, image=rgb_float, bg_label=0, alpha=0.45, kind="overlay")
    return {
        "hematoxylin": hematoxylin,
        "hematoxylin_threshold": threshold,
        "nuclei_candidate_mask": candidates,
        "nuclei_distance": distance,
        "nuclei_markers": markers,
        "nuclei_labels": labels,
        "nuclei_overlay": np.clip(overlay * 255.0, 0, 255).astype(np.uint8),
        "nuclei_measurements": measurements,
        "candidate_object_count": int(np.unique(labels).size - (1 if labels.size and np.any(labels == 0) else 0)),
    }


def basic_metrics(rgb: np.ndarray, gray: np.ndarray, tissue: np.ndarray, candidate_count: int) -> dict[str, Any]:
    rgb = _require_rgb(rgb)
    gray = _require_gray(gray)
    if gray.shape != rgb.shape[:2] or np.asarray(tissue).shape != rgb.shape[:2]:
        raise ValueError("RGB image, grayscale image, and tissue mask dimensions must match.")
    if not gray.size:
        raise ValueError("Cannot calculate metrics for an empty image.")
    hsv = color.rgb2hsv(util.img_as_float(to_uint8(rgb)))
    return {
        "width_px": int(rgb.shape[1]),
        "height_px": int(rgb.shape[0]),
        "mean_gray_intensity": round(float(np.mean(gray)), 3),
        "std_gray_intensity": round(float(np.std(gray)), 3),
        "gray_intensity_min": int(np.min(gray)),
        "gray_intensity_max": int(np.max(gray)),
        "mean_saturation": round(float(np.mean(hsv[:, :, 1])), 5),
        "tissue_coverage_percent": round(100 * float(np.mean(tissue)), 3),
        "candidate_nuclei_or_objects": int(candidate_count),
        "note": "Candidate object count is heuristic and requires visual validation; it is not a diagnostic measurement.",
    }


def image_to_png_bytes(image: np.ndarray) -> bytes:
    arr = np.asarray(image)
    if arr.ndim == 2:
        save_arr = to_uint8(arr)
        pil = Image.fromarray(save_arr, mode="L")
    elif arr.ndim == 3 and arr.shape[2] >= 3:
        save_arr = to_uint8(arr[:, :, :3])
        pil = Image.fromarray(save_arr, mode="RGB")
    else:
        save_arr = to_uint8(arr)
        pil = Image.fromarray(save_arr)
    stream = BytesIO()
    pil.save(stream, format="PNG", optimize=True)
    return stream.getvalue()


def build_export_zip(
    filename: str,
    original_rgb: np.ndarray,
    outputs: dict[str, Any],
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    settings: dict[str, Any],
) -> bytes:
    """Build a ZIP archive containing derived PNGs, measurements and metadata."""
    buffer = BytesIO()
    stem = Path(filename.replace("\\", "/")).stem or "histopathology_image"
    # Avoid path separators and control characters in uploaded filenames.
    stem = "".join(char if char.isalnum() or char in "-_." else "_" for char in stem).strip("._")
    stem = stem or "histopathology_image"
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{stem}/original.png", image_to_png_bytes(original_rgb))
        for name, value in outputs.items():
            if isinstance(value, np.ndarray) and value.ndim in (2, 3):
                if name == "nuclei_labels":
                    label_buffer = BytesIO()
                    np.save(label_buffer, value.astype(np.int32, copy=False))
                    archive.writestr(f"{stem}/{name}.npy", label_buffer.getvalue())
                elif name not in {"hematoxylin"}:
                    archive.writestr(f"{stem}/{name}.png", image_to_png_bytes(value))
                elif name == "hematoxylin":
                    archive.writestr(f"{stem}/{name}_display.png", image_to_png_bytes(scale_for_display(value)))
        measurements = outputs.get("nuclei_measurements")
        if isinstance(measurements, pd.DataFrame):
            archive.writestr(f"{stem}/object_measurements.csv", measurements.to_csv(index=False).encode("utf-8"))
        archive.writestr(f"{stem}/summary_metrics.csv", pd.DataFrame([metrics]).to_csv(index=False).encode("utf-8"))
        report = {
            "filename": filename,
            "metadata": metadata,
            "metrics": metrics,
            "settings": settings,
            "limitations": [
                "Tissue masking is a color-thresholding heuristic and may miss tissue or include background.",
                "Hematoxylin-based candidate nuclei segmentation and watershed labels are heuristic, not validated detections.",
                "No machine-learning model or medical diagnosis is performed.",
                "Area measurements are in pixels because physical pixel calibration was not supplied.",
            ],
        }
        archive.writestr(f"{stem}/analysis_report.json", json.dumps(report, indent=2, default=str).encode("utf-8"))
    return buffer.getvalue()
