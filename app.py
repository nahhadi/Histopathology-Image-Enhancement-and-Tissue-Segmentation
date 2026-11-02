from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from skimage import color

from histopathology.pipeline import (
    basic_metrics,
    build_export_zip,
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


st.set_page_config(
    page_title="HistoScope | Classical DIP",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.8rem; padding-bottom: 2.4rem; max-width: 1500px;}
      .hero {padding: 1.35rem 1.5rem; border-radius: 18px; border: 1px solid rgba(128,128,128,.24);
             background: linear-gradient(115deg, rgba(67,97,130,.16), rgba(65,120,106,.08)); margin-bottom: 1.1rem;}
      .eyebrow {text-transform: uppercase; letter-spacing: .14em; font-size: .74rem; opacity: .72; font-weight: 700;}
      .hero h1 {margin: .2rem 0 .25rem 0; font-size: 2.1rem;}
      .hero p {margin: 0; opacity: .82; max-width: 850px;}
      .soft-note {padding: .8rem 1rem; border-radius: 12px; border: 1px solid rgba(128,128,128,.25); font-size: .92rem;}
      div[data-testid="stMetric"] {background: rgba(128,128,128,.07); padding: .7rem .85rem; border-radius: 12px;}
      .small-muted {font-size: .86rem; opacity: .74;}
    </style>
    <div class="hero">
      <div class="eyebrow">Classical digital image processing · No ML</div>
      <h1>HistoScope</h1>
      <p>Explore histopathology images with color analysis, enhancement, spatial and frequency filtering,
      edge detection, morphological tissue masking and heuristic nuclei-candidate analysis.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="soft-note">🔎 <b>Research/education tool:</b> outputs are algorithmic visualizations, not medical diagnoses. '
    'Tissue and nuclei masks are heuristic and should be checked against the original image.</div>',
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_load(data: bytes, max_dimension: int):
    return load_image(data, max_dimension)


@st.cache_data(show_spinner=False, max_entries=8)
def cached_analysis(data: bytes, max_dimension: int, clahe_clip: float, gamma: float,
                    median_kernel: int, gaussian_sigma: float, analysis_source: str,
                    edge_method: str, edge_percentile: float, frequency_type: str,
                    frequency_mode: str, cutoff: float, butterworth_order: int,
                    min_saturation: float, morphology_radius: int, min_object_area: int,
                    nuclei_min_area: int, peak_distance: int):
    rgb, metadata = load_image(data, max_dimension)
    gray = grayscale(rgb)
    contrast = contrast_outputs(gray, clahe_clip=clahe_clip, gamma=gamma)
    source_lookup = {
        "Original grayscale": contrast["grayscale"],
        "Histogram equalization": contrast["histogram_equalized"],
        "CLAHE": contrast["clahe"],
        "Contrast stretch": contrast["contrast_stretched"],
        "Gamma corrected": contrast["gamma_corrected"],
    }
    analysis_gray = source_lookup[analysis_source]
    filters_result = spatial_filters(analysis_gray, median_kernel, gaussian_sigma)
    edges = edge_outputs(analysis_gray, edge_method, edge_percentile)
    frequency = frequency_filter(analysis_gray, frequency_type, frequency_mode, cutoff, butterworth_order)
    tissue = tissue_mask(rgb, min_saturation, morphology_radius, min_object_area)
    nuclei = nuclei_candidates(rgb, tissue["tissue_mask"], nuclei_min_area, peak_distance)

    boundary_overlay = rgb.copy()
    boundary = np.asarray(tissue["tissue_boundary"], dtype=bool)
    boundary_overlay[boundary] = np.array([255, 45, 68], dtype=np.uint8)

    outputs = {}
    outputs.update(contrast)
    outputs.update(filters_result)
    outputs.update(edges)
    outputs.update(frequency)
    outputs.update(tissue)
    outputs.update(nuclei)
    outputs["analysis_input"] = analysis_gray
    outputs["tissue_boundary_overlay"] = boundary_overlay
    metrics = basic_metrics(rgb, gray, tissue["tissue_mask"], nuclei["candidate_object_count"])
    metrics.update({
        "edge_detector": edge_method,
        "analysis_source": analysis_source,
        "tissue_saturation_threshold": round(float(tissue["tissue_threshold"]), 5),
        "hematoxylin_threshold": round(float(nuclei["hematoxylin_threshold"]), 5),
    })
    settings = {
        "max_dimension": max_dimension,
        "clahe_clip_limit": clahe_clip,
        "gamma": gamma,
        "median_kernel": median_kernel,
        "gaussian_sigma": gaussian_sigma,
        "analysis_source": analysis_source,
        "edge_method": edge_method,
        "edge_threshold_percentile": edge_percentile,
        "frequency_filter": frequency_type,
        "frequency_mode": frequency_mode,
        "frequency_cutoff": cutoff,
        "butterworth_order": butterworth_order,
        "minimum_saturation": min_saturation,
        "morphology_radius": morphology_radius,
        "minimum_tissue_object_area": min_object_area,
        "minimum_nuclei_candidate_area": nuclei_min_area,
        "watershed_peak_distance": peak_distance,
    }
    return rgb, metadata, outputs, metrics, settings


with st.sidebar:
    st.header("Analysis controls")
    st.caption("Changes update the displayed analysis. Large images are resized for responsiveness.")
    max_dimension = st.select_slider("Maximum analysis dimension", options=[512, 768, 1024, 1600, 2048, 3072], value=1600)
    st.divider()
    st.subheader("Enhancement")
    clahe_clip = st.slider("CLAHE clip limit", 0.5, 5.0, 2.0, 0.5)
    gamma = st.slider("Gamma correction", 0.4, 2.0, 0.9, 0.1)
    median_kernel = st.select_slider("Median filter kernel", options=[3, 5, 7], value=3)
    gaussian_sigma = st.slider("Gaussian sigma", 0.2, 3.0, 1.0, 0.2)
    analysis_source = st.selectbox(
        "Image used for edges/frequency analysis",
        ["CLAHE", "Original grayscale", "Histogram equalization", "Contrast stretch", "Gamma corrected"],
        index=0,
    )
    st.divider()
    st.subheader("Edge detection")
    edge_method = st.selectbox("Operator", ["All three", "Roberts", "Prewitt", "Sobel"], index=0)
    edge_percentile = st.slider("Adaptive edge threshold percentile", 50.0, 99.0, 75.0, 1.0)
    st.divider()
    st.subheader("Frequency domain")
    frequency_type = st.selectbox("Filter type", ["Gaussian", "Ideal", "Butterworth"], index=0)
    frequency_mode = st.selectbox("Filter mode", ["Low-pass", "High-pass"], index=0)
    cutoff = st.slider("Normalized cutoff frequency", 0.02, 0.30, 0.08, 0.01)
    butterworth_order = st.slider("Butterworth order", 1, 6, 2)
    st.divider()
    st.subheader("Segmentation")
    min_saturation = st.slider("Minimum saturation threshold", 0.00, 0.40, 0.05, 0.01)
    morphology_radius = st.slider("Morphological disk radius", 1, 10, 3)
    min_object_area = st.slider("Minimum tissue component area (px)", 10, 1000, 80, 10)
    nuclei_min_area = st.slider("Minimum candidate nuclei area (px)", 3, 200, 12, 1)
    peak_distance = st.slider("Watershed peak separation (px)", 1, 20, 5, 1)

st.markdown("### 1 · Upload your images")
uploads = st.file_uploader(
    "Choose one or more histopathology images",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
    accept_multiple_files=True,
    help="Supported formats: PNG, JPEG, BMP and TIFF. For multi-page TIFF, the first page is used.",
)

if not uploads:
    col_a, col_b = st.columns([1.25, 1])
    with col_a:
        st.subheader("What this app does")
        st.markdown(
            """
            - RGB and HSV channel analysis
            - Histogram equalization, CLAHE, gamma and contrast stretching
            - Median and Gaussian spatial filtering
            - Roberts, Prewitt and Sobel edge maps
            - Ideal, Butterworth and Gaussian frequency filters
            - HSV-based tissue/background mask with morphology
            - Hematoxylin-based candidate nuclei mask and watershed labels
            - Object measurements, CSV and ZIP export
            """
        )
    with col_b:
        st.subheader("Getting started")
        st.markdown("Upload an H&E-stained image using the uploader above. When run on your own computer, processing occurs on that computer. If deployed to a hosted service, uploads are processed on that host. No third-party AI service is called.")
        st.info("For the most meaningful results, use representative, well-focused images and visually inspect every mask.")
    st.stop()

labels = [f"{index + 1}. {uploaded.name}" for index, uploaded in enumerate(uploads)]
selected_label = st.selectbox("Image to inspect", labels, index=0)
selected_index = labels.index(selected_label)
selected_upload = uploads[selected_index]
file_bytes = selected_upload.getvalue()

try:
    with st.spinner("Processing image with classical DIP algorithms…"):
        rgb, metadata, outputs, metrics, settings = cached_analysis(
            file_bytes, max_dimension, clahe_clip, gamma, median_kernel, gaussian_sigma,
            analysis_source, edge_method, edge_percentile, frequency_type, frequency_mode,
            cutoff, butterworth_order, min_saturation, morphology_radius, min_object_area,
            nuclei_min_area, peak_distance,
        )
except Exception as exc:
    st.error(f"Could not process this image: {exc}")
    st.exception(exc)
    st.stop()

st.markdown(f"### 2 · Results for `{selected_upload.name}`")
if metadata["downsampled"]:
    st.caption(
        f"Analysis resized from {metadata['original_width']}×{metadata['original_height']} to "
        f"{metadata['analysis_width']}×{metadata['analysis_height']} pixels for responsiveness. "
        "Measurements are in analysis-image pixels."
    )
else:
    st.caption(f"Analysis resolution: {metadata['analysis_width']}×{metadata['analysis_height']} pixels.")

metric_cols = st.columns(4)
metric_cols[0].metric("Image width", f"{metrics['width_px']:,} px")
metric_cols[1].metric("Image height", f"{metrics['height_px']:,} px")
metric_cols[2].metric("Estimated tissue area", f"{metrics['tissue_coverage_percent']:.2f}%")
metric_cols[3].metric("Candidate objects", f"{metrics['candidate_nuclei_or_objects']:,}")

with st.expander("How to interpret these metrics", expanded=False):
    st.write("Tissue coverage is the percentage of pixels in the heuristic tissue mask. Candidate-object count comes from hematoxylin thresholding and watershed; it is not a validated nuclei count. No diagnosis is performed.")

tabs = st.tabs(["Overview", "Enhancement", "Color analysis", "Filters & edges", "Frequency domain", "Segmentation", "Measurements & export"])

with tabs[0]:
    left, right = st.columns(2)
    with left:
        st.image(rgb, caption="Selected image (analysis resolution)", use_container_width=True)
    with right:
        st.image(outputs["grayscale"], caption="Grayscale representation", use_container_width=True)
        st.write({
            "File format": metadata["format"],
            "Original dimensions": f"{metadata['original_width']} × {metadata['original_height']}",
            "Analysis dimensions": f"{metadata['analysis_width']} × {metadata['analysis_height']}",
            "Mean grayscale intensity": metrics["mean_gray_intensity"],
            "Grayscale standard deviation": metrics["std_gray_intensity"],
        })
    st.subheader("Grayscale histogram")
    histogram, _ = np.histogram(outputs["grayscale"], bins=256, range=(0, 256))
    hist_frame = pd.DataFrame({"Intensity": np.arange(256), "Pixel count": histogram}).set_index("Intensity")
    st.line_chart(hist_frame)

with tabs[1]:
    st.subheader("Contrast enhancement")
    enhancement_items = [
        ("grayscale", "Original grayscale"),
        ("histogram_equalized", "Histogram equalization"),
        ("clahe", "CLAHE (local contrast)"),
        ("contrast_stretched", "Contrast stretching"),
        ("gamma_corrected", f"Gamma correction ({gamma:.1f})"),
    ]
    grid = st.columns(3)
    for idx, (key, title) in enumerate(enhancement_items):
        with grid[idx % 3]:
            st.image(outputs[key], caption=title, use_container_width=True)
    st.caption("Histogram equalization and CLAHE alter intensity distributions; inspect whether they reveal detail without exaggerating staining variation or noise.")

with tabs[2]:
    hsv = color.rgb2hsv(rgb.astype(np.float32) / 255.0)
    color_cols = st.columns(4)
    color_cols[0].image(rgb, caption="RGB original", use_container_width=True)
    color_cols[1].image(rgb[:, :, 0], caption="Red channel", use_container_width=True)
    color_cols[2].image(rgb[:, :, 1], caption="Green channel", use_container_width=True)
    color_cols[3].image(rgb[:, :, 2], caption="Blue channel", use_container_width=True)
    hsv_cols = st.columns(3)
    hsv_cols[0].image(hsv[:, :, 0], caption="Hue", use_container_width=True)
    hsv_cols[1].image(hsv[:, :, 1], caption="Saturation", use_container_width=True)
    hsv_cols[2].image(hsv[:, :, 2], caption="Value / brightness", use_container_width=True)
    st.caption("HSV can make color-based separation more convenient, but saturation alone is not a reliable tissue classifier for every stain and scanner.")

with tabs[3]:
    st.subheader("Spatial filtering")
    filter_cols = st.columns(3)
    filter_cols[0].image(outputs["analysis_input"], caption=f"Processing input: {analysis_source}", use_container_width=True)
    filter_cols[1].image(outputs["median_filtered"], caption=f"Median filter ({median_kernel}×{median_kernel})", use_container_width=True)
    filter_cols[2].image(outputs["gaussian_filtered"], caption=f"Gaussian filter (σ={gaussian_sigma:.1f})", use_container_width=True)
    st.subheader("Edge detection")
    edge_keys = [("roberts", "Roberts"), ("prewitt", "Prewitt"), ("sobel", "Sobel")]
    edge_cols = st.columns(3)
    active_count = 0
    for prefix, title in edge_keys:
        mag_key = f"{prefix}_magnitude"
        mask_key = f"{prefix}_edges"
        if mag_key in outputs and mask_key in outputs:
            with edge_cols[active_count % 3]:
                st.image(outputs[mag_key], caption=f"{title} gradient magnitude", use_container_width=True)
                st.image(to_uint8(outputs[mask_key]), caption=f"{title} thresholded edge map", use_container_width=True)
            active_count += 1
    st.caption("The edge threshold is selected from the image's gradient distribution. Edge density is not a measure of segmentation accuracy.")

with tabs[4]:
    st.subheader(f"Fourier-domain {frequency_mode.lower()} filtering")
    freq_left, freq_right = st.columns(2)
    freq_left.image(outputs["dft_spectrum"], caption="Centered DFT magnitude spectrum (log display)", use_container_width=True)
    freq_right.image(outputs["frequency_filtered"], caption=f"{frequency_type} {frequency_mode.lower()} output", use_container_width=True)
    st.write({"Filter": frequency_type, "Mode": frequency_mode, "Normalized cutoff": cutoff, "Butterworth order": butterworth_order})
    st.caption("Low-pass filters suppress high spatial frequencies and may blur fine details. High-pass results emphasize rapid intensity changes and can look dark or noise-sensitive.")

with tabs[5]:
    st.subheader("Tissue/background mask")
    tcols = st.columns(3)
    tcols[0].image(outputs["tissue_mask_raw"], caption="Raw saturation threshold", use_container_width=True)
    tcols[1].image(outputs["tissue_mask_opened"], caption="After opening", use_container_width=True)
    tcols[2].image(outputs["tissue_mask"], caption="Cleaned tissue mask", use_container_width=True)
    st.image(outputs["tissue_boundary_overlay"], caption="Tissue-mask boundary overlay (red)", use_container_width=True)
    st.write(f"Otsu-derived saturation threshold with configured floor: **{outputs['tissue_threshold']:.4f}**")
    st.divider()
    st.subheader("Hematoxylin-based candidate nuclei / object analysis")
    ncols = st.columns(3)
    ncols[0].image(outputs["hematoxylin"], caption="Hematoxylin stain component (display-scaled)", use_container_width=True)
    ncols[1].image(outputs["nuclei_candidate_mask"], caption="Thresholded candidate mask", use_container_width=True)
    ncols[2].image(outputs["nuclei_overlay"], caption="Watershed labels overlaid on image", use_container_width=True)
    st.write(f"Hematoxylin threshold: **{outputs['hematoxylin_threshold']:.5f}** · Candidate regions: **{outputs['candidate_object_count']}**")
    st.warning("This stain-thresholding and watershed output is a heuristic candidate segmentation. It may merge, split, miss or falsely identify structures; it is not a validated nuclei detector.")

with tabs[6]:
    st.subheader("Image-level measurements")
    summary = pd.DataFrame([metrics]).drop(columns=["note"], errors="ignore")
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.subheader("Detected candidate-object measurements")
    measurements = outputs["nuclei_measurements"]
    if isinstance(measurements, pd.DataFrame) and not measurements.empty:
        st.dataframe(measurements, use_container_width=True, hide_index=True, height=320)
        st.download_button(
            "Download object measurements CSV",
            data=measurements.to_csv(index=False).encode("utf-8"),
            file_name=f"{Path(selected_upload.name).stem}_object_measurements.csv",
            mime="text/csv",
        )
    else:
        st.info("No candidate objects passed the current threshold and size settings. Try adjusting the segmentation controls, then inspect the mask visually.")
    zip_bytes = build_export_zip(selected_upload.name, rgb, outputs, metadata, metrics, settings)
    st.download_button(
        "Download complete analysis bundle (.zip)",
        data=zip_bytes,
        file_name=f"{Path(selected_upload.name).stem}_histoscope_results.zip",
        mime="application/zip",
        type="primary",
    )
    st.caption("The ZIP contains PNG outputs, object measurements (when available), summary metrics and a JSON report of settings and limitations.")

if len(uploads) > 1:
    with st.expander(f"Batch summary for all {len(uploads)} uploaded images"):
        st.write("Build a quick batch summary using the same segmentation parameters. Results are heuristic and should be visually inspected.")
        if st.button("Run batch summary", type="secondary"):
            rows = []
            progress = st.progress(0)
            for index, uploaded in enumerate(uploads):
                try:
                    batch_rgb, batch_meta = cached_load(uploaded.getvalue(), max_dimension)
                    batch_gray = grayscale(batch_rgb)
                    batch_tissue = tissue_mask(batch_rgb, min_saturation, morphology_radius, min_object_area)
                    batch_nuclei = nuclei_candidates(batch_rgb, batch_tissue["tissue_mask"], nuclei_min_area, peak_distance)
                    row = basic_metrics(batch_rgb, batch_gray, batch_tissue["tissue_mask"], batch_nuclei["candidate_object_count"])
                    row["filename"] = uploaded.name
                    row["original_width"] = batch_meta["original_width"]
                    row["original_height"] = batch_meta["original_height"]
                    rows.append(row)
                except Exception as exc:
                    rows.append({"filename": uploaded.name, "error": str(exc)})
                progress.progress((index + 1) / len(uploads))
            batch_df = pd.DataFrame(rows)
            st.dataframe(batch_df, use_container_width=True, hide_index=True)
            st.download_button(
                "Download batch summary CSV",
                data=batch_df.to_csv(index=False).encode("utf-8"),
                file_name="histoscope_batch_summary.csv",
                mime="text/csv",
            )

st.divider()
st.markdown(
    "<div class='small-muted'>HistoScope is an educational research prototype built from conventional image-processing methods. "
    "It is not intended for clinical diagnosis or treatment decisions.</div>",
    unsafe_allow_html=True,
)
