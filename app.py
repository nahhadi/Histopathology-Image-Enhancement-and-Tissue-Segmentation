from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from skimage import color

from histopathology.pipeline import (
    basic_metrics,
    build_export_zip,
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


logger = logging.getLogger(__name__)


st.set_page_config(
    page_title="HistoScope | Research Workstation",
    page_icon="H",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root {color-scheme:dark;}
      html, body, [data-testid="stAppViewContainer"] {font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;}
      .stApp {background:#0B0F17;color:#E8EDF5;}
      [data-testid="stHeader"] {background:rgba(11,15,23,.94);}
      .block-container {padding:1rem clamp(.8rem,3vw,2.4rem) 2rem;max-width:1660px;}
      [data-testid="stSidebar"] {background:#101722;border-right:1px solid #263244;}
      [data-testid="stSidebar"] > div {padding-top:.65rem;}
      [data-testid="stSidebar"] [data-testid="stRadio"] label {padding:.28rem .48rem;border:1px solid transparent;border-radius:8px;margin:.05rem 0;}
      [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {background:#151D29;}
      [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {background:#1A2638;border-color:#31425d;color:#E8EDF5;}
      [data-testid="stSidebar"] [data-testid="stExpander"] {border-color:#263244;border-radius:8px;margin:.28rem 0;}
      [data-testid="stSidebar"] [data-testid="stExpander"] summary {font-size:.88rem;}
      .brand {font-size:1.02rem;font-weight:700;letter-spacing:-.02em;padding:.25rem 0 .05rem;}
      .brand-sub {color:#929EAF;font-size:.68rem;letter-spacing:.08em;margin-bottom:.65rem;}
      .topbar {display:flex;align-items:center;justify-content:space-between;gap:1rem;padding:.3rem 0 .65rem;margin-bottom:.8rem;border-bottom:1px solid #263244;}
      .topbar-title {font-size:.78rem;color:#929EAF;letter-spacing:.07em;text-transform:uppercase;}
      .topbar-file {font-size:.9rem;color:#E8EDF5;font-weight:620;overflow-wrap:anywhere;}
      .page-title {font-size:1.48rem;line-height:1.25;font-weight:660;letter-spacing:-.025em;margin:0 0 .18rem;color:#E8EDF5;}
      .page-intro {color:#929EAF;margin:0 0 .85rem;font-size:.9rem;}
      .section-title {font-size:1rem;line-height:1.35;font-weight:640;margin:.15rem 0 .45rem;color:#E8EDF5;}
      .meta-label {color:#929EAF!important;font-size:.78rem!important;}
      .small-muted {font-size:.78rem;color:#929EAF;}
      div[data-testid="stMetric"] {height:100%;min-height:96px;background:#151D29;border:1px solid #263244;padding:.7rem .8rem;border-radius:9px;}
      div[data-testid="stMetricLabel"] {color:#929EAF;font-size:.78rem;}
      div[data-testid="stMetricValue"] {font-size:1.24rem;}
      div[data-testid="stImage"] {display:flex;justify-content:center;}
      div[data-testid="stImage"] img {border-radius:7px;max-height:68vh;object-fit:contain;}
      [data-testid="stDataFrame"] {border:1px solid #263244;border-radius:8px;overflow:hidden;}
      div.stButton > button, div.stDownloadButton > button {border-radius:8px;font-weight:580;}
      div.stButton > button[kind="primary"], div.stDownloadButton > button[kind="primary"] {background:#7298FF;border-color:#7298FF;color:#0B0F17;}
      [data-testid="stFileUploader"] section {padding:.45rem .65rem;border:1px dashed #34445c;border-radius:9px;background:#111a27;}
      [data-testid="stFileUploader"] small {color:#929EAF;}
      @media (max-width:900px) {
        .block-container {padding:.8rem .7rem 1.5rem;}
        .page-title {font-size:1.32rem;}
        div[data-testid="stMetric"] {min-height:86px;padding:.6rem;}
        div[data-testid="stMetricValue"] {font-size:1.05rem;}
      }
      @media (max-width:600px) {
        .block-container {padding:.65rem .55rem 1.25rem;}
        .page-title {font-size:1.22rem;}
        .topbar {align-items:flex-start;flex-direction:column;gap:.2rem;}
        div[data-testid="stMetric"] {min-height:74px;}
      }
    </style>
    """,
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


PAGES = [
    "Overview", "Image Workspace", "Enhancement", "Spatial Analysis",
    "Frequency Domain", "Tissue Segmentation", "Nuclei Analysis",
    "Measurements", "Reports and Exports",
]
PAGE_ICONS = {
    "Overview": "⌂", "Image Workspace": "▧", "Enhancement": "◐",
    "Spatial Analysis": "⌁", "Frequency Domain": "◌", "Tissue Segmentation": "◧",
    "Nuclei Analysis": "⊙", "Measurements": "▤", "Reports and Exports": "⇩",
}
PAGE_DESCRIPTIONS = {
    "Overview": "A concise view of the current image and analysis status.",
    "Image Workspace": "Inspect the image, compare views, and review color channels.",
    "Enhancement": "Compare global and local contrast transformations.",
    "Spatial Analysis": "Review spatial filters and classical edge operators.",
    "Frequency Domain": "Inspect the centered DFT and frequency-filter response.",
    "Tissue Segmentation": "Review saturation thresholding and each morphology stage.",
    "Nuclei Analysis": "Inspect hematoxylin candidates, distance values and watershed labels.",
    "Measurements": "Review image-level coverage and candidate-object measurements.",
    "Reports and Exports": "Review settings and download CSV or complete analysis files.",
}

with st.sidebar:
    st.markdown('<div class="brand">HistoScope</div><div class="brand-sub">CLASSICAL DIP WORKSTATION</div>', unsafe_allow_html=True)
    active_page = st.radio(
        "Navigation", PAGES, format_func=lambda page: f"{PAGE_ICONS[page]}   {page}",
        label_visibility="collapsed", key="active_page",
    )
    st.divider()
    st.caption("Analysis controls")
    with st.expander("Image processing", expanded=False):
        max_dimension = st.select_slider("Maximum analysis dimension", options=[512, 768, 1024, 1600, 2048, 3072], value=1600)
    with st.expander("Enhancement and filtering", expanded=False):
        clahe_clip = st.slider("CLAHE clip limit", 0.5, 5.0, 2.0, 0.5)
        gamma = st.slider("Gamma correction", 0.4, 2.0, 0.9, 0.1)
        median_kernel = st.select_slider("Median filter kernel", options=[3, 5, 7], value=3)
        gaussian_sigma = st.slider("Gaussian sigma", 0.2, 3.0, 1.0, 0.2)
        analysis_source = st.selectbox(
            "Input for edges and frequency analysis",
            ["CLAHE", "Original grayscale", "Histogram equalization", "Contrast stretch", "Gamma corrected"],
            index=0,
        )
        edge_method = st.selectbox("Edge operator", ["All three", "Roberts", "Prewitt", "Sobel"], index=0)
        edge_percentile = st.slider("Adaptive edge threshold percentile", 50.0, 99.0, 75.0, 1.0)
    with st.expander("Frequency domain", expanded=False):
        frequency_type = st.selectbox("Filter type", ["Gaussian", "Ideal", "Butterworth"], index=0)
        frequency_mode = st.selectbox("Filter mode", ["Low-pass", "High-pass"], index=0)
        cutoff = st.slider("Normalized cutoff frequency", 0.02, 0.30, 0.08, 0.01)
        butterworth_order = st.slider("Butterworth order", 1, 6, 2)
    with st.expander("Segmentation", expanded=False):
        min_saturation = st.slider("Minimum saturation threshold", 0.00, 0.40, 0.05, 0.01)
        morphology_radius = st.slider("Morphological disk radius", 1, 10, 3)
        min_object_area = st.slider("Minimum tissue component area (px)", 10, 1000, 80, 10)
        nuclei_min_area = st.slider("Minimum candidate nuclei area (px)", 3, 200, 12, 1)
        peak_distance = st.slider("Watershed peak separation (px)", 1, 20, 5, 1)
    st.divider()
    st.caption("Research and education only. Outputs are heuristic and are not diagnostic.")

st.markdown(
    f'<div class="topbar"><div><div class="topbar-title">HistoScope / {active_page}</div>'
    '<div class="topbar-file">Classical image processing workspace</div></div>'
    '<div class="small-muted">Local analysis · Classical DIP</div></div>',
    unsafe_allow_html=True,
)
st.markdown(f'<div class="page-title">{active_page}</div><div class="page-intro">{PAGE_DESCRIPTIONS[active_page]}</div>', unsafe_allow_html=True)

with st.expander("Add or change images", expanded=not bool(st.session_state.get("image_upload"))):
    uploads = st.file_uploader(
        "Open image files",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        accept_multiple_files=True,
        help="PNG, JPEG, BMP and TIFF are supported. For multi-page TIFF, the first page is used.",
        key="image_upload",
    )
if not uploads:
    st.info("Upload one or more histopathology images to begin. Processing stays within this application instance.")
    st.markdown("#### Available analysis modules")
    intro_cols = st.columns(3)
    intro_cols[0].markdown("**Enhancement and filtering**  \nEqualization, CLAHE, gamma, spatial filters and edge operators.")
    intro_cols[1].markdown("**Tissue and nuclei candidates**  \nHSV-based tissue masks and heuristic hematoxylin watershed analysis.")
    intro_cols[2].markdown("**Measurements and reports**  \nCoverage, candidate region measurements, CSV and ZIP exports.")
    st.stop()

labels = [f"{index + 1:02d} - {uploaded.name}" for index, uploaded in enumerate(uploads)]
select_col, selection_help = st.columns([1.25, 2.75], vertical_alignment="bottom")
selected_label = select_col.selectbox("Current image", labels, index=0, key="selected_image")
selected_index = labels.index(selected_label)
selected_upload = uploads[selected_index]
selection_help.caption(f"{len(uploads)} image{'s' if len(uploads) != 1 else ''} in this workspace · Choose an image to update all analysis views.")
file_bytes = selected_upload.getvalue()

try:
    with st.spinner("Analyzing image with classical DIP algorithms..."):
        rgb, metadata, outputs, metrics, settings = cached_analysis(
            file_bytes, max_dimension, clahe_clip, gamma, median_kernel, gaussian_sigma,
            analysis_source, edge_method, edge_percentile, frequency_type, frequency_mode,
            cutoff, butterworth_order, min_saturation, morphology_radius, min_object_area,
            nuclei_min_area, peak_distance,
        )
except Exception:
    logger.exception("Image processing failed for %r", selected_upload.name)
    st.error(f"Could not process {selected_upload.name!r}. Check that the file is a valid supported image. Details are available in the server log.")
    st.stop()

def navigate_to_reports() -> None:
    st.session_state["active_page"] = "Reports and Exports"


file_header, report_action = st.columns([5, 1.25], vertical_alignment="center")
file_header.markdown(
    f'<div class="topbar-file">{selected_upload.name}</div>'
    f'<div class="small-muted">{metadata["analysis_width"]} × {metadata["analysis_height"]} px · {metadata["format"]} · Analysis ready</div>',
    unsafe_allow_html=True,
)
report_action.button("Reports & exports", use_container_width=True, on_click=navigate_to_reports)

if metadata["downsampled"]:
    st.caption(
        f"Analysis image: {metadata['analysis_width']} × {metadata['analysis_height']} px · "
        f"Original: {metadata['original_width']} × {metadata['original_height']} px · "
        "Measurements use analysis-image pixels."
    )
else:
    st.caption(f"Analysis resolution: {metadata['analysis_width']} × {metadata['analysis_height']} px · {metadata['format']}")

def section_title(text: str, caption: str | None = None) -> None:
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)
    if caption:
        st.caption(caption)


def metadata_rows(items: list[tuple[str, object]]) -> None:
    """Render image metadata as readable label/value rows rather than a dict."""
    for label, value in items:
        label_col, value_col = st.columns([1, 1.35], vertical_alignment="center")
        label_col.markdown(f'<span class="meta-label">{label}</span>', unsafe_allow_html=True)
        value_col.write(str(value))


if active_page == "Overview":
    st.markdown("#### Analysis snapshot")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Analysis dimensions", f"{metrics['width_px']:,} × {metrics['height_px']:,} px")
    metric_cols[1].metric("Tissue coverage", f"{metrics['tissue_coverage_percent']:.2f}%")
    metric_cols[2].metric("Candidate objects", f"{metrics['candidate_nuclei_or_objects']:,}")
    metric_cols[3].metric("Processing status", "Ready")
    left, right = st.columns([1.7, 1])
    with left:
        with st.container(border=True):
            st.image(rgb, caption=selected_upload.name, width="stretch")
    with right:
        with st.container(border=True):
            st.markdown("#### Image details")
            metadata_rows([
                ("File name", selected_upload.name),
                ("Format", metadata["format"]),
                ("Original dimensions", f"{metadata['original_width']:,} × {metadata['original_height']:,} px"),
                ("Analysis dimensions", f"{metadata['analysis_width']:,} × {metadata['analysis_height']:,} px"),
                ("Mean intensity", f"{metrics['mean_gray_intensity']:.2f} / 255"),
            ])
            st.caption(f"Tissue-mask method: {outputs['tissue_segmentation_strategy']}")
    st.caption("The analysis image preserves its aspect ratio. Select the image preview's expand control for fullscreen inspection.")
    section_title("Intensity distribution", "The histogram is calculated from the analysis-resolution grayscale image.")
    histogram, _ = np.histogram(outputs["grayscale"], bins=256, range=(0, 256))
    st.line_chart(pd.DataFrame({"Pixel count": histogram}, index=np.arange(256)), height=220)
    history = st.session_state.setdefault("analysis_history", [])
    if not history or history[-1] != selected_upload.name:
        history.append(selected_upload.name)
    with st.expander("Recent images in this session"):
        st.write(" · ".join(history[-5:]))

elif active_page == "Image Workspace":
    section_title("Image viewer", "Fit preserves the full image. Zoom and pan create a display-only crop; they do not alter analysis data. Use the image expand control for fullscreen.")
    left, right = st.columns([1.35, 1])
    with left:
        comparison_choice = st.selectbox(
            "Comparison view",
            ["Grayscale", "Histogram equalization", "CLAHE", "Contrast stretch", "Gamma corrected"],
            key="workspace_comparison",
        )
        comparison_key = {
            "Grayscale": "grayscale",
            "Histogram equalization": "histogram_equalized",
            "CLAHE": "clahe",
            "Contrast stretch": "contrast_stretched",
            "Gamma corrected": "gamma_corrected",
        }[comparison_choice]
        fit_col, zoom_col = st.columns([1, 3], vertical_alignment="bottom")
        if fit_col.button("Fit to view", use_container_width=True, key="workspace_fit"):
            st.session_state["workspace_zoom"] = 1.0
        zoom = zoom_col.slider("Zoom", 1.0, 4.0, 1.0, 0.25, key="workspace_zoom", help="1× fits the full image; higher values crop around the selected pan point.")
        if zoom > 1:
            center_row = st.slider("Vertical pan", 0, 100, 50, key="workspace_pan_y") / 100
            center_col = st.slider("Horizontal pan", 0, 100, 50, key="workspace_pan_x") / 100
            height, width = rgb.shape[:2]
            crop_h, crop_w = max(1, round(height / zoom)), max(1, round(width / zoom))
            row0 = min(max(0, round(center_row * height - crop_h / 2)), height - crop_h)
            col0 = min(max(0, round(center_col * width - crop_w / 2)), width - crop_w)
            view_rgb = rgb[row0:row0 + crop_h, col0:col0 + crop_w].copy()
            view_compare = outputs[comparison_key][row0:row0 + crop_h, col0:col0 + crop_w].copy()
        else:
            view_rgb, view_compare = rgb, outputs[comparison_key]
        show_cols = st.columns(2)
        show_cols[0].image(view_rgb, caption="Original RGB", width="stretch")
        show_cols[1].image(view_compare, caption=comparison_choice, width="stretch")
        st.caption(f"Original analysis dimensions: {rgb.shape[1]:,} × {rgb.shape[0]:,} px · Current view: {view_rgb.shape[1]:,} × {view_rgb.shape[0]:,} px")
    with right:
        section_title("RGB channels")
        rgb_cols = st.columns(3)
        for i, name in enumerate(("Red", "Green", "Blue")):
            rgb_cols[i].image(rgb[:, :, i], caption=name, width="stretch")
        section_title("HSV channels")
        hsv = color.rgb2hsv(rgb.astype(np.float32) / 255.0)
        hsv_cols = st.columns(3)
        for i, name in enumerate(("Hue", "Saturation", "Value")):
            hsv_cols[i].image(scale_for_display(hsv[:, :, i]), caption=name, width="stretch")
    st.caption("HSV saturation is a useful exploratory channel, but it is not a universal tissue classifier.")

elif active_page == "Enhancement":
    section_title("Intensity enhancement", "Controls for CLAHE and gamma are in the sidebar. Each preview is derived from the same grayscale input.")
    enhancement_items = [
        ("grayscale", "Original grayscale"),
        ("histogram_equalized", "Histogram equalization"),
        ("clahe", f"CLAHE · clip {clahe_clip:.1f}"),
        ("contrast_stretched", "Contrast stretching"),
        ("gamma_corrected", f"Gamma correction · γ {gamma:.1f}"),
    ]
    for pair in (enhancement_items[:2], enhancement_items[2:4], enhancement_items[4:]):
        cols = st.columns(len(pair))
        for col, (key, title) in zip(cols, pair):
            col.image(outputs[key], caption=title, width="stretch")
    st.caption("Equalization and CLAHE change intensity distributions. Inspect amplified detail for noise or staining variation.")

elif active_page == "Spatial Analysis":
    section_title("Spatial filtering", f"Edge analysis input: {analysis_source}. Median kernel: {median_kernel} × {median_kernel}; Gaussian σ: {gaussian_sigma:.1f}.")
    filter_cols = st.columns(3)
    filter_cols[0].image(outputs["analysis_input"], caption=f"Input · {analysis_source}", width="stretch")
    filter_cols[1].image(outputs["median_filtered"], caption=f"Median · {median_kernel} × {median_kernel}", width="stretch")
    filter_cols[2].image(outputs["gaussian_filtered"], caption=f"Gaussian · σ {gaussian_sigma:.1f}", width="stretch")
    section_title("Gradient operators", f"Operator: {edge_method} · adaptive threshold percentile: {edge_percentile:.0f}.")
    edge_keys = [("roberts", "Roberts"), ("prewitt", "Prewitt"), ("sobel", "Sobel")]
    edge_cols = st.columns(3)
    for col, (prefix, title) in zip(edge_cols, edge_keys):
        if f"{prefix}_magnitude" in outputs:
            col.image(outputs[f"{prefix}_magnitude"], caption=f"{title} gradient magnitude", width="stretch")
            show_mask_in_col = display_mask(outputs[f"{prefix}_edges"])
            col.image(show_mask_in_col, caption=f"{title} binary edge map", width="stretch")
    st.caption("Binary edge maps use white pixels for detected edges. Edge density is not a measure of segmentation quality.")

elif active_page == "Frequency Domain":
    section_title("Discrete Fourier transform", "The spectrum is centered with fftshift and log-scaled for display; filtering uses the numerical transform.")
    freq_cols = st.columns(3)
    freq_cols[0].image(outputs["grayscale"], caption="Original grayscale", width="stretch")
    freq_cols[1].image(outputs["dft_spectrum"], caption="Centered DFT magnitude spectrum", width="stretch")
    frequency_caption = (
        f"{frequency_type} high-pass response · display-scaled"
        if frequency_mode == "High-pass" else f"{frequency_type} low-pass output"
    )
    freq_cols[2].image(outputs["frequency_filtered"], caption=frequency_caption, width="stretch")
    st.write({"Filter type": frequency_type, "Mode": frequency_mode, "Normalized cutoff": cutoff, "Butterworth order": butterworth_order})
    st.caption("Low-pass filters suppress high spatial frequencies and may blur fine detail. High-pass responses emphasize rapid intensity changes and may amplify noise.")

elif active_page == "Tissue Segmentation":
    section_title("Tissue and background", "White is classified foreground; black is background. Coverage is reported for each Boolean mask stage.")
    source_cols = st.columns(2)
    source_cols[0].image(rgb, caption="Original RGB", width="stretch")
    source_cols[1].image(to_uint8(outputs["saturation"]), caption="HSV saturation channel", width="stretch")
    stage_items = [
        ("tissue_mask_raw", "Raw saturation threshold", "tissue_mask_raw_coverage_percent"),
        ("tissue_mask_opened", "After morphological opening", "tissue_mask_opened_coverage_percent"),
        ("tissue_mask_closed", "After morphological closing", "tissue_mask_closed_coverage_percent"),
        ("tissue_mask", "Final cleaned mask", "tissue_mask_coverage_percent"),
    ]
    stage_cols = st.columns(4)
    for col, (key, title, coverage_key) in zip(stage_cols, stage_items):
        col.image(display_mask(outputs[key]), caption=f"{title} · {outputs[coverage_key]:.2f}%", width="stretch")
    st.image(outputs["tissue_boundary_overlay"], caption="Tissue boundary overlay", width="stretch")
    st.write({
        "Otsu saturation threshold": round(float(outputs["tissue_otsu_threshold"]), 5),
        "Configured threshold floor": round(float(outputs["tissue_threshold_floor"]), 5),
        "Effective saturation threshold": round(float(outputs["tissue_threshold"]), 5),
        "Raw-mask perimeter coverage": round(float(outputs["tissue_mask_raw_border_coverage_percent"]), 2),
        "Active segmentation path": outputs["tissue_segmentation_strategy"],
        "Small-hole fill limit (pixels)": int(outputs["tissue_hole_fill_area_limit_pixels"]),
    })
    if outputs["tissue_segmentation_strategy"] != "HSV saturation":
        st.info("Most perimeter pixels passed the raw saturation threshold. Morphology therefore uses an Otsu RGB-distance mask from the estimated border background; compare it with the original.")
    if outputs["tissue_mask_coverage_percent"] == 0:
        st.warning("No tissue foreground remains after thresholding and morphology. Check the source image and processing controls.")
    st.caption("The tissue mask is an educational heuristic and may be incorrect for unusual backgrounds, weak staining, folds or tissue touching most image borders.")

elif active_page == "Nuclei Analysis":
    section_title("Hematoxylin and watershed", "Candidate segmentation is heuristic and is not a validated nuclei detector or clinical count.")
    h_cols = st.columns(3)
    h_cols[0].image(scale_for_display(outputs["hematoxylin"]), caption="Hematoxylin component · display-scaled", width="stretch")
    h_cols[1].image(display_mask(outputs["nuclei_candidate_mask"]), caption=f"Binary candidates · {100*np.mean(outputs['nuclei_candidate_mask']):.2f}% foreground", width="stretch")
    h_cols[2].image(scale_for_display(outputs["nuclei_distance"]), caption="Euclidean distance transform", width="stretch")
    w_cols = st.columns(3)
    marker_preview = color.label2rgb(outputs["nuclei_markers"], bg_label=0, bg_color=(0, 0, 0))
    label_preview = color.label2rgb(outputs["nuclei_labels"], bg_label=0, bg_color=(0, 0, 0))
    w_cols[0].image(np.clip(marker_preview * 255, 0, 255).astype(np.uint8), caption="Watershed markers", width="stretch")
    w_cols[1].image(np.clip(label_preview * 255, 0, 255).astype(np.uint8), caption="Final watershed labels", width="stretch")
    w_cols[2].image(outputs["nuclei_overlay"], caption="Segmentation overlay", width="stretch")
    st.write({"Hematoxylin threshold": round(float(outputs["hematoxylin_threshold"]), 5), "Candidate object count": int(outputs["candidate_object_count"])})
    st.warning("Thresholding and watershed may merge, split, miss or falsely include structures. Inspect every result against the source image.")

elif active_page == "Measurements":
    section_title("Image-level measurements")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Tissue coverage", f"{metrics['tissue_coverage_percent']:.2f}%")
    metric_cols[1].metric("Candidate objects", f"{metrics['candidate_nuclei_or_objects']:,}")
    metric_cols[2].metric("Mean intensity", f"{metrics['mean_gray_intensity']:.2f}")
    metric_cols[3].metric("Mean saturation", f"{metrics['mean_saturation']:.4f}")
    measurements = outputs["nuclei_measurements"]
    section_title("Candidate-object measurements", "Areas and lengths use analysis-image pixels; physical calibration is not available.")
    if not measurements.empty:
        st.dataframe(measurements, width="stretch", hide_index=True, height=360)
        area_summary = measurements.nlargest(30, "area_pixels").sort_values("area_pixels", ascending=True)
        st.bar_chart(area_summary.set_index("object_id")[["area_pixels"]], horizontal=True, height=360)
        st.download_button(
            "Download measurements CSV",
            data=measurements.to_csv(index=False).encode("utf-8"),
            file_name=f"{Path(selected_upload.name).stem}_object_measurements.csv",
            mime="text/csv",
        )
    else:
        st.info("No candidate objects passed the current threshold and size settings.")
    st.caption("Object measurements describe heuristic watershed regions, not validated cells or nuclei.")

elif active_page == "Reports and Exports":
    section_title("Analysis report", "The downloadable bundle includes image outputs, settings, summary metrics and object measurements.")
    report_cols = st.columns(2)
    with report_cols[0]:
        st.markdown("#### Image metadata")
        metadata_rows([
            ("File format", metadata["format"]),
            ("Original dimensions", f"{metadata['original_width']:,} × {metadata['original_height']:,} px"),
            ("Analysis dimensions", f"{metadata['analysis_width']:,} × {metadata['analysis_height']:,} px"),
            ("Downsampled for analysis", "Yes" if metadata["downsampled"] else "No"),
        ])
        st.markdown("#### Summary metrics")
        st.dataframe(pd.DataFrame([metrics]).drop(columns=["note"], errors="ignore"), width="stretch", hide_index=True)
    with report_cols[1]:
        st.markdown("#### Processing settings")
        metadata_rows([(key.replace("_", " ").capitalize(), value) for key, value in settings.items()])
        st.markdown("#### Measurements")
        measurements = outputs["nuclei_measurements"]
        st.write(f"{len(measurements)} candidate-object rows")
        if not measurements.empty:
            st.dataframe(measurements.head(12), width="stretch", hide_index=True)
    csv_bytes = measurements.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download measurements CSV",
        data=csv_bytes,
        file_name=f"{Path(selected_upload.name).stem}_object_measurements.csv",
        mime="text/csv",
    )
    zip_bytes = build_export_zip(selected_upload.name, rgb, outputs, metadata, metrics, settings)
    st.download_button(
        "Download complete analysis bundle (.zip)",
        data=zip_bytes,
        file_name=f"{Path(selected_upload.name).stem}_histoscope_results.zip",
        mime="application/zip",
        type="primary",
    )
    st.caption("The ZIP contains PNG outputs, measurements, summary metrics and a JSON record of settings and limitations.")

    if len(uploads) > 1:
        with st.expander(f"Batch analysis · {len(uploads)} images"):
            st.write("The batch report applies the same segmentation parameters to each image. Results are heuristic and should be checked individually.")
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
                    except Exception:
                        logger.exception("Batch processing failed for %r", uploaded.name)
                        rows.append({"filename": uploaded.name, "error": "Could not process this image; see server log."})
                    progress.progress((index + 1) / len(uploads))
                batch_df = pd.DataFrame(rows)
                st.dataframe(batch_df, width="stretch", hide_index=True)
                st.download_button(
                    "Download batch summary CSV",
                    data=batch_df.to_csv(index=False).encode("utf-8"),
                    file_name="histoscope_batch_summary.csv",
                    mime="text/csv",
                )

st.divider()
st.markdown(
    '<div class="small-muted">HistoScope is an educational research workstation built from conventional image-processing methods. It is not intended for diagnosis or treatment decisions.</div>',
    unsafe_allow_html=True,
)
