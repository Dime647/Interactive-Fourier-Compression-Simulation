"""
app.py — Streamlit dashboard for interactive 2-D Fourier image compression.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from backend import (
    CompressionMode,
    compress_radial_cutoff,
    compress_top_k,
    compute_metrics,
    forward_fft,
    inverse_fft,
    load_preset,
    load_uploaded,
    log_magnitude,
    render_residual,
    render_spectrum,
    PRESET_IMAGES,
)

# ──────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="2-D Fourier Compression Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject minimal custom CSS for metric cards
st.markdown(
    """
    <style>
    .metric-card {
        background: #1e2130;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
        border: 1px solid #2d3250;
    }
    .metric-card .label {
        color: #8b92a5;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-card .value {
        color: #e0e0e0;
        font-size: 1.35rem;
        font-weight: 700;
        margin-top: 2px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def metric_card(label: str, value: str) -> str:
    """Return HTML for a styled metric card."""
    return (
        f'<div class="metric-card">'
        f'<div class="label">{label}</div>'
        f'<div class="value">{value}</div>'
        f'</div>'
    )


# ──────────────────────────────────────────────
# Sidebar — interactive controls
# ──────────────────────────────────────────────

with st.sidebar:
    st.title("🔬 Controls")
    st.divider()

    # ── Input source ──
    st.subheader("📷 Image Source")
    source = st.radio(
        "Choose input",
        ["Preset Image", "Upload Custom"],
        horizontal=True,
        label_visibility="collapsed",
    )

    image: np.ndarray | None = None

    if source == "Preset Image":
        preset_name = st.selectbox("Preset", list(PRESET_IMAGES.keys()))
        image = load_preset(preset_name)
    else:
        uploaded = st.file_uploader(
            "Upload (.png / .jpg)",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
        )
        if uploaded is not None:
            image = load_uploaded(uploaded.read())

    st.divider()

    # ── Compression mode ──
    st.subheader("⚙️ Compression Mode")
    mode_label = st.radio(
        "Algorithm",
        ["Top-K (Magnitude %)", "Radial Cutoff (Low-Pass)"],
        label_visibility="collapsed",
    )
    mode = (
        CompressionMode.TOP_K
        if "Top-K" in mode_label
        else CompressionMode.RADIAL_CUTOFF
    )

    # ── Compression parameter ──
    if mode == CompressionMode.TOP_K:
        retain_pct = st.slider(
            "Retain top % of coefficients",
            min_value=0.01,
            max_value=100.0,
            value=10.0,
            step=0.01,
            format="%.2f%%",
        )
    else:
        if image is not None:
            max_radius = min(image.shape) // 2
        else:
            max_radius = 128
        cutoff_radius = st.slider(
            "Cutoff radius (px from DC)",
            min_value=1,
            max_value=max_radius,
            value=max(1, max_radius // 4),
        )

    st.divider()

    # ── Visualisation toggles ──
    st.subheader("🎨 Display Options")
    show_residual = st.checkbox("Show residual error map", value=True)
    show_spectra = st.checkbox("Show frequency spectra", value=True)
    residual_gain = st.slider(
        "Residual amplification gain",
        min_value=1.0,
        max_value=20.0,
        value=5.0,
        step=0.5,
    )

# ──────────────────────────────────────────────
# Main area
# ──────────────────────────────────────────────

st.markdown(
    "# 🌊 2-D Fourier Image Compression Lab\n"
    "*Explore how discarding frequency-domain coefficients affects spatial reconstruction*"
)

if image is None:
    st.info("⬅️ Select a preset image or upload one from the sidebar to begin.")
    st.stop()

# ── Compute forward FFT (cached on image identity) ──
@st.cache_data(show_spinner=False)
def _cached_fft(img_bytes: bytes, shape: tuple) -> np.ndarray:
    img = np.frombuffer(img_bytes, dtype=np.float64).reshape(shape)
    return forward_fft(img)


spectrum = _cached_fft(image.tobytes(), image.shape)

# ── Apply compression ──
if mode == CompressionMode.TOP_K:
    masked_spectrum, mask = compress_top_k(spectrum, retain_pct)
else:
    masked_spectrum, mask = compress_radial_cutoff(spectrum, cutoff_radius)

# ── Reconstruct ──
reconstructed = inverse_fft(masked_spectrum)

# ── Metrics ──
metrics = compute_metrics(image, reconstructed, spectrum, mask)


# ──────────────────────────────────────────────
# §0  Real-time quantitative telemetry
# ──────────────────────────────────────────────

st.markdown("### 📊 Quantitative Telemetry")
m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    st.markdown(
        metric_card("Coefficients Zeroed", f"{metrics.coefficients_zeroed_pct:.2f}%"),
        unsafe_allow_html=True,
    )
with m2:
    st.markdown(
        metric_card("Retained Energy", f"{metrics.retained_energy_pct:.2f}%"),
        unsafe_allow_html=True,
    )
with m3:
    st.markdown(
        metric_card("MSE", f"{metrics.mse:.6f}"),
        unsafe_allow_html=True,
    )
with m4:
    psnr_str = f"{metrics.psnr_db:.2f} dB" if np.isfinite(metrics.psnr_db) else "∞ dB"
    st.markdown(
        metric_card("PSNR", psnr_str),
        unsafe_allow_html=True,
    )
with m5:
    st.markdown(
        metric_card("SSIM", f"{metrics.ssim_index:.4f}"),
        unsafe_allow_html=True,
    )

st.divider()

# ──────────────────────────────────────────────
# §1  Panel A — Spatial domain comparison
# ──────────────────────────────────────────────

st.markdown("### 🖼️ Panel A — Spatial Domain Comparison")
col_orig, col_recon = st.columns(2)

with col_orig:
    st.image(image, caption="Original", use_container_width=True, clamp=True)

with col_recon:
    st.image(
        reconstructed,
        caption="Reconstructed (compressed)",
        use_container_width=True,
        clamp=True,
    )

# ──────────────────────────────────────────────
# §2  Panel B — Frequency domain inspection
# ──────────────────────────────────────────────

if show_spectra:
    st.divider()
    st.markdown("### 📡 Panel B — Frequency Domain Inspection")
    col_full, col_masked = st.columns(2)

    full_log = log_magnitude(spectrum)
    masked_log = log_magnitude(masked_spectrum)

    with col_full:
        st.image(
            render_spectrum(full_log, "Full Log-Magnitude Spectrum"),
            caption="Full spectrum  log(1 + |F(u,v)|)",
            use_container_width=True,
        )

    with col_masked:
        st.image(
            render_spectrum(masked_log, "Masked Spectrum (retained)"),
            caption="Retained spectrum after compression",
            use_container_width=True,
        )

# ──────────────────────────────────────────────
# §3  Panel C — Residual / degradation analysis
# ──────────────────────────────────────────────

if show_residual:
    st.divider()
    st.markdown("### 🔥 Panel C — Degradation & Residual Analysis")
    residual_rgb = render_residual(image, reconstructed, gain=residual_gain)
    st.image(
        residual_rgb,
        caption=f"|Original − Reconstructed| × {residual_gain:.0f}",
        use_container_width=True,
    )

# ──────────────────────────────────────────────
# §4  Pedagogical insights
# ──────────────────────────────────────────────

st.divider()

with st.expander("📚 **Signals & Systems Insights**", expanded=False):
    st.markdown(
        r"""
#### 1. The DC Coefficient and Average Brightness

The DC component $F(0, 0)$ equals the sum of all pixel intensities in the
image.  Retaining it during compression preserves the **global average
brightness**; discarding it shifts the reconstructed image to zero mean —
an immediately visible artifact.

$$F(0,0) = \sum_{m=0}^{M-1}\sum_{n=0}^{N-1} f(m,n)$$

---

#### 2. Ringing Artifacts and the Gibbs Phenomenon

When high-frequency components are abruptly zeroed (especially with an
*ideal* low-pass cutoff), the reconstruction exhibits oscillatory
overshoot near sharp edges.  This is the **Gibbs phenomenon**: truncating
a Fourier series at a discontinuity always produces ~9 % overshoot that
does not diminish with more terms — it only moves closer to the edge.

In the spatial domain, the ideal low-pass filter's impulse response is a
2-D **sinc** function with infinite sidelobes.  Convolution with these
sidelobes produces the visible ringing.

> **Tip:** Compare the *Checkerboard* preset at a tight radial cutoff to
> see dramatic ringing along every edge.

---

#### 3. DFT vs. DCT — Why JPEG Uses the Cosine Transform

The Discrete Fourier Transform treats the image as **periodic** in both
dimensions.  Unless the image tiles seamlessly, this implicit periodicity
creates artificial discontinuities at the boundaries, spreading energy
into high-frequency coefficients that carry no real image information.

The **Discrete Cosine Transform (DCT)** uses a symmetric (even)
extension, which avoids boundary jumps and concentrates energy far more
efficiently into the low-frequency region.  This is the core reason
JPEG's 8 × 8 block DCT achieves better compression than a naïve DFT
approach at equivalent quality.

$$\text{DCT:} \quad X_k = \sum_{n=0}^{N-1} x_n \cos\!\left[\frac{\pi}{N}\left(n + \tfrac{1}{2}\right)k\right]$$

---

#### 4. Parseval's Theorem and Energy Compaction

Parseval's theorem guarantees that the **total energy** in the spatial
domain equals the total energy in the frequency domain (up to a
normalisation constant):

$$\sum_{m,n} |f(m,n)|^2 = \frac{1}{MN}\sum_{u,v} |F(u,v)|^2$$

This means the "Retained Energy %" metric shown above tells you
precisely how much of the image's power is preserved, regardless of
which domain you measure it in.  Natural images typically concentrate
> 95 % of energy in < 5 % of their DFT coefficients — the basis of
transform-domain compression.
        """
    )

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────

st.divider()
st.caption(
    "2-D Fourier Compression Lab · Built with Streamlit, NumPy, "
    "scikit-image, and Matplotlib"
)
