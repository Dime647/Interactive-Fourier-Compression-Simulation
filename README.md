# 🌊 Interactive 2-D Fourier Image Compression Simulation

An interactive dashboard built with **Streamlit** that demonstrates image compression using the **Two-Dimensional Discrete Fourier Transform (2-D DFT)**. Manipulate frequency-domain coefficients in real time and observe spatial reconstruction, artifacts, and quantitative fidelity metrics.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

- **Two compression algorithms:**
  - **Top-K Magnitude Thresholding** — retain the strongest N% of frequency coefficients
  - **Radial Low-Pass Cutoff** — zero out all frequencies beyond a radius from DC center

- **Real-time dashboard with 3 synchronized panels:**
  - Spatial domain comparison (original vs. reconstructed)
  - Frequency domain inspection (full vs. masked log-magnitude spectrum)
  - Residual error map with adjustable amplification (inferno colormap)

- **5 quantitative metrics updated live:**
  - Coefficients Zeroed (%) | Retained Energy (%) | MSE | PSNR (dB) | SSIM

- **Preset test images:** Cameraman, Synthetic Shapes, Checkerboard + custom upload

- **Pedagogical insights panel** covering DC coefficient, Gibbs phenomenon, DFT vs DCT, and Parseval's theorem

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Dime647/Interactive-Fourier-Compression-Simulation.git
cd Interactive-Fourier-Compression-Simulation
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the dashboard
```bash
streamlit run app.py
```

Opens at **http://localhost:8501**

---

## 📁 Project Structure

```
├── app.py               # Streamlit frontend (UI layout, controls, panels)
├── backend.py            # Numerical engine (FFT, compression, metrics)
├── requirements.txt      # Python dependencies
└── README.md
```

| Module | Responsibility |
|--------|---------------|
| `backend.py` | 2-D FFT, Top-K & radial compression, MSE/PSNR/SSIM computation, Matplotlib renderers |
| `app.py` | Sidebar controls, metric cards, 3 display panels, educational insights |

---

## 🔬 Experiments to Try

| # | Setup | What You'll Learn |
|---|-------|-------------------|
| 1 | Top-K: slide 100% → 5% | Natural images keep >95% energy in <5% of coefficients |
| 2 | Checkerboard + Radial cutoff r≈20 | Gibbs phenomenon — ringing at sharp edges |
| 3 | Top-K at 0.05% | DC coefficient preserves average brightness |
| 4 | Compare both modes at 95% zeroed | Top-K preserves edges better than radial cutoff |

---

## 🛠️ Tech Stack

- **NumPy** — Vectorized FFT and array operations
- **SciPy** — Scientific computing utilities
- **scikit-image** — SSIM metric and test images
- **Matplotlib** — Spectrum and residual map rendering
- **OpenCV** — Image I/O and resizing
- **Streamlit** — Interactive web dashboard

---

## 📜 License

MIT License — free to use, modify, and distribute.
