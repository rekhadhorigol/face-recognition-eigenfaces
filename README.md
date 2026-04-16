# Face Recognition Using Eigenfaces

A complete **face recognition pipeline** that combines deep learning-based facial embeddings with classical dimensionality reduction (PCA/Eigenfaces) to identify individuals from images.

**How It Works:**

The system uses a two-stage approach — **DeepFace (FaceNet)** extracts rich 128-dimensional identity embeddings from each face, and **PCA** then compresses these into a compact eigenface space where similarity matching is performed.

**Pipeline Breakdown:**

- **Setup & Model Loading** — Installs dependencies, downloads OpenCV's Haar Cascade XML for face detection, and warms up the FaceNet model (~90MB, downloaded once)
- **Face Detection & Preprocessing** — Multi-scale Haar Cascade detection with fallback centre-crop if no face is found; includes **eye detection and geometric alignment** (rotation correction using eye landmark angle) and **CLAHE** (Contrast Limited Adaptive Histogram Equalization) for lighting normalization
- **Dataset Loading** — Accepts a **.zip** folder of person-wise images; processes each image through the full face pipeline and logs detection success vs. fallback counts per person
- **DeepFace Embedding Extraction** — Generates a 128-dimensional FaceNet embedding per face crop, stacked into an embedding matrix A of shape (n_images * 128)
- **Mean Face & Centering** — Computes the mean embedding across all training images and centers the matrix (A_centered = A - mean_face)
- **PCA / Eigenfaces** — Fits PCA on centered embeddings (up to 20 components); visualizes variance explained via a **Scree plot** and a **variance-per-component bar chart**
- **Outlier Removal** — Projects all training images into PCA space; removes outlier crops per person using centroid distance thresholding (mean + 1.5*std), displays removed crops for inspection
- **Gallery Building** — Constructs a robust per-person gallery using the mean projection of all clean (non-outlier) crops; also stores individual clean projections for nearest-neighbour matching
- **2D Scatter Visualization** — Projects all training embeddings to 2D PCA space and plots per-person clusters to visually assess separability
- **Test Image Recognition** — Uploads a test face, extracts its DeepFace embedding, projects it into eigenface space, and computes similarity using a **combined score: 60% centroid cosine similarity + 40% nearest-neighbour cosine similarity**
- **Per-Person Threshold Calibration** — Each person gets their own acceptance threshold (mean score − 1.5*std of their own training scores, with a safety floor of 35%) to avoid penalizing people with naturally lower similarity scores
- **Decision & Visualization** — Displays a horizontal bar chart of all similarity scores with per-person threshold markers; outputs either a confident identity match or flags the test image as **Unknown**
- **Diagnostic View** — Reports face detection quality per person (detected vs. fallback counts with a visual bar) and displays all training crops side by side for quality inspection

**Key Concepts:**
- Eigenfaces via PCA on deep embeddings (not raw pixels)
- Eye alignment using geometric rotation for pose normalization
- CLAHE for illumination robustness
- Per-person adaptive thresholding to reduce false accepts
- Combined centroid + nearest-neighbour cosine similarity scoring

**Tools & Libraries:** numpy, matplotlib, opencv-python (Haar Cascade, CLAHE, eye alignment), DeepFace (FaceNet), scikit-learn (PCA), PIL, zipfile

**Colab link:**
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OsvnQtY23rwnfFVGNKCw_bO5qmDM0ISN?usp=sharing)
