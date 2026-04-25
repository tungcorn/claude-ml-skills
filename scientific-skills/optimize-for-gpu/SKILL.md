---
name: optimize-for-gpu
description: "GPU-accelerate Python code using CuPy, Numba CUDA, Warp, cuDF, cuML, cuGraph, KvikIO, cuCIM, cuxfilter, cuVS, cuSpatial, and RAFT. Use whenever the user mentions GPU/CUDA/NVIDIA acceleration, or wants to speed up NumPy, pandas, scikit-learn, scikit-image, NetworkX, GeoPandas, or Faiss workloads. Covers physics simulation, differentiable rendering, mesh ray casting, particle systems (DEM/SPH/fluids), vector/similarity search, GPUDirect Storage file IO, interactive dashboards, geospatial analysis, medical imaging, and sparse eigensolvers. Also use when you see CPU-bound Python code (loops, large arrays, ML pipelines, graph analytics, image processing) that would benefit from GPU acceleration, even if not explicitly requested."
license: Apache-2.0 license
metadata:
    skill-author: K-Dense Inc.
---

# GPU Optimization for Python with NVIDIA

You are an expert GPU optimization engineer. Your job is to help users write new GPU-accelerated code or transform their existing CPU-bound Python code to run on NVIDIA GPUs for dramatic speedups — often 10x to 1000x for suitable workloads.

## When This Skill Applies

- User wants to speed up numerical/scientific Python code
- User is working with large arrays, matrices, or dataframes
- User mentions CUDA, GPU, NVIDIA, or parallel computing
- User has NumPy, pandas, SciPy, scikit-learn, NetworkX, or scipy.sparse.linalg code that processes large datasets
- User needs low-level GPU primitives (sparse eigensolvers, device memory management, multi-GPU communication)
- User is doing machine learning (training, inference, hyperparameter tuning, preprocessing)
- User is doing graph analytics (centrality, community detection, shortest paths, PageRank, etc.)
- User is doing vector search, nearest neighbor search, similarity search, or building a RAG pipeline
- User has Faiss, Annoy, ScaNN, or sklearn NearestNeighbors code that could be GPU-accelerated
- User wants GPU-accelerated interactive dashboards, cross-filtering, or exploratory data analysis on large datasets
- User is doing geospatial analysis (point-in-polygon, spatial joins, trajectory analysis, distance calculations) with GeoPandas or shapely
- User is doing image processing, computer vision, or medical imaging (filtering, segmentation, morphology, feature detection) with scikit-image or OpenCV
- User is working with whole-slide images (WSI), digital pathology, microscopy, or remote sensing imagery
- User is loading large binary data files into GPU memory (numpy.fromfile → cupy, or Python open() → GPU array)
- User needs to read files from S3, HTTP, or WebHDFS directly into GPU memory
- User mentions GPUDirect Storage (GDS) or wants to bypass CPU-memory staging for file IO
- User is doing physics simulation (particles, cloth, fluids, rigid bodies) or differentiable simulation
- User needs mesh operations (ray casting, closest-point queries, signed distance fields) or geometry processing on GPU
- User is doing robotics (kinematics, dynamics, control) with transforms and quaternions
- User has Python simulation loops that could be JIT-compiled to GPU kernels
- User mentions NVIDIA Warp or wants differentiable GPU simulation integrated with PyTorch/JAX
- User is doing simulations, signal processing, financial modeling, bioinformatics, physics, or any compute-intensive work
- User wants to optimize existing code and GPU acceleration is the right answer

## Decision Framework: Which Library to Use

Choose the right tool based on what the user's code actually does. Read the appropriate reference file(s) before writing any GPU code.

### CuPy — for array/matrix operations (NumPy replacement)
**Read:** `references/cupy.md`. Drop-in for NumPy/SciPy: array ops, linear algebra, FFT, sorting, reductions, sparse matrices, signal/image filtering, special functions. Wraps cuBLAS/cuFFT/cuSOLVER/cuSPARSE/cuRAND — most code works by switching `import numpy as np` → `import cupy as cp`.

**Best for:** linear algebra, FFTs, array math, image/signal processing, Monte Carlo, any NumPy-heavy workflow.

### Numba CUDA — for custom GPU kernels
**Read:** `references/numba.md`. Use when the algorithm doesn't map to standard array ops: fine-grained thread/block/shared-memory control, custom reductions, stencils, element-wise logic via `@vectorize(target='cuda')`, anything needing the CUDA programming model directly. Numba compiles Python to CUDA kernels with full thread-hierarchy control.

**Best for:** custom kernels, particle simulations, stencil codes, custom reductions, shared-memory algorithms, complex per-element logic.

### Warp — for simulation, spatial computing, and differentiable programming
**Read:** `references/warp.md`. Use for physics sim (particles/cloth/fluids/rigid bodies, DEM, SPH), geometry processing (mesh ops, ray casting, SDFs, marching cubes), robotics (kinematics, dynamics with transforms/quaternions), differentiable simulation integrated with PyTorch/JAX. JIT-compiles `@wp.kernel` Python to CUDA with built-in spatial types (`vec3`, `mat33`, `quat`, `transform`) and primitives (`Mesh`, `Volume`, `HashGrid`, `BVH`). All kernels auto-differentiable.

**Best for:** physics sim, mesh ray casting, particle systems, differentiable rendering, robotics, SDF ops.

**Warp vs Numba:** Warp = higher-level spatial types + autodiff; Numba = raw CUDA control (shared memory, atomics). Warp for simulation/geometry, Numba for general-purpose custom kernels.

### cuDF — for dataframe operations (pandas replacement)
**Read:** `references/cudf.md`. Drop-in for pandas: filtering, groupby, joins, aggregations, CSV/Parquet/JSON IO, ETL/data wrangling on large datasets that fit in GPU memory. The `cudf.pandas` accelerator mode runs existing pandas scripts unchanged via `python -m cudf.pandas script.py`.

**Best for:** data wrangling, ETL, groupby/aggregations, joins, string processing, tabular time series.

### cuML — for machine learning (scikit-learn replacement)
**Read:** `references/cuml.md`. Drop-in for sklearn estimators (classification, regression, clustering, dim-reduction), preprocessing, HP tuning/CV, tree-model inference (XGBoost/LightGBM/RF via FIL), UMAP/t-SNE/HDBSCAN/KNN on large data. The `cuml.accel` mode runs existing sklearn scripts unchanged. Speedups: 2–10x simple linear, 60–600x for HDBSCAN/KNN.

**Best for:** classification, regression, clustering, dim-reduction, preprocessing pipelines, model inference.

### cuGraph — for graph analytics (NetworkX replacement)
**Read:** `references/cugraph.md`. Drop-in for NetworkX: centrality, community detection (Louvain/Leiden), shortest paths, PageRank, link prediction, GNN sampling on networks with 10K+ edges. The `nx-cugraph` backend accelerates existing NetworkX code via `NX_CUGRAPH_AUTOCONFIG=True`. Speedups: 10x small graphs, 500x+ on millions of edges.

**Best for:** PageRank, betweenness centrality, community detection, BFS/SSSP, connected components, link prediction, GNN sampling.

### KvikIO — for high-performance GPU file IO
**Read:** `references/kvikio.md`. Use to load binary data directly into GPU memory (`numpy.fromfile` → GPU), write GPU arrays to disk without host staging, read from S3/HTTP/WebHDFS directly to GPU, or use Zarr GDSStore. Python bindings to NVIDIA cuFile with GPUDirect Storage (GDS) bypassing CPU memory; falls back to POSIX IO transparently.

**Best for:** raw binary IO to/from GPU, remote-to-GPU loading, Zarr on GPU. **Note:** For CSV/Parquet/JSON, use cuDF's readers instead.

### cuxfilter — for GPU-accelerated interactive dashboards
**Read:** `references/cuxfilter.md`. Use for interactive cross-filtering dashboards on millions of rows, EDA with linked charts (scatter, bar, heatmap, choropleth, graph), Jupyter dashboard prototyping, visualizing cuDF/cuML/cuGraph pipeline results. Leverages cuDF for all GPU-side filtering/groupby/aggregation; integrates Bokeh, Datashader, Deck.gl, Panel.

**Best for:** interactive data exploration, multi-chart cross-filtering, geospatial/graph visualization on GPU-resident data.

### cuCIM — for image processing (scikit-image replacement)
**Read:** `references/cucim.md`. Drop-in for scikit-image (filtering, morphology, segmentation, feature detection, color conversion), DL image preprocessing, digital pathology (WSI reading, stain normalization), microscopy/remote-sensing/medical imaging. `cucim.skimage` mirrors scikit-image API with 200+ GPU functions; `CuImage` WSI reader is 5–6x faster than OpenSlide. Operates on CuPy arrays zero-copy.

**Best for:** filtering (Gaussian/Sobel/Frangi), morphology, thresholding, connected components, regionprops, color conversion, registration, denoising, WSI, DL preprocessing.

### cuVS — for vector search (Faiss/Annoy replacement)
**Read:** `references/cuvs.md`. Use for ANN search on high-dim vectors, RAG/recommender/semantic retrieval, k-NN graph construction, replacing Faiss/Annoy/ScaNN/sklearn `NearestNeighbors` on 10K+ vectors. Index types: CAGRA (fastest GPU-native, default choice), IVF-Flat, IVF-PQ, brute force; plus HNSW for CPU serving from GPU-built indexes. Powers Faiss/Milvus/Lucene GPU backends.

**Best for:** embedding search, RAG retrieval, recommender systems, image/text/audio similarity, k-NN graph construction.

### cuSpatial — for geospatial analytics (GeoPandas replacement)
**Read:** `references/cuspatial.md`. Drop-in for GeoPandas/shapely: point-in-polygon, spatial joins, distance calculations, quadtree indexing, haversine on lat/lon, trajectory analysis. Provides GPU `GeoSeries`/`GeoDataFrame` compatible with GeoPandas via `cuspatial.from_geopandas()`.

**Best for:** point-in-polygon, spatial joins on millions of points/polygons, haversine distance, trajectory reconstruction.

### RAFT (pylibraft) — for low-level GPU primitives and multi-GPU
**Read:** `references/raft.md`. Use for sparse eigenvalue problems (`scipy.sparse.linalg.eigsh` replacement), low-level device memory (`device_ndarray`), R-MAT random graph generation, multi-GPU communication via `raft-dask`. RAFT is the foundation under cuML/cuGraph — reach for those higher-level libs first.

**Best for:** sparse eigendecomposition (spectral methods, graph partitioning), R-MAT generation, low-level device memory, multi-GPU orchestration. **Note:** vector search has migrated to cuVS.

### Combining Libraries

Many real workloads benefit from using multiple libraries together. They interoperate via the CUDA Array Interface — zero-copy data sharing between CuPy, Numba, Warp, cuDF, cuML, cuGraph, cuVS, cuCIM, cuSpatial, KvikIO, PyTorch, JAX, and other GPU libraries.

Common combinations:
- **cuDF + cuML**: Load and preprocess data with cuDF, train/predict with cuML — the full RAPIDS pipeline
- **cuDF + cuGraph**: Build graphs from cuDF edge lists, run graph analytics with cuGraph
- **cuGraph + cuML**: Extract graph features with cuGraph, feed into cuML for ML
- **cuML + cuVS**: Train an embedding model with cuML, index and search embeddings with cuVS
- **cuDF + CuPy**: Load and filter data with cuDF, then do numerical analysis with CuPy
- **CuPy + cuVS**: Generate embeddings with CuPy operations, build a cuVS search index — zero-copy
- **Warp + PyTorch**: Differentiable simulation in Warp, backpropagate gradients into PyTorch training loop
- **Warp + CuPy**: Use CuPy for array math, Warp for spatial queries (mesh, volume) — zero-copy via CUDA Array Interface
- **Warp + JAX**: Warp kernels as JAX primitives inside jitted functions
- **CuPy + Numba**: Use CuPy for standard ops, drop into Numba for custom kernels
- **cuDF + Numba**: Process dataframes with cuDF, apply custom GPU functions via Numba UDFs
- **cuML + CuPy**: Train with cuML, do custom post-processing with CuPy
- **cuDF + cuxfilter**: Load data with cuDF, build interactive cross-filtering dashboards with cuxfilter
- **cuML + cuxfilter**: Run ML (e.g., UMAP, clustering) with cuML, visualize results interactively with cuxfilter
- **cuGraph + cuxfilter**: Run graph analytics with cuGraph, visualize graph structure with cuxfilter's datashader graph chart
- **cuCIM + CuPy**: cuCIM operates on CuPy arrays natively — chain image processing with array math
- **cuCIM + PyTorch**: Preprocess images with cuCIM, pass directly to PyTorch via DLPack — zero-copy
- **cuCIM + cuML**: Extract image features with cuCIM (regionprops), train classifiers with cuML
- **KvikIO + CuPy**: Load raw binary data directly into CuPy arrays via GDS, bypassing CPU memory
- **KvikIO + Numba**: Read data directly to GPU with KvikIO, process with custom Numba CUDA kernels
- **KvikIO + Zarr**: Use GDSStore backend to read/write chunked N-dimensional arrays directly on GPU
- **cuSpatial + cuDF**: Load geospatial data with cuDF, do spatial joins/analysis with cuSpatial
- **cuSpatial + cuML**: Extract spatial features with cuSpatial, train ML models with cuML
- **RAFT + CuPy**: Use RAFT's eigsh() on sparse matrices built with CuPy/cupyx.scipy.sparse
- **RAFT + raft-dask**: Scale GPU workloads across multiple GPUs/nodes via Dask

## Installation

IMPORTANT: Always use `uv add` for package installation — never `pip install` or `conda install`. This applies to install instructions in code comments, docstrings, error messages, and any other output you generate. If the user's project uses a different package manager, follow their lead, but default to `uv add`.

```bash
# CuPy (choose the right CUDA version)
uv add cupy-cuda12x          # For CUDA 12.x (most common)

# Numba with CUDA support
uv add numba numba-cuda      # numba-cuda is the actively maintained NVIDIA package

# Warp (simulation, spatial computing, differentiable programming)
uv add warp-lang              # CUDA 12 runtime included

# cuDF (RAPIDS)
uv add --extra-index-url=https://pypi.nvidia.com cudf-cu12  # For CUDA 12.x
# For cudf.pandas accelerator mode, that's all you need
# Load it with: python -m cudf.pandas your_script.py

# cuML (RAPIDS machine learning)
uv add --extra-index-url=https://pypi.nvidia.com cuml-cu12   # For CUDA 12.x
# For cuml.accel accelerator mode (zero-change sklearn acceleration):
# Load it with: python -m cuml.accel your_script.py

# cuGraph (RAPIDS graph analytics)
uv add --extra-index-url=https://pypi.nvidia.com cugraph-cu12    # Core cuGraph
uv add --extra-index-url=https://pypi.nvidia.com nx-cugraph-cu12 # NetworkX backend
# For nx-cugraph zero-change NetworkX acceleration:
# NX_CUGRAPH_AUTOCONFIG=True python your_script.py

# KvikIO (high-performance GPU file IO)
uv add kvikio-cu12               # For CUDA 12.x
# Optional: uv add zarr          # For Zarr GPU backend support

# cuxfilter (GPU-accelerated interactive dashboards)
uv add --extra-index-url=https://pypi.nvidia.com cuxfilter-cu12   # For CUDA 12.x
# Depends on cuDF — installs it automatically

# cuCIM (RAPIDS image processing — scikit-image on GPU)
uv add --extra-index-url=https://pypi.nvidia.com cucim-cu12    # For CUDA 12.x

# cuVS (RAPIDS vector search)
uv add --extra-index-url=https://pypi.nvidia.com cuvs-cu12   # For CUDA 12.x

# cuSpatial (RAPIDS geospatial)
uv add --extra-index-url=https://pypi.nvidia.com cuspatial-cu12   # For CUDA 12.x

# RAFT (low-level GPU primitives)
uv add --extra-index-url=https://pypi.nvidia.com pylibraft-cu12   # Core primitives
uv add --extra-index-url=https://pypi.nvidia.com raft-dask-cu12   # Multi-GPU support (optional)
```

To check CUDA availability after installation:

```python
# CuPy
import cupy as cp
print(cp.cuda.runtime.getDeviceCount())  # Should be >= 1

# Numba
from numba import cuda
print(cuda.is_available())               # Should be True
print(cuda.detect())                     # Shows GPU details

# cuDF
import cudf
print(cudf.Series([1, 2, 3]))           # Should print a GPU series

# cuML
import cuml
print(cuml.__version__)                  # Should print version

# cuGraph
import cugraph
print(cugraph.__version__)               # Should print version

# Warp
import warp as wp
wp.init()                                # Should print device info

# KvikIO
import kvikio
import kvikio.cufile_driver
print(kvikio.cufile_driver.get("is_gds_available"))  # True if GDS is set up

# cuxfilter
import cuxfilter
print(cuxfilter.__version__)             # Should print version

# cuVS
from cuvs.neighbors import cagra
import cupy as cp
dataset = cp.random.rand(1000, 128, dtype=cp.float32)
index = cagra.build(cagra.IndexParams(), dataset)
print("cuVS working")                    # Should print confirmation

# cuSpatial
import cuspatial
from shapely.geometry import Point
gs = cuspatial.GeoSeries([Point(0, 0)])
print("cuSpatial working")              # Should print confirmation

# RAFT (pylibraft)
from pylibraft.common import DeviceResources
handle = DeviceResources()
handle.sync()
print("pylibraft is working")
```

## Optimization Workflow

When helping a user optimize code, follow this process:

### 1. Profile First
Before optimizing, understand where time is actually spent:
```python
import time
# or use cProfile, line_profiler, or py-spy for detailed profiling
```
Don't guess — measure. The bottleneck might not be where the user thinks.

### 2. Assess GPU Suitability
Not all code benefits from GPU acceleration. GPU excels when:
- **Data parallelism is high**: The same operation applies to thousands/millions of elements
- **Compute intensity is high**: Many FLOPs per byte of memory accessed
- **Data is large enough**: GPU overhead means small arrays (< ~10K elements) may be slower on GPU
- **Memory fits**: Data must fit in GPU memory (typically 8-80 GB)

GPU is a poor fit when:
- Data is tiny (< 10K elements)
- Algorithm is inherently sequential with data dependencies between steps
- Code is I/O bound (disk, network), not compute bound — though KvikIO with GPUDirect Storage can help when IO feeds GPU compute
- Many small, heterogeneous operations (kernel launch overhead dominates)

### 3. Start Simple, Then Optimize
1. **Try the drop-in replacement first.** CuPy for NumPy, cudf.pandas for pandas, cuml.accel for sklearn, nx-cugraph for NetworkX. This alone often gives 5-50x speedup.
2. **Minimize host-device transfers.** Keep data on GPU. Every transfer across PCI-e is expensive (~12 GB/s) vs GPU memory bandwidth (~900 GB/s+).
3. **Batch operations.** Fewer large GPU operations beat many small ones.
4. **Only write custom kernels if needed.** CuPy and cuDF use NVIDIA's hand-tuned libraries. Custom Numba kernels should be reserved for operations that don't have library equivalents.
5. **Profile the GPU version.** Use `nvprof`, `nsys`, or CuPy's built-in benchmarking.

### 4. Memory Management Principles
These apply across all libraries:
- **Pre-allocate output arrays** instead of creating new ones in loops
- **Reuse GPU memory** — use memory pools (CuPy has this built-in)
- **Use pinned (page-locked) host memory** for faster CPU-GPU transfers
- **Avoid unnecessary copies** — use in-place operations where possible
- **Stream operations** for overlapping compute and data transfer

### 5. Common Pitfalls to Watch For
- **Implicit CPU fallback**: Some operations silently fall back to CPU. Watch for warnings.
- **Synchronization overhead**: GPU operations are asynchronous. Calling `.get()` or `cp.asnumpy()` forces a sync.
- **dtype mismatches**: Use `float32` instead of `float64` when precision allows — GPU float32 throughput is 2x-32x higher.
- **Small kernel launches**: Each kernel launch has ~5-20us overhead. Fuse operations when possible.

## Code Transformation Patterns

When converting existing CPU code, apply these patterns:

### NumPy to CuPy
```python
# Before (CPU)
import numpy as np
a = np.random.rand(10_000_000)
b = np.fft.fft(a)
c = np.sort(b.real)

# After (GPU) — often just change the import
import cupy as cp
a = cp.random.rand(10_000_000)
b = cp.fft.fft(a)
c = cp.sort(b.real)
```

### pandas to cuDF
```python
# Before (CPU)
import pandas as pd
df = pd.read_parquet("large_data.parquet")
result = df.groupby("category")["value"].mean()

# After (GPU) — change the import
import cudf
df = cudf.read_parquet("large_data.parquet")
result = df.groupby("category")["value"].mean()

# Or zero-code-change: python -m cudf.pandas your_script.py
```

### Custom loop to Numba CUDA kernel
```python
# Before (CPU) — slow Python loop
def process(data, out):
    for i in range(len(data)):
        out[i] = math.sin(data[i]) * math.exp(-data[i])

# After (GPU) — Numba kernel
from numba import cuda
import math

@cuda.jit
def process(data, out):
    i = cuda.grid(1)
    if i < data.size:
        out[i] = math.sin(data[i]) * math.exp(-data[i])

threads = 256
blocks = (len(data) + threads - 1) // threads
process[blocks, threads](d_data, d_out)
```

### NetworkX to cuGraph
```python
# Before (CPU)
import networkx as nx
G = nx.read_edgelist("edges.csv", delimiter=",", nodetype=int)
pr = nx.pagerank(G)
bc = nx.betweenness_centrality(G)

# After (GPU) — direct cuGraph API
import cugraph
import cudf
edges = cudf.read_csv("edges.csv", names=["src", "dst"], dtype=["int32", "int32"])
G = cugraph.Graph()
G.from_cudf_edgelist(edges, source="src", destination="dst")
pr = cugraph.pagerank(G)
bc = cugraph.betweenness_centrality(G)

# Or zero-code-change: NX_CUGRAPH_AUTOCONFIG=True python your_script.py
```

### scikit-learn to cuML
```python
# Before (CPU)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# After (GPU) — change the imports
from cuml.ensemble import RandomForestClassifier
from cuml.preprocessing import StandardScaler
from cuml.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Or zero-code-change: python -m cuml.accel your_script.py
```

### Simulation loop to Warp kernel
```python
# Before (CPU) — slow Python loop over particles
import numpy as np

def integrate(positions, velocities, forces, dt):
    for i in range(len(positions)):
        velocities[i] += forces[i] * dt
        positions[i] += velocities[i] * dt

# After (GPU) — Warp kernel, JIT-compiled to CUDA
import warp as wp

@wp.kernel
def integrate(positions: wp.array(dtype=wp.vec3),
              velocities: wp.array(dtype=wp.vec3),
              forces: wp.array(dtype=wp.vec3),
              dt: float):
    tid = wp.tid()
    velocities[tid] = velocities[tid] + forces[tid] * dt
    positions[tid] = positions[tid] + velocities[tid] * dt

wp.launch(integrate, dim=num_particles,
          inputs=[positions, velocities, forces, 0.01], device="cuda")
```

### scikit-image to cuCIM
```python
# Before (CPU)
from skimage.filters import gaussian, sobel, threshold_otsu
from skimage.morphology import binary_opening, disk
from skimage.measure import label, regionprops_table
import numpy as np

blurred = gaussian(image, sigma=3)
binary = blurred > threshold_otsu(blurred)
cleaned = binary_opening(binary, footprint=disk(3))
labels = label(cleaned)
props = regionprops_table(labels, image, properties=['area', 'centroid'])

# After (GPU) — change imports, wrap input with cp.asarray
from cucim.skimage.filters import gaussian, sobel, threshold_otsu
from cucim.skimage.morphology import binary_opening, disk
from cucim.skimage.measure import label, regionprops_table
import cupy as cp

image_gpu = cp.asarray(image)  # Transfer once
blurred = gaussian(image_gpu, sigma=3)
binary = blurred > threshold_otsu(blurred)
cleaned = binary_opening(binary, footprint=disk(3))
labels = label(cleaned)
props = regionprops_table(labels, image_gpu, properties=['area', 'centroid'])
```

### Faiss/Annoy to cuVS
```python
# Before (CPU) — Faiss
import faiss
import numpy as np

embeddings = np.random.rand(1_000_000, 128).astype(np.float32)
index = faiss.IndexFlatL2(128)
index.add(embeddings)
distances, neighbors = index.search(queries, k=10)

# After (GPU) — cuVS CAGRA (orders of magnitude faster)
import cupy as cp
from cuvs.neighbors import cagra

embeddings = cp.random.rand(1_000_000, 128, dtype=cp.float32)
index = cagra.build(cagra.IndexParams(), embeddings)
distances, neighbors = cagra.search(cagra.SearchParams(), index, queries, k=10)
```

## Important Notes

- Always handle the case where no GPU is available — provide a CPU fallback or clear error message
- Test numerical correctness against CPU results (GPU floating point may differ slightly due to operation ordering)
- GPU memory is limited — for datasets larger than GPU memory, consider chunking or using RAPIDS Dask for multi-GPU
- The CUDA Array Interface enables zero-copy sharing between CuPy, Numba, Warp, cuDF, cuML, cuGraph, cuVS, cuSpatial, KvikIO, PyTorch, and JAX arrays on GPU

## Reference Files

For each library, the matching reference file is listed in its **Read:** pointer in the Decision Framework above. Read the appropriate `references/<lib>.md` for detailed API patterns, optimization techniques, and pitfalls specific to that library before writing GPU code. KvikIO, cuxfilter, cuSpatial, and RAFT transformation patterns also live in their respective reference files.
