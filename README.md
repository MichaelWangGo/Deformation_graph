# Deformation Graph

An unofficial implementation of **Embedded Deformation (ED)** for 3D shape manipulation.

This repository provides a lightweight ED pipeline including:

- Deformation graph node sampling
- Vertex-node blending weight computation
- Graph regularization term construction
- Gauss-Newton + Levenberg-Marquardt optimization
- Mesh / point cloud IO and deformed geometry export

## Reference

- Paper: https://dl.acm.org/doi/10.1145/1276377.1276478

```bibtex
@incollection{sumner2007embedded,
  title={Embedded deformation for shape manipulation},
  author={Sumner, Robert W and Schmid, Johannes and Pauly, Mark},
  booktitle={ACM SIGGRAPH 2007 papers},
  pages={80--es},
  year={2007}
}
```

## Project Structure

```text
Deformation_graph/
├── deformation_graph.py    # core implementation and helper functions
└── README.md
```

## Requirements

Required:

- Python 3.8+
- numpy
- torch
- scipy

Optional (for richer IO or mesh simplification fallback paths):

- open3d
- trimesh

Install dependencies:

```bash
pip install numpy torch scipy open3d trimesh
```

## Quick Start

### 1. Python API (recommended)

```python
import numpy as np
from deformation_graph import (
    load_mesh_or_pointcloud,
    build_deformation_graph,
    solve_embedded_deformation_gauss_newton,
    save_mesh_or_pointcloud,
)

# 1) load geometry
data = load_mesh_or_pointcloud("input.obj")

# 2) build deformation graph
graph = build_deformation_graph(
    mesh=data,
    method="uniform",     # "uniform" or "simplify"
    num_nodes=150,
    k=4,
    fixed_vertex_ids=None,
    seed=0,
)

# 3) define handle constraints (example)
handle_ids = np.array([0, 10, 20], dtype=np.int64)
handle_targets = data.vertices[handle_ids] + np.array(
    [[0.0, 0.2, 0.0],
     [0.0, 0.2, 0.0],
     [0.0, 0.2, 0.0]],
    dtype=np.float64,
)

# 4) optimize
result = solve_embedded_deformation_gauss_newton(
    vertices=data.vertices,
    graph=graph,
    handle_vertex_ids=handle_ids,
    handle_target_pos=handle_targets,
    w_rot=1.0,
    w_reg=10.0,
    w_con=100.0,
    num_iters=20,
    lm_lambda=1e-4,
    verbose=True,
)

# 5) save deformed geometry
save_mesh_or_pointcloud(
    "output.obj",
    result["deformed_vertices"],
    data.faces,
    data.colors,
)
```

### 2. Built-in script example

`deformation_graph.py` contains a runnable `__main__` block with hard-coded paths.
Before running, edit input/output paths in the file, then execute:

```bash
python deformation_graph.py
```

## Key Parameters

- `method`: graph node sampling strategy.
  - `uniform`: sample directly on vertices (FPS / poisson-like)
  - `simplify`: prefer mesh simplification-based nodes when available
- `num_nodes`: number of graph nodes (ignored when `radius` is used)
- `radius`: optional radius-based sampling mode
- `k`: number of influencing graph nodes per vertex
- `w_rot`, `w_reg`, `w_con`: rotation, regularization, and constraint weights
- `num_iters`: number of optimization iterations
- `lm_lambda`: damping factor for LM-style stabilization

## Notes

- `handle_vertex_ids` and `handle_target_pos` are mandatory constraints for deformation.
- If `scipy` is not installed, KD-tree based neighbor queries will fail.
- For point cloud input, graph construction still works; some mesh-specific paths depend on triangle faces.

## License

No standalone license file is currently included in this folder. Add your preferred license if you plan to distribute this module independently.
