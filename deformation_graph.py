import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

try:
    import open3d as o3d
except ImportError:
    o3d = None

try:
    import trimesh
except ImportError:
    trimesh = None


# ============================================================
# Data containers
# ============================================================

@dataclass
class MeshData:
    vertices: np.ndarray          # (N, 3)
    faces: Optional[np.ndarray]   # (M, 3) or None
    colors: Optional[np.ndarray] = None  # (N, 3) float in [0, 1] or None


@dataclass
class DeformationGraph:
    nodes: np.ndarray             # (G, 3)
    edges: np.ndarray             # (E, 2)
    knn_idx: np.ndarray           # (N, K)
    knn_w: np.ndarray             # (N, K)
    fixed_mask: np.ndarray        # (G,) bool


# ============================================================
# Utility
# ============================================================

def to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def build_kdtree(points: np.ndarray):
    if cKDTree is None:
        raise ImportError("scipy is required: pip install scipy")
    return cKDTree(points)


def knn_query(src: np.ndarray, dst: np.ndarray, k: int):
    tree = build_kdtree(dst)
    dists, idx = tree.query(src, k=k)
    if k == 1:
        dists = dists[:, None]
        idx = idx[:, None]
    return dists, idx


# ============================================================
# Sampling
# ============================================================

def farthest_point_sampling(points: np.ndarray, num_samples: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    N = points.shape[0]
    assert num_samples <= N

    sampled = np.zeros(num_samples, dtype=np.int64)
    sampled[0] = rng.integers(0, N)

    min_dist2 = np.full(N, np.inf, dtype=np.float64)
    last = points[sampled[0]]

    for i in range(1, num_samples):
        d2 = np.sum((points - last[None, :]) ** 2, axis=1)
        min_dist2 = np.minimum(min_dist2, d2)
        sampled[i] = np.argmax(min_dist2)
        last = points[sampled[i]]
    return sampled


def poisson_disk_like_sampling(points: np.ndarray, radius: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = np.asarray(points)
    N = points.shape[0]
    order = rng.permutation(N)
    tree = build_kdtree(points)

    alive = np.ones(N, dtype=bool)
    chosen = []

    for idx in order:
        if not alive[idx]:
            continue
        chosen.append(idx)
        neigh = tree.query_ball_point(points[idx], radius)
        alive[np.array(neigh, dtype=np.int64)] = False
        alive[idx] = False

    return np.array(chosen, dtype=np.int64)


def sample_graph_nodes_uniform(
    mesh: MeshData,
    target_num_nodes: Optional[int] = None,
    radius: Optional[float] = None,
    seed: int = 0
) -> np.ndarray:
    V = mesh.vertices
    if target_num_nodes is not None:
        idx = farthest_point_sampling(V, target_num_nodes, seed=seed)
        return V[idx]
    elif radius is not None:
        idx = poisson_disk_like_sampling(V, radius=radius, seed=seed)
        return V[idx]
    else:
        raise ValueError("Either target_num_nodes or radius must be provided.")


def sample_graph_nodes_mesh_simplification(
    mesh: MeshData,
    target_num_nodes: int
) -> np.ndarray:
    V = mesh.vertices
    F = mesh.faces

    if F is None:
        return sample_graph_nodes_uniform(mesh, target_num_nodes=target_num_nodes)

    if o3d is not None:
        try:
            mesh_o3d = o3d.geometry.TriangleMesh()
            mesh_o3d.vertices = o3d.utility.Vector3dVector(V)
            mesh_o3d.triangles = o3d.utility.Vector3iVector(F)
            simplified = mesh_o3d.simplify_quadric_decimation(
                target_number_of_triangles=max(target_num_nodes * 2, 4)
            )
            nodes = np.asarray(simplified.vertices)
            if len(nodes) > target_num_nodes:
                idx = farthest_point_sampling(nodes, target_num_nodes)
                nodes = nodes[idx]
            elif len(nodes) < target_num_nodes:
                extra = target_num_nodes - len(nodes)
                idx = farthest_point_sampling(V, extra)
                nodes = np.concatenate([nodes, V[idx]], axis=0)
            return nodes.astype(np.float64)
        except Exception:
            pass

    if trimesh is not None:
        try:
            tm = trimesh.Trimesh(vertices=V, faces=F, process=False)
            if hasattr(tm, "simplify_quadric_decimation"):
                simp = tm.simplify_quadric_decimation(face_count=max(target_num_nodes * 2, 4))
                nodes = np.asarray(simp.vertices)
                if len(nodes) > target_num_nodes:
                    idx = farthest_point_sampling(nodes, target_num_nodes)
                    nodes = nodes[idx]
                elif len(nodes) < target_num_nodes:
                    extra = target_num_nodes - len(nodes)
                    idx = farthest_point_sampling(V, extra)
                    nodes = np.concatenate([nodes, V[idx]], axis=0)
                return nodes.astype(np.float64)
        except Exception:
            pass

    return sample_graph_nodes_uniform(mesh, target_num_nodes=target_num_nodes)


# ============================================================
# Graph construction
# ============================================================

def compute_vertex_node_weights(
    vertices: np.ndarray,
    nodes: np.ndarray,
    k: int = 4,
    eps: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Paper-like:
        w_j(v_i) = (1 - ||v_i-g_j|| / dmax)^2
    where dmax is distance to (k+1)-th nearest node.
    """
    dists_k1, idx_k1 = knn_query(vertices, nodes, k=k + 1)
    dmax = dists_k1[:, -1:] + eps
    dists = dists_k1[:, :k]
    idx = idx_k1[:, :k]
    w = (1.0 - dists / dmax) ** 2
    w = w / (np.sum(w, axis=1, keepdims=True) + eps)
    return idx.astype(np.int64), w.astype(np.float64)


def build_graph_edges_from_vertex_support(knn_idx: np.ndarray) -> np.ndarray:
    edge_set = set()
    N, K = knn_idx.shape
    for i in range(N):
        ids = knn_idx[i]
        for a in range(K):
            for b in range(a + 1, K):
                u, v = int(ids[a]), int(ids[b])
                if u == v:
                    continue
                if u > v:
                    u, v = v, u
                edge_set.add((u, v))
    if len(edge_set) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.array(list(edge_set), dtype=np.int64)


def build_fixed_mask_from_fixed_vertices(
    fixed_vertex_ids: Optional[np.ndarray],
    knn_idx: np.ndarray,
    num_nodes: int
) -> np.ndarray:
    fixed_mask = np.zeros(num_nodes, dtype=bool)
    if fixed_vertex_ids is None or len(fixed_vertex_ids) == 0:
        return fixed_mask
    ids = knn_idx[fixed_vertex_ids].reshape(-1)
    fixed_mask[np.unique(ids)] = True
    return fixed_mask


def build_deformation_graph(
    mesh: MeshData,
    method: str = "uniform",
    num_nodes: int = 200,
    radius: Optional[float] = None,
    k: int = 4,
    fixed_vertex_ids: Optional[np.ndarray] = None,
    seed: int = 0
) -> DeformationGraph:
    # Step 1: sample graph nodes from input geometry.
    # - uniform: FPS/poisson-like sampling directly on vertices
    # - simplify: simplify mesh first then use simplified vertices as graph nodes
    if method == "uniform":
        nodes = sample_graph_nodes_uniform(
            mesh,
            target_num_nodes=num_nodes if radius is None else None,
            radius=radius,
            seed=seed
        )
    elif method == "simplify":
        nodes = sample_graph_nodes_mesh_simplification(mesh, target_num_nodes=num_nodes)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Step 2: compute per-vertex KNN node support and normalized blending weights.
    # These weights are reused both during optimization constraints and forward deformation.
    knn_idx, knn_w = compute_vertex_node_weights(mesh.vertices, nodes, k=k)
    # Step 3: connect graph nodes if they co-support at least one surface vertex.
    # This produces the regularization edges used by as-rigid-as-possible smoothing.
    edges = build_graph_edges_from_vertex_support(knn_idx)
    # Step 4: mark graph nodes as fixed if they are associated with fixed mesh vertices.
    # Fixed nodes are removed from unknowns and stay at identity transform.
    fixed_mask = build_fixed_mask_from_fixed_vertices(fixed_vertex_ids, knn_idx, len(nodes))

    return DeformationGraph(
        nodes=nodes,
        edges=edges,
        knn_idx=knn_idx,
        knn_w=knn_w,
        fixed_mask=fixed_mask
    )


# ============================================================
# Parameter packing
# Each node j has:
#   A_j: 3x3  -> 9 params
#   t_j: 3    -> 3 params
# total 12 params / node
# For fixed nodes, params are removed from unknown vector.
# ============================================================

class VariableIndexer:
    def __init__(self, num_nodes: int, fixed_mask: np.ndarray):
        self.num_nodes = num_nodes
        self.fixed_mask = fixed_mask.astype(bool)

        self.free_nodes = np.where(~self.fixed_mask)[0]
        self.fixed_nodes = np.where(self.fixed_mask)[0]

        self.node_to_varbase = {}
        offset = 0
        # Pack variables node by node as [A(9), t(3)] only for free nodes.
        # This keeps the linear system compact when many nodes are fixed.
        for j in self.free_nodes:
            self.node_to_varbase[int(j)] = offset
            offset += 12
        self.num_vars = offset

    def is_free(self, j: int) -> bool:
        return not self.fixed_mask[j]

    def base(self, j: int) -> int:
        return self.node_to_varbase[int(j)]

    def a_index(self, j: int, row: int, col: int) -> int:
        return self.base(j) + row * 3 + col

    def t_index(self, j: int, dim: int) -> int:
        return self.base(j) + 9 + dim


def initial_x_from_graph(graph: DeformationGraph) -> Tuple[np.ndarray, VariableIndexer]:
    indexer = VariableIndexer(len(graph.nodes), graph.fixed_mask)
    x0 = np.zeros(indexer.num_vars, dtype=np.float64)

    for j in indexer.free_nodes:
        b = indexer.base(j)
        A0 = np.eye(3).reshape(-1)
        t0 = np.zeros(3)
        x0[b:b+9] = A0
        x0[b+9:b+12] = t0
    return x0, indexer


def unpack_node_transform(
    x: np.ndarray,
    indexer: VariableIndexer,
    node_id: int
) -> Tuple[np.ndarray, np.ndarray]:
    if not indexer.is_free(node_id):
        return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

    b = indexer.base(node_id)
    A = x[b:b+9].reshape(3, 3)
    t = x[b+9:b+12]
    return A, t


def unpack_all_transforms(
    x: np.ndarray,
    indexer: VariableIndexer,
    num_nodes: int
) -> Tuple[np.ndarray, np.ndarray]:
    A_all = np.tile(np.eye(3, dtype=np.float64)[None, :, :], (num_nodes, 1, 1))
    t_all = np.zeros((num_nodes, 3), dtype=np.float64)

    for j in indexer.free_nodes:
        b = indexer.base(j)
        A_all[j] = x[b:b+9].reshape(3, 3)
        t_all[j] = x[b+9:b+12]
    return A_all, t_all


# ============================================================
# Residual/Jacobian builder
# ============================================================

class EDGaussNewtonSystem:
    """
    Build residual vector r and Jacobian J for:
        E = w_rot * E_rot + w_reg * E_reg + w_con * E_con

    Residual design:
    1) Rotation residual per free node j:
       Let columns of A_j be c1,c2,c3.
       Residuals:
         c1^T c2
         c1^T c3
         c2^T c3
         c1^T c1 - 1
         c2^T c2 - 1
         c3^T c3 - 1

    2) Regularization residual per edge (j,k):
         A_j (g_k - g_j) + g_j + t_j - (g_k + t_k)   in R^3

    3) Constraint residual per handle vertex i:
         sum_j w_ij [ A_j (v_i - g_j) + g_j + t_j ] - q_i   in R^3
    """
    def __init__(
        self,
        vertices: np.ndarray,
        graph: DeformationGraph,
        handle_vertex_ids: np.ndarray,
        handle_target_pos: np.ndarray,
        w_rot: float = 1.0,
        w_reg: float = 10.0,
        w_con: float = 100.0,
    ):
        self.V = vertices.astype(np.float64)
        self.G = graph.nodes.astype(np.float64)
        self.E = graph.edges.astype(np.int64)
        self.knn_idx = graph.knn_idx.astype(np.int64)
        self.knn_w = graph.knn_w.astype(np.float64)
        self.fixed_mask = graph.fixed_mask.astype(bool)

        self.handle_vertex_ids = handle_vertex_ids.astype(np.int64)
        self.handle_target_pos = handle_target_pos.astype(np.float64)

        self.w_rot = float(w_rot)
        self.w_reg = float(w_reg)
        self.w_con = float(w_con)

        self.indexer = VariableIndexer(len(self.G), self.fixed_mask)

    def num_residuals(self) -> int:
        n_free = len(self.indexer.free_nodes)
        n_rot = 6 * n_free
        n_reg = 3 * len(self.E)
        n_con = 3 * len(self.handle_vertex_ids)
        return n_rot + n_reg + n_con

    def build(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, slice]]:
        """
        Returns:
            r: (R,)
            J: (R, num_vars)
            slices: dict of slices for rot/reg/con blocks
        """
        num_vars = self.indexer.num_vars
        R_total = self.num_residuals()

        r = np.zeros(R_total, dtype=np.float64)
        J = np.zeros((R_total, num_vars), dtype=np.float64)

        # Materialize all node transforms from the packed unknown vector.
        # Fixed nodes are identity + zero translation by design.
        A_all, t_all = unpack_all_transforms(x, self.indexer, len(self.G))

        cursor = 0

        # ----------------------------------------------------
        # 1) Rotation residuals
        # ----------------------------------------------------
        # Enforce each local affine matrix A_j to stay close to rotation:
        # - orthogonality between column pairs
        # - unit length for each column
        # This stabilizes deformation and avoids shear/scale drift.
        rot_start = cursor
        sqrt_wrot = math.sqrt(self.w_rot)

        def aidx_colmajor_in_rowflat(col, row):
            # A[row, col] in row-major flattened array
            return row * 3 + col

        for j in self.indexer.free_nodes:
            A = A_all[j]
            c1 = A[:, 0]
            c2 = A[:, 1]
            c3 = A[:, 2]

            # 6 residuals per node (soft orthonormality constraints).
            vals = np.array([
                np.dot(c1, c2),
                np.dot(c1, c3),
                np.dot(c2, c3),
                np.dot(c1, c1) - 1.0,
                np.dot(c2, c2) - 1.0,
                np.dot(c3, c3) - 1.0,
            ], dtype=np.float64)

            r[cursor:cursor+6] = sqrt_wrot * vals

            b = self.indexer.base(j)

            # Fill analytic Jacobian for better numerical behavior than finite differences.
            # residual 0: c1^T c2, d/dc1 = c2, d/dc2 = c1
            for row in range(3):
                J[cursor + 0, b + aidx_colmajor_in_rowflat(0, row)] = sqrt_wrot * c2[row]
                J[cursor + 0, b + aidx_colmajor_in_rowflat(1, row)] = sqrt_wrot * c1[row]

            # residual 1: c1^T c3
            for row in range(3):
                J[cursor + 1, b + aidx_colmajor_in_rowflat(0, row)] = sqrt_wrot * c3[row]
                J[cursor + 1, b + aidx_colmajor_in_rowflat(2, row)] = sqrt_wrot * c1[row]

            # residual 2: c2^T c3
            for row in range(3):
                J[cursor + 2, b + aidx_colmajor_in_rowflat(1, row)] = sqrt_wrot * c3[row]
                J[cursor + 2, b + aidx_colmajor_in_rowflat(2, row)] = sqrt_wrot * c2[row]

            # residual 3: c1^T c1 - 1
            for row in range(3):
                J[cursor + 3, b + aidx_colmajor_in_rowflat(0, row)] = sqrt_wrot * 2.0 * c1[row]

            # residual 4: c2^T c2 - 1
            for row in range(3):
                J[cursor + 4, b + aidx_colmajor_in_rowflat(1, row)] = sqrt_wrot * 2.0 * c2[row]

            # residual 5: c3^T c3 - 1
            for row in range(3):
                J[cursor + 5, b + aidx_colmajor_in_rowflat(2, row)] = sqrt_wrot * 2.0 * c3[row]

            cursor += 6

        rot_end = cursor

        # ----------------------------------------------------
        # 2) Regularization residuals
        # ----------------------------------------------------
        # Neighboring graph nodes should transform their relative offset consistently:
        #   A_j (g_k-g_j) + g_j + t_j  ~=  g_k + t_k
        # This is the smoothness term that couples local transforms over graph edges.
        reg_start = cursor
        sqrt_wreg = math.sqrt(self.w_reg)

        for e in range(len(self.E)):
            j, k = int(self.E[e, 0]), int(self.E[e, 1])
            gj = self.G[j]
            gk = self.G[k]
            d = gk - gj  # (3,)

            Aj = A_all[j]
            tj = t_all[j]
            tk = t_all[k]

            # 3D vector residual for one edge.
            val = Aj @ d + gj + tj - (gk + tk)  # (3,)
            r[cursor:cursor+3] = sqrt_wreg * val

            # Derivatives wrt source node (j): matrix and translation.
            if self.indexer.is_free(j):
                bj = self.indexer.base(j)

                # row 0 residual depends on first row of Aj
                J[cursor + 0, bj + 0] = sqrt_wreg * d[0]
                J[cursor + 0, bj + 1] = sqrt_wreg * d[1]
                J[cursor + 0, bj + 2] = sqrt_wreg * d[2]

                J[cursor + 1, bj + 3] = sqrt_wreg * d[0]
                J[cursor + 1, bj + 4] = sqrt_wreg * d[1]
                J[cursor + 1, bj + 5] = sqrt_wreg * d[2]

                J[cursor + 2, bj + 6] = sqrt_wreg * d[0]
                J[cursor + 2, bj + 7] = sqrt_wreg * d[1]
                J[cursor + 2, bj + 8] = sqrt_wreg * d[2]

                # wrt tj
                J[cursor + 0, bj + 9]  = sqrt_wreg
                J[cursor + 1, bj + 10] = sqrt_wreg
                J[cursor + 2, bj + 11] = sqrt_wreg

            # Derivative wrt target node translation (k).
            if self.indexer.is_free(k):
                bk = self.indexer.base(k)
                J[cursor + 0, bk + 9]  = -sqrt_wreg
                J[cursor + 1, bk + 10] = -sqrt_wreg
                J[cursor + 2, bk + 11] = -sqrt_wreg

            cursor += 3

        reg_end = cursor

        # ----------------------------------------------------
        # 3) Constraint residuals
        # ----------------------------------------------------
        # Data term: selected handle vertices must move close to target positions.
        # Vertex deformation is blended from K nearby graph nodes by knn_w weights.
        con_start = cursor
        sqrt_wcon = math.sqrt(self.w_con)

        for p in range(len(self.handle_vertex_ids)):
            vid = int(self.handle_vertex_ids[p])
            v = self.V[vid]
            q = self.handle_target_pos[p]

            idxs = self.knn_idx[vid]
            ws = self.knn_w[vid]

            # Forward warp of one constrained vertex under current graph transforms.
            f = np.zeros(3, dtype=np.float64)

            for kk in range(len(idxs)):
                j = int(idxs[kk])
                w = float(ws[kk])

                gj = self.G[j]
                Aj = A_all[j]
                tj = t_all[j]
                d = v - gj

                f += w * (Aj @ d + gj + tj)

            # Residual in world space: predicted handle position minus target.
            val = f - q
            r[cursor:cursor+3] = sqrt_wcon * val

            for kk in range(len(idxs)):
                j = int(idxs[kk])
                w = float(ws[kk])

                if not self.indexer.is_free(j):
                    continue

                bj = self.indexer.base(j)
                d = v - self.G[j]

                # Jacobian wrt local affine block A_j.
                J[cursor + 0, bj + 0] += sqrt_wcon * w * d[0]
                J[cursor + 0, bj + 1] += sqrt_wcon * w * d[1]
                J[cursor + 0, bj + 2] += sqrt_wcon * w * d[2]

                J[cursor + 1, bj + 3] += sqrt_wcon * w * d[0]
                J[cursor + 1, bj + 4] += sqrt_wcon * w * d[1]
                J[cursor + 1, bj + 5] += sqrt_wcon * w * d[2]

                J[cursor + 2, bj + 6] += sqrt_wcon * w * d[0]
                J[cursor + 2, bj + 7] += sqrt_wcon * w * d[1]
                J[cursor + 2, bj + 8] += sqrt_wcon * w * d[2]

                # Jacobian wrt local translation t_j.
                J[cursor + 0, bj + 9]  += sqrt_wcon * w
                J[cursor + 1, bj + 10] += sqrt_wcon * w
                J[cursor + 2, bj + 11] += sqrt_wcon * w

            cursor += 3

        con_end = cursor

        assert cursor == R_total

        slices = {
            "rot": slice(rot_start, rot_end),
            "reg": slice(reg_start, reg_end),
            "con": slice(con_start, con_end),
        }

        return r, J, slices


# ============================================================
# Solver
# ============================================================

def evaluate_block_energies(r: np.ndarray, slices: Dict[str, slice]) -> Dict[str, float]:
    out = {}
    total = 0.0
    for k, s in slices.items():
        val = 0.5 * float(np.dot(r[s], r[s]))
        out[k] = val
        total += val
    out["total"] = total
    return out


def solve_linear_system(H: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    Solve H dx = g
    """
    try:
        dx = np.linalg.solve(H, g)
    except np.linalg.LinAlgError:
        dx = np.linalg.lstsq(H, g, rcond=None)[0]
    return dx


def solve_embedded_deformation_gauss_newton(
    vertices: np.ndarray,
    graph: DeformationGraph,
    handle_vertex_ids: np.ndarray,
    handle_target_pos: np.ndarray,
    w_rot: float = 1.0,
    w_reg: float = 10.0,
    w_con: float = 100.0,
    num_iters: int = 30,
    lm_lambda: float = 1e-4,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Gauss-Newton / Levenberg-Marquardt style solver:
        (J^T J + lambda I) dx = -J^T r
    """
    # Build the least-squares system provider (residuals + analytic Jacobian).
    system = EDGaussNewtonSystem(
        vertices=vertices,
        graph=graph,
        handle_vertex_ids=handle_vertex_ids,
        handle_target_pos=handle_target_pos,
        w_rot=w_rot,
        w_reg=w_reg,
        w_con=w_con
    )

    # Initialize unknowns: each free node starts from identity transform.
    x, indexer = initial_x_from_graph(graph)

    # Fast path: all nodes are fixed, no optimization variables.
    if indexer.num_vars == 0:
        A_all = np.tile(np.eye(3)[None], (len(graph.nodes), 1, 1))
        t_all = np.zeros((len(graph.nodes), 3), dtype=np.float64)
        deformed = deform_vertices(vertices, graph, A_all, t_all)
        return {
            "x": x,
            "A": A_all,
            "t": t_all,
            "deformed_vertices": deformed
        }

    # LM damping controls step aggressiveness:
    # larger lambda -> closer to gradient descent, smaller -> Gauss-Newton.
    lam = float(lm_lambda)

    r, J, slices = system.build(x)
    energy = evaluate_block_energies(r, slices)

    if verbose:
        print(f"[Init] total={energy['total']:.8f}, rot={energy['rot']:.8f}, "
              f"reg={energy['reg']:.8f}, con={energy['con']:.8f}")

    for it in range(num_iters):
        # Normal equation for least squares: (J^T J + lambda I) dx = -J^T r.
        H = J.T @ J
        g = J.T @ r

        H_lm = H + lam * np.eye(H.shape[0], dtype=np.float64)
        dx = solve_linear_system(H_lm, -g)

        # Trial update.
        x_new = x + dx
        r_new, J_new, slices_new = system.build(x_new)
        energy_new = evaluate_block_energies(r_new, slices_new)

        accepted = energy_new["total"] < energy["total"]

        if accepted:
            # Accept and slightly decrease damping when the objective improves.
            x = x_new
            r, J, slices = r_new, J_new, slices_new
            energy = energy_new
            lam = max(lam * 0.5, 1e-8)
        else:
            # Reject and increase damping to get a more conservative next step.
            lam = min(lam * 10.0, 1e8)

        if verbose:
            step_norm = float(np.linalg.norm(dx))
            flag = "accept" if accepted else "reject"
            print(
                f"[Iter {it:03d}] {flag} "
                f"total={energy['total']:.8f}, "
                f"rot={energy['rot']:.8f}, reg={energy['reg']:.8f}, con={energy['con']:.8f}, "
                f"|dx|={step_norm:.6e}, lambda={lam:.3e}"
            )

    A_all, t_all = unpack_all_transforms(x, indexer, len(graph.nodes))
    deformed = deform_vertices(vertices, graph, A_all, t_all)

    return {
        "x": x,
        "A": A_all,
        "t": t_all,
        "deformed_vertices": deformed
    }


# ============================================================
# Forward deformation
# ============================================================

def deform_vertices(
    vertices: np.ndarray,
    graph: DeformationGraph,
    A_all: np.ndarray,
    t_all: np.ndarray
) -> np.ndarray:
    V = vertices.astype(np.float64)
    G = graph.nodes.astype(np.float64)
    knn_idx = graph.knn_idx
    knn_w = graph.knn_w

    N, K = knn_idx.shape
    out = np.zeros((N, 3), dtype=np.float64)

    for i in range(N):
        v = V[i]
        acc = np.zeros(3, dtype=np.float64)
        # Embedded deformation blend:
        # v' = sum_j w_ij * (A_j (v - g_j) + g_j + t_j)
        # where j traverses the K supporting nodes of vertex i.
        for kk in range(K):
            j = int(knn_idx[i, kk])
            w = float(knn_w[i, kk])
            gj = G[j]
            Aj = A_all[j]
            tj = t_all[j]
            acc += w * (Aj @ (v - gj) + gj + tj)
        out[i] = acc
    return out


def deform_normals(
    normals: np.ndarray,
    graph: DeformationGraph,
    A_all: np.ndarray
) -> np.ndarray:
    """
    Approximate normal transform:
        n'_i = sum_j w_ij * A_j^{-T} n_i
    then normalize.
    """
    N = normals.shape[0]
    K = graph.knn_idx.shape[1]
    out = np.zeros((N, 3), dtype=np.float64)

    for i in range(N):
        n = normals[i]
        acc = np.zeros(3, dtype=np.float64)
        for kk in range(K):
            j = int(graph.knn_idx[i, kk])
            w = float(graph.knn_w[i, kk])
            Aj = A_all[j]
            try:
                n2 = np.linalg.inv(Aj).T @ n
            except np.linalg.LinAlgError:
                n2 = n
            acc += w * n2
        norm = np.linalg.norm(acc) + 1e-12
        out[i] = acc / norm

    return out


# ============================================================
# I/O helpers
# ============================================================

def load_mesh_or_pointcloud(path: str) -> MeshData:
    """
    Supports:
      - mesh via open3d/trimesh
      - point cloud via open3d
    """

    try:
        mesh = o3d.io.read_triangle_mesh(path)
        if len(mesh.vertices) > 0:
            V = np.asarray(mesh.vertices, dtype=np.float64)
            F = np.asarray(mesh.triangles, dtype=np.int64)
            C = None
            # Keep vertex colors so they can be written back after deformation.
            if mesh.has_vertex_colors() and len(mesh.vertex_colors) == len(V):
                C = np.asarray(mesh.vertex_colors, dtype=np.float64)
            if len(F) == 0:
                F = None
            return MeshData(vertices=V, faces=F, colors=C)
    except Exception:
        print(f"Failed to load as mesh: {path}")

    try:
        pcd = o3d.io.read_point_cloud(path)
        if len(pcd.points) > 0:
            V = np.asarray(pcd.points, dtype=np.float64)
            C = None
            if pcd.has_colors() and len(pcd.colors) == len(V):
                C = np.asarray(pcd.colors, dtype=np.float64)
            return MeshData(vertices=V, faces=None, colors=C)
    except Exception:
        print(f"Failed to load as point cloud: {path}")

    # if trimesh is not None:
    #     try:
    #         obj = trimesh.load(path, process=False)
    #         if isinstance(obj, trimesh.Trimesh):
    #             V = np.asarray(obj.vertices, dtype=np.float64)
    #             F = np.asarray(obj.faces, dtype=np.int64)
    #             if len(F) == 0:
    #                 F = None
    #             return MeshData(vertices=V, faces=F)
    #         elif hasattr(obj, "geometry") and len(obj.geometry) > 0:
    #             # scene -> merge
    #             meshes = []
    #             for _, g in obj.geometry.items():
    #                 if isinstance(g, trimesh.Trimesh):
    #                     meshes.append(g)
    #             if len(meshes) > 0:
    #                 merged = trimesh.util.concatenate(meshes)
    #                 V = np.asarray(merged.vertices, dtype=np.float64)
    #                 F = np.asarray(merged.faces, dtype=np.int64)
    #                 if len(F) == 0:
    #                     F = None
    #                 return MeshData(vertices=V, faces=F)
    #     except Exception:
    #         pass

    # raise RuntimeError(f"Failed to load file: {path}")


def save_mesh_or_pointcloud(
    path: str,
    vertices: np.ndarray,
    faces: Optional[np.ndarray] = None,
    colors: Optional[np.ndarray] = None,
):
    color_ok = colors is not None and len(colors) == len(vertices)

    if o3d is not None:
        if faces is not None:
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            if color_ok:
                mesh.vertex_colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
            ok = o3d.io.write_triangle_mesh(path, mesh)
            if ok:
                return
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vertices)
            if color_ok:
                pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
            ok = o3d.io.write_point_cloud(path, pcd)
            if ok:
                return

    if trimesh is not None:
        color_u8 = None
        if color_ok:
            c = np.asarray(colors)
            if np.issubdtype(c.dtype, np.floating):
                c = np.clip(c, 0.0, 1.0) * 255.0
            c = np.clip(c, 0, 255).astype(np.uint8)
            if c.ndim == 2 and c.shape[1] == 3:
                alpha = np.full((c.shape[0], 1), 255, dtype=np.uint8)
                c = np.concatenate([c, alpha], axis=1)
            color_u8 = c

        if faces is not None:
            tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            if color_u8 is not None:
                tm.visual.vertex_colors = color_u8
            tm.export(path)
            return
        else:
            pc = trimesh.points.PointCloud(vertices, colors=color_u8)
            pc.export(path)
            return

    raise RuntimeError("No available writer. Install open3d or trimesh.")


# ============================================================
# Example
# ============================================================

def example_point_cloud(mesh: Optional[MeshData] = None):

    pts = mesh.vertices

    graph = build_deformation_graph(
        mesh,
        method="uniform",
        num_nodes=120,
        k=4,
        fixed_vertex_ids=np.arange(0, 300),
        seed=0
    )

    # Select a few vertex indices as deformation handles to be position-constrained.
    handle_ids = np.array([1000, 1200, 1500, 1800], dtype=np.int64)
    # Move each handle upward along +Y by 0.02 to form the target positions.
    handle_targets = pts[handle_ids] + np.array([0.0, 0.02, 0.0])[None, :]

    result = solve_embedded_deformation_gauss_newton(
        vertices=pts,
        graph=graph,
        handle_vertex_ids=handle_ids,
        handle_target_pos=handle_targets,
        w_rot=1.0,
        w_reg=10.0,
        w_con=100.0,
        num_iters=20,
        lm_lambda=1e-4,
        verbose=True
    )

    return result


def example_mesh(vertices: np.ndarray, faces: np.ndarray):
    mesh = MeshData(vertices=vertices, faces=faces)

    graph = build_deformation_graph(
        mesh,
        method="simplify",
        num_nodes=150,
        k=4,
        fixed_vertex_ids=None,
        seed=0
    )

    handle_ids = np.array([0, 10, 20], dtype=np.int64)
    handle_targets = vertices[handle_ids] + np.array([[0, 0.2, 0],
                                                      [0, 0.2, 0],
                                                      [0, 0.2, 0]], dtype=np.float64)

    result = solve_embedded_deformation_gauss_newton(
        vertices=vertices,
        graph=graph,
        handle_vertex_ids=handle_ids,
        handle_target_pos=handle_targets,
        w_rot=1.0,
        w_reg=10.0,
        w_con=100.0,
        num_iters=20,
        lm_lambda=1e-4,
        verbose=True
    )
    return result


# ============================================================
# Command-line style helper
# ============================================================

def run_deformation(
    input_path: str,
    output_path: str,
    method: str = "uniform",
    num_nodes: int = 150,
    k: int = 4,
    fixed_vertex_ids: Optional[np.ndarray] = None,
    handle_vertex_ids: Optional[np.ndarray] = None,
    handle_target_pos: Optional[np.ndarray] = None,
    radius: Optional[float] = None,
    w_rot: float = 1.0,
    w_reg: float = 10.0,
    w_con: float = 100.0,
    num_iters: int = 20,
    lm_lambda: float = 1e-4,
):
    data = load_mesh_or_pointcloud(input_path)

    if handle_vertex_ids is None or handle_target_pos is None:
        raise ValueError("handle_vertex_ids and handle_target_pos must be provided.")

    graph = build_deformation_graph(
        data,
        method=method,
        num_nodes=num_nodes,
        radius=radius,
        k=k,
        fixed_vertex_ids=fixed_vertex_ids,
        seed=0
    )

    result = solve_embedded_deformation_gauss_newton(
        vertices=data.vertices,
        graph=graph,
        handle_vertex_ids=handle_vertex_ids,
        handle_target_pos=handle_target_pos,
        w_rot=w_rot,
        w_reg=w_reg,
        w_con=w_con,
        num_iters=num_iters,
        lm_lambda=lm_lambda,
        verbose=True
    )

    save_mesh_or_pointcloud(
        output_path,
        result["deformed_vertices"],
        data.faces,
        data.colors,
    )

    return result


if __name__ == "__main__":
    mesh_path = "/workspace/Any6D/results/demo_mustard/refine_init_mesh_demo.obj"  # Replace with your input file path
    # output_path = "output.obj"  # Replace with your desired output file path
    mesh = load_mesh_or_pointcloud(mesh_path)
    result = example_point_cloud(mesh)

    output_path = "/workspace/InstantMesh/outputs/instant-mesh-large/demo_deformed.obj"
    save_mesh_or_pointcloud(output_path, result["deformed_vertices"], mesh.faces, mesh.colors)