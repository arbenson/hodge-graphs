import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    # Python equivalent imports
    import marimo
    import numpy as np
    import scipy.sparse as sp
    from scipy.sparse.linalg import lsqr
    from itertools import combinations
    import time
    import re
    return combinations, lsqr, np, re, sp, time


@app.cell
def _(np, sp):
    # spones equivalent
    def spones(A):
        """Convert sparse matrix to one where all non-zero entries become 1."""
        A_coo = A.tocoo()
        data = np.ones_like(A_coo.data, dtype=A_coo.data.dtype)
        return sp.coo_matrix(
            (data, (A_coo.row, A_coo.col)), shape=A_coo.shape
        ).tocsc()
    return (spones,)


@app.cell
def _(re, sp):
    # CORRECTED: Function to read .paj files
    def read_paj_sparse(fname):
        with open(fname, "r") as f:
            content = f.read()

        sections = content.split("*network")
        labels_content = sections[1] if len(sections) > 1 else content
        n_match = re.search(r"\*vertices\s+(\d+)", labels_content)
        n = int(n_match.group(1))
        labels = re.findall(r'\s*\d+\s+"([^"]+)"', labels_content)

        I, J, V = [], [], []
        arcs_match = re.search(r"\*arcs(.*?)(?:\*|$)", content, re.DOTALL)
        if arcs_match:
            arcs_str = arcs_match.group(1).strip()
            for line in arcs_str.split("\n"):
                parts = line.split()
                if len(parts) >= 3:
                    I.append(int(parts[0]) - 1)
                    J.append(int(parts[1]) - 1)
                    V.append(float(parts[2]))

        A = sp.csc_matrix((V, (I, J)), shape=(n, n), dtype=float)
        return A, labels
    return (read_paj_sparse,)


@app.cell
def _(sp):
    # Gradient matrix from graph adjacency matrix
    def grad_mat(A):
        edge_map = {}
        I, J, V = [], [], []
        curr_edge_ind = 0
        num_vertices = A.shape[1]

        rows, cols = sp.triu(A, k=1).nonzero()
        for i, j in zip(rows, cols):
            edge_map[(i, j)] = curr_edge_ind
            I.extend([curr_edge_ind, curr_edge_ind])
            J.extend([i, j])
            V.extend([-1, 1])
            curr_edge_ind += 1

        grad = sp.csc_matrix(
            (V, (I, J)), shape=(len(edge_map), num_vertices), dtype=float
        )
        return grad, edge_map
    return (grad_mat,)


@app.cell
def _(combinations, np, sp, spones):
    # Curl matrix from graph adjacency matrix
    def curl_mat(A, edge_map):
        num_vertices = A.shape[1]
        degrees = np.array(spones(A).sum(axis=0)).flatten()
        deg_order = np.argsort(degrees)
        deg_rank = np.empty_like(deg_order)
        deg_rank[deg_order] = np.arange(num_vertices)

        tri_map = {}
        curr_tri_ind = 0
        I, J, V = [], [], []

        for i in range(num_vertices):
            pos = deg_rank[i]
            neighbors = A[i, :].nonzero()[1]
            higher_deg_neighbors = [v for v in neighbors if deg_rank[v] > pos]

            for j, k in combinations(higher_deg_neighbors, 2):
                if A[j, k] > 0.0:
                    a, b, c = sorted([i, j, k])
                    if (
                        (a, b) in edge_map
                        and (a, c) in edge_map
                        and (b, c) in edge_map
                    ):
                        tri_map[(a, b, c)] = curr_tri_ind
                        I.extend([curr_tri_ind, curr_tri_ind, curr_tri_ind])
                        J.extend([
                            edge_map[(a, b)],
                            edge_map[(a, c)],
                            edge_map[(b, c)],
                        ])
                        V.extend([1, -1, 1])
                        curr_tri_ind += 1

        curl = sp.csc_matrix(
            (V, (I, J)), shape=(len(tri_map), len(edge_map)), dtype=float
        )
        return curl, tri_map
    return (curl_mat,)


@app.cell
def _(lsqr):
    # Hodge decomposition function
    def hodge_decomp(X, grad, curl):
        f = lsqr(grad, X, atol=1e-10, btol=1e-10, iter_lim=1000)[0]
        Phi = lsqr(curl.T, X, atol=1e-10, btol=1e-10, iter_lim=1000)[0]
        X_H = X - curl.T.dot(Phi) - grad.dot(f)
        return f, Phi, X_H
    return (hodge_decomp,)


@app.cell
def _(curl_mat, grad_mat, read_paj_sparse, time):
    # Read data and build matrices, with timing
    _start_time = time.time()
    A, labels = read_paj_sparse("Florida.paj")
    print(f"Read data: {time.time() - _start_time:.6f} seconds")

    Asym = A.maximum(A.T)
    _start_time = time.time()
    grad, edge_map = grad_mat(Asym)
    print(f"Gradient matrix construction: {time.time() - _start_time:.6f} seconds")

    _start_time = time.time()
    curl, tri_map = curl_mat(Asym, edge_map)
    print(f"Curl matrix construction: {time.time() - _start_time:.6f} seconds")

    curlgrad = curl.dot(grad)
    print(
        f"Sanity check: min(curl @ grad) = {curlgrad.min():.1f}, max(curl @ grad) = {curlgrad.max():.1f}"
    )
    return A, curl, edge_map, grad, labels, tri_map


@app.cell
def _(np, sp):
    # Edge flow extraction function
    def edge_flow(T, edge_map):
        X = np.zeros(len(edge_map))
        B = sp.triu(T, k=1).tocoo()
        for i, j, v in zip(B.row, B.col, B.data):
            if (i, j) in edge_map:
                X[edge_map[(i, j)]] = v
        return X
    return (edge_flow,)


@app.cell
def _(A, curl, edge_flow, edge_map, grad, hodge_decomp, time):
    # Standard decomposition with timing
    _start_time = time.time()
    X = edge_flow(A - A.T, edge_map)
    print(f"Edge flow calculation: {time.time() - _start_time:.6f} seconds")

    _start_time = time.time()
    f, Phi, X_H = hodge_decomp(X, grad, curl)
    print(f"Hodge decomposition: {time.time() - _start_time:.6f} seconds")

    x1 = grad.dot(f)
    x2 = curl.T.dot(Phi)
    print(
        f"Orthogonality check (should be near zero): ({x1.dot(x2):.2e}, {x1.dot(X_H):.2e}, {x2.dot(X_H):.2e})"
    )
    return Phi, X, f


@app.cell
def _(X, f, grad, labels, np):
    # Potential function ranking and analysis
    ranking = np.array(labels)[np.argsort(f)[::-1]]
    print(f"Top 5 ranked nodes: {[str(_) for _ in ranking[:5]]}")
    print(f"Bottom 5 ranked nodes: {[str(_) for _ in ranking[-5:]]}")

    frac_potential = np.linalg.norm(grad.dot(f)) ** 2 / np.linalg.norm(X) ** 2
    print(f"Fraction of flow from potential: {frac_potential:.4f}")
    return


@app.cell
def _(A, curl, edge_flow, edge_map, grad, hodge_decomp, labels, np, time):
    # Sign-only analysis with timing
    A_diff = A - A.T
    A_diff.data = np.sign(A_diff.data)
    X_sign = edge_flow(A_diff, edge_map)

    _start_time = time.time()
    f_sign, _, _ = hodge_decomp(X_sign, grad, curl)
    print(f"Sign-only decomposition: {time.time() - _start_time:.6f} seconds")

    ranking_sign = np.array(labels)[np.argsort(f_sign)[::-1]]
    print(f"Top 5 (sign-only): {[str(_) for _ in ranking_sign[:5]]}")
    print(f"Bottom 5 (sign-only): {[str(_) for _ in ranking_sign[-5:]]}")

    frac_potential_sign = (
        np.linalg.norm(grad.dot(f_sign)) ** 2 / np.linalg.norm(X_sign) ** 2
    )
    print(f"Fraction from potential (sign-only): {frac_potential_sign:.4f}")
    return


@app.cell
def _(
    A,
    curl,
    edge_flow,
    edge_map,
    grad,
    hodge_decomp,
    labels,
    np,
    spones,
    time,
):
    # Asymmetric flow analysis with timing
    A1 = spones(A)
    B = A1.multiply(A1.T)
    U = A1 - B

    M = U - U.T
    M.data = np.sign(M.data)
    X_asym = edge_flow(M, edge_map)

    _start_time = time.time()
    f_asym, Phi_asym, _ = hodge_decomp(X_asym, grad, curl)
    print(
        f"Asymmetric flows decomposition: {time.time() - _start_time:.6f} seconds"
    )

    ranking_asym = np.array(labels)[np.argsort(f_asym)[::-1]]
    print(f"Top 5 (asymmetric): {[str(_) for _ in ranking_asym[:5]]}")
    print(f"Bottom 5 (asymmetric): {[str(_) for _ in ranking_asym[-5:]]}")

    frac_potential_asym = (
        np.linalg.norm(grad.dot(f_asym)) ** 2 / np.linalg.norm(X_asym) ** 2
    )
    print(f"Fraction from potential (asymmetric): {frac_potential_asym:.4f}")
    return


@app.cell
def _(Phi, labels, np, tri_map):
    # Find and display inconsistent triangles
    ordered_tris = np.array(list(tri_map.keys()))[
        np.argsort(list(tri_map.values()))
    ]
    sorted_by_phi = ordered_tris[np.argsort(np.abs(Phi))[::-1]]

    print("Top 5 inconsistent triangles:")
    for i in range(5):
        triangle_labels = np.array(labels)[list(sorted_by_phi[i])]
        print(f"- {[str(_) for _ in triangle_labels]}")
    return


@app.cell
def _(A, labels, np):
    # Analyze connections for "Parrotfish"
    pf_idx = labels.index("Parrotfish")
    incoming_indices = A[:, pf_idx].nonzero()[0]
    outgoing_indices = A[pf_idx, :].nonzero()[1]

    incoming_labels = list(np.array(labels)[incoming_indices])
    outgoing_labels = list(np.array(labels)[outgoing_indices])

    print(f"Parrotfish eats: {[str(_) for _ in incoming_labels]}")
    print(f"Parrotfish is eaten by: {[str(_) for _ in outgoing_labels]}")
    return


@app.cell
def _(
    curl_mat,
    edge_flow,
    grad_mat,
    hodge_decomp,
    np,
    read_paj_sparse,
    spones,
    time,
):
    # Analysis of Michigan.paj (will be skipped if file not found)
    try:
        start_time = time.time()
        A_mich, labels_mich = read_paj_sparse("Michigan.paj")
        print(f"Read Michigan data: {time.time() - start_time:.6f} seconds")

        Asym_mich = A_mich.maximum(A_mich.T)
        grad_mich, edge_map_mich = grad_mat(Asym_mich)
        curl_mich, _ = curl_mat(Asym_mich, edge_map_mich)

        A1_mich = spones(A_mich)
        B_mich = A1_mich.multiply(A1_mich.T)
        U_mich = A1_mich - B_mich
        M_mich = U_mich - U_mich.T
        M_mich.data = np.sign(M_mich.data)
        X_mich = edge_flow(M_mich, edge_map_mich)

        start_time = time.time()
        f_mich, _, _ = hodge_decomp(X_mich, grad_mich, curl_mich)
        print(f"Michigan decomposition: {time.time() - start_time:.6f} seconds")

        ranking_mich = np.array(labels_mich)[np.argsort(f_mich)[::-1]]
        frac_potential_mich = (
            np.linalg.norm(grad_mich.dot(f_mich)) ** 2
            / np.linalg.norm(X_mich) ** 2
        )

        print("\n--- Michigan Data Analysis ---")
        print(f"Top 5 (Michigan): {[str(_) for _ in ranking_mich[:5]]}")
        print(f"Bottom 5 (Michigan): {[str(_) for _ in ranking_mich[-5:]]}")
        print(f"Fraction from potential (Michigan): {frac_potential_mich:.4f}")
    except FileNotFoundError:
        print("\nAnalysis of Michigan.paj skipped: File not found.")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
