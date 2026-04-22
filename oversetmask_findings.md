# Overset Mask Findings: Dirichlet Treatment & AMReX Overset Mask

## 1. How Dirichlet Boundaries Are Treated in the Curl-Curl Solver

The equation is `curl(alpha * curl(E)) + beta * E = f`, discretised on staggered (edge-centred) grids.

**Index types (3D):**
- `E_x` at `(0,1,1)` — cell-centred in x, nodal in y,z
- `E_y` at `(1,0,1)` — cell-centred in y, nodal in x,z
- `E_z` at `(1,1,0)` — cell-centred in z, nodal in x,y

**Which edges are Dirichlet?**
Component `E_d` is Dirichlet on a boundary face with normal `n` if and only if `E_d` is *nodal* in the `n`-direction. Concretely:
- `E_x` is Dirichlet at y-boundaries and z-boundaries (but NOT at x-boundaries, since it's cell-centred in x)
- `E_y` is Dirichlet at x-boundaries and z-boundaries
- `E_z` is Dirichlet at x-boundaries and y-boundaries

This is encoded in `CurlCurlDirichletInfo`, which stores `dirichlet_lo[d]` and `dirichlet_hi[d]` — the nodal-index coordinates of each Dirichlet face. Non-Dirichlet faces use sentinel values (`int::lowest()` / `int::max()`) so checks never fire.

### A. Unigrid (no AMR, no MG)

The solver implements Dirichlet BCs by treating boundary edges as **identity rows** in the discrete operator. In pseudocode:

**Matrix-vector product `A*x`:**
```
for each edge (d,i,j,k):
    if is_dirichlet_edge(d,i,j,k):
        (A*x)_{d,i,j,k} = 0       // identity row (implicitly: 1 * x = x_prescribed)
    else:
        (A*x)_{d,i,j,k} = [curl(alpha*curl(x)) + beta*x]_{d,i,j,k}   // normal stencil
```

**Residual `r = f - A*x`:**
```
for each edge (d,i,j,k):
    if is_dirichlet_edge(d,i,j,k):
        r_{d,i,j,k} = 0            // Dirichlet is exactly satisfied
    else:
        r_{d,i,j,k} = f_{d,i,j,k} - (A*x)_{d,i,j,k}
```

**Smoother (4-colour Gauss-Seidel):**
The GS4 smoother groups 6 unknowns around each node
(`v = [Ex(i-1,j,k), Ex(i,j,k), Ey(i,j-1,k), Ey(i,j,k), Ez(i,j,k-1), Ez(i,j,k)]`)
and solves a local 6x6 system `A_local * v = b_local`.

```
for each node (i,j,k) of the current colour:
    if is_dirichlet_node(i,j,k):
        return   // skip entirely — don't modify any Dirichlet edges
    else:
        assemble 6x6 local system
        solve via precomputed LU factorisation
        update the 6 surrounding edge values
```

**`setDirichletNodesToZero` (called by MLMG on the correction):**
```
for each MultiFab in the Array<MultiFab,3>:
    for each boundary face with Dirichlet BC:
        if field is nodal in the face-normal direction:
            set field = 0 on that boundary face
```

This ensures the correction satisfies *homogeneous* Dirichlet BCs.

**Mathematical summary (unigrid):**
The discrete system is:

    [ I    0  ] [ E_boundary ] = [ E_prescribed ]
    [ S_ib L  ] [ E_interior ]   [ f_interior   ]

where `I` is the identity on Dirichlet edges, `L` is the interior curl-curl + beta stencil, and `S_ib` couples interior equations to boundary values. The residual at Dirichlet edges is always 0, the smoother never touches them, and the correction is always 0 there.

### B. No AMR, but MG

The MG V-cycle adds restriction (fine->coarse) and prolongation (coarse->fine) operators. All operations preserve the Dirichlet structure.

**Restriction `r_coarse = R(r_fine)`:**
```
for each coarse edge (d,i,j,k):
    (ii,jj,kk) = (2i, 2j, 2k)   // corresponding fine-grid index
    if is_dirichlet_edge(d, ii, jj, kk):
        r_coarse(i,j,k) = 0      // residual at Dirichlet edges is 0
    else:
        r_coarse(i,j,k) = weighted_average of fine-grid values
```

The restriction stencil is a standard staggered-grid weighted average. For E_x (cell-centred in x, nodal in y,z), the stencil averages over the y,z nodal directions with weights [1,2,1]/4 and sums the two x-offset fine values, giving a 2x3x3 stencil in 3D (total weight 1/32).

Key point: the stencil is dimension-specific — it averages in the *nodal* directions of that component but sums in the *cell-centred* direction (since two fine cells map to one coarse cell in that direction).

**Prolongation (interpolation of correction) `e_fine += P(e_coarse)`:**
```
for each fine edge (d,i,j,k):
    if is_dirichlet_edge(d, i, j, k):
        // skip — do NOT add correction at Dirichlet edges
    else:
        e_fine(i,j,k) += bilinear_interp(e_coarse, i, j, k)
```

The prolongation uses bilinear interpolation in the nodal directions of each component.

**Apply and smooth:** Identical to unigrid, but `getDirichletInfo(amrlev, mglev)` adjusts boundary indices for each MG level's geometry (the domain shrinks as we coarsen).

**V-cycle pseudocode (single-level MG):**
```
mgVcycle(mglev_top):
    for mglev = top to bottom-1:     // down-leg
        cor[mglev] = 0
        smooth(cor[mglev], res[mglev])      // Dirichlet edges untouched
        rescor = res[mglev] - A*cor[mglev]  // Dirichlet edges -> 0
        res[mglev+1] = R(rescor)            // Dirichlet edges -> 0

    bottom_solve(cor[bottom], res[bottom])

    for mglev = bottom-1 to top:      // up-leg
        cor[mglev] += P(cor[mglev+1])       // skip Dirichlet edges in prolongation
        smooth(cor[mglev], res[mglev])      // Dirichlet edges untouched
```

Throughout, the correction at Dirichlet edges remains 0, so the prescribed boundary values are preserved exactly.

### C. MG with AMR

AMR adds coarse-fine (C/F) boundaries, handled via a **ghost-nodes strategy**.

**C/F boundary masks (`m_cfmask`):**
Three `iMultiFab`s (one per edge component), with values:
- 0 = valid or interior ghost cell
- 1 = layer-1 C/F ghost (one cell into the coarse-grid region) — **relaxed** by the smoother
- 2 = layer-2 C/F ghost (two cells into coarse-grid region) — **frozen** (Dirichlet-like, set from interpolated coarse data or zero)

**C/F boundary data:**
Coarse-level E data is interpolated to fine resolution and stored in `m_cf_data[amrlev]`. This provides inhomogeneous C/F boundary values.

**`applyCFBC` pseudocode:**
```
for each edge component d:
    for each ghost cell of fine level amrlev:
        if cfmask == 1 (layer-1):
            if inhomogeneous (mglev == 0, solution mode):
                fill from interpolated coarse data
            // else: leave as-is (will be relaxed)
        if cfmask == 2 (layer-2):
            if inhomogeneous:
                fill from interpolated coarse data
            else:
                set to 0 (homogeneous Dirichlet)
```

**`apply` with C/F boundaries:**
When computing `A*x` in correction mode on a fine AMR level:
1. The iteration box is grown by 1 to include layer-1 ghosts (so the operator is computed there too).
2. A **Galerkin diagonal correction** is applied at edges whose stencil reaches into layer-2 (frozen) ghost cells. Since those ghosts are zero, the off-diagonal coupling to them vanishes, and the diagonal must be compensated:
```
for each edge (d,i,j,k) in the grown box:
    if any neighbour n of the stencil has cfmask(n) == 2:
        (A*x)_{d,i,j,k} += alpha_face * (1/dx^2) * x_{d,i,j,k}
            // for each such frozen neighbour
```

**Smoother with C/F boundaries:**
The smoothing box is grown by 1 to include layer-1 C/F ghosts, so they are relaxed by the GS4 smoother just like interior edges.

**Composite V-cycle (`oneIter`):**
```
// Down sweep: fine AMR levels to coarse
for alev = finest to 1:
    miniCycle(alev)                 // MG V-cycle on this AMR level
    sol[alev] += cor[alev]
    computeResWithCrseSolFineCor(alev-1, alev)  // coarse residual accounting for fine correction

// Coarsest AMR level
mgVcycle(0, 0) or mgFcycle()
sol[0] += cor[0]

// Up sweep: coarse AMR levels to fine
for alev = 1 to finest:
    cor[alev] = P_amr(cor[alev-1])   // interpolate AMR correction
    sol[alev] += cor[alev]
    computeResWithCrseCorFineCor(alev) // update residual
    miniCycle(alev)                    // post-smoothing V-cycle
    sol[alev] += cor[alev]

averageDownAndSync(sol)
```

**Reflux / syncCFResidual:**
At C/F boundaries, the residual on the coarse level is corrected using a direct composite stencil. For each flux direction at a coarse node near the C/F boundary:
- Fine-covered neighbours use the fine-level flux (fine alpha, fine E, fine dx)
- Non-covered neighbours use the coarse-level flux

Domain-boundary Dirichlet BCs carry through all levels unchanged — `getDirichletInfo(amrlev, mglev)` simply adjusts the boundary index coordinates for each level's geometry.

---

## 2. How the Existing AMReX Overset Mask Implementation Works

AMReX has two overset-mask implementations: one in cell-centred solvers (`MLCellABecLap`), and one in nodal solvers (`MLNodeLinOp`). Both use the convention:
- mask = 1: **unknown** (solve for this DOF)
- mask = 0: **known** (prescribed value, treated as internal Dirichlet)

### Cell-Centred Solver (MLCellABecLap / MLABecLaplacian)

This solver handles `a*alpha*phi - b*beta*div(B*grad(phi)) = rhs`, with the overset mask specifying internal Dirichlet DOFs on the cell-centred grid.

#### A. Unigrid (no AMR, no MG)

**Setup (called once before solve):**

1. **`applyOverset`** — called from `MLMG::prepareForSolve`:
```
for each cell (i,j,k):
    if mask(i,j,k) == 0:
        rhs(i,j,k) = 0
```

2. **B-coefficient rescaling** at faces between known and unknown cells (only on coarsened MG levels; not needed at mglev=0):
```
for each face (d,i,j,k):
    if exactly one of the two adjacent cells is known:
        B_face *= osfac
```
(At mglev=0, osfac=1 so no rescaling.)

**During the solve:**

3. **`adotx_os` (matrix-vector product):**
```
for each cell (i,j,k):
    if mask(i,j,k) == 0:
        (A*phi)(i,j,k) = 0       // identity row
    else:
        (A*phi)(i,j,k) = normal stencil
```

4. **`gsrb_os` (Gauss-Seidel Red-Black smoother):**
```
for each cell (i,j,k) of current colour:
    if mask(i,j,k) == 0:
        phi(i,j,k) = 0           // pin correction to zero
    else:
        phi(i,j,k) = standard GS update
```

**How it preserves prescribed values:**
- The MLMG smoother operates on the *correction* (not the solution directly)
- The correction at known cells is pinned to 0
- Therefore `sol = initial_guess + correction` never changes at known cells
- The user provides prescribed values in the initial guess
- `rhs = 0` and `A*x = 0` at known cells => residual = 0 there
- Unknown cells adjacent to known cells see the prescribed values through the normal stencil, coupling them correctly

**Mathematical model:**
Identical in structure to the domain Dirichlet treatment: known cells get identity rows, the RHS is zeroed, and the correction is pinned to zero. The difference is that these constraints are *interior* rather than at domain boundaries.

#### B. No AMR, but MG

**Mask coarsening:**
The mask is coarsened from fine MG level to coarse. For cell-centred data, each coarse cell corresponds to 2^d fine cells:
```
coarse_mask(i,j,k) = sum of fine_mask values in the 2^d refinement patch
if coarse_mask == 2^d:      // all fine cells unknown
    coarse_mask = 1
else if coarse_mask == 0:   // all fine cells known
    coarse_mask = 0
else:                        // MIXED — error at amrlev=0
    coarse_mask = partial sum (treated as error, stops coarsening)
```

**Coarsening stops** when mixed cells are encountered. The maximum MG coarsening level is limited to the deepest level with no mixed cells. Implication: the overset region boundary must be aligned with the MG coarsening hierarchy (every 2^mglev cells).

**B-coefficient rescaling on coarser MG levels:**
At coarser MG levels, the effective distance from a face to the "overset boundary" changes. The face B-coefficient between a known and unknown coarse cell is rescaled:
```
fac = 2^mglev
osfac = 2 * fac / (fac + 1)
```

| mglev | fac | osfac |
|-------|-----|-------|
| 0     | 1   | 1.0   |
| 1     | 2   | 4/3   |
| 2     | 4   | 8/5   |
| 3     | 8   | 16/9  |

This corrects the second-order gradient stencil for the actual distance between the coarse cell centre and the effective overset boundary, which sits at a sub-cell position on coarsened grids.

**Smoother and adotx:** The `_os` variants are used at every MG level, checking the coarsened mask.

**V-cycle pseudocode with overset:**
Same as standard V-cycle, but:
- `adotx` uses `_os` variant (identity row at mask=0)
- Smoother uses `_os` variant (pins correction to 0 at mask=0)
- Restriction naturally produces 0 residual at coarsened known cells (since the fine residual was already 0 there)
- Prolongation adds the coarse correction to the fine correction; at known cells, both are 0

#### C. MG with AMR

**Mask on finer AMR levels:**
Each AMR level maintains its own overset mask, coarsened through its MG hierarchy.

**`applyOverset` is called once per AMR level** during setup:
```
for alev = 0 to finest:
    for each cell (i,j,k) at mglev=0:
        if mask[alev](i,j,k) == 0:
            rhs[alev](i,j,k) = 0
```

**Composite solve:**
The overset mask operates independently on each AMR level. The composite V-cycle structure is unchanged — overset cells act as internal Dirichlet constraints that pin the solution. The C/F boundary treatment is orthogonal to the overset treatment.

### Nodal Solver (MLNodeLinOp)

The nodal solver takes a fundamentally different approach: **the overset mask is merged directly into the Dirichlet mask**.

#### A. Unigrid (no AMR, no MG)

**`setOversetMask`:**
```
for each node (i,j,k):
    dirichlet_mask(i,j,k) = 1 - overset_mask(i,j,k)
    // overset_mask: 1=unknown, 0=known
    // dirichlet_mask: 0=unknown, 1=Dirichlet
m_overset_dirichlet_mask = true
```

From this point on, overset nodes are treated identically to domain-boundary Dirichlet nodes.

**`solutionResidual`:**
```
A*x = Fapply(x)     // standard nodal Laplacian
for each node (i,j,k):
    if dirichlet_mask(i,j,k) == 1:   // known (domain BC or overset)
        resid(i,j,k) = 0
    else:
        resid(i,j,k) = rhs(i,j,k) - (A*x)(i,j,k)
```

**`setDirichletNodesToZero`:**
```
for each node (i,j,k):
    if dirichlet_mask(i,j,k) == 1:
        field(i,j,k) = 0             // zero the correction
```

**Singularity detection:**
If overset mask is present (`m_overset_dirichlet_mask == true`), the system is NOT treated as singular even if there are no domain-boundary Dirichlet BCs. The overset constraints provide sufficient rank.

#### B. No AMR, but MG

**Mask coarsening:**
```
for mglev = 1 to num_mg_levels:
    if m_overset_dirichlet_mask and mglev > 0:
        average_down_nodal(dirichlet_mask[mglev-1], dirichlet_mask[mglev])
    // Then: add domain-boundary Dirichlet nodes
    mlndlap_set_dirichlet_mask(dirichlet_mask[mglev], ...)
```

The nodal mask coarsening uses `average_down_nodal`, which for ratio-2 coarsening is simple **injection**: `coarse(i,j,k) = fine(2i,2j,2k)`. The coarse node inherits the Dirichlet status of the coincident fine node. This avoids the mixed-cell problem entirely since there is always exactly one fine node at the coarse node position.

The domain-boundary Dirichlet marking is then applied on top, so boundary nodes are always marked regardless of the overset mask.

**Smoother and apply:** Use the combined Dirichlet mask (overset + domain boundary) uniformly. No special `_os` variants needed — the Dirichlet mask already covers everything.

#### C. MG with AMR

**Per-level masks:**
Each AMR level has its own Dirichlet mask hierarchy. The overset mask is merged at `mglev=0`, then coarsened down through MG levels.

**Composite solve:** Same structure as without overset. The overset nodes are simply additional Dirichlet constraints in the existing framework.

---

## Summary: Key Differences Between Cell-Centred and Nodal Overset

| Aspect | Cell-centred (MLCellABecLap) | Nodal (MLNodeLinOp) |
|--------|-------------------------------|---------------------|
| Storage | Separate `m_overset_mask` | Merged into `m_dirichlet_mask` |
| Coarsening | Sum-and-threshold (all-or-nothing), stops on mixed | `average_down_nodal` (injection from coincident fine node) |
| Mixed cells | Error / limits coarsening | No mixed-cell issue (nodal averaging) |
| B-coef rescaling | Yes, `osfac` at coarsened MG levels | Not applicable (no B-coefficients in same sense) |
| adotx | Separate `_os` kernel variant | Uses Dirichlet mask in standard kernel |
| Smoother | Separate `_os` kernel variant | Uses Dirichlet mask in standard smoother |
| RHS zeroing | `applyOverset` (separate step) | Part of residual computation |

## Relevance to Curl-Curl Solver

The curl-curl solver (`MLCurlCurl_CNS`) inherits from `MLLinOpT<Array<MultiFab,3>>` — it is neither cell-centred nor nodal, but edge-centred on staggered grids. It already has a custom Dirichlet treatment through `CurlCurlDirichletInfo`, which handles *domain-boundary* Dirichlet conditions only.

The closest analogy for adding overset support is the **cell-centred approach**, since:
1. Edge variables are cell-centred in one direction (similar to cell-centred DOFs)
2. The overset mask would need to be per-component (3 masks with matching edge centerings)
3. B-coefficient rescaling may be needed for the curl-curl stencil at the known/unknown interface
4. The mask coarsening logic would need adaptation for staggered grids
5. The existing `CurlCurlDirichletInfo` check-based approach (comparing against boundary indices) would need to be extended or replaced with a mask-based approach for interior overset regions
