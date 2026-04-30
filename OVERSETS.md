# Overset Mask + Multigrid in AMReX MLLinOp

Reference for adding overset-mask support to a new `MLLinOp` subclass. Covers cell-centred (`MLABecLaplacian`) and nodal (`MLNodeLaplacian`) paths with no AMR. Curl-curl ignored. Pseudocode + maths only.

Convention everywhere: mask = 1 → unknown (solve), mask = 0 → known (prescribed). Prescribed values live in the user-supplied initial guess of `sol`.

---

## 1. MG algorithm refresher (relevant call sites in MLMG)

`MLMG::solve`:
1. `MLMG::prepareForSolve` (`AMReX_MLMG.H:1031`)
   - `linop.prepareForSolve()` (line 1045) — subclass coeff averaging, mask coarsening, singular-flag update, B-coef rescale.
   - For each AMR level: `linop.applyOverset(alev, rhs[alev])` (line 1104) — RHS pinning hook.
   - `linop.scaleRHS` / metric / Robin BC etc.
2. Iteration loop calls `MLMG::oneIter` (1270) → `mgVcycle(0, 0)` (1301).
3. `mgVcycle` (1350):
   - **Down leg** (mglev = 0..nbottom-1):
     - `linop.solutionResidual` (top, only at start) **or** residual already in `res[mglev]`.
     - `linop.smooth(amrlev, mglev, cor, res, skip_fb, nu1)` (1369) — pre-smooth on **correction equation** A·cor = res, starting from cor=0.
     - Compute residual-of-correction `rescor = res − A·cor` via `correctionResidual`.
     - `linop.restriction(amrlev, mglev+1, res_coarse, rescor)` (1382).
   - **Bottom**: `bottomSolve` (1394). Default smoother variant calls `linop.smooth(...)` repeatedly.
   - **Up leg** (mglev = nbottom-1..0):
     - `linop.interpolation(amrlev, mglev, fine_cor, crse_cor)` (1832) — `fine_cor += P·crse_cor`.
     - `linop.smooth(... cor, res, false, nu2)` (1437) — post-smooth.
4. Final: `sol += cor` at each level.

Two state modes drive `apply`:
- `Solution` mode: `solutionResidual` ⇒ `apply(BCMode::Inhomogeneous, Solution)` ⇒ `r = b − A·x`.
- `Correction` mode: `smooth` and `correctionResidual` ⇒ `apply(Homogeneous, Correction)`. The smoother operates on `cor` with cor=0 initial guess and homogeneous physical BCs.

**Why this matters for overset:** the *correction* at known cells/nodes must stay identically zero for every iteration; then `sol = sol₀ + cor` preserves prescribed values exactly. Every MG primitive must respect this invariant.

Two implementation strategies:
- **Cell-centred** uses an `_os` kernel branch and an `applyOverset` RHS-zeroing step.
- **Nodal** absorbs overset into the existing Dirichlet machinery; no `_os` kernel and no `applyOverset`.

---

## 2. Cell-centred path (`MLCellLinOp → MLCellABecLap → MLABecLaplacian`)

Operator: `L φ = α·a·φ − β·∇·(B ∇φ)` on cells. RHS `rhs`. Stencil per cell uses face B-coeffs.

### 2.1 Data

`MLCellABecLap::m_overset_mask` — `Vector<Vector<unique_ptr<iMultiFab>>>` per `(amrlev, mglev)` (`AMReX_MLCellABecLap.H:86`). Built once at construction; populated in:

`MLCellABecLap::define(...)` overload taking `a_overset_mask` (`.H:111`):
```
m_overset_mask[alev][0] := copy(user mask)
for mglev = 1..nmglevs-1:
    fine = m_overset_mask[alev][mglev-1]
    coarse = new iMultiFab on coarse BoxArray
    nerrors = coarsen_overset_mask(coarse, fine)
    if nerrors > 0:
        max_overset_mask_coarsening_level = mglev - 1
        break
truncate m_overset_mask[alev] to that depth
NMGLevels effectively reduced by mixed-cell barrier
```

Coarsening kernel `coarsen_overset_mask` (`AMReX_MLCellABecLap_3D_K.H:8`):
```
sum = Σ fine_mask over 2^d block
if sum == 2^d: coarse = 1
elif sum == 0: coarse = 0
else:          ++nerrors  (mixed → caps MG depth)
```

### 2.2 Hook map (cell-centred)

| MLLinOp virtual | Override location | Action | Why |
|---|---|---|---|
| `prepareForSolve` | `MLABecLaplacian::prepareForSolve` (`.H:423`) | parent → `averageDownCoeffs` → `update_singular_flags` | Standard. B-coef rescale (below) + mask hierarchy already done in `define`. |
| (helper, not virtual) `MLABecLaplacian::averageDownCoeffs` | `.H:656–714` | Per `mglev > 0`: if `m_overset_mask[alev][mglev]` exists, multiply face coeff by `osfac = 2·fac/(fac+1)` where `fac = 2^mglev`, on faces where exactly one adjacent cell is masked (`osm(i)+osm(i±1) == 1`) | Coarse-cell centre sits a sub-cell distance from the effective overset boundary; the gradient stencil `(φᵢ − φᵢ₋₁)/Δx` would be wrong without rescale. Faces interior-to-interior or known-to-known unaffected. |
| `applyOverset` | `MLCellABecLap::applyOverset` (`.H:634`) | `for cell c at mglev=0: if mask(c)==0 then rhs(c) := 0`. Called once per AMR level from `MLMG::prepareForSolve`. | Forces residual at known cells to be 0 once `apply` returns 0 there (see `Fapply` below). Without this step `r = b − 0 ≠ 0` and the smoother would try to "fix" known cells. |
| `apply` ⇒ `Fapply` | `MLABecLaplacian::Fapply` (`.H:801`) | Branch on `m_overset_mask[alev][mglev]`: if present, kernel `mlabeclap_adotx_os` else `mlabeclap_adotx`. | Produces an *identity row that maps masked DOFs to 0*. Crucial that the result is **0** (not "skip"): the residual `b − A·x` then evaluates to `0 − 0 = 0` at known cells. |
| `smooth` ⇒ `Fsmooth` | `MLABecLaplacian::Fsmooth` (`.H:884`) | GSRB and Jacobi paths each branch on the mask: `_os` kernels (`abec_gsrb_os`, `abec_jacobi_os`). | The smoother updates `cor`. At a known cell the kernel **assigns** `cor := 0` (overwrite, not skip). Skipping would leave whatever was in the buffer; assignment guarantees the invariant. |
| `correctionResidual` | inherited (`MLCellLinOp`) | `r' = b' − A·cor`, where for the down-leg `b' = res` and the smoother just produced `cor`. At known cells `cor = 0` (pinned) and `A·cor = 0` (identity row) so `r'(known) = b'(known) = 0` (already zero from above). | No special treatment needed — invariants from `Fapply` + `Fsmooth` + `applyOverset` propagate. |
| `restriction` | inherited; cell-centred restriction is volume-weighted average | At coarse mask-known cells, all 2^d fine residuals are 0 (no mixed cells, by coarsening rule) ⇒ coarse residual is 0. | Mixed-cell ban is precisely what makes plain restriction safe. |
| `interpolation` | inherited; piecewise-constant or linear injection | At fine mask-known cells the fine `cor` is overwritten next, but to keep invariants the next smoother call pins those cells to 0 immediately. Coarse `cor` at masked is 0 anyway. | No `_os` interpolation needed because the post-smooth pins cells. |
| `setDirichletNodesToZero` | `MLCellABecLap::setDirichletNodesToZero` (`.H:257`) | `if mask==0 then mf := 0` | GMRES bottom solver applies operator to vectors that may have non-zero entries at masked cells; this strips them. |
| `isSingular` | parent `MLABecLaplacian::update_singular_flags` (`.H:740`) | `m_is_singular[alev] = domain_covered ∧ no_dirichlet_BC ∧ ¬overset_mask ∧ α·a == 0` | An overset mask provides interior Dirichlet rank ⇒ system non-singular even without domain BCs. |
| `normalize` | inherited (`MLABecLaplacian::normalize`, `.H:1247`) | unchanged; divides by diagonal | Used only for bottom CG. Masked cells already pinned by smoothers/`apply`, so dividing zero by diagonal is harmless. No `_os` needed. |
| `getSolvabilityOffset` / `fixSolvabilityByOffset` | inherited | n/a (only used when fully singular) | Overset prevents singular case. |

### 2.3 `_os` kernel semantics (cell-centred)

`mlabeclap_adotx_os(i,j,k,n)` (`AMReX_MLABecLap_3D_K.H:32`):
```
if osm(i,j,k) == 0:
    y(i,j,k,n) := 0           // identity row (assign, not skip)
else:
    y(i,j,k,n) := standard 7-point β-Laplacian stencil
                  reading x at neighbours (which may be masked,
                  carrying their prescribed values)
```

`abec_gsrb_os(i,j,k,n,redblack)` (`:268`):
```
if (i+j+k+redblack) % 2 == 0:
    if osm(i,j,k) == 0:
        phi(i,j,k,n) := 0      // pin correction (assign)
    else:
        γ      = α·a + Σ dh·(B_face)
        γ_eff  = γ − coarse-fine BC absorption terms (m_, f_)
        ρ      = Σ dh·B_face·phi(neighbour)
        res    = rhs − (γ·phi − ρ)
        phi   += ω/γ_eff · res         (ω = 1.15)
```
Note: `phi` here is the correction passed by `smooth`. At masked cells assignment to 0 is the invariant; at unmasked cells the standard GS update pulls in masked neighbours whose value is whatever the smoother left there — for cor that's 0; for sol that's the prescribed value; both are correct.

`abec_jacobi_os` (`:379`): same guard, `phi += (2/3)·(rhs − A·phi)/γ_eff`.

`overset_rescale_bcoef_{x,y,z}(box, b_face, osm, osfac)` (`:741–791`):
```
if osm(i−1,j,k) + osm(i,j,k) == 1:        // exactly one side known
    bX(i,j,k) *= osfac
```
applied at every `mglev > 0` — see `MLABecLaplacian.H:659–714`.

### 2.4 Singular-flag update

`MLABecLaplacian::update_singular_flags` (`.H:740`): the existence of `m_overset_mask[alev][0]` flips the system out of singular (line 751). Without this, a pure-Neumann domain with overset would be wrongly nullspace-detected.

---

## 3. Nodal path (`MLNodeLinOp → MLNodeLaplacian`)

Operator: nodal Laplacian `L φ = ∇·σ∇φ` (or stencil variant). Overset is *folded into* the existing Dirichlet mask `m_dirichlet_mask`. There is no separate `_os` kernel and no `applyOverset` override.

### 3.1 Data

`MLNodeLinOp::m_dirichlet_mask` (`.H:138`) — `Vector<Vector<unique_ptr<iMultiFab>>>` per `(amrlev, mglev)`. Convention here: **dmsk = 1 → known/Dirichlet**, **dmsk = 0 → unknown** (opposite of the cell-centred sign).

`MLNodeLinOp::m_overset_dirichlet_mask` (`.H:155`) — bool, set true once `setOversetMask` runs.

### 3.2 Hook map (nodal)

| MLLinOp virtual | Override location | Action | Why |
|---|---|---|---|
| (entry) `setOversetMask` | `MLNodeLinOp::setOversetMask` (`.cpp:485`) | At amrlev mglev=0: `dmsk(i,j,k) := 1 − omsk(i,j,k)`, then `m_overset_dirichlet_mask = true`. | Convention flip: overset 1=unknown → Dirichlet 1=known. From here on the rest of the linop is unaware whether a Dirichlet node came from a domain BC or from overset. |
| `prepareForSolve` | `MLNodeLaplacian::prepareForSolve` (`.cpp:453`) → calls parent `MLNodeLinOp::prepareForSolve` (`.cpp:100`) → `buildMasks` (`.cpp:287`) → coeff averaging → stencil build | `buildMasks` builds the full `m_dirichlet_mask` hierarchy (see below) and sets `m_is_bottom_singular`. | Mask must be ready before any `apply` runs. |
| (helper) `MLNodeLinOp::buildMasks` | `.cpp:287` | (a) Coarsen overset part: for `mglev = 1..`, if `m_overset_dirichlet_mask` then `average_down_nodal(dmask[mglev-1], dmask[mglev], 2)` — pure injection: `coarse(I) := fine(2I)`. (b) For every mglev, kernel `mlndlap_set_dirichlet_mask` ORs in domain-boundary Dirichlet nodes derived from coarsened cell-mask `ccm`. (c) Singularity: `m_is_bottom_singular = domain_covered ∧ no_dirichlet_BC ∧ ¬m_overset_dirichlet_mask`. | Nodal injection avoids the cell-centred mixed-cell problem: a coarse node *coincides* with one fine node, so its Dirichlet status is unambiguous. Overset adds rank → not singular. |
| `solutionResidual` | `MLNodeLinOp::solutionResidual` (`.cpp:123`) | `apply(amrlev, 0, resid, x, Inhomog, Solution)`; then per node `if dmsk(i,j,k) then resid := 0 else resid := b − resid`. | Explicit residual zeroing at all Dirichlet nodes (domain or overset) — analogous to cell-centred `applyOverset` but folded into the residual step rather than RHS. |
| `correctionResidual` | `MLNodeLinOp::correctionResidual` (`.cpp:152`) | `apply(..., Homog, Correction)`; `resid = b − resid` (no explicit guard). At masked nodes `apply` returns 0 (kernel guard) ⇒ `resid(masked) = b(masked)`. As long as `b(masked) = 0` already (from previous `solutionResidual` zero-out and from restriction below), invariant holds. | The `apply`-side guard is sufficient; no extra dmsk check needed because the input `b` is already zero at masked nodes. |
| `apply` ⇒ `Fapply` | `MLNodeLaplacian::Fapply` (`MLNodeLaplacian_misc.cpp:199`) | Standard nodal stencil kernels (`mlndlap_adotx_*`) consume `dmsk`: `if dmsk(i,j,k) then out := 0 else out := stencil(in)`. | Same invariant as cell-centred: identity row maps Dirichlet DOFs to 0 in apply output. |
| `smooth` ⇒ `Fsmooth` | `MLNodeLaplacian::Fsmooth` (`misc.cpp:347`) | Gauss-Seidel kernels (`mlndlap_gauss_seidel_*`): `if dmsk(i,j,k) then sol := 0 else GS update`. | Pin correction at known nodes (assignment, not skip). |
| `restriction` | `MLNodeLaplacian::restriction` (`.cpp:472`) | Pass fine `dmsk_fine` into restriction kernel; the kernel skips contributions from masked fine nodes (which are already 0 anyway) and respects boundary stencils. | Don't propagate ghost values from pinned nodes into coarse residual averaging weights. |
| `interpolation` | `MLNodeLaplacian::interpolation` (`.cpp:586`) | Pass fine `dmsk_fine`; kernel `if !dmsk(fine) then fine_cor += P·crse_cor else skip`. | Don't perturb fine prescribed nodes. (And the post-smooth would re-pin them anyway — this is defence in depth.) |
| `setDirichletNodesToZero` | `MLNodeLinOp::setDirichletNodesToZero` (`.cpp:466`) | `if dmsk then mf := 0` (per mglev). | Used by GMRES + by the smoother harness when needed. |
| `isSingular` / `isBottomSingular` | `MLNodeLinOp` (`.H:67`) | Return flags set inside `buildMasks`. | Overset mask flips bottom-singular off (line 300). |
| `applyOverset` | **NOT overridden** | base no-op | RHS zero-out is folded into `solutionResidual`, so the explicit hook is unused. |
| `xdoty` / `getSolvabilityOffset` | `MLNodeLinOp::xdoty` (`.cpp:181`), `getSolvabilityOffset` (`.cpp:225`) | Use `m_bottom_dot_mask` / `m_coarse_dot_mask` which are themselves built from `dmsk` (Dirichlet nodes carry weight 0). | Inner products and solvability fixes must exclude pinned DOFs. |

### 3.3 Why the nodal approach is simpler

- Storage: one mask covers two BC sources.
- Coarsening: nodal injection (`average_down_nodal` with ratio 2 = `coarse(I)=fine(2I)`) is well-defined; no mixed-cell barrier ⇒ MG can coarsen as deep as the geometry allows.
- No B-coef rescaling: the σ-coeff lives on cells, and the nodal stencil is structured so the geometric distance to a Dirichlet node is correct at every level.
- No new kernels: existing Dirichlet-aware kernels do the job.

---

## 4. Side-by-side

| Aspect | Cell-centred (`MLABecLaplacian`) | Nodal (`MLNodeLaplacian`) |
|---|---|---|
| Mask sign | 1=unknown, 0=known | 0=unknown, 1=Dirichlet (after `1−omsk`) |
| Storage | `m_overset_mask` (separate) | merged into `m_dirichlet_mask` |
| Coarsening | sum-and-threshold; mixed → cap MG | nodal injection; no cap |
| RHS-zero hook | `applyOverset` (called by MLMG) | folded into `solutionResidual` |
| Apply zeroing | `_os` kernel (`mlabeclap_adotx_os`) | dmsk guard inside standard `mlndlap_adotx_*` |
| Smoother zeroing | `_os` kernel (`abec_gsrb_os`, `abec_jacobi_os`) | dmsk guard inside `mlndlap_gauss_seidel_*` |
| B-coef rescale | yes, `osfac = 2·fac/(fac+1)` at mglev>0 | n/a |
| Restriction | inherited (mixed-cell ban makes it safe) | overridden, dmsk-aware |
| Interpolation | inherited (post-smooth re-pins) | overridden, dmsk-aware |
| Singularity flip | `update_singular_flags` checks `m_overset_mask[alev][0]` | `buildMasks` checks `m_overset_dirichlet_mask` |
| `setDirichletNodesToZero` | `MLCellABecLap` override | `MLNodeLinOp` override |

---

## 5. Mathematical invariant (both paths)

Let `K = {DOFs with mask=known}`, `U = complement`. Decompose `sol = sol_K ⊕ sol_U` and similarly for `cor`, `rhs`. The MG cycle satisfies, by construction:

```
sol_K stays equal to user-supplied initial guess for all iterations
cor_K ≡ 0 at every smoother / interpolation / restriction step
A_KK · cor_K + A_KU · cor_U  ≡  0     (because A_KK = I, cor_K = 0)
res_K ≡ 0
rhs_K ≡ 0  (cell-centred: applyOverset; nodal: solutionResidual zero-out)
```

Therefore the only constraint on `cor_U` is `A_UU · cor_U = res_U − A_UK · sol_K_initial_guess_correction = res_U` (because the prescribed values enter the unknown rows through the off-diagonal `A_UK · sol_K` term *only* during `apply` on `sol`, and that contribution is folded into `res_U` once at the start).

---

## 6. Minimum hook list to add overset to a new MLLinOp subclass

If you mimic the cell-centred style:
1. Member `m_overset_mask[alev][mglev]`.
2. Builder method (analogue of `MLCellABecLap::define`) that copies the user mask and coarsens it; cap MG depth on mixed cells.
3. Override `applyOverset(amrlev, rhs)` — zero RHS at masked DOFs.
4. Override `Fapply` — branch to `_os` kernel that assigns 0 at masked, normal stencil elsewhere.
5. Override `Fsmooth` — branch to `_os` smoother kernel that assigns 0 at masked.
6. (If you have face-coefficients) coefficient rescale at `mglev > 0` analogous to `osfac`.
7. Override `setDirichletNodesToZero` — assign 0 at masked.
8. Update singular-flag logic — overset implies non-singular.
9. Verify `restriction`/`interpolation` work as-is given mixed-cell ban; otherwise add `_os` versions.

If you mimic the nodal style:
1. Decide the Dirichlet-mask convention; merge overset via `dmsk = 1 − omsk` in a `setOversetMask`.
2. Coarsen via injection in `buildMasks` or equivalent.
3. Make `solutionResidual` zero residual at Dirichlet/overset nodes.
4. Make all stencil kernels (`Fapply`, `Fsmooth`, `restriction`, `interpolation`) consume the mask and assign 0 at masked DOFs.
5. Override `setDirichletNodesToZero`.
6. Flip singular flag when overset mask present.
