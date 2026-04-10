# CurlCurl_VariableCoeffs Test — Handover Document

## What This Test Does

A standalone AMReX test for the `MLCurlCurl_CNS` linear solver. It simulates magnetic diffusion through a cylindrical shell with variable resistivity and permeability, using the AmrLevel framework.

**Physics:** A transverse magnetic field B₀x̂ diffuses through a conducting annular shell surrounding a permeable core. The implicit update solves:

```
curl(α curl(E)) + β E = curl(B/μ)
```

where α = dt/μ (face-centred), β = 1/η (edge-centred), and B is updated via `B_new = B_old - dt curl(E)`.

## File Layout

| File | Role |
|------|------|
| `main.cpp` | Standard `Amr` driver loop |
| `LevelBld.cpp` | Factory that returns `AmrLevelCurlCurl` instances |
| `AmrLevelCurlCurl.H` | AmrLevel subclass header — state types, member declarations |
| `AmrLevelCurlCurl.cpp` | All AmrLevel methods: setup, init, advance, post_timestep, etc. |
| `Prob_K.H` | GPU kernels: IC, material properties (tanh-smoothed), boundary fill functors |
| `AMReX_MLCurlCurl_CNS.H/cpp`, `MLCurlCurl_CNS_K.H` | The custom curl-curl solver (local copies, will eventually merge into AMReX) |
| `GNUmakefile` | Build config — **must** have `AMREX_NO_PROBINIT = TRUE` |
| `Make.package` | Source/header lists |
| `inputs` | Default runtime parameters |
| `HANDOVER.md` | This file |

## Build

```bash
cd amrex/Tests/LinearSolvers/CurlCurl_VariableCoeffs
make -j6 DIM=2 -s          # optimised
make -j6 DIM=2 DEBUG=TRUE -s  # debug
```

Critical build flags:
- `AMREX_NO_PROBINIT = TRUE` — without this, AMReX calls a Fortran `amrex_probinit` that doesn't exist, causing a null-pointer segfault during `Amr::init()`.
- `BL_NO_FORT = TRUE` — no Fortran sources.

## State Variables

```
Bx_Type (0)   — face-centred in x: IndexType(NODE, CELL), 1 component
By_Type (1)   — face-centred in y: IndexType(CELL, NODE), 1 component
Bz_Type (2)   — [3D only] face-centred in z: IndexType(CELL, CELL, NODE), 1 component
Bcc_Type      — cell-centred: 3 components (Bx_cc, By_cc, Bz_cc)
MatProp_Type  — cell-centred: 2 components (mu_rel, eta), for plotfiles only
```

In 2D, Bx_cc and By_cc are averaged from face values. **Bz_cc is evolved** (not averaged) via `dBz/dt = -(dEy/dx - dEx/dy)`.

MatProp_Type is filled once in `initData()` and `fillMatProps()` (after regrid) from the analytical `getMuRel`/`getEta` functions. It is NOT touched during `advance()`.

## Material Properties

Both η and μ use tanh-smoothed profiles controlled by `prob.intf_width` and `prob.mu_intf_width` respectively. The old sharp-interface versions are commented out in `Prob_K.H` but preserved.

Coefficients (α, β, RHS) are evaluated by calling `getMuRel`/`getEta` directly at face/node/cell-centre physical positions — this works because the profiles are smooth. The old cell-index-based harmonic-averaging code is commented out but preserved in `AmrLevelCurlCurl.cpp`.

## Advance Loop (AmrLevelCurlCurl::advance)

1. Swap time levels
2. Fill ghost cells of old data — uses `FillPatchSingleLevel` for face types (NOT `FillPatch`, which would trigger the `face_divfree_interp` abort on level > 0)
3. Build edge-centred E, RHS, β and face-centred α MultiFabs. **`rhs` must be initialised to zero** — AMReX MultiFabs are NOT zero-initialised on construction.
4. Compute coefficients and RHS from old B
5. Set up `MLCurlCurl_CNS` with variable α and β, solve with `MLMGT`. On level > 0, `setCoarseFineBC` passes the coarse level's stored E field.
6. Update face B: `B_new = B_old - dt curl(E)`
7. Update cell-centred Bz (2D evolved) and Bx_cc, By_cc (averaged)
8. Store E field in `s_crse_edge_E[level]` for coarse-fine BC of finer levels
9. Accumulate E into `EdgeFluxRegister` for AMR refluxing

## Solver Setup

```cpp
mlcc.setScalars(1.0, 1.0);    // scalar multipliers (both 1.0)
mlcc.setAlpha({...});          // variable α = dt/μ, replaces scalar α
mlcc.setBeta({...});           // variable β = 1/η
```

When `setAlpha` is called, the scalar alpha from `setScalars` is **ignored** for the curl-curl term. The operator becomes: `curl(α_array curl(E)) + β_scalar × β_array E`.

## AMR Support — Current State

### What works
- `errorEst`: tags cells by radius range (`prob.refine_min_r`, `prob.refine_max_r`)
- `init()` (new level from coarse): custom `fillFacesFromCoarse()` uses `FaceDivFree::interp_arr()` directly, matching CNS pattern at `Core/AmrLevel.cpp:6520-6561`
- `init(AmrLevel& old)` (regrid): custom `fillFacesFromOldAndCoarse()` uses `FillPatchTwoLevels` array overload, matching CNS pattern at `Core/AmrLevel.cpp:6612-6629`
- `post_regrid`: reconstructs `EdgeFluxRegister` after regrid
- `post_timestep`: refluxes face B via `EdgeFluxRegister::Reflux`, then `average_down_faces` and `average_down` for Bcc
- Ghost fill in `advance()`: uses `FillPatchSingleLevel` for face types to avoid the `face_divfree_interp` abort
- Coarse-fine BC for solver: `setCoarseFineBC` with stored coarse E

### Known issues
- **Fine-level solver convergence:** The MLMG solver converges very slowly on fine AMR levels. The curl-curl operator's MG hierarchy has issues near C/F boundaries. The CNS code has similar difficulties and limits MG coarsening depth. May need PCG or GMRES preconditioned solver on fine levels, or a composite solve.
- **3D not implemented:** The 3D branches in advance (B update, RHS computation) are stubbed out with TODOs.
- **Composite solve not implemented:** Only level-by-level solve exists. A composite solve (all levels simultaneously) was planned but deferred.
- **Debug prints:** `AmrLevelCurlCurl.cpp` prints coefficient norms (`max alpha/beta/rhs`) every timestep. Remove when no longer needed.

## Boundary Conditions

- Domain BCs: `LinOpBCType::Dirichlet` on all faces (non-periodic domain)
- Physical BC fill: `GpuBndryFuncFab<FaceBFill>` and `GpuBndryFuncFab<CellBFill>` functors fill ghost cells with the analytical dipole+uniform field (time-independent)
- The analytical field is only correct far from the shell; the domain must be large enough

## Refluxing with EdgeFluxRegister

Uses `EdgeFluxRegister` (not `FluxRegister`). The raw E field is passed to `CrseAdd`/`FineAdd` — they multiply by `dt` internally. `Reflux` corrects coarse face B at C/F boundaries via `B -= curl(E_fine×dt - E_crse×dt)`. With no subcycling, `dt_fine == dt_crse`.

## Face-Centred Interpolation — Critical Pitfall

`face_divfree_interp` is registered as the interpolator for face B states. Its `interp()` method (single FArrayBox) **aborts** — it only works via `interp_arr()` which takes `Array<FArrayBox*, AMREX_SPACEDIM>` (all face directions simultaneously).

This means:
- **Never call `FillPatch`/`FillCoarsePatch`** on face-centred state types when level > 0 — it triggers the abort
- Use `FillPatchSingleLevel` for same-level ghost fills (safe, no interpolation)
- Use `fillFacesFromCoarse()` / `fillFacesFromOldAndCoarse()` for coarse-to-fine
- The CNS code has the same constraint; see `Core/AmrLevel.cpp:6520-6630`

## Reference Material

- **CNS coefficient assembly:** `DiffusiveMethod/ImplicitFD/ImplicitFD.cpp`, function `buildCurlCurlInputs` (line 3500) and `solveGlobalResistiveCurlCurl` (line 3880)
- **CNS B update from E:** `ConstrainedTransport/ConstrainedTransport.cpp`, function `updateSMMagField` (line 49)
- **CNS face-centred state registration:** `Core/AmrLevel.cpp`, line 715
- **CNS EdgeFluxRegister usage:** `Core/AmrLevel.cpp`, lines 4700-4850
- **CNS custom FillPatch for faces (from coarse):** `Core/AmrLevel.cpp`, line 6520 (`makeLevelFromCoarseForCT`)
- **CNS custom FillPatch for faces (regrid):** `Core/AmrLevel.cpp`, line 6568 (`makeLevelFromExistingForCT`)
- **AMReX Advection_AmrLevel tutorial:** `amrex/Tests/Amr/Advection_AmrLevel/` — the AmrLevel pattern this test follows
