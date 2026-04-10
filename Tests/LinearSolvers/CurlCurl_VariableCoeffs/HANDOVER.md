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
| `Prob_K.H` | GPU kernels: IC, material properties, boundary fill functors |
| `AMReX_MLCurlCurl_CNS.H/cpp`, `MLCurlCurl_CNS_K.H` | The custom curl-curl solver (local copies, will eventually merge into AMReX) |
| `GNUmakefile` | Build config — **must** have `AMREX_NO_PROBINIT = TRUE` |
| `Make.package` | Source/header lists |
| `inputs` | Default runtime parameters |

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
Bx_Type (0)  — face-centred in x: IndexType(NODE, CELL), 1 component
By_Type (1)  — face-centred in y: IndexType(CELL, NODE), 1 component
Bz_Type (2)  — [3D only] face-centred in z: IndexType(CELL, CELL, NODE), 1 component
Bcc_Type     — cell-centred: 3 components (Bx_cc, By_cc, Bz_cc)
```

In 2D, Bx_cc and By_cc are averaged from face values. **Bz_cc is evolved** (not averaged) via `dBz/dt = -(dEy/dx - dEx/dy)`.

## Advance Loop (AmrLevelCurlCurl::advance)

1. Swap time levels
2. FillPatch old data (ghost cells)
3. Build edge-centred E, RHS, β and face-centred α MultiFabs
4. Compute coefficients and RHS from old B (see "Coefficient Assembly" below)
5. Set up `MLCurlCurl_CNS` with variable α and β, solve with `MLMGT`
6. Update face B: `B_new = B_old - dt curl(E)`
7. Update cell-centred Bz (2D evolved) and Bx_cc, By_cc (averaged)
8. Accumulate E into `EdgeFluxRegister` for AMR refluxing

## Coefficient Assembly — Critical Details

The coefficient computation follows the pattern in `DiffusiveMethod/ImplicitFD/ImplicitFD.cpp` (the `buildCurlCurlInputs` function, lines 3500–3860). Key rules:

### Property evaluation
Material properties (η, μ) must be evaluated at **cell-centre positions** using cell indices, then averaged to edges/faces/nodes. Do NOT evaluate `getEta`/`getMuRel` at arbitrary physical positions (e.g. node positions) — this gives wrong classifications at material interfaces. Use helper lambdas like `getCellMu(i, j)` that convert cell index to position internally.

**Exception:** The user has modified the β (eta) evaluation to call `getEta` directly at the edge/node position. This works because `getEta` uses smooth tanh profiles (`prob.intf_width`). If `getMuRel` is also smoothed in the future, the same approach could be used for μ. For sharp interfaces, cell-index-based evaluation with harmonic averaging is required.

### β = 1/η (edge-centred)
- Ex edge (0,1): η at the edge position (smoothed) or harmonic avg of cells (i, j-1) and (i, j)
- Ey edge (1,0): η at the edge position (smoothed) or harmonic avg of cells (i-1, j) and (i, j)
- Ez node (1,1): η at the node position (smoothed) or harmonic avg of 4 surrounding cells

### α = dt/μ (face-centred)
- x-face: `dt / harmonicAvg(μ(i-1,j), μ(i,j))`
- y-face: `dt / harmonicAvg(μ(i,j-1), μ(i,j))`
- 2D cell-centre (z-component): `dt / μ(i,j)`

### RHS = curl(B/μ) = curl(H)
- **RHS_Ez** (node): `d(By/μ)/dx - d(Bx/μ)/dy`. The μ used to divide each face B value is the harmonic average of the two cells flanking the node in the relevant direction, matching the CNS pattern at `ImplicitFD.cpp:3796-3804`:
  - `By(i,j)/μ_avg(cell(i,j-1), cell(i,j))` — cells flanking the node in y
  - `Bx(i,j)/μ_avg(cell(i-1,j), cell(i,j))` — cells flanking the node in x
- **RHS_Ex** (2D): `d(Bz_cc/μ_cc)/dy` — uses cell-centred Bz and cell-centred μ
- **RHS_Ey** (2D): `-d(Bz_cc/μ_cc)/dx` — same pattern

### Initialisation of MultiFabs
**`rhs` must be initialised to zero** before the coefficient loop. AMReX MultiFabs are NOT zero-initialised on construction. Without this, ghost cells at domain boundaries contain garbage, which corrupts the solve.

## Solver Setup

```cpp
mlcc.setScalars(1.0, 1.0);    // scalar multipliers (both 1.0)
mlcc.setAlpha({...});          // variable α replaces scalar α
mlcc.setBeta({...});           // variable β multiplied by scalar β
```

When `setAlpha` is called, the scalar alpha from `setScalars` is **ignored** for the curl-curl term. The operator becomes: `curl(α_array curl(E)) + β_scalar × β_array E`.

## Boundary Conditions

- Domain BCs: `LinOpBCType::Dirichlet` on all faces (non-periodic domain)
- Physical BC fill: `GpuBndryFuncFab<FaceBFill>` and `GpuBndryFuncFab<CellBFill>` functors fill ghost cells with the analytical dipole+uniform field (time-independent)
- The analytical field is only correct far from the shell; the domain must be large enough

## AMR Support

### Interpolation
Face-centred states use `face_divfree_interp`. **This interpolator only works via `interp_arr()`** (all AMREX_SPACEDIM face arrays simultaneously). The standard `FillPatch` calls `interp()` which **aborts**. For AMR level > 0, the `init()` methods currently use the default `FillPatch`/`FillCoarsePatch` — this will crash with AMR. A custom fill-patch that calls `face_divfree_interp.interp_arr()` directly is needed (see `Core/AmrLevel.cpp:6547` in CNSAMReX for the pattern).

### Coarse-fine BC for solver
On level > 0, the coarse level's E field is passed to `mlcc.setCoarseFineBC()`. The E field is stored per-level in the static `s_crse_edge_E` vector after each solve.

### Refluxing
Uses `EdgeFluxRegister` (not `FluxRegister`). The raw E field is passed to `CrseAdd`/`FineAdd` — they multiply by `dt` internally. `Reflux` corrects coarse face B at C/F boundaries via `B -= curl(E_fine×dt - E_crse×dt)`.

### Subcycling
Disabled (`amr.subcycling_mode = None`). All levels use the same dt.

## Known Issues / TODO

1. **Stability with variable μ:** The simulation was blowing up with sharp mu interfaces. The eta interface has been smoothed with tanh profiles (`prob.intf_width`). The mu interface (`getMuRel`) still uses a sharp step function — smoothing it similarly may be needed.

2. **3D not implemented:** The 3D branches in advance (B update, RHS computation) are stubbed out with TODOs. The solver itself supports 3D.

3. **AMR FillPatch:** `init(AmrLevel& old)` and `init()` use the default `FillPatch`/`FillCoarsePatch` which will abort for face-centred states with `face_divfree_interp` when `max_level > 0` and regridding occurs. Need custom fill that calls `interp_arr` directly.

4. **Composite solve:** Only level-by-level solve is implemented. A composite solve (all levels simultaneously) was planned but deferred.

5. **Debug prints:** Lines 524-532 in `AmrLevelCurlCurl.cpp` print coefficient norms every timestep. Remove when no longer needed.

## Reference Material

- **CNS coefficient assembly:** `DiffusiveMethod/ImplicitFD/ImplicitFD.cpp`, function `buildCurlCurlInputs` (line 3500) and `solveGlobalResistiveCurlCurl` (line 3880)
- **CNS B update from E:** `ConstrainedTransport/ConstrainedTransport.cpp`, function `updateSMMagField` (line 49)
- **CNS face-centred state registration:** `Core/AmrLevel.cpp`, line 715
- **CNS EdgeFluxRegister usage:** `Core/AmrLevel.cpp`, lines 4700-4850
- **CNS custom FillPatch for faces:** `Core/AmrLevel.cpp`, line 6547 (`face.interp_arr(...)`)
- **AMReX Advection_AmrLevel tutorial:** `amrex/Tests/Amr/Advection_AmrLevel/` — the AmrLevel pattern this test follows
- **Plan file:** `.claude/plans/majestic-chasing-walrus.md` — original design plan with full rationale
