#include "AmrLevelCurlCurl.H"
#if USE_CUSTOM_CURLCURL
#include "AMReX_MLCurlCurl_CNS.H"
#else
#include "AMReX_MLCurlCurl.H"
#endif

#include <AMReX_ParmParse.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_FillPatchUtil.H>
#include <AMReX_Interpolater.H>
#include <AMReX_StateData.H>
#include <AMReX_MLMG.H>
#include <AMReX_TagBox.H>
#include <AMReX_VisMF.H>
#include <AMReX_PhysBCFunct.H>

using namespace amrex;

// ---------------------------------------------------------------------------
// Static data
// ---------------------------------------------------------------------------

ProbParm* AmrLevelCurlCurl::h_prob_parm = nullptr;
ProbParm* AmrLevelCurlCurl::d_prob_parm = nullptr;

Vector<Array<MultiFab, 3>> AmrLevelCurlCurl::s_crse_edge_E;

int  AmrLevelCurlCurl::s_verbose = 1;
int  AmrLevelCurlCurl::s_bottom_verbose = 0;
int  AmrLevelCurlCurl::s_max_iter = 1000;
int  AmrLevelCurlCurl::s_max_coarsening_level = 100;
Real AmrLevelCurlCurl::s_tol_rel = 1.0e-10;
Real AmrLevelCurlCurl::s_tol_abs = 0.0;

// ---------------------------------------------------------------------------
// Constructors / Destructor
// ---------------------------------------------------------------------------

AmrLevelCurlCurl::AmrLevelCurlCurl () = default;

AmrLevelCurlCurl::AmrLevelCurlCurl (Amr& papa,
                                    int lev,
                                    const Geometry& level_geom,
                                    const BoxArray& bl,
                                    const DistributionMapping& dm,
                                    Real time)
  : AmrLevel(papa, lev, level_geom, bl, dm, time)
{
  if (level > 0)
  {
    m_edge_flux_reg = std::make_unique<EdgeFluxRegister>(
      grids, getLevel(level - 1).boxArray(),
      dmap, getLevel(level - 1).DistributionMap(),
      geom, getLevel(level - 1).Geom(),
      1);
  }
}

AmrLevelCurlCurl::~AmrLevelCurlCurl () = default;

// ---------------------------------------------------------------------------
// Read parameters
// ---------------------------------------------------------------------------

void
AmrLevelCurlCurl::readParams ()
{
  static bool done = false;
  if (done) { return; }
  done = true;

  {
    ParmParse pp("prob");
    pp.query("B0", h_prob_parm->B0);
    pp.query("mu_r_core", h_prob_parm->mu_r_core);
    pp.query("core_radius", h_prob_parm->core_radius);
    pp.query("shell_radius", h_prob_parm->shell_radius);
    pp.query("eta_shell", h_prob_parm->eta_shell);
    pp.query("eta_free_space", h_prob_parm->eta_free_space);
    pp.query("fixed_dt", h_prob_parm->fixed_dt);
    pp.query("refine_max_r", h_prob_parm->refine_max_r);
    pp.query("refine_min_r", h_prob_parm->refine_min_r);
    // core_position is left at origin for now
  }

  {
    ParmParse pp("solver");
    pp.query("verbose", s_verbose);
    pp.query("bottom_verbose", s_bottom_verbose);
    pp.query("max_iter", s_max_iter);
    pp.query("max_coarsening_level", s_max_coarsening_level);
    pp.query("tol_rel", s_tol_rel);
    pp.query("tol_abs", s_tol_abs);
  }
}

// ---------------------------------------------------------------------------
// State variable setup
// ---------------------------------------------------------------------------

void
AmrLevelCurlCurl::variableSetUp ()
{
  BL_ASSERT(desc_lst.size() == 0);

  h_prob_parm = new ProbParm{};
  d_prob_parm = static_cast<ProbParm*>(The_Arena()->alloc(sizeof(ProbParm)));

  readParams();

  Gpu::copy(Gpu::hostToDevice, h_prob_parm, h_prob_parm + 1, d_prob_parm);

  // Face-centred index types
  Array<IndexType, AMREX_SPACEDIM> faceIdxTypes;
  for (int d = 0; d < AMREX_SPACEDIM; ++d)
  {
    faceIdxTypes[d] = IndexType::TheCellType();
    faceIdxTypes[d].setType(d, IndexType::NODE);
  }

  const int nGhostFace = 1;
  const int nGhostCell = 1;

  // --- Face-centred Bx ---
  desc_lst.addDescriptor(Bx_Type, faceIdxTypes[0],
                         StateDescriptor::Point, nGhostFace, 1,
                         &face_divfree_interp);

  // --- Face-centred By ---
  desc_lst.addDescriptor(By_Type, faceIdxTypes[1],
                         StateDescriptor::Point, nGhostFace, 1,
                         &face_divfree_interp);

#if (AMREX_SPACEDIM == 3)
  // --- Face-centred Bz ---
  desc_lst.addDescriptor(Bz_Type, faceIdxTypes[2],
                         StateDescriptor::Point, nGhostFace, 1,
                         &face_divfree_interp);
#endif

  // --- Cell-centred Bcc (Bx, By, Bz) ---
  desc_lst.addDescriptor(Bcc_Type, IndexType::TheCellType(),
                         StateDescriptor::Point, nGhostCell, 3,
                         &cell_cons_interp);

  // Boundary conditions: ext_dir on all faces for all states
  int lo_bc[AMREX_SPACEDIM];
  int hi_bc[AMREX_SPACEDIM];
  for (int d = 0; d < AMREX_SPACEDIM; ++d)
  {
    lo_bc[d] = BCType::ext_dir;
    hi_bc[d] = BCType::ext_dir;
  }
  BCRec bc(lo_bc, hi_bc);

  // Face Bx boundary fill
  {
    StateDescriptor::BndryFunc bx_fill(
      GpuBndryFuncFab<FaceBFill>(FaceBFill{*h_prob_parm, 0}));
    bx_fill.setRunOnGPU(true);
    desc_lst.setComponent(Bx_Type, 0, "Bx_face", bc, bx_fill);
  }

  // Face By boundary fill
  {
    StateDescriptor::BndryFunc by_fill(
      GpuBndryFuncFab<FaceBFill>(FaceBFill{*h_prob_parm, 1}));
    by_fill.setRunOnGPU(true);
    desc_lst.setComponent(By_Type, 0, "By_face", bc, by_fill);
  }

#if (AMREX_SPACEDIM == 3)
  // Face Bz boundary fill
  {
    StateDescriptor::BndryFunc bz_fill(
      GpuBndryFuncFab<FaceBFill>(FaceBFill{*h_prob_parm, 2}));
    bz_fill.setRunOnGPU(true);
    desc_lst.setComponent(Bz_Type, 0, "Bz_face", bc, bz_fill);
  }
#endif

  // Cell-centred Bcc boundary fill
  {
    StateDescriptor::BndryFunc bcc_fill(
      GpuBndryFuncFab<CellBFill>(CellBFill{*h_prob_parm}));
    bcc_fill.setRunOnGPU(true);
    Vector<std::string> names = {"Bx", "By", "Bz"};
    Vector<BCRec> bcs(3, bc);
    desc_lst.setComponent(Bcc_Type, 0, names, bcs, bcc_fill);
  }

  // --- Cell-centred material properties (mu, eta) for plotfiles ---
  desc_lst.addDescriptor(MatProp_Type, IndexType::TheCellType(),
                         StateDescriptor::Point, 0, 2,
                         &cell_cons_interp);
  {
    StateDescriptor::BndryFunc mp_fill(
      [] (Box const&, FArrayBox&, int, int,
          Geometry const&, Real,
          const Vector<BCRec>&, int, int) { });
    mp_fill.setRunOnGPU(true);
    Vector<std::string> mp_names = {"mu_rel", "eta"};
    Vector<BCRec> mp_bcs(2, bc);
    desc_lst.setComponent(MatProp_Type, 0, mp_names, mp_bcs, mp_fill);
  }
}

void
AmrLevelCurlCurl::variableCleanUp ()
{
  desc_lst.clear();
  delete h_prob_parm;
  The_Arena()->free(d_prob_parm);
  h_prob_parm = nullptr;
  d_prob_parm = nullptr;
  s_crse_edge_E.clear();
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

void
AmrLevelCurlCurl::initData ()
{
  const auto prob = *h_prob_parm;
  const auto dx = geom.CellSizeArray();
  const auto problo = geom.ProbLoArray();

  // Initialise face Bx using vector potential
  MultiFab& Bx_new = get_new_data(Bx_Type);
  for (MFIter mfi(Bx_new, TilingIfNotGPU()); mfi.isValid(); ++mfi)
  {
    const Box& bx = mfi.tilebox();
    auto bxArr = Bx_new.array(mfi);
    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
      // Bx face: node in x, cell in y
      Real x = problo[0] + static_cast<Real>(i) * dx[0];
      Real y = problo[1] + (static_cast<Real>(j) + 0.5) * dx[1];
      Real yp = y + 0.5 * dx[1];
      Real ym = y - 0.5 * dx[1];
      bxArr(i, j, k) = (vectorPotentialAz(x, yp, prob)
                       - vectorPotentialAz(x, ym, prob)) / dx[1];
    });
  }

  // Initialise face By using vector potential
  MultiFab& By_new = get_new_data(By_Type);
  for (MFIter mfi(By_new, TilingIfNotGPU()); mfi.isValid(); ++mfi)
  {
    const Box& bx = mfi.tilebox();
    auto byArr = By_new.array(mfi);
    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
      // By face: cell in x, node in y
      Real x = problo[0] + (static_cast<Real>(i) + 0.5) * dx[0];
      Real y = problo[1] + static_cast<Real>(j) * dx[1];
      Real xp = x + 0.5 * dx[0];
      Real xm = x - 0.5 * dx[0];
      byArr(i, j, k) = -(vectorPotentialAz(xp, y, prob)
                        - vectorPotentialAz(xm, y, prob)) / dx[0];
    });
  }

#if (AMREX_SPACEDIM == 3)
  MultiFab& Bz_new = get_new_data(Bz_Type);
  Bz_new.setVal(0.0);
#endif

#if AMREX_SPACEDIM == 2
  // Compute cell-centred Bz
  MultiFab& Bcc_new = get_new_data(Bcc_Type);
  Bcc_new.setVal(0.0);
#endif
  // Cell-centred B
  computeBcc();

  // Material properties (mu, eta) for plotfiles
  MultiFab& mp = get_new_data(MatProp_Type);
  for (MFIter mfi(mp, TilingIfNotGPU()); mfi.isValid(); ++mfi)
  {
    const Box& bx = mfi.tilebox();
    auto mpArr = mp.array(mfi);
    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
      mpArr(i, j, k, 0) = getMuRel(i, j, k, problo, dx, prob);
      mpArr(i, j, k, 1) = getEta(i, j, k, problo, dx, prob);
    });
  }
}

void
AmrLevelCurlCurl::init (AmrLevel& old)
{
  auto* oldlev = static_cast<AmrLevelCurlCurl*>(&old);

  Real dt_new = parent->dtLevel(level);
  Real cur_time = oldlev->state[Bx_Type].curTime();
  Real prev_time = oldlev->state[Bx_Type].prevTime();
  Real dt_old = cur_time - prev_time;
  setTimeLevel(cur_time, dt_old, dt_new);

  // Cell-centred states: standard FillPatch is safe
  FillPatch(old, get_new_data(Bcc_Type), 0, cur_time, Bcc_Type, 0, 3);
  FillPatch(old, get_new_data(MatProp_Type), 0, cur_time, MatProp_Type, 0, 2);

  // Face-centred states: custom divfree fill-patch
  fillFacesFromOldAndCoarse(old);

  // Recompute MatProp at fine resolution
  fillMatProps();
}

void
AmrLevelCurlCurl::init ()
{
  Real dt = parent->dtLevel(level);
  Real cur_time = getLevel(level - 1).state[Bx_Type].curTime();
  Real prev_time = getLevel(level - 1).state[Bx_Type].prevTime();
  Real dt_old = (cur_time - prev_time)
              / static_cast<Real>(parent->MaxRefRatio(level - 1));

  setTimeLevel(cur_time, dt_old, dt);

  // Cell-centred states: standard FillCoarsePatch is safe
  FillCoarsePatch(get_new_data(Bcc_Type), 0, cur_time, Bcc_Type, 0, 3);
  FillCoarsePatch(get_new_data(MatProp_Type), 0, cur_time, MatProp_Type, 0, 2);

  // Face-centred states: custom divfree interp from coarse
  fillFacesFromCoarse();

  // Recompute MatProp at fine resolution
  fillMatProps();
}

// ---------------------------------------------------------------------------
// Time stepping
// ---------------------------------------------------------------------------

Real
AmrLevelCurlCurl::advance (Real time, Real dt, int /*iteration*/, int /*ncycle*/)
{
  // Swap time levels
  for (int k = 0; k < NUM_STATE_TYPE; ++k)
  {
    if (k == MatProp_Type) { continue; }
    state[k].allocOldData();
    state[k].swapTimeLevels(dt);
  }

  const auto dx = geom.CellSizeArray();
  const auto problo = geom.ProbLoArray();
  const auto prob = *h_prob_parm;
  const int finest_level = parent->finestLevel();

  // Old face B data
  MultiFab& Bx_old = get_old_data(Bx_Type);
  MultiFab& By_old = get_old_data(By_Type);
  MultiFab& Bcc_old = get_old_data(Bcc_Type);

  // Fill ghost cells of old data.
  // Use FillPatchSingleLevel for face types to avoid coarse-fine
  // interpolation (face_divfree_interp requires array-based path).
  for (int ft : {Bx_Type, By_Type})
  {
    MultiFab& mf = get_old_data(ft);
    StateDataPhysBCFunct physbc(get_state_data(ft), 0, geom);
    FillPatchSingleLevel(mf, mf.nGrowVect(), time,
                         {&mf}, {time}, 0, 0, 1, geom, physbc, 0);
  }
  FillPatch(*this, Bcc_old, Bcc_old.nGrow(), time, Bcc_Type, 0, 3);
  AMREX_ASSERT(!Bcc_old.contains_nan());

  // -----------------------------------------------------------------------
  // Build edge-centred MultiFabs for E, RHS, beta and face-centred alpha
  // -----------------------------------------------------------------------
  constexpr int ng = 2;

  Array<IntVect, 3> etype;
#if (AMREX_SPACEDIM == 3)
  etype = {IntVect(0, 1, 1), IntVect(1, 0, 1), IntVect(1, 1, 0)};
#elif (AMREX_SPACEDIM == 2)
  etype = {IntVect(0, 1), IntVect(1, 0), IntVect(1, 1)};
#endif

  Array<MultiFab, 3> Efield, rhs, betaCoef;
  Array<iMultiFab, 3> overset_mask;
#if USE_CUSTOM_CURLCURL
  Array<MultiFab, 3> alphaCoef;
#else
  MultiFab nodalAlpha;
#endif
  for (int idim = 0; idim < 3; ++idim)
  {
    BoxArray edgeBA = convert(grids, etype[idim]);
    Efield[idim].define(edgeBA, dmap, 1, ng);
    rhs[idim].define(edgeBA, dmap, 1, ng);
    betaCoef[idim].define(edgeBA, dmap, 1, 0);
    Efield[idim].setVal(0.0);
    rhs[idim].setVal(0.0);
    
    overset_mask[idim].define(edgeBA, dmap, 1, 0);
    overset_mask[idim].setVal(1);

#if USE_CUSTOM_CURLCURL
    if (AMREX_SPACEDIM < 3 && idim == 2)
    {
      alphaCoef[idim].define(grids, dmap, 1, ng);
    }
    else
    {
      BoxArray faceBA = convert(grids, IntVect::TheDimensionVector(idim));
      alphaCoef[idim].define(faceBA, dmap, 1, ng);
    }
#endif
  }
#if not(USE_CUSTOM_CURLCURL)
  {
    BoxArray nba = convert(grids, IndexType::TheNodeType());
    nodalAlpha.define(nba, dmap, 1, 0);
    nodalAlpha.setVal(0.0);
  }
#endif

  // -----------------------------------------------------------------------
  // Helper: get eta and mu at cell centre (ic, jc) from cell indices.
  // -----------------------------------------------------------------------
  auto getCellEta = [=] AMREX_GPU_DEVICE (int ic, int jc, int kc) -> Real
  {
    return getEta(ic, jc, kc, problo, dx, prob);
  };
  auto getCellMu = [=] AMREX_GPU_DEVICE (int ic, int jc, int kc) -> Real
  {
    return getMuRel(ic, jc, kc, problo, dx, prob);
  };
  constexpr Real threshold = 1e-10;
  constexpr Real inv_threshold = 1/threshold;
  auto getBeta = [] AMREX_GPU_DEVICE (Real eta) -> Real
  {
    return (eta > threshold) ? 1.0 / eta : inv_threshold;
  };
  auto isIdeal = [] AMREX_GPU_DEVICE (Real eta) -> bool
  {
    return eta <= threshold;
  };

  // -----------------------------------------------------------------------
  // Compute RHS, beta, alpha
  // -----------------------------------------------------------------------
  for (MFIter mfi(Bcc_old, TilingIfNotGPU()); mfi.isValid(); ++mfi)
  {
    auto bxArr = Bx_old.const_array(mfi);
    auto byArr = By_old.const_array(mfi);
    auto bccArr = Bcc_old.const_array(mfi);

    // --- Ex component: centring (0,1) in 2D = (cell-x, node-y) ---
    // Beta: harmonic avg of eta at cells (i, j-1) and (i, j)
    // RHS: d(Bz/mu)/dy using cell-centred Bz
    {
      const Box& exBox = mfi.tilebox(etype[0]);
      auto rhsEx = rhs[0].array(mfi);
      auto betaEx = betaCoef[0].array(mfi);
      auto osm = overset_mask[0].array(mfi);

      ParallelFor(exBox, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {
        const Real eta_up = getCellEta(i,j,k);
        const Real eta_down = getCellEta(i,j-1,k);
        const Real eta_face = harmonicAvg(eta_up,eta_down);
        betaEx(i, j, k) = getBeta(eta_face);
        osm(i,j,k) = (isIdeal(eta_up) || isIdeal(eta_down)) ? 0 : 1;

#if (AMREX_SPACEDIM == 2)
        // Bz_cc lives at cell centres; divide by mu at each cell centre
        rhsEx(i, j, k) = (bccArr(i, j, k, 2) / getCellMu(i,j,k)
                         - bccArr(i, j - 1, k, 2) / getCellMu(i,j-1,k)) / dx[1];
#else
        rhsEx(i, j, k) = 0.0; // TODO: 3D
#endif
      });
    }

    // --- Ey component: centring (1,0) in 2D = (node-x, cell-y) ---
    // Beta: harmonic avg of eta at cells (i-1, j) and (i, j)
    // RHS: -d(Bz/mu)/dx using cell-centred Bz
    {
      const Box& eyBox = mfi.tilebox(etype[1]);
      auto rhsEy = rhs[1].array(mfi);
      auto betaEy = betaCoef[1].array(mfi);
      auto osm = overset_mask[1].array(mfi);

      ParallelFor(eyBox, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {
        const Real eta_right = getCellEta(i,j,k);
        const Real eta_left = getCellEta(i-1,j,k);
        const Real eta_face = harmonicAvg(eta_right,eta_left);
        betaEy(i, j, k) = getBeta(eta_face);
        osm(i,j,k) = (isIdeal(eta_right) || isIdeal(eta_left)) ? 0 : 1;

#if (AMREX_SPACEDIM == 2)
        // Bz_cc lives at cell centres; divide by mu at each cell centre
        rhsEy(i, j, k) = -(bccArr(i, j, k, 2) / getCellMu(i,j,k)
                          - bccArr(i - 1, j, k, 2) / getCellMu(i-1,j,k)) / dx[0];
#else
        rhsEy(i, j, k) = 0.0; // TODO: 3D
#endif
      });
    }

    // --- Ez component: centring (1,1) in 2D = (node-x, node-y) ---
    // Beta: harmonic avg of eta over 4 surrounding cells
    // RHS: d(By/mu)/dx - d(Bx/mu)/dy using face-centred B
    {
      const Box& ezBox = mfi.tilebox(etype[2]);
      auto rhsEz = rhs[2].array(mfi);
      auto betaEz = betaCoef[2].array(mfi);
      auto osm = overset_mask[2].array(mfi);

      ParallelFor(ezBox, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {
        // 4 surrounding cells: SW=(i-1,j-1), SE=(i,j-1), NW=(i-1,j), NE=(i,j)
        Real e_sw = getCellEta(i - 1, j - 1, k);
        Real e_se = getCellEta(i, j - 1, k);
        Real e_nw = getCellEta(i - 1, j, k);
        Real e_ne = getCellEta(i, j, k);
        Real eta_inv = 0.25 * (1.0 / (e_sw) + 1.0 / (e_se)
                              + 1.0 / (e_nw) + 1.0 / (e_ne));
        betaEz(i, j, k) = getBeta(1.0 / (eta_inv));

        osm(i,j,k) = (isIdeal(e_sw) || isIdeal(e_se) || isIdeal(e_nw) || isIdeal(e_ne)) ? 0 : 1;

        // Mu at face positions flanking the Ez node.
        Real mu_se_ne = harmonicAvg(getCellMu(i, j - 1, k), getCellMu(i, j, k));
        Real mu_sw_nw = harmonicAvg(getCellMu(i - 1, j - 1, k), getCellMu(i - 1, j, k));
        Real mu_nw_ne = harmonicAvg(getCellMu(i - 1, j, k), getCellMu(i, j, k));
        Real mu_sw_se = harmonicAvg(getCellMu(i - 1, j - 1, k), getCellMu(i, j - 1, k));

        Real dHydx = (byArr(i, j, k) / mu_se_ne
                    - byArr(i - 1, j, k) / mu_sw_nw) / dx[0];
        Real dHxdy = (bxArr(i, j, k) / mu_nw_ne
                    - bxArr(i, j - 1, k) / mu_sw_se) / dx[1];
        rhsEz(i, j, k) = dHydx - dHxdy;
      });
    }

    // --- Alpha coefficients ---
    // Harmonic avg of cell-centred mu
    // x-face: alpha = dt / harmonic_avg(mu(i-1,j), mu(i,j))
    // y-face: alpha = dt / harmonic_avg(mu(i,j-1), mu(i,j))
    // 2D cell-centre: alpha = dt / mu(i,j)
#if USE_CUSTOM_CURLCURL
    {
      const Box& xBox = grow(mfi.tilebox(IntVect::TheDimensionVector(0)), ng);
      auto ax = alphaCoef[0].array(mfi);
      ParallelFor(xBox, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {
        ax(i, j, k) = dt / harmonicAvg(getCellMu(i - 1, j, k), getCellMu(i, j, k));
      });
    
      const Box& yBox = grow(mfi.tilebox(IntVect::TheDimensionVector(1)), ng);
      auto ay = alphaCoef[1].array(mfi);
      ParallelFor(yBox, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {
        ay(i, j, k) = dt / harmonicAvg(getCellMu(i, j - 1, k), getCellMu(i, j, k));
      });
    
      const Box& zBox = grow(mfi.tilebox(), ng);
      auto az = alphaCoef[2].array(mfi);
      ParallelFor(zBox, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {
        az(i, j, k) = dt / getCellMu(i, j, k);
      });
    }
#else
    {
      const auto& nbx = mfi.nodaltilebox();
      const auto& an = nodalAlpha.array(mfi);
      ParallelFor(nbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
      {
        const Real mu1 = getCellMu(i,j,k);
        const Real mu2 = getCellMu(i-1,j,k);
        const Real mu3 = getCellMu(i,j-1,k);
        const Real mu4 = getCellMu(i-1,j-1,k);
        const Real mu_inv = 0.25 * (Real(1) / mu1 + Real(1) / mu2 + Real(1) / mu3 + Real(1) / mu4);
        an(i,j,k) = dt * mu_inv;
      });
    }
#endif
  }

  // Fill ghost cells
  for (int idim = 0; idim < 3; ++idim)
  {
    rhs[idim].FillBoundary(geom.periodicity());
    betaCoef[idim].FillBoundary(geom.periodicity());
#if USE_CUSTOM_CURLCURL
    alphaCoef[idim].FillBoundary(geom.periodicity());
#endif
  }
#if not(USE_CUSTOM_CURLCURL)
  nodalAlpha.FillBoundary(geom.periodicity());
#endif

  // -----------------------------------------------------------------------
  // Set up and solve MLCurlCurl
  // -----------------------------------------------------------------------
  LPInfo info;
  info.setMaxCoarseningLevel(s_max_coarsening_level);

#if USE_CUSTOM_CURLCURL
  MLCurlCurl_CNS mlcc({geom}, {grids}, {dmap}, info);
#else
  MLCurlCurl mlcc({geom}, {grids}, {dmap}, info);
#endif

  Array<LinOpBCType, AMREX_SPACEDIM> loBC, hiBC;
  for (int d = 0; d < AMREX_SPACEDIM; ++d)
  {
    loBC[d] = LinOpBCType::Dirichlet;
    hiBC[d] = LinOpBCType::Dirichlet;
  }
  mlcc.setDomainBC(loBC, hiBC);

  mlcc.setScalars(1.0, 1.0);
#if USE_CUSTOM_CURLCURL
  AMREX_ASSERT(!alphaCoef[0].contains_nan());
  AMREX_ASSERT(!alphaCoef[1].contains_nan());
  AMREX_ASSERT(!alphaCoef[2].contains_nan());
#else
  AMREX_ASSERT(!nodalAlpha.contains_nan());
#endif
  AMREX_ASSERT(!betaCoef[0].contains_nan());
  AMREX_ASSERT(!betaCoef[1].contains_nan());
  AMREX_ASSERT(!betaCoef[2].contains_nan());
  AMREX_ASSERT(!rhs[0].contains_nan());
  AMREX_ASSERT(!rhs[1].contains_nan());
  AMREX_ASSERT(!rhs[2].contains_nan());

#if USE_CUSTOM_CURLCURL
  mlcc.setAlpha({Array<MultiFab const*, 3>{&alphaCoef[0],
                                           &alphaCoef[1],
                                           &alphaCoef[2]}});
#else
  mlcc.setAlpha({&nodalAlpha});
#endif
  mlcc.setBeta({Array<MultiFab const*, 3>{&betaCoef[0],
                                          &betaCoef[1],
                                          &betaCoef[2]}});

  // Coarse-fine BC for level > 0
  if (level > 0 && s_crse_edge_E[level - 1][0].ok())
  {
    mlcc.setCoarseFineBC(&s_crse_edge_E[level - 1],
                         parent->refRatio(level - 1)[0]);
  }

  mlcc.setLevelBC(0, nullptr);
  mlcc.prepareRHS({&rhs});

  using V = Array<MultiFab, 3>;
  MLMGT<V> mlmg(mlcc);
  mlmg.setVerbose(s_verbose);
  mlmg.setBottomVerbose(s_bottom_verbose);
  mlmg.setMaxIter(s_max_iter);

  mlmg.solve({&Efield}, {&rhs}, s_tol_rel, s_tol_abs);

  // Fill E ghost cells
  for (int idim = 0; idim < 3; ++idim)
  {
    Efield[idim].FillBoundary(geom.periodicity());
  }

  // Store E for coarse-fine BC of finer levels
  if (static_cast<int>(s_crse_edge_E.size()) <= level)
  {
    s_crse_edge_E.resize(level + 1);
  }
  for (int idim = 0; idim < 3; ++idim)
  {
    s_crse_edge_E[level][idim].define(Efield[idim].boxArray(),
                                      Efield[idim].DistributionMap(), 1, 0);
    MultiFab::Copy(s_crse_edge_E[level][idim], Efield[idim], 0, 0, 1, 0);
  }

  // -----------------------------------------------------------------------
  // Update face B: B_new = B_old - dt * curl(E)
  // -----------------------------------------------------------------------
  MultiFab& Bx_new = get_new_data(Bx_Type);
  MultiFab& By_new = get_new_data(By_Type);

#if (AMREX_SPACEDIM == 2)
  // 2D: only Ez matters for face B update
  for (MFIter mfi(Bx_new, TilingIfNotGPU()); mfi.isValid(); ++mfi)
  {
    auto bxOld = Bx_old.const_array(mfi);
    auto bxNew = Bx_new.array(mfi);
    auto ez = Efield[2].const_array(mfi);
    const Real dtdy = dt / dx[1];

    const Box& bx = mfi.tilebox();
    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
      bxNew(i, j, k) = bxOld(i, j, k) - dtdy * (ez(i, j + 1, k) - ez(i, j, k));
    });
  }

  for (MFIter mfi(By_new, TilingIfNotGPU()); mfi.isValid(); ++mfi)
  {
    auto byOld = By_old.const_array(mfi);
    auto byNew = By_new.array(mfi);
    auto ez = Efield[2].const_array(mfi);
    const Real dtdx = dt / dx[0];

    const Box& bx = mfi.tilebox();
    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
      byNew(i, j, k) = byOld(i, j, k) + dtdx * (ez(i + 1, j, k) - ez(i, j, k));
    });
  }
#elif (AMREX_SPACEDIM == 3)
  // TODO: 3D B update
  MultiFab& Bz_new = get_new_data(Bz_Type);
  amrex::ignore_unused(Bz_new);
#endif

  // -----------------------------------------------------------------------
  // Update cell-centred Bz (evolved in 2D) and Bx_cc, By_cc (averaged)
  // -----------------------------------------------------------------------
  MultiFab& Bcc_new = get_new_data(Bcc_Type);

#if (AMREX_SPACEDIM == 2)
  // Bz_cc evolved: Bz_new = Bz_old - dt * (dEy/dx - dEx/dy)
  for (MFIter mfi(Bcc_new, TilingIfNotGPU()); mfi.isValid(); ++mfi)
  {
    auto bccOld = Bcc_old.const_array(mfi);
    auto bccNew = Bcc_new.array(mfi);
    auto ex = Efield[0].const_array(mfi);
    auto ey = Efield[1].const_array(mfi);
    const Real dtdx_loc = dt / dx[0];
    const Real dtdy_loc = dt / dx[1];

    const Box& bx = mfi.tilebox();
    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
      bccNew(i, j, k, 2) = bccOld(i, j, k, 2)
        - dtdx_loc * (ey(i + 1, j, k) - ey(i, j, k))
        + dtdy_loc * (ex(i, j + 1, k) - ex(i, j, k));
    });
  }
#endif

  // Bx_cc, By_cc: averaged from faces
  for (MFIter mfi(Bcc_new, TilingIfNotGPU()); mfi.isValid(); ++mfi)
  {
    auto bccNew = Bcc_new.array(mfi);
    auto bxNew = Bx_new.const_array(mfi);
    auto byNew = By_new.const_array(mfi);

    const Box& bx = mfi.tilebox();
    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
      bccNew(i, j, k, 0) = 0.5 * (bxNew(i, j, k) + bxNew(i + 1, j, k));
      bccNew(i, j, k, 1) = 0.5 * (byNew(i, j, k) + byNew(i, j + 1, k));
#if (AMREX_SPACEDIM == 3)
      auto bzNew = Bz_new.const_array(mfi);
      bccNew(i, j, k, 2) = 0.5 * (bzNew(i, j, k) + bzNew(i, j, k + 1));
#endif
    });
  }

  // -----------------------------------------------------------------------
  // EdgeFluxRegister accumulation
  // -----------------------------------------------------------------------
  EdgeFluxRegister* fine_reg = (level < finest_level)
    ? getLevel(level + 1).m_edge_flux_reg.get() : nullptr;
  EdgeFluxRegister* curr_reg = (level > 0)
    ? m_edge_flux_reg.get() : nullptr;

  if (fine_reg)
  {
    fine_reg->reset();
  }

  if (fine_reg || curr_reg)
  {
    for (MFIter mfi(Efield[2], false); mfi.isValid(); ++mfi)
    {
#if (AMREX_SPACEDIM == 2)
      if (fine_reg) { fine_reg->CrseAdd(mfi, Efield[2][mfi], dt); }
      if (curr_reg) { curr_reg->FineAdd(mfi, Efield[2][mfi], dt); }
#elif (AMREX_SPACEDIM == 3)
      Array<FArrayBox const*, 3> E_ptrs = {&Efield[0][mfi],
                                            &Efield[1][mfi],
                                            &Efield[2][mfi]};
      if (fine_reg) { fine_reg->CrseAdd(mfi, E_ptrs, dt); }
      if (curr_reg) { curr_reg->FineAdd(mfi, E_ptrs, dt); }
#endif
    }
  }

  return dt;
}

// ---------------------------------------------------------------------------
// dt computation
// ---------------------------------------------------------------------------

void
AmrLevelCurlCurl::computeInitialDt (int finest_level,
                                    int /*sub_cycle*/,
                                    Vector<int>& n_cycle,
                                    const Vector<IntVect>& /*ref_ratio*/,
                                    Vector<amrex::Real>& dt_level,
                                    Real stop_time)
{
  if (level > 0) { return; }

  Real dt_0 = h_prob_parm->fixed_dt;

  const Real eps = 0.001 * dt_0;
  Real cur_time = state[Bx_Type].curTime();
  if (stop_time >= 0.0)
  {
    if ((cur_time + dt_0) > (stop_time - eps))
    {
      dt_0 = stop_time - cur_time;
    }
  }

  int n_factor = 1;
  for (int i = 0; i <= finest_level; ++i)
  {
    n_factor *= n_cycle[i];
    dt_level[i] = dt_0 / static_cast<Real>(n_factor);
  }
}

void
AmrLevelCurlCurl::computeNewDt (int finest_level,
                                int /*sub_cycle*/,
                                Vector<int>& n_cycle,
                                const Vector<IntVect>& /*ref_ratio*/,
                                Vector<amrex::Real>& /*dt_min*/,
                                Vector<amrex::Real>& dt_level,
                                Real stop_time,
                                int /*post_regrid_flag*/)
{
  if (level > 0) { return; }

  Real dt_0 = h_prob_parm->fixed_dt;

  const Real eps = 0.001 * dt_0;
  Real cur_time = state[Bx_Type].curTime();
  if (stop_time >= 0.0)
  {
    if ((cur_time + dt_0) > (stop_time - eps))
    {
      dt_0 = stop_time - cur_time;
    }
  }

  int n_factor = 1;
  for (int i = 0; i <= finest_level; ++i)
  {
    n_factor *= n_cycle[i];
    dt_level[i] = dt_0 / static_cast<Real>(n_factor);
  }
}

// ---------------------------------------------------------------------------
// Post-timestep operations
// ---------------------------------------------------------------------------

void
AmrLevelCurlCurl::post_timestep (int /*iteration*/)
{
  if (level < parent->finestLevel())
  {
    // Reflux face B at C/F boundary
    Array<MultiFab*, AMREX_SPACEDIM> B_crse = {
      AMREX_D_DECL(&get_new_data(Bx_Type),
                    &get_new_data(By_Type),
                    &get_new_data(Bz_Type))
    };
    getLevel(level + 1).m_edge_flux_reg->Reflux(B_crse);

    // Average down face B
    auto& fine_lev = getLevel(level + 1);
    Array<MultiFab const*, AMREX_SPACEDIM> fine_face = {
      AMREX_D_DECL(&fine_lev.get_new_data(Bx_Type),
                    &fine_lev.get_new_data(By_Type),
                    &fine_lev.get_new_data(Bz_Type))
    };
    Array<MultiFab*, AMREX_SPACEDIM> crse_face = {
      AMREX_D_DECL(&get_new_data(Bx_Type),
                    &get_new_data(By_Type),
                    &get_new_data(Bz_Type))
    };
    average_down_faces(fine_face, crse_face,
                       parent->refRatio(level), geom);

    // Recompute Bcc and average down
    computeBcc();

    MultiFab& fine_Bcc = fine_lev.get_new_data(Bcc_Type);
    MultiFab& crse_Bcc = get_new_data(Bcc_Type);
    average_down(fine_Bcc, crse_Bcc, fine_lev.geom, geom,
                 0, 3, parent->refRatio(level));
  }

  if (level < parent->finestLevel())
  {
    getLevel(level + 1).resetFillPatcher();
  }
}

void
AmrLevelCurlCurl::post_regrid (int /*lbase*/, int /*new_finest*/)
{
  if (level > 0)
  {
    m_edge_flux_reg = std::make_unique<EdgeFluxRegister>(
      grids, getLevel(level - 1).boxArray(),
      dmap, getLevel(level - 1).DistributionMap(),
      geom, getLevel(level - 1).Geom(), 1);
  }
}

void
AmrLevelCurlCurl::post_init (Real /*stop_time*/)
{
  if (level > 0) { return; }

  int finest = parent->finestLevel();
  for (int k = finest - 1; k >= 0; --k)
  {
    getLevel(k).avgDown();
  }

  // Allocate coarse E storage
  s_crse_edge_E.resize(finest + 1);
}

// ---------------------------------------------------------------------------
// Error estimation
// ---------------------------------------------------------------------------

void
AmrLevelCurlCurl::errorEst (TagBoxArray& tags,
                            int /*clearval*/,
                            int /*tagval*/,
                            Real /*time*/,
                            int /*n_error_buf*/,
                            int /*ngrow*/)
{
  const auto prob = *h_prob_parm;
  MultiFab& Bcc = get_new_data(Bcc_Type);
  const char tagval = TagBox::SET;

  const auto& dx = geom.CellSizeArray();
  const auto& problo = geom.ProbLoArray();
  const Real refine_min_r_sq = prob.refine_min_r*prob.refine_min_r;
  const Real refine_max_r_sq = prob.refine_max_r*prob.refine_max_r;

  for (MFIter mfi(Bcc, TilingIfNotGPU()); mfi.isValid(); ++mfi)
  {
    const Box& bx = mfi.tilebox();
    auto tagArr = tags.array(mfi);

    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
      const Real xr = (i+0.5)*dx[0]+problo[0] - prob.core_position[0];
      const Real yr = (j+0.5)*dx[1]+problo[1] - prob.core_position[1];
      const Real r_sq = xr*xr + yr*yr;
      if (r_sq >= refine_min_r_sq && r_sq <= refine_max_r_sq)
      {
        tagArr(i, j, k) = tagval;
      }
    });
  }
}

// ---------------------------------------------------------------------------
// Plotfile
// ---------------------------------------------------------------------------

void
AmrLevelCurlCurl::writePlotFile (const std::string& dir,
                                 std::ostream& os,
                                 VisMF::How how)
{
  AmrLevel::writePlotFile(dir, os, how);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

void
AmrLevelCurlCurl::computeBcc ()
{
  MultiFab& Bcc = get_new_data(Bcc_Type);
  MultiFab& Bx = get_new_data(Bx_Type);
  MultiFab& By = get_new_data(By_Type);

  for (MFIter mfi(Bcc, TilingIfNotGPU()); mfi.isValid(); ++mfi)
  {
    const Box& bx = mfi.tilebox();
    auto bccArr = Bcc.array(mfi);
    auto bxArr = Bx.const_array(mfi);
    auto byArr = By.const_array(mfi);

    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
      bccArr(i, j, k, 0) = 0.5 * (bxArr(i, j, k) + bxArr(i + 1, j, k));
      bccArr(i, j, k, 1) = 0.5 * (byArr(i, j, k) + byArr(i, j + 1, k));
    });
  }

#if (AMREX_SPACEDIM == 3)
  MultiFab& Bz = get_new_data(Bz_Type);
  for (MFIter mfi(Bcc, TilingIfNotGPU()); mfi.isValid(); ++mfi)
  {
    const Box& bx = mfi.tilebox();
    auto bccArr = Bcc.array(mfi);
    auto bzArr = Bz.const_array(mfi);
    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
      bccArr(i, j, k, 2) = 0.5 * (bzArr(i, j, k) + bzArr(i, j, k + 1));
    });
  }
#else
  // In 2D, Bz_cc is NOT averaged from faces (it's evolved).
  // Don't overwrite it here. It was set in initData or advance.
#endif
}

void
AmrLevelCurlCurl::avgDown ()
{
  if (level == parent->finestLevel()) { return; }

  auto& fine_lev = getLevel(level + 1);

  // Average down face B
  Array<MultiFab const*, AMREX_SPACEDIM> fine_face = {
    AMREX_D_DECL(&fine_lev.get_new_data(Bx_Type),
                  &fine_lev.get_new_data(By_Type),
                  &fine_lev.get_new_data(Bz_Type))
  };
  Array<MultiFab*, AMREX_SPACEDIM> crse_face = {
    AMREX_D_DECL(&get_new_data(Bx_Type),
                  &get_new_data(By_Type),
                  &get_new_data(Bz_Type))
  };
  average_down_faces(fine_face, crse_face,
                     parent->refRatio(level), geom);

  // Average down cell-centred Bcc
  MultiFab& fine_Bcc = fine_lev.get_new_data(Bcc_Type);
  MultiFab& crse_Bcc = get_new_data(Bcc_Type);
  average_down(fine_Bcc, crse_Bcc, fine_lev.geom, geom,
               0, 3, parent->refRatio(level));
}

// ---------------------------------------------------------------------------
// AMR helpers: divergence-free face fill-patch
// ---------------------------------------------------------------------------

void
AmrLevelCurlCurl::fillFacesFromCoarse ()
{
  AMREX_ASSERT(level > 0);

  const int nGhost = get_new_data(Bx_Type).nGrow();
  const auto& crse_lev = getLevel(level - 1);
  const IntVect ratio = crse_ratio;

  // State type indices for each face direction
  const int face_types[AMREX_SPACEDIM] = {AMREX_D_DECL(Bx_Type, By_Type, Bz_Type)};

  // 1. Copy coarse data onto coarsened-fine grids
  Array<MultiFab, AMREX_SPACEDIM> crse_copy;
  for (int d = 0; d < AMREX_SPACEDIM; ++d)
  {
    BoxArray cba = convert(grids, IntVect::TheDimensionVector(d));
    cba.coarsen(ratio);
    crse_copy[d].define(cba, dmap, 1, nGhost,
                        MFInfo(), crse_lev.get_new_data(face_types[d]).Factory());
    crse_copy[d].ParallelCopy(crse_lev.get_new_data(face_types[d]),
                              0, 0, 1, nGhost, nGhost,
                              crse_lev.Geom().periodicity());
    crse_copy[d].FillBoundary(crse_lev.Geom().periodicity());
  }

  // 2. Divergence-free interpolation from coarse to fine
  FaceDivFree face_interp;

  // Build BCRec array (ext_dir on all faces, same for all directions)
  int lo_bc[AMREX_SPACEDIM], hi_bc[AMREX_SPACEDIM];
  for (int d = 0; d < AMREX_SPACEDIM; ++d)
  {
    lo_bc[d] = BCType::ext_dir;
    hi_bc[d] = BCType::ext_dir;
  }
  BCRec bc(lo_bc, hi_bc);
  Vector<Array<BCRec, AMREX_SPACEDIM>> bcrec_arr(1);
  for (int d = 0; d < AMREX_SPACEDIM; ++d)
  {
    bcrec_arr[0][d] = bc;
  }

  for (MFIter mfi(get_new_data(Bcc_Type), MFItInfo().SetDynamic(true));
       mfi.isValid(); ++mfi)
  {
    const Box& bx = grow(mfi.validbox(), nGhost);

    Array<FArrayBox*, AMREX_SPACEDIM> crse_ptrs;
    Array<FArrayBox*, AMREX_SPACEDIM> fine_ptrs;
    Array<IArrayBox*, AMREX_SPACEDIM> mask_ptrs;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
      crse_ptrs[d] = &crse_copy[d][mfi];
      fine_ptrs[d] = &get_new_data(face_types[d])[mfi];
      mask_ptrs[d] = nullptr;
    }

    face_interp.interp_arr(crse_ptrs, 0, fine_ptrs, 0, 1, bx, ratio,
                           mask_ptrs,
                           crse_lev.Geom(), geom,
                           bcrec_arr, 0, 0, RunOn::Cpu);
  }
}

void
AmrLevelCurlCurl::fillFacesFromOldAndCoarse (AmrLevel& old)
{
  AMREX_ASSERT(level > 0);

  const int face_types[AMREX_SPACEDIM] = {AMREX_D_DECL(Bx_Type, By_Type, Bz_Type)};

  // Build arrays of MultiFab pointers for fine-new, fine-old, coarse
  Array<MultiFab*, AMREX_SPACEDIM> fine_new;
  Array<MultiFab*, AMREX_SPACEDIM> fine_old;
  Array<MultiFab*, AMREX_SPACEDIM> crse;
  for (int d = 0; d < AMREX_SPACEDIM; ++d)
  {
    fine_new[d] = &get_new_data(face_types[d]);
    fine_old[d] = &old.get_new_data(face_types[d]);
    crse[d] = &getLevel(level - 1).get_new_data(face_types[d]);
  }

  // Build physical BC functors
  auto& crse_lev = getLevel(level - 1);
  Array<StateDataPhysBCFunct, AMREX_SPACEDIM> fine_bc_func = {AMREX_D_DECL(
    StateDataPhysBCFunct(get_state_data(face_types[0]), 0, geom),
    StateDataPhysBCFunct(get_state_data(face_types[1]), 0, geom),
    StateDataPhysBCFunct(get_state_data(face_types[2]), 0, geom))};
  Array<StateDataPhysBCFunct, AMREX_SPACEDIM> crse_bc_func = {AMREX_D_DECL(
    StateDataPhysBCFunct(crse_lev.get_state_data(face_types[0]), 0, crse_lev.Geom()),
    StateDataPhysBCFunct(crse_lev.get_state_data(face_types[1]), 0, crse_lev.Geom()),
    StateDataPhysBCFunct(crse_lev.get_state_data(face_types[2]), 0, crse_lev.Geom()))};

  // Build BCRec arrays
  int lo_bc[AMREX_SPACEDIM], hi_bc[AMREX_SPACEDIM];
  for (int d2 = 0; d2 < AMREX_SPACEDIM; ++d2)
  {
    lo_bc[d2] = BCType::ext_dir;
    hi_bc[d2] = BCType::ext_dir;
  }
  BCRec bc(lo_bc, hi_bc);
  Array<Vector<BCRec>, AMREX_SPACEDIM> bcrec_arr;
  for (int d = 0; d < AMREX_SPACEDIM; ++d)
  {
    bcrec_arr[d].resize(1, bc);
  }

  const IntVect nghost = fine_new[0]->nGrowVect();
  const Real cur_time = state[Bx_Type].curTime();

  Interpolater* mapper = &face_divfree_interp;

  FillPatchTwoLevels(
    fine_new,
    nghost,
    cur_time,
    {crse},
    {cur_time},
    {fine_old},
    {cur_time},
    0, 0, 1,
    getLevel(level - 1).Geom(),
    geom,
    crse_bc_func, 0,
    fine_bc_func, 0,
    crse_ratio,
    mapper,
    bcrec_arr,
    0);
}

void
AmrLevelCurlCurl::fillMatProps ()
{
  const auto prob = *h_prob_parm;
  const auto dx_arr = geom.CellSizeArray();
  const auto problo = geom.ProbLoArray();

  MultiFab& mp = get_new_data(MatProp_Type);
  for (MFIter mfi(mp, TilingIfNotGPU()); mfi.isValid(); ++mfi)
  {
    const Box& bx = mfi.tilebox();
    auto mpArr = mp.array(mfi);
    ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
      Real x = problo[0] + (static_cast<Real>(i) + 0.5) * dx_arr[0];
      Real y = problo[1] + (static_cast<Real>(j) + 0.5) * dx_arr[1];
      mpArr(i, j, k, 0) = getMuRel(x, y, prob);
      mpArr(i, j, k, 1) = getEta(x, y, prob);
    });
  }
}
