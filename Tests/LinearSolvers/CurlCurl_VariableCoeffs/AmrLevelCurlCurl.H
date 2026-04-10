#ifndef AMRLEVELCURLCURL_H
#define AMRLEVELCURLCURL_H

#include <AMReX_AmrLevel.H>
#include <AMReX_EdgeFluxRegister.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PhysBCFunct.H>

#include "Prob_K.H"

#include <memory>

enum StateType {
  Bx_Type = 0,
  By_Type,
#if (AMREX_SPACEDIM == 3)
  Bz_Type,
#endif
  Bcc_Type,
  MatProp_Type,   // cell-centred mu and eta (2 components), for plotfiles
  NUM_STATE_TYPE
};

class AmrLevelCurlCurl
  : public amrex::AmrLevel
{
public:

  AmrLevelCurlCurl ();

  AmrLevelCurlCurl (amrex::Amr& papa,
                    int lev,
                    const amrex::Geometry& level_geom,
                    const amrex::BoxArray& bl,
                    const amrex::DistributionMapping& dm,
                    amrex::Real time);

  ~AmrLevelCurlCurl () override;

  AmrLevelCurlCurl (AmrLevelCurlCurl const&) = delete;
  AmrLevelCurlCurl& operator= (AmrLevelCurlCurl const&) = delete;
  AmrLevelCurlCurl (AmrLevelCurlCurl&&) noexcept = delete;
  AmrLevelCurlCurl& operator= (AmrLevelCurlCurl&&) noexcept = delete;

  //! Define data descriptors.
  static void variableSetUp ();

  //! Cleanup data descriptors at end of run.
  static void variableCleanUp ();

  //! Initialize grid data at problem start-up.
  void initData () override;

  //! Initialize data on this level from another AmrLevelCurlCurl (during regrid).
  void init (amrex::AmrLevel& old) override;

  //! Initialize data on this level after regridding if old level did not previously exist.
  void init () override;

  //! Advance grids at this level in time.
  amrex::Real advance (amrex::Real time,
                       amrex::Real dt,
                       int iteration,
                       int ncycle) override;

  //! Compute initial dt.
  void computeInitialDt (int finest_level,
                         int sub_cycle,
                         amrex::Vector<int>& n_cycle,
                         const amrex::Vector<amrex::IntVect>& ref_ratio,
                         amrex::Vector<amrex::Real>& dt_level,
                         amrex::Real stop_time) override;

  //! Compute new dt.
  void computeNewDt (int finest_level,
                     int sub_cycle,
                     amrex::Vector<int>& n_cycle,
                     const amrex::Vector<amrex::IntVect>& ref_ratio,
                     amrex::Vector<amrex::Real>& dt_min,
                     amrex::Vector<amrex::Real>& dt_level,
                     amrex::Real stop_time,
                     int post_regrid_flag) override;

  //! Do work after timestep.
  void post_timestep (int iteration) override;

  //! Do work after regrid.
  void post_regrid (int lbase, int new_finest) override;

  //! Do work after init.
  void post_init (amrex::Real stop_time) override;

  //! Error estimation for regridding.
  void errorEst (amrex::TagBoxArray& tags,
                 int clearval,
                 int tagval,
                 amrex::Real time,
                 int n_error_buf = 0, int ngrow = 0) override;

  //! Write plotfile.
  void writePlotFile (const std::string& dir,
                      std::ostream& os,
                      amrex::VisMF::How how) override;

  //! Per-level problem parameters (host copy).
  static ProbParm* h_prob_parm;
  //! Per-level problem parameters (device copy).
  static ProbParm* d_prob_parm;

  //! Edge flux register for CT refluxing.
  std::unique_ptr<amrex::EdgeFluxRegister> m_edge_flux_reg;

#if !defined(AMREX_USE_CUDA)
protected:
#endif

  static void readParams ();

  //! Compute cell-centred B by averaging face values (and evolving Bz in 2D).
  void computeBcc ();

  //! Average-down face B and cell B from fine level.
  void avgDown ();

  //! Fill face-centred B from coarse level using divergence-free interpolation.
  void fillFacesFromCoarse ();

  //! Fill face-centred B from old level and coarse level during regrid.
  void fillFacesFromOldAndCoarse (amrex::AmrLevel& old);

  //! Fill MatProp_Type from analytical mu/eta functions.
  void fillMatProps ();

  AmrLevelCurlCurl& getLevel (int lev)
  {
    return *static_cast<AmrLevelCurlCurl*>(&parent->getLevel(lev));
  }

  //! Per-level E field stored after solve, for coarse-fine BC of finer levels.
  static amrex::Vector<amrex::Array<amrex::MultiFab, 3>> s_crse_edge_E;

  //! Solver parameters.
  static int s_verbose;
  static int s_bottom_verbose;
  static int s_max_iter;
  static int s_max_coarsening_level;
  static amrex::Real s_tol_rel;
  static amrex::Real s_tol_abs;
};

#endif
