#ifndef PROB_K_H
#define PROB_K_H

#include <AMReX_Array.H>
#include <AMReX_REAL.H>
#include <AMReX_IntVect.H>
#include <AMReX_Array4.H>
#include <AMReX_Geometry.H>
#include <AMReX_BCRec.H>
#include <cmath>

#define USE_CUSTOM_CURLCURL 0

struct ProbParm
{
  amrex::Real B0 = 1.0;
  amrex::Real mu_r_core = 100.0;
  amrex::Real core_radius = 0.35;
  amrex::Real shell_radius = 0.5;
  amrex::Real eta_shell = 0.01;
  amrex::Real eta_free_space = 1.0e5;
  amrex::Real fixed_dt = 0.01;
  amrex::Real refine_max_r = 0.55;
  amrex::Real refine_min_r = 0.3;
  amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> core_position = {AMREX_D_DECL(0.0, 0.0, 0.0)};
};

// ---------------------------------------------------------------------------
// Physics kernels
// ---------------------------------------------------------------------------

//! Vector potential Az for the dipole+uniform initial condition.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real vectorPotentialAz (amrex::Real x, amrex::Real y,
                               const ProbParm& prob)
{
  const amrex::Real xr = x - prob.core_position[0];
  const amrex::Real yr = y - prob.core_position[1];
  const amrex::Real rsq = xr * xr + yr * yr + 1.0e-50;
  const amrex::Real r2 = prob.shell_radius;
  if (std::sqrt(rsq) > r2)
  {
    return prob.B0 * yr * (1.0 - r2 * r2 / rsq);
  }
  return 0.0;
}

//! Resistivity
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real getEta (amrex::Real x, amrex::Real y,
                    const ProbParm& prob)
{
  const amrex::Real xr = x - prob.core_position[0];
  const amrex::Real yr = y - prob.core_position[1];
  const amrex::Real r = std::sqrt(xr * xr + yr * yr);

  if (r < prob.core_radius || r > prob.shell_radius)
  {
    return prob.eta_free_space;
  }
  else
  {
    return prob.eta_shell;
  }
}

//! Relative permeability
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real getMuRel (amrex::Real x, amrex::Real y,
                      const ProbParm& prob)
{
  const amrex::Real xr = x - prob.core_position[0];
  const amrex::Real yr = y - prob.core_position[1];
  const amrex::Real r = std::sqrt(xr * xr + yr * yr);
  if (r < prob.core_radius)
  {
    return prob.mu_r_core;
  }
  return 1.0;
}

//! Get resistivity for a cell
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real getEta (int i, int j, int /*k*/,
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
                    amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
                    const ProbParm& prob)
{
  const amrex::Real xc = problo[0] + (amrex::Real(i) + amrex::Real(0.5)) * dx[0];
  const amrex::Real yc = problo[1] + (amrex::Real(j) + amrex::Real(0.5)) * dx[1];
  return getEta(xc,yc,prob);
}

//! Get relative permeability for a cell
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real getMuRel (int i, int j, int /*k*/,
                      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& problo,
                      amrex::GpuArray<amrex::Real, AMREX_SPACEDIM> const& dx,
                      const ProbParm& prob)
{
  const amrex::Real xc = problo[0] + (amrex::Real(i) + amrex::Real(0.5)) * dx[0];
  const amrex::Real yc = problo[1] + (amrex::Real(j) + amrex::Real(0.5)) * dx[1];
  return getMuRel(xc,yc,prob);
}

//! Analytical Bx for the dipole+uniform field (valid for r > r2).
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real analyticalBx (amrex::Real x, amrex::Real y,
                          const ProbParm& prob)
{
  const amrex::Real xr = x - prob.core_position[0];
  const amrex::Real yr = y - prob.core_position[1];
  const amrex::Real rsq = xr * xr + yr * yr + 1.0e-50;
  const amrex::Real r2sq = prob.shell_radius * prob.shell_radius;
  return prob.B0 * (1.0 - r2sq * (xr * xr - yr * yr) / (rsq * rsq));
}

//! Analytical By for the dipole+uniform field (valid for r > r2).
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real analyticalBy (amrex::Real x, amrex::Real y,
                          const ProbParm& prob)
{
  const amrex::Real xr = x - prob.core_position[0];
  const amrex::Real yr = y - prob.core_position[1];
  const amrex::Real rsq = xr * xr + yr * yr + 1.0e-50;
  const amrex::Real r2sq = prob.shell_radius * prob.shell_radius;
  return -2.0 * prob.B0 * r2sq * xr * yr / (rsq * rsq);
}

//! Harmonic average of two values.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real harmonicAvg (amrex::Real a, amrex::Real b)
{
  return 2.0 * a * b / (a + b + 1.0e-50);
}

// ---------------------------------------------------------------------------
// Boundary fill functors for GpuBndryFuncFab
// ---------------------------------------------------------------------------

//! Fills face-centred Bx or By ghost cells with the analytical dipole field.
struct FaceBFill
{
  ProbParm prob;
  int face_dir;

  AMREX_GPU_DEVICE
  void operator() (const amrex::IntVect& iv,
                   amrex::Array4<amrex::Real> const& dest,
                   int dcomp, int numcomp,
                   amrex::GeometryData const& geomdata,
                   amrex::Real /*time*/,
                   const amrex::BCRec* bcr,
                   int /*bcomp*/, int /*orig_comp*/) const
  {
    using namespace amrex;

    const auto* lo = geomdata.ProbLo();
    const auto* dx = geomdata.CellSize();
    const auto& domain = geomdata.Domain();

    // Compute physical position of this face centre
    Real x, y;
    if (face_dir == 0)
    {
      // Bx face: node in x, cell in y
      x = lo[0] + static_cast<Real>(iv[0]) * dx[0];
      y = lo[1] + (static_cast<Real>(iv[1]) + 0.5) * dx[1];
    }
    else
    {
      // By face: cell in x, node in y
      x = lo[0] + (static_cast<Real>(iv[0]) + 0.5) * dx[0];
      y = lo[1] + static_cast<Real>(iv[1]) * dx[1];
    }

    // Build face domain box
    IntVect ftype(0);
    ftype[face_dir] = 1;
    Box fdomain = amrex::convert(domain, IndexType(ftype));

    for (int n = 0; n < numcomp; ++n)
    {
      // Only fill ext_dir boundaries
      bool dominated = false;
      for (int d = 0; d < AMREX_SPACEDIM; ++d)
      {
        if ((iv[d] < fdomain.smallEnd(d) && bcr[n].lo(d) == BCType::ext_dir) ||
            (iv[d] > fdomain.bigEnd(d)   && bcr[n].hi(d) == BCType::ext_dir))
        {
          dominated = true;
        }
      }
      if (dominated)
      {
        if (face_dir == 0)
        {
          dest(iv, dcomp + n) = analyticalBx(x, y, prob);
        }
        else
        {
          dest(iv, dcomp + n) = analyticalBy(x, y, prob);
        }
      }
    }
  }
};

//! Fills cell-centred Bcc ghost cells (Bx, By, Bz) with analytical values.
struct CellBFill
{
  ProbParm prob;

  AMREX_GPU_DEVICE
  void operator() (const amrex::IntVect& iv,
                   amrex::Array4<amrex::Real> const& dest,
                   int dcomp, int numcomp,
                   amrex::GeometryData const& geomdata,
                   amrex::Real /*time*/,
                   const amrex::BCRec* bcr,
                   int /*bcomp*/, int /*orig_comp*/) const
  {
    using namespace amrex;

    const auto* lo = geomdata.ProbLo();
    const auto* dx = geomdata.CellSize();
    const auto& domain = geomdata.Domain();

    Real x = lo[0] + (static_cast<Real>(iv[0]) + 0.5) * dx[0];
    Real y = lo[1] + (static_cast<Real>(iv[1]) + 0.5) * dx[1];

    bool dominated = false;
    for (int d = 0; d < AMREX_SPACEDIM; ++d)
    {
      if ((iv[d] < domain.smallEnd(d) && bcr[0].lo(d) == BCType::ext_dir) ||
          (iv[d] > domain.bigEnd(d)   && bcr[0].hi(d) == BCType::ext_dir))
      {
        dominated = true;
      }
    }
    if (dominated)
    {
      if (dcomp + 0 < dcomp + numcomp) {
        dest(iv, dcomp + 0) = analyticalBx(x, y, prob);
      }
      if (numcomp > 1) {
        dest(iv, dcomp + 1) = analyticalBy(x, y, prob);
      }
      if (numcomp > 2) {
        dest(iv, dcomp + 2) = 0.0;  // Bz = 0
      }
    }
  }
};

#endif
