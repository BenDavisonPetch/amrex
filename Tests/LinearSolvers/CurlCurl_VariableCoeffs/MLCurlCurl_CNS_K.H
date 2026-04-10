#ifndef MLCURLCURL_CNS_K_H_
#define MLCURLCURL_CNS_K_H_
#include <AMReX_Config.H>

#include <AMReX_Array4.H>
#include <AMReX_LUSolver.H>
#include <AMReX_PCGSolver.H>

namespace amrex {

/* Index types
 * E_x        : (0,1,1)
 * E_y        : (1,0,1)
 * E_z        : (1,1,0)
 * (curl E)_x : (1,0,0)
 * (curl E)_y : (0,1,0)
 * (curl E)_z : (0,0,1)
 */

/*
  Notes for gs4:

  Interior nodes:

    For v = [Ex(i-1,j,k), Ex(i,j,k), Ey(i,j-1,k), Ey(i,j,k), Ez(i,j,k-1), Ez(i,j,k)]^T,
    we have A*v = b, where

    A00 = alpha*(dyy+dzz)*2 + beta
    A01 = 0
    A02 = -alpha*dxy
    A03 =  alpha*dxy
    A04 = -alpha*dxz
    A05 =  alpha*dxz

    A10 = 0
    A11 = alpha*(dyy+dzz)*2 + beta
    A12 =  alpha*dxy
    A13 = -alpha*dxy
    A14 =  alpha*dxz
    A15 = -alpha*dxz

    A20 = -alpha*dxy
    A21 =  alpha*dxy
    A22 = alpha*(dxx+dzz)*2 + beta
    A23 = 0
    A24 = -alpha*dyz
    A25 =  alpha*dyz

    A30 =  alpha*dxy
    A31 = -alpha*dxy
    A32 = 0
    A33 = alpha*(dxx+dzz)*2 + beta
    A34 =  alpha*dyz
    A35 = -alpha*dyz

    A40 = -alpha*dxz
    A41 =  alpha*dxz
    A42 = -alpha*dyz
    A43 =  alpha*dyz
    A44 = alpha*(dxx+dyy)*2 + beta
    A45 = 0

    A50 =  alpha*dxz
    A51 = -alpha*dxz
    A52 =  alpha*dyz
    A53 = -alpha*dyz
    A54 = 0
    A55 = alpha*(dxx+dyy)*2 + beta

    b0 = rhsx(i-1,j,k) - (alpha*ccex), where
      ccex = - dyy * (ex(i-1,j-1,k  ) +
                      ex(i-1,j+1,k  ))
             - dzz * (ex(i-1,j  ,k+1) +
                      ex(i-1,j  ,k-1))
             + dxy * (ey(i-1,j-1,k  )
                    - ey(i-1,j  ,k  ))
             + dxz * (ez(i-1,j  ,k-1)
                    - ez(i-1,j  ,k  ))
    b1 = rhsx(i,j,k) - (alpha*ccex), where
      ccex = - dyy * ( ex(i  ,j-1,k  ) +
                       ex(i  ,j+1,k  ))
             - dzz * ( ex(i  ,j  ,k+1) +
                       ex(i  ,j  ,k-1))
             + dxy * (-ey(i+1,j-1,k  )
                     + ey(i+1,j  ,k  ))
             + dxz * (-ez(i+1,j  ,k-1)
                     + ez(i+1,j  ,k  ));
    b2 = rhsy(i,j-1,k) - alpha*ccey, where
      ccey = - dxx * (ey(i-1,j-1,k  ) +
                      ey(i+1,j-1,k  ))
             - dzz * (ey(i  ,j-1,k-1) +
                      ey(i  ,j-1,k+1))
             + dxy * (ex(i-1,j-1,k  )
                    - ex(i  ,j-1,k  ))
             + dyz * (ez(i  ,j-1,k-1)
                    - ez(i  ,j-1,k  ))
    b3 = rhsy(i,j,k) - alpha*ccey, where
      ccey = - dxx * ( ey(i-1,j  ,k  ) +
                       ey(i+1,j  ,k  ))
             - dzz * ( ey(i  ,j  ,k-1) +
                       ey(i  ,j  ,k+1))
             + dxy * (-ex(i-1,j+1,k  )
                     + ex(i  ,j+1,k  ))
             + dyz * (-ez(i  ,j+1,k-1)
                     + ez(i  ,j+1,k  ));
    b4 = rhsz(i,j,k-1) - alpha*ccez, where
      ccez = - dxx * (ez(i-1,j  ,k-1) +
                      ez(i+1,j  ,k-1))
             - dyy * (ez(i  ,j-1,k-1) +
                      ez(i  ,j+1,k-1))
             + dxz * (ex(i-1,j  ,k-1)
                    - ex(i  ,j  ,k-1))
             + dyz * (ey(i  ,j-1,k-1)
                    - ey(i  ,j  ,k-1))
    b5 = rhsz(i,j,k) - alpha*ccez, where
      ccez = - dxx * ( ez(i-1,j  ,k  ) +
                       ez(i+1,j  ,k  ))
             - dyy * ( ez(i  ,j-1,k  ) +
                       ez(i  ,j+1,k  ))
             + dxz * (-ex(i-1,j  ,k+1)
                     + ex(i  ,j  ,k+1))
             + dyz * (-ey(i  ,j-1,k+1)
                     + ey(i  ,j  ,k+1));

    dxx = 1/(dx*dx)
    dyy = 1/(dy*dy)
    dzz = 1/(dz*dz)
    dxy = 1/(dx*dy)
    dxz = 1/(dx*dz)
    dyz = 1/(dy*dz)

  For Dirichlet boundary nodes, we don't do anything.

  For symmetric boundary nodes, we treat it as interior nodes because the
  rhs outside the domain has been filled properly.

  In 2D,

    For v = [Ex(i-1,j,k), Ex(i,j,k), Ey(i,j-1,k), Ey(i,j,k)]^T,
    we have A*v = b, where

    A00 = alpha*dyy*2 + beta
    A01 = 0
    A02 = -alpha*dxy
    A03 =  alpha*dxy

    A10 = 0
    A11 = alpha*dyy*2 + beta
    A12 =  alpha*dxy
    A13 = -alpha*dxy

    A20 = -alpha*dxy
    A21 =  alpha*dxy
    A22 = alpha*dxx*2 + beta
    A23 = 0

    A30 =  alpha*dxy
    A31 = -alpha*dxy
    A32 = 0
    A33 = alpha*dxx*2 + beta

    b0 = rhsx(i-1,j,k) - (alpha*ccex), where
      ccex = - dyy * (ex(i-1,j-1,k  ) +
                      ex(i-1,j+1,k  ))
             + dxy * (ey(i-1,j-1,k  )
                    - ey(i-1,j  ,k  ))
    b1 = rhsx(i,j,k) - (alpha*ccex), where
      ccex = - dyy * ( ex(i  ,j-1,k  ) +
                       ex(i  ,j+1,k  ))
             + dxy * (-ey(i+1,j-1,k  )
                     + ey(i+1,j  ,k  ))
    b2 = rhsy(i,j-1,k) - alpha*ccey, where
      ccey = - dxx * (ey(i-1,j-1,k  ) +
                      ey(i+1,j-1,k  ))
             + dxy * (ex(i-1,j-1,k  )
                    - ex(i  ,j-1,k  ))
    b3 = rhsy(i,j,k) - alpha*ccey, where
      ccey = - dxx * ( ey(i-1,j  ,k  ) +
                       ey(i+1,j  ,k  ))
             + dxy * (-ex(i-1,j+1,k  )
                     + ex(i  ,j+1,k  ))
*/

struct CurlCurlDirichletInfo
{
    IntVect dirichlet_lo;
    IntVect dirichlet_hi;

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool is_dirichlet_node (int i, int j, int k) const
    {
#if (AMREX_SPACEDIM == 1)
        amrex::ignore_unused(j,k);
        return (i == dirichlet_lo[0]) || (i == dirichlet_hi[0]);
#elif (AMREX_SPACEDIM == 2)
        amrex::ignore_unused(k);
        return (i == dirichlet_lo[0]) || (i == dirichlet_hi[0])
            || (j == dirichlet_lo[1]) || (j == dirichlet_hi[1]);
#else
        return (i == dirichlet_lo[0]) || (i == dirichlet_hi[0])
            || (j == dirichlet_lo[1]) || (j == dirichlet_hi[1])
            || (k == dirichlet_lo[2]) || (k == dirichlet_hi[2]);
#endif
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool is_dirichlet_x_edge (int, int j, int k) const // NOLINT(readability-convert-member-functions-to-static)
    {
#if (AMREX_SPACEDIM == 1)
        amrex::ignore_unused(j, k);
        return false;
#elif (AMREX_SPACEDIM == 2)
        amrex::ignore_unused(k);
        return (j == dirichlet_lo[1]) || (j == dirichlet_hi[1]);
#else
        return (j == dirichlet_lo[1]) || (j == dirichlet_hi[1])
            || (k == dirichlet_lo[2]) || (k == dirichlet_hi[2]);
#endif
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool is_dirichlet_y_edge (int i, int, int k) const
    {
#if (AMREX_SPACEDIM < 3)
        amrex::ignore_unused(k);
        return (i == dirichlet_lo[0]) || (i == dirichlet_hi[0]);
#else
        return (i == dirichlet_lo[0]) || (i == dirichlet_hi[0])
            || (k == dirichlet_lo[2]) || (k == dirichlet_hi[2]);
#endif
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool is_dirichlet_z_edge (int i, int j, int) const
    {
#if (AMREX_SPACEDIM == 1)
        amrex::ignore_unused(j);
        return (i == dirichlet_lo[0]) || (i == dirichlet_hi[0]);
#else
        return (i == dirichlet_lo[0]) || (i == dirichlet_hi[0])
            || (j == dirichlet_lo[1]) || (j == dirichlet_hi[1]);
#endif
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool is_dirichlet_edge (int dim, int i, int j, int k) const
    {
        if (dim == 0) {
            return is_dirichlet_x_edge(i,j,k);
        } else if (dim == 1) {
            return is_dirichlet_y_edge(i,j,k);
        } else {
            return is_dirichlet_z_edge(i,j,k);
        }
    }
};

struct CurlCurlSymmetryInfo
{
    IntVect symmetry_lo;
    IntVect symmetry_hi;

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool xlo_is_symmetric (int i) const
    {
        return i == symmetry_lo[0];
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool xhi_is_symmetric (int i) const
    {
        return i == symmetry_hi[0];
    }

#if (AMREX_SPACEDIM > 1)
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool ylo_is_symmetric (int j) const
    {
        return j == symmetry_lo[1];
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool yhi_is_symmetric (int j) const
    {
        return j == symmetry_hi[1];
    }
#endif

#if (AMREX_SPACEDIM == 3)
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool zlo_is_symmetric (int k) const
    {
        return k == symmetry_lo[2];
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    bool zhi_is_symmetric (int k) const
    {
        return k == symmetry_hi[2];
    }
#endif

    [[nodiscard]] bool is_symmetric (int dir, int side, int idx) const
    {
#if (AMREX_SPACEDIM == 1)
        amrex::ignore_unused(dir);
        return (side == 0) ? xlo_is_symmetric(idx) : xhi_is_symmetric(idx);
#elif (AMREX_SPACEDIM == 2)
        if (dir == 0) {
            return (side == 0) ? xlo_is_symmetric(idx) : xhi_is_symmetric(idx);
        } else {
            return (side == 0) ? ylo_is_symmetric(idx) : yhi_is_symmetric(idx);
        }
#else
        if (dir == 0) {
            return (side == 0) ? xlo_is_symmetric(idx) : xhi_is_symmetric(idx);
        } else if (dir == 1) {
            return (side == 0) ? ylo_is_symmetric(idx) : yhi_is_symmetric(idx);
        } else {
            return (side == 0) ? zlo_is_symmetric(idx) : zhi_is_symmetric(idx);
        }
#endif
    }
};

enum struct CurlCurlStateType { x, b, r }; // x, b & r=b-Ax

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void mlcurlcurl_cns_adotx_x (int i, int j, int k, Array4<Real> const& Ax,
                         Array4<Real const> const& ex,
                         Array4<Real const> const& ey,
                         Array4<Real const> const& ez,
                         Real beta, GpuArray<Real,AMREX_SPACEDIM> const& adxinv)
{
#if (AMREX_SPACEDIM == 1)
    amrex::ignore_unused(ey,ez,adxinv);
    Real ccex = 0;
#elif (AMREX_SPACEDIM == 2)
    amrex::ignore_unused(ez);
    Real dyy = adxinv[1] * adxinv[1];
    Real dxy = adxinv[0] * adxinv[1];
    Real ccex =  ex(i  ,j  ,k  ) * dyy * Real(2.0)
        - dyy * (ex(i  ,j-1,k  ) +
                 ex(i  ,j+1,k  ))
        + dxy * (ey(i  ,j-1,k  )
               - ey(i  ,j  ,k  )
               - ey(i+1,j-1,k  )
               + ey(i+1,j  ,k  ));
#elif (AMREX_SPACEDIM == 3)
    Real dyy = adxinv[1] * adxinv[1];
    Real dzz = adxinv[2] * adxinv[2];
    Real dxy = adxinv[0] * adxinv[1];
    Real dxz = adxinv[0] * adxinv[2];
    Real ccex =  ex(i  ,j  ,k  ) * (dyy+dzz)*Real(2.0)
        - dyy * (ex(i  ,j-1,k  ) +
                 ex(i  ,j+1,k  ))
        - dzz * (ex(i  ,j  ,k+1) +
                 ex(i  ,j  ,k-1))
        + dxy * (ey(i  ,j-1,k  )
               - ey(i  ,j  ,k  )
               - ey(i+1,j-1,k  )
               + ey(i+1,j  ,k  ))
        + dxz * (ez(i  ,j  ,k-1)
               - ez(i  ,j  ,k  )
               - ez(i+1,j  ,k-1)
               + ez(i+1,j  ,k  ));
#endif
    Ax(i,j,k) = ccex + beta * ex(i,j,k);
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void mlcurlcurl_cns_adotx_y (int i, int j, int k, Array4<Real> const& Ay,
                         Array4<Real const> const& ex,
                         Array4<Real const> const& ey,
                         Array4<Real const> const& ez,
                         Real beta, GpuArray<Real,AMREX_SPACEDIM> const& adxinv)
{
#if (AMREX_SPACEDIM == 1)
    amrex::ignore_unused(ex,ez);
    Real dxx = adxinv[0] * adxinv[0];
    Real ccey =  ey(i  ,j  ,k  ) * dxx * Real(2.0)
        - dxx * (ey(i-1,j  ,k  ) +
                 ey(i+1,j  ,k  ));
#elif (AMREX_SPACEDIM == 2)
    amrex::ignore_unused(ez);
    Real dxx = adxinv[0] * adxinv[0];
    Real dxy = adxinv[0] * adxinv[1];
    Real ccey =  ey(i  ,j  ,k  ) * dxx * Real(2.0)
        - dxx * (ey(i-1,j  ,k  ) +
                 ey(i+1,j  ,k  ))
        + dxy * (ex(i-1,j  ,k  )
               - ex(i  ,j  ,k  )
               - ex(i-1,j+1,k  )
               + ex(i  ,j+1,k  ));
#elif (AMREX_SPACEDIM == 3)
    Real dxx = adxinv[0] * adxinv[0];
    Real dzz = adxinv[2] * adxinv[2];
    Real dxy = adxinv[0] * adxinv[1];
    Real dyz = adxinv[1] * adxinv[2];
    Real ccey =  ey(i  ,j  ,k  ) * (dxx+dzz)*Real(2.0)
        - dxx * (ey(i-1,j  ,k  ) +
                 ey(i+1,j  ,k  ))
        - dzz * (ey(i  ,j  ,k-1) +
                 ey(i  ,j  ,k+1))
        + dxy * (ex(i-1,j  ,k  )
               - ex(i  ,j  ,k  )
               - ex(i-1,j+1,k  )
               + ex(i  ,j+1,k  ))
        + dyz * (ez(i  ,j  ,k-1)
               - ez(i  ,j  ,k  )
               - ez(i  ,j+1,k-1)
               + ez(i  ,j+1,k  ));
#endif
    Ay(i,j,k) = ccey + beta * ey(i,j,k);
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void mlcurlcurl_cns_adotx_z (int i, int j, int k, Array4<Real> const& Az,
                         Array4<Real const> const& ex,
                         Array4<Real const> const& ey,
                         Array4<Real const> const& ez,
                         Real beta, GpuArray<Real,AMREX_SPACEDIM> const& adxinv)
{
#if (AMREX_SPACEDIM == 1)
    amrex::ignore_unused(ex,ey);
    Real dxx = adxinv[0] * adxinv[0];
    Real ccez =  ez(i  ,j  ,k  ) * dxx*Real(2.0)
        - dxx * (ez(i-1,j  ,k  ) +
                 ez(i+1,j  ,k  ));
#elif (AMREX_SPACEDIM == 2)
    amrex::ignore_unused(ex,ey);
    Real dxx = adxinv[0] * adxinv[0];
    Real dyy = adxinv[1] * adxinv[1];
    Real ccez =  ez(i  ,j  ,k  ) * (dxx+dyy)*Real(2.0)
        - dxx * (ez(i-1,j  ,k  ) +
                 ez(i+1,j  ,k  ))
        - dyy * (ez(i  ,j-1,k  ) +
                 ez(i  ,j+1,k  ));
#elif (AMREX_SPACEDIM == 3)
    Real dxx = adxinv[0] * adxinv[0];
    Real dyy = adxinv[1] * adxinv[1];
    Real dxz = adxinv[0] * adxinv[2];
    Real dyz = adxinv[1] * adxinv[2];
    Real ccez =  ez(i  ,j  ,k  ) * (dxx+dyy)*Real(2.0)
        - dxx * (ez(i-1,j  ,k  ) +
                 ez(i+1,j  ,k  ))
        - dyy * (ez(i  ,j-1,k  ) +
                 ez(i  ,j+1,k  ))
        + dxz * (ex(i-1,j  ,k  )
               - ex(i  ,j  ,k  )
               - ex(i-1,j  ,k+1)
               + ex(i  ,j  ,k+1))
        + dyz * (ey(i  ,j-1,k  )
               - ey(i  ,j  ,k  )
               - ey(i  ,j-1,k+1)
               + ey(i  ,j  ,k+1));
#endif
    Az(i,j,k) = ccez + beta * ez(i,j,k);
}

//! Variable-alpha x-component: curl(alpha(x) curl E) + beta E.
/**
 * \param i,j,k Cell index
 * \param Ax Output x-component
 * \param ex,ey,ez E-field edge components
 * \param beta Beta coefficient (scalar; caller may pass per-cell value)
 * \param alphay Alpha on y-faces (centering (0,1,0) in 3D)
 * \param alphaz Alpha on z-faces (centering (0,0,1) in 3D, (0,0) in 2D)
 * \param dxinv Raw inverse cell sizes (NOT scaled by sqrt(alpha))
 */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void mlcurlcurl_cns_adotx_x (int i, int j, int k, Array4<Real> const& Ax,
                              Array4<Real const> const& ex,
                              Array4<Real const> const& ey,
                              Array4<Real const> const& ez,
                              Real beta,
                              Array4<Real const> const& alphay,
                              Array4<Real const> const& alphaz,
                              GpuArray<Real,AMREX_SPACEDIM> const& dxinv)
{
#if (AMREX_SPACEDIM == 1)
    amrex::ignore_unused(ey,ez,alphay,alphaz,dxinv);
    Real ccex = Real(0.0);
#elif (AMREX_SPACEDIM == 2)
    amrex::ignore_unused(ez,alphay);
    Real dyy = dxinv[1] * dxinv[1];
    Real dxy = dxinv[0] * dxinv[1];
    Real az_hi = alphaz(i,j  ,k);
    Real az_lo = alphaz(i,j-1,k);
    Real ccex = (az_hi + az_lo) * dyy * ex(i,j,k)
        - az_hi * dyy * ex(i,j+1,k)
        - az_lo * dyy * ex(i,j-1,k)
        + az_hi * dxy * (ey(i+1,j  ,k) - ey(i  ,j  ,k))
        + az_lo * dxy * (ey(i  ,j-1,k) - ey(i+1,j-1,k));
#elif (AMREX_SPACEDIM == 3)
    Real dyy = dxinv[1] * dxinv[1];
    Real dzz = dxinv[2] * dxinv[2];
    Real dxy = dxinv[0] * dxinv[1];
    Real dxz = dxinv[0] * dxinv[2];
    Real az_hi = alphaz(i,j  ,k);
    Real az_lo = alphaz(i,j-1,k);
    Real ay_hi = alphay(i,j,k  );
    Real ay_lo = alphay(i,j,k-1);
    Real ccex = (az_hi + az_lo) * dyy * ex(i,j,k)
              + (ay_hi + ay_lo) * dzz * ex(i,j,k)
        - az_hi * dyy * ex(i,j+1,k)
        - az_lo * dyy * ex(i,j-1,k)
        - ay_hi * dzz * ex(i,j,k+1)
        - ay_lo * dzz * ex(i,j,k-1)
        + az_hi * dxy * (ey(i+1,j  ,k) - ey(i  ,j  ,k))
        + az_lo * dxy * (ey(i  ,j-1,k) - ey(i+1,j-1,k))
        + ay_hi * dxz * (ez(i+1,j,k  ) - ez(i  ,j,k  ))
        + ay_lo * dxz * (ez(i  ,j,k-1) - ez(i+1,j,k-1));
#endif
    Ax(i,j,k) = ccex + beta * ex(i,j,k);
}

//! Variable-alpha y-component: curl(alpha(x) curl E) + beta E.
/**
 * \param i,j,k Cell index
 * \param Ay Output y-component
 * \param ex,ey,ez E-field edge components
 * \param beta Beta coefficient (scalar; caller may pass per-cell value)
 * \param alphax Alpha on x-faces (centering (1,0,0) in 3D)
 * \param alphaz Alpha on z-faces (centering (0,0,1) in 3D, (0,0) in 2D)
 * \param dxinv Raw inverse cell sizes (NOT scaled by sqrt(alpha))
 */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void mlcurlcurl_cns_adotx_y (int i, int j, int k, Array4<Real> const& Ay,
                              Array4<Real const> const& ex,
                              Array4<Real const> const& ey,
                              Array4<Real const> const& ez,
                              Real beta,
                              Array4<Real const> const& alphax,
                              Array4<Real const> const& alphaz,
                              GpuArray<Real,AMREX_SPACEDIM> const& dxinv)
{
#if (AMREX_SPACEDIM == 1)
    amrex::ignore_unused(ex,ez,alphax);
    Real dxx = dxinv[0] * dxinv[0];
    Real az_hi = alphaz(i  ,j,k);
    Real az_lo = alphaz(i-1,j,k);
    Real ccey = (az_hi + az_lo) * dxx * ey(i,j,k)
        - az_hi * dxx * ey(i+1,j,k)
        - az_lo * dxx * ey(i-1,j,k);
#elif (AMREX_SPACEDIM == 2)
    amrex::ignore_unused(ez,alphax);
    Real dxx = dxinv[0] * dxinv[0];
    Real dxy = dxinv[0] * dxinv[1];
    Real az_hi = alphaz(i  ,j,k);
    Real az_lo = alphaz(i-1,j,k);
    Real ccey = (az_hi + az_lo) * dxx * ey(i,j,k)
        - az_hi * dxx * ey(i+1,j,k)
        - az_lo * dxx * ey(i-1,j,k)
        + az_hi * dxy * (ex(i  ,j+1,k) - ex(i  ,j  ,k))
        + az_lo * dxy * (ex(i-1,j  ,k) - ex(i-1,j+1,k));
#elif (AMREX_SPACEDIM == 3)
    Real dxx = dxinv[0] * dxinv[0];
    Real dzz = dxinv[2] * dxinv[2];
    Real dxy = dxinv[0] * dxinv[1];
    Real dyz = dxinv[1] * dxinv[2];
    Real az_hi = alphaz(i  ,j,k);
    Real az_lo = alphaz(i-1,j,k);
    Real ax_hi = alphax(i,j,k  );
    Real ax_lo = alphax(i,j,k-1);
    Real ccey = (az_hi + az_lo) * dxx * ey(i,j,k)
              + (ax_hi + ax_lo) * dzz * ey(i,j,k)
        - az_hi * dxx * ey(i+1,j,k)
        - az_lo * dxx * ey(i-1,j,k)
        - ax_hi * dzz * ey(i,j,k+1)
        - ax_lo * dzz * ey(i,j,k-1)
        + az_hi * dxy * (ex(i  ,j+1,k) - ex(i  ,j  ,k))
        + az_lo * dxy * (ex(i-1,j  ,k) - ex(i-1,j+1,k))
        + ax_hi * dyz * (ez(i,j+1,k  ) - ez(i,j  ,k  ))
        + ax_lo * dyz * (ez(i,j  ,k-1) - ez(i,j+1,k-1));
#endif
    Ay(i,j,k) = ccey + beta * ey(i,j,k);
}

//! Variable-alpha z-component: curl(alpha(x) curl E) + beta E.
/**
 * \param i,j,k Cell index
 * \param Az Output z-component
 * \param ex,ey,ez E-field edge components
 * \param beta Beta coefficient (scalar; caller may pass per-cell value)
 * \param alphax Alpha on x-faces (centering (1,0,0) in 3D, (1,0) in 2D)
 * \param alphay Alpha on y-faces (centering (0,1,0) in 3D, (0,1) in 2D)
 * \param dxinv Raw inverse cell sizes (NOT scaled by sqrt(alpha))
 */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void mlcurlcurl_cns_adotx_z (int i, int j, int k, Array4<Real> const& Az,
                              Array4<Real const> const& ex,
                              Array4<Real const> const& ey,
                              Array4<Real const> const& ez,
                              Real beta,
                              Array4<Real const> const& alphax,
                              Array4<Real const> const& alphay,
                              GpuArray<Real,AMREX_SPACEDIM> const& dxinv)
{
#if (AMREX_SPACEDIM == 1)
    amrex::ignore_unused(ex,ey);
    Real dxx = dxinv[0] * dxinv[0];
    Real ay_hi = alphay(i  ,j,k);
    Real ay_lo = alphay(i-1,j,k);
    Real ccez = (ay_hi + ay_lo) * dxx * ez(i,j,k)
        - ay_hi * dxx * ez(i+1,j,k)
        - ay_lo * dxx * ez(i-1,j,k);
#elif (AMREX_SPACEDIM == 2)
    amrex::ignore_unused(ex,ey);
    Real dxx = dxinv[0] * dxinv[0];
    Real dyy = dxinv[1] * dxinv[1];
    Real ay_hi = alphay(i  ,j,k);
    Real ay_lo = alphay(i-1,j,k);
    Real ax_hi = alphax(i,j  ,k);
    Real ax_lo = alphax(i,j-1,k);
    Real ccez = (ay_hi + ay_lo) * dxx * ez(i,j,k)
              + (ax_hi + ax_lo) * dyy * ez(i,j,k)
        - ay_hi * dxx * ez(i+1,j,k)
        - ay_lo * dxx * ez(i-1,j,k)
        - ax_hi * dyy * ez(i,j+1,k)
        - ax_lo * dyy * ez(i,j-1,k);
#elif (AMREX_SPACEDIM == 3)
    Real dxx = dxinv[0] * dxinv[0];
    Real dyy = dxinv[1] * dxinv[1];
    Real dxz = dxinv[0] * dxinv[2];
    Real dyz = dxinv[1] * dxinv[2];
    Real ay_hi = alphay(i  ,j,k);
    Real ay_lo = alphay(i-1,j,k);
    Real ax_hi = alphax(i,j  ,k);
    Real ax_lo = alphax(i,j-1,k);
    Real ccez = (ay_hi + ay_lo) * dxx * ez(i,j,k)
              + (ax_hi + ax_lo) * dyy * ez(i,j,k)
        - ay_hi * dxx * ez(i+1,j,k)
        - ay_lo * dxx * ez(i-1,j,k)
        - ax_hi * dyy * ez(i,j+1,k)
        - ax_lo * dyy * ez(i,j-1,k)
        + ay_hi * dxz * (ex(i  ,j,k+1) - ex(i  ,j,k  ))
        + ay_lo * dxz * (ex(i-1,j,k  ) - ex(i-1,j,k+1))
        + ax_hi * dyz * (ey(i,j  ,k+1) - ey(i,j  ,k  ))
        + ax_lo * dyz * (ey(i,j-1,k  ) - ey(i,j-1,k+1));
#endif
    Az(i,j,k) = ccez + beta * ez(i,j,k);
}

//
// Variable alpha component, with an overset mask
//

//! Variable-alpha x-component: curl(alpha(x) curl E) + beta E.
/**
 * \param i,j,k Cell index
 * \param Ax Output x-component
 * \param ex,ey,ez E-field edge components
 * \param beta Beta coefficient (scalar; caller may pass per-cell value)
 * \param alphay Alpha on y-faces (centering (0,1,0) in 3D)
 * \param alphaz Alpha on z-faces (centering (0,0,1) in 3D, (0,0) in 2D)
 * \param osm The overset mask (value 0 => known, 1 => unknown)
 * \param dxinv Raw inverse cell sizes (NOT scaled by sqrt(alpha))
 */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void mlcurlcurl_cns_adotx_x_os (int i, int j, int k, Array4<Real> const& Ax,
                                Array4<Real const> const& ex,
                                Array4<Real const> const& ey,
                                Array4<Real const> const& ez,
                                Real beta,
                                Array4<Real const> const& alphay,
                                Array4<Real const> const& alphaz,
                                Array4<int const> const& osm,
                                GpuArray<Real,AMREX_SPACEDIM> const& dxinv)
{
    if (osm(i,j,k) == 0)
    {
        // value is known. set A*x=0 so that residual = 0
        // (rhs has already been set to 0 for this cell)
        Ax(i,j,k) = Real(0);
        return;
    }
#if (AMREX_SPACEDIM == 1)
    amrex::ignore_unused(ey,ez,alphay,alphaz,dxinv);
    Real ccex = Real(0.0);
#elif (AMREX_SPACEDIM == 2)
    amrex::ignore_unused(ez,alphay);
    Real dyy = dxinv[1] * dxinv[1];
    Real dxy = dxinv[0] * dxinv[1];
    Real az_hi = alphaz(i,j  ,k);
    Real az_lo = alphaz(i,j-1,k);
    Real ccex = (az_hi + az_lo) * dyy * ex(i,j,k)
        - az_hi * dyy * ex(i,j+1,k)
        - az_lo * dyy * ex(i,j-1,k)
        + az_hi * dxy * (ey(i+1,j  ,k) - ey(i  ,j  ,k))
        + az_lo * dxy * (ey(i  ,j-1,k) - ey(i+1,j-1,k));
#elif (AMREX_SPACEDIM == 3)
    Real dyy = dxinv[1] * dxinv[1];
    Real dzz = dxinv[2] * dxinv[2];
    Real dxy = dxinv[0] * dxinv[1];
    Real dxz = dxinv[0] * dxinv[2];
    Real az_hi = alphaz(i,j  ,k);
    Real az_lo = alphaz(i,j-1,k);
    Real ay_hi = alphay(i,j,k  );
    Real ay_lo = alphay(i,j,k-1);
    Real ccex = (az_hi + az_lo) * dyy * ex(i,j,k)
              + (ay_hi + ay_lo) * dzz * ex(i,j,k)
        - az_hi * dyy * ex(i,j+1,k)
        - az_lo * dyy * ex(i,j-1,k)
        - ay_hi * dzz * ex(i,j,k+1)
        - ay_lo * dzz * ex(i,j,k-1)
        + az_hi * dxy * (ey(i+1,j  ,k) - ey(i  ,j  ,k))
        + az_lo * dxy * (ey(i  ,j-1,k) - ey(i+1,j-1,k))
        + ay_hi * dxz * (ez(i+1,j,k  ) - ez(i  ,j,k  ))
        + ay_lo * dxz * (ez(i  ,j,k-1) - ez(i+1,j,k-1));
#endif
    Ax(i,j,k) = ccex + beta * ex(i,j,k);
}

//! Variable-alpha y-component: curl(alpha(x) curl E) + beta E.
/**
 * \param i,j,k Cell index
 * \param Ay Output y-component
 * \param ex,ey,ez E-field edge components
 * \param beta Beta coefficient (scalar; caller may pass per-cell value)
 * \param alphax Alpha on x-faces (centering (1,0,0) in 3D)
 * \param alphaz Alpha on z-faces (centering (0,0,1) in 3D, (0,0) in 2D)
 * \param osm The overset mask (value 0 => known, 1 => unknown)
 * \param dxinv Raw inverse cell sizes (NOT scaled by sqrt(alpha))
 */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void mlcurlcurl_cns_adotx_y_os (int i, int j, int k, Array4<Real> const& Ay,
                              Array4<Real const> const& ex,
                              Array4<Real const> const& ey,
                              Array4<Real const> const& ez,
                              Real beta,
                              Array4<Real const> const& alphax,
                              Array4<Real const> const& alphaz,
                              Array4<int const> const& osm,
                              GpuArray<Real,AMREX_SPACEDIM> const& dxinv)
{
    if (osm(i,j,k) == 0)
    {
        // value is known. set A*x=0 so that residual = 0
        // (rhs has already been set to 0 for this cell)
        Ay(i,j,k) = Real(0);
        return;
    }
#if (AMREX_SPACEDIM == 1)
    amrex::ignore_unused(ex,ez,alphax);
    Real dxx = dxinv[0] * dxinv[0];
    Real az_hi = alphaz(i  ,j,k);
    Real az_lo = alphaz(i-1,j,k);
    Real ccey = (az_hi + az_lo) * dxx * ey(i,j,k)
        - az_hi * dxx * ey(i+1,j,k)
        - az_lo * dxx * ey(i-1,j,k);
#elif (AMREX_SPACEDIM == 2)
    amrex::ignore_unused(ez,alphax);
    Real dxx = dxinv[0] * dxinv[0];
    Real dxy = dxinv[0] * dxinv[1];
    Real az_hi = alphaz(i  ,j,k);
    Real az_lo = alphaz(i-1,j,k);
    Real ccey = (az_hi + az_lo) * dxx * ey(i,j,k)
        - az_hi * dxx * ey(i+1,j,k)
        - az_lo * dxx * ey(i-1,j,k)
        + az_hi * dxy * (ex(i  ,j+1,k) - ex(i  ,j  ,k))
        + az_lo * dxy * (ex(i-1,j  ,k) - ex(i-1,j+1,k));
#elif (AMREX_SPACEDIM == 3)
    Real dxx = dxinv[0] * dxinv[0];
    Real dzz = dxinv[2] * dxinv[2];
    Real dxy = dxinv[0] * dxinv[1];
    Real dyz = dxinv[1] * dxinv[2];
    Real az_hi = alphaz(i  ,j,k);
    Real az_lo = alphaz(i-1,j,k);
    Real ax_hi = alphax(i,j,k  );
    Real ax_lo = alphax(i,j,k-1);
    Real ccey = (az_hi + az_lo) * dxx * ey(i,j,k)
              + (ax_hi + ax_lo) * dzz * ey(i,j,k)
        - az_hi * dxx * ey(i+1,j,k)
        - az_lo * dxx * ey(i-1,j,k)
        - ax_hi * dzz * ey(i,j,k+1)
        - ax_lo * dzz * ey(i,j,k-1)
        + az_hi * dxy * (ex(i  ,j+1,k) - ex(i  ,j  ,k))
        + az_lo * dxy * (ex(i-1,j  ,k) - ex(i-1,j+1,k))
        + ax_hi * dyz * (ez(i,j+1,k  ) - ez(i,j  ,k  ))
        + ax_lo * dyz * (ez(i,j  ,k-1) - ez(i,j+1,k-1));
#endif
    Ay(i,j,k) = ccey + beta * ey(i,j,k);
}

//! Variable-alpha z-component: curl(alpha(x) curl E) + beta E.
/**
 * \param i,j,k Cell index
 * \param Az Output z-component
 * \param ex,ey,ez E-field edge components
 * \param beta Beta coefficient (scalar; caller may pass per-cell value)
 * \param alphax Alpha on x-faces (centering (1,0,0) in 3D, (1,0) in 2D)
 * \param alphay Alpha on y-faces (centering (0,1,0) in 3D, (0,1) in 2D)
 * \param osm The overset mask (value 0 => known, 1 => unknown)
 * \param dxinv Raw inverse cell sizes (NOT scaled by sqrt(alpha))
 */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void mlcurlcurl_cns_adotx_z (int i, int j, int k, Array4<Real> const& Az,
                              Array4<Real const> const& ex,
                              Array4<Real const> const& ey,
                              Array4<Real const> const& ez,
                              Real beta,
                              Array4<Real const> const& alphax,
                              Array4<Real const> const& alphay,
                              Array4<int const> const& osm,
                              GpuArray<Real,AMREX_SPACEDIM> const& dxinv)
{
    if (osm(i,j,k) == 0)
    {
        // value is known. set A*x=0 so that residual = 0
        // (rhs has already been set to 0 for this cell)
        Az(i,j,k) = Real(0);
        return;
    }
#if (AMREX_SPACEDIM == 1)
    amrex::ignore_unused(ex,ey);
    Real dxx = dxinv[0] * dxinv[0];
    Real ay_hi = alphay(i  ,j,k);
    Real ay_lo = alphay(i-1,j,k);
    Real ccez = (ay_hi + ay_lo) * dxx * ez(i,j,k)
        - ay_hi * dxx * ez(i+1,j,k)
        - ay_lo * dxx * ez(i-1,j,k);
#elif (AMREX_SPACEDIM == 2)
    amrex::ignore_unused(ex,ey);
    Real dxx = dxinv[0] * dxinv[0];
    Real dyy = dxinv[1] * dxinv[1];
    Real ay_hi = alphay(i  ,j,k);
    Real ay_lo = alphay(i-1,j,k);
    Real ax_hi = alphax(i,j  ,k);
    Real ax_lo = alphax(i,j-1,k);
    Real ccez = (ay_hi + ay_lo) * dxx * ez(i,j,k)
              + (ax_hi + ax_lo) * dyy * ez(i,j,k)
        - ay_hi * dxx * ez(i+1,j,k)
        - ay_lo * dxx * ez(i-1,j,k)
        - ax_hi * dyy * ez(i,j+1,k)
        - ax_lo * dyy * ez(i,j-1,k);
#elif (AMREX_SPACEDIM == 3)
    Real dxx = dxinv[0] * dxinv[0];
    Real dyy = dxinv[1] * dxinv[1];
    Real dxz = dxinv[0] * dxinv[2];
    Real dyz = dxinv[1] * dxinv[2];
    Real ay_hi = alphay(i  ,j,k);
    Real ay_lo = alphay(i-1,j,k);
    Real ax_hi = alphax(i,j  ,k);
    Real ax_lo = alphax(i,j-1,k);
    Real ccez = (ay_hi + ay_lo) * dxx * ez(i,j,k)
              + (ax_hi + ax_lo) * dyy * ez(i,j,k)
        - ay_hi * dxx * ez(i+1,j,k)
        - ay_lo * dxx * ez(i-1,j,k)
        - ax_hi * dyy * ez(i,j+1,k)
        - ax_lo * dyy * ez(i,j-1,k)
        + ay_hi * dxz * (ex(i  ,j,k+1) - ex(i  ,j,k  ))
        + ay_lo * dxz * (ex(i-1,j,k  ) - ex(i-1,j,k+1))
        + ax_hi * dyz * (ey(i,j  ,k+1) - ey(i,j  ,k  ))
        + ax_lo * dyz * (ey(i,j-1,k  ) - ey(i,j-1,k+1));
#endif
    Az(i,j,k) = ccez + beta * ez(i,j,k);
}

#if (AMREX_SPACEDIM == 1)
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void mlcurlcurl_cns_1D (int i, int j, int k,
                    Array4<Real> const& ex,
                    Array4<Real> const& ey,
                    Array4<Real> const& ez,
                    Array4<Real const> const& rhsx,
                    Array4<Real const> const& rhsy,
                    Array4<Real const> const& rhsz,
                    Real beta,
                    GpuArray<Real,AMREX_SPACEDIM> const& adxinv,
                    int color,
                    CurlCurlDirichletInfo const& dinfo, bool valid_x)
{
    if (dinfo.is_dirichlet_node(i,j,k)) {return; }

    Real dxx = adxinv[0] * adxinv[0];

    int my_color = i%2;

    if (my_color == color)
    {
        if (valid_x) {
            ex(i,j,k) = rhsx(i,j,k) / beta;
        }

        Real gamma_y = dxx * Real(2.0) + beta;
        Real ccey = - dxx * (ey(i-1,j  ,k  ) +
                             ey(i+1,j  ,k  ) );
        Real res_y = rhsy(i,j,k) - ( gamma_y * ey(i,j,k) + ccey );
        ey(i,j,k) += res_y/gamma_y;

        Real gamma_z = dxx * Real(2.0) + beta;
        Real ccez = -dxx * (ez(i-1,j  ,k  ) +
                            ez(i+1,j  ,k  ) );
        Real res_z = rhsz(i,j,k) - ( gamma_z * ez(i,j,k) + ccez );
        ez(i,j,k) += res_z/gamma_z;
    }
}


AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void mlcurlcurl_cns_1D (int i, int j, int k,
                    Array4<Real> const& ex,
                    Array4<Real> const& ey,
                    Array4<Real> const& ez,
                    Array4<Real const> const& rhsx,
                    Array4<Real const> const& rhsy,
                    Array4<Real const> const& rhsz,
                    Array4<Real const> const& betax,
                    Array4<Real const> const& betay,
                    Array4<Real const> const& betaz,
                    GpuArray<Real,AMREX_SPACEDIM> const& adxinv,
                    int color,
                    CurlCurlDirichletInfo const& dinfo, bool valid_x)
{
    if (dinfo.is_dirichlet_node(i,j,k)) {return; }
    Real dxx = adxinv[0] * adxinv[0];

    int my_color = i%2;

    if (my_color == color)
    {
        if (valid_x) {
            ex(i,j,k) = rhsx(i,j,k) / betax(i,j,k);
        }

        Real gamma_y = dxx * Real(2.0) + betay(i,j,k);
        Real ccey = - dxx * (ey(i-1,j  ,k  ) +
                             ey(i+1,j  ,k  ) );
        Real res_y = rhsy(i,j,k) - ( gamma_y * ey(i,j,k) + ccey );
        ey(i,j,k) += res_y/gamma_y;

        Real gamma_z = dxx * Real(2.0) + betaz(i,j,k);
        Real ccez = -dxx * (ez(i-1,j  ,k  ) +
                            ez(i+1,j  ,k  ) );
        Real res_z = rhsz(i,j,k) - ( gamma_z * ez(i,j,k) + ccez );
        ez(i,j,k) += res_z/gamma_z;
    }
}

//! Variable-alpha 1D Gauss-Seidel smoother with scalar beta.
/**
 * \param i,j,k Cell index
 * \param ex,ey,ez E-field edge components (modified in place)
 * \param rhsx,rhsy,rhsz Right-hand side
 * \param beta Scalar beta coefficient
 * \param alphay Alpha on y-faces (cell-centred in 1D)
 * \param alphaz Alpha on z-faces (cell-centred in 1D)
 * \param dxinv Raw inverse cell sizes
 * \param color Gauss-Seidel color (0 or 1)
 * \param dinfo Dirichlet boundary info
 * \param valid_x True if i is within the cell-centred domain for Ex
 */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void mlcurlcurl_cns_1D (int i, int j, int k,
                    Array4<Real> const& ex,
                    Array4<Real> const& ey,
                    Array4<Real> const& ez,
                    Array4<Real const> const& rhsx,
                    Array4<Real const> const& rhsy,
                    Array4<Real const> const& rhsz,
                    Real beta,
                    Array4<Real const> const& alphay,
                    Array4<Real const> const& alphaz,
                    GpuArray<Real,AMREX_SPACEDIM> const& dxinv,
                    int color,
                    CurlCurlDirichletInfo const& dinfo, bool valid_x)
{
    if (dinfo.is_dirichlet_node(i,j,k)) { return; }

    Real dxx = dxinv[0] * dxinv[0];

    int my_color = i%2;

    if (my_color == color)
    {
        if (valid_x) {
            ex(i,j,k) = rhsx(i,j,k) / beta;
        }

        Real az_hi = alphaz(i  ,j,k);
        Real az_lo = alphaz(i-1,j,k);
        Real gamma_y = (az_hi + az_lo) * dxx + beta;
        Real ccey = - az_hi * dxx * ey(i+1,j,k)
                    - az_lo * dxx * ey(i-1,j,k);
        Real res_y = rhsy(i,j,k) - ( gamma_y * ey(i,j,k) + ccey );
        ey(i,j,k) += res_y/gamma_y;

        Real ay_hi = alphay(i  ,j,k);
        Real ay_lo = alphay(i-1,j,k);
        Real gamma_z = (ay_hi + ay_lo) * dxx + beta;
        Real ccez = - ay_hi * dxx * ez(i+1,j,k)
                    - ay_lo * dxx * ez(i-1,j,k);
        Real res_z = rhsz(i,j,k) - ( gamma_z * ez(i,j,k) + ccez );
        ez(i,j,k) += res_z/gamma_z;
    }
}

//! Variable-alpha 1D Gauss-Seidel smoother with variable beta.
/**
 * \param i,j,k Cell index
 * \param ex,ey,ez E-field edge components (modified in place)
 * \param rhsx,rhsy,rhsz Right-hand side
 * \param betax,betay,betaz Per-cell beta coefficients
 * \param alphay Alpha on y-faces (cell-centred in 1D)
 * \param alphaz Alpha on z-faces (cell-centred in 1D)
 * \param dxinv Raw inverse cell sizes
 * \param color Gauss-Seidel color (0 or 1)
 * \param dinfo Dirichlet boundary info
 * \param valid_x True if i is within the cell-centred domain for Ex
 */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void mlcurlcurl_cns_1D (int i, int j, int k,
                    Array4<Real> const& ex,
                    Array4<Real> const& ey,
                    Array4<Real> const& ez,
                    Array4<Real const> const& rhsx,
                    Array4<Real const> const& rhsy,
                    Array4<Real const> const& rhsz,
                    Array4<Real const> const& betax,
                    Array4<Real const> const& betay,
                    Array4<Real const> const& betaz,
                    Array4<Real const> const& alphay,
                    Array4<Real const> const& alphaz,
                    GpuArray<Real,AMREX_SPACEDIM> const& dxinv,
                    int color,
                    CurlCurlDirichletInfo const& dinfo, bool valid_x)
{
    if (dinfo.is_dirichlet_node(i,j,k)) { return; }

    Real dxx = dxinv[0] * dxinv[0];

    int my_color = i%2;

    if (my_color == color)
    {
        if (valid_x) {
            ex(i,j,k) = rhsx(i,j,k) / betax(i,j,k);
        }

        Real az_hi = alphaz(i  ,j,k);
        Real az_lo = alphaz(i-1,j,k);
        Real gamma_y = (az_hi + az_lo) * dxx + betay(i,j,k);
        Real ccey = - az_hi * dxx * ey(i+1,j,k)
                    - az_lo * dxx * ey(i-1,j,k);
        Real res_y = rhsy(i,j,k) - ( gamma_y * ey(i,j,k) + ccey );
        ey(i,j,k) += res_y/gamma_y;

        Real ay_hi = alphay(i  ,j,k);
        Real ay_lo = alphay(i-1,j,k);
        Real gamma_z = (ay_hi + ay_lo) * dxx + betaz(i,j,k);
        Real ccez = - ay_hi * dxx * ez(i+1,j,k)
                    - ay_lo * dxx * ez(i-1,j,k);
        Real res_z = rhsz(i,j,k) - ( gamma_z * ez(i,j,k) + ccez );
        ez(i,j,k) += res_z/gamma_z;
    }
}

#endif

#if (AMREX_SPACEDIM > 1)

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void mlcurlcurl_cns_gs4_lu (int i, int j, int k,
                        Array4<Real> const& ex,
                        Array4<Real> const& ey,
                        Array4<Real> const& ez,
                        Array4<Real const> const& rhsx,
                        Array4<Real const> const& rhsy,
                        Array4<Real const> const& rhsz,
#if (AMREX_SPACEDIM == 2)
                        Real beta,
#endif
                        GpuArray<Real,AMREX_SPACEDIM> const& adxinv,
                        int color, LUSolver<AMREX_SPACEDIM*2,Real> const& lusolver,
                        CurlCurlDirichletInfo const& dinfo,
                        CurlCurlSymmetryInfo const& sinfo)
{
    if (dinfo.is_dirichlet_node(i,j,k)) { return; }

    int my_color = i%2 + 2*(j%2);
    if (k%2 != 0) {
        my_color = 3 - my_color;
    }

#if (AMREX_SPACEDIM == 2)

    Real dxx = adxinv[0] * adxinv[0];
    Real dyy = adxinv[1] * adxinv[1];

    if (((my_color == 0 || my_color == 3) && (color == 0 || color == 3)) ||
        ((my_color == 1 || my_color == 2) && (color == 1 || color == 2)))
    {
        Real gamma = (dxx+dyy)*Real(2.0) + beta;
        Real ccez = - dxx * (ez(i-1,j  ,k  ) +
                             ez(i+1,j  ,k  ))
                    - dyy * (ez(i  ,j-1,k  ) +
                             ez(i  ,j+1,k  ));
        Real res = rhsz(i,j,k) - (gamma*ez(i,j,k) + ccez);
        constexpr Real omega = Real(1.15);
        ez(i,j,k) += omega/gamma * res;
    }

    if (my_color != color) { return; }

    Real dxy = adxinv[0]*adxinv[1];

    GpuArray<Real,6> b
        {rhsx(i-1,j,k) - (-dyy * ( ex(i-1,j-1,k  ) +
                                   ex(i-1,j+1,k  ))
                         + dxy * ( ey(i-1,j-1,k  )
                                  -ey(i-1,j  ,k  ))),
         rhsx(i  ,j,k) - (-dyy * ( ex(i  ,j-1,k  ) +
                                   ex(i  ,j+1,k  ))
                         + dxy * (-ey(i+1,j-1,k  )
                                  +ey(i+1,j  ,k  ))),
         rhsy(i,j-1,k) - (-dxx * ( ey(i-1,j-1,k  ) +
                                   ey(i+1,j-1,k  ))
                         + dxy * ( ex(i-1,j-1,k  )
                                  -ex(i  ,j-1,k  ))),
         rhsy(i,j  ,k) - (-dxx * ( ey(i-1,j  ,k  ) +
                                   ey(i+1,j  ,k  ))
                         + dxy * (-ex(i-1,j+1,k  )
                                  +ex(i  ,j+1,k  )))};

    if (sinfo.xlo_is_symmetric(i)) {
        b[0] = -b[1];
    } else if (sinfo.xhi_is_symmetric(i)) {
        b[1] = -b[0];
    }

    if (sinfo.ylo_is_symmetric(j)) {
        b[2] = -b[3];
    } else if (sinfo.yhi_is_symmetric(j)) {
        b[3] = -b[2];
    }

    GpuArray<Real,4> x;
    lusolver(x.data(), b.data());
    ex(i-1,j  ,k  ) = x[0];
    ex(i  ,j  ,k  ) = x[1];
    ey(i  ,j-1,k  ) = x[2];
    ey(i  ,j  ,k  ) = x[3];

#elif (AMREX_SPACEDIM == 3)

    if (my_color != color) { return; }

    Real dxx = adxinv[0]*adxinv[0];
    Real dyy = adxinv[1]*adxinv[1];
    Real dzz = adxinv[2]*adxinv[2];
    Real dxy = adxinv[0]*adxinv[1];
    Real dxz = adxinv[0]*adxinv[2];
    Real dyz = adxinv[1]*adxinv[2];

    GpuArray<Real,6> b
        {rhsx(i-1,j,k) - (-dyy * ( ex(i-1,j-1,k  ) +
                                   ex(i-1,j+1,k  ))
                         - dzz * ( ex(i-1,j  ,k+1) +
                                   ex(i-1,j  ,k-1))
                         + dxy * ( ey(i-1,j-1,k  )
                                  -ey(i-1,j  ,k  ))
                         + dxz * ( ez(i-1,j  ,k-1)
                                  -ez(i-1,j  ,k  ))),
         rhsx(i  ,j,k) - (-dyy * ( ex(i  ,j-1,k  ) +
                                   ex(i  ,j+1,k  ))
                         - dzz * ( ex(i  ,j  ,k+1) +
                                   ex(i  ,j  ,k-1))
                         + dxy * (-ey(i+1,j-1,k  )
                                  +ey(i+1,j  ,k  ))
                         + dxz * (-ez(i+1,j  ,k-1)
                                  +ez(i+1,j  ,k  ))),
         rhsy(i,j-1,k) - (-dxx * ( ey(i-1,j-1,k  ) +
                                   ey(i+1,j-1,k  ))
                         - dzz * ( ey(i  ,j-1,k-1) +
                                   ey(i  ,j-1,k+1))
                         + dxy * ( ex(i-1,j-1,k  )
                                  -ex(i  ,j-1,k  ))
                         + dyz * ( ez(i  ,j-1,k-1)
                                  -ez(i  ,j-1,k  ))),
         rhsy(i,j  ,k) - (-dxx * ( ey(i-1,j  ,k  ) +
                                   ey(i+1,j  ,k  ))
                         - dzz * ( ey(i  ,j  ,k-1) +
                                   ey(i  ,j  ,k+1))
                         + dxy * (-ex(i-1,j+1,k  )
                                  +ex(i  ,j+1,k  ))
                         + dyz * (-ez(i  ,j+1,k-1)
                                  +ez(i  ,j+1,k  ))),
         rhsz(i,j,k-1) - (-dxx * ( ez(i-1,j  ,k-1) +
                                   ez(i+1,j  ,k-1))
                         - dyy * ( ez(i  ,j-1,k-1) +
                                   ez(i  ,j+1,k-1))
                         + dxz * ( ex(i-1,j  ,k-1)
                                  -ex(i  ,j  ,k-1))
                         + dyz * ( ey(i  ,j-1,k-1)
                                  -ey(i  ,j  ,k-1))),
         rhsz(i,j,k  ) - (-dxx * ( ez(i-1,j  ,k  ) +
                                   ez(i+1,j  ,k  ))
                         - dyy * ( ez(i  ,j-1,k  ) +
                                   ez(i  ,j+1,k  ))
                         + dxz * (-ex(i-1,j  ,k+1)
                                  +ex(i  ,j  ,k+1))
                         + dyz * (-ey(i  ,j-1,k+1)
                                  +ey(i  ,j  ,k+1)))};

    if (sinfo.xlo_is_symmetric(i)) {
        b[0] = -b[1];
    } else if (sinfo.xhi_is_symmetric(i)) {
        b[1] = -b[0];
    }

    if (sinfo.ylo_is_symmetric(j)) {
        b[2] = -b[3];
    } else if (sinfo.yhi_is_symmetric(j)) {
        b[3] = -b[2];
    }

    if (sinfo.zlo_is_symmetric(k)) {
        b[4] = -b[5];
    } else if (sinfo.zhi_is_symmetric(k)) {
        b[5] = -b[4];
    }

    GpuArray<Real,6> x;
    lusolver(x.data(), b.data());
    ex(i-1,j  ,k  ) = x[0];
    ex(i  ,j  ,k  ) = x[1];
    ey(i  ,j-1,k  ) = x[2];
    ey(i  ,j  ,k  ) = x[3];
    ez(i  ,j  ,k-1) = x[4];
    ez(i  ,j  ,k  ) = x[5];
#endif
}

template <bool PCG>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void mlcurlcurl_cns_gs4 (int i, int j, int k,
                     Array4<Real> const& ex,
                     Array4<Real> const& ey,
                     Array4<Real> const& ez,
                     Array4<Real const> const& rhsx,
                     Array4<Real const> const& rhsy,
                     Array4<Real const> const& rhsz,
                     GpuArray<Real,AMREX_SPACEDIM> const& adxinv,
                     int color,
                     Array4<Real const> const& betax,
                     Array4<Real const> const& betay,
                     Array4<Real const> const& betaz,
                     CurlCurlDirichletInfo const& dinfo,
                     CurlCurlSymmetryInfo const& sinfo)
{
    if (dinfo.is_dirichlet_node(i,j,k)) { return; }

    int my_color = i%2 + 2*(j%2);
    if (k%2 != 0) {
        my_color = 3 - my_color;
    }

#if (AMREX_SPACEDIM == 2)

    Real dxx = adxinv[0] * adxinv[0];
    Real dyy = adxinv[1] * adxinv[1];

    if (((my_color == 0 || my_color == 3) && (color == 0 || color == 3)) ||
        ((my_color == 1 || my_color == 2) && (color == 1 || color == 2)))
    {
        Real gamma = (dxx+dyy)*Real(2.0) + betaz(i,j,k);
        Real ccez = - dxx * (ez(i-1,j  ,k  ) +
                             ez(i+1,j  ,k  ))
                    - dyy * (ez(i  ,j-1,k  ) +
                             ez(i  ,j+1,k  ));
        Real res = rhsz(i,j,k) - (gamma*ez(i,j,k) + ccez);
        constexpr Real omega = Real(1.15);
        ez(i,j,k) += omega/gamma * res;
    }

    if (my_color != color) { return; }

    Real dxy = adxinv[0]*adxinv[1];

    GpuArray<Real,6> b
        {rhsx(i-1,j,k) - (-dyy * ( ex(i-1,j-1,k  ) +
                                   ex(i-1,j+1,k  ))
                         + dxy * ( ey(i-1,j-1,k  )
                                  -ey(i-1,j  ,k  ))),
         rhsx(i  ,j,k) - (-dyy * ( ex(i  ,j-1,k  ) +
                                   ex(i  ,j+1,k  ))
                         + dxy * (-ey(i+1,j-1,k  )
                                  +ey(i+1,j  ,k  ))),
         rhsy(i,j-1,k) - (-dxx * ( ey(i-1,j-1,k  ) +
                                   ey(i+1,j-1,k  ))
                         + dxy * ( ex(i-1,j-1,k  )
                                  -ex(i  ,j-1,k  ))),
         rhsy(i,j  ,k) - (-dxx * ( ey(i-1,j  ,k  ) +
                                   ey(i+1,j  ,k  ))
                         + dxy * (-ex(i-1,j+1,k  )
                                  +ex(i  ,j+1,k  )))};

    GpuArray<Real,4> beta;

    if (sinfo.xlo_is_symmetric(i)) {
        b[0] = -b[1];
        beta[0] = beta[1] = betax(i,j,k);
    } else if (sinfo.xhi_is_symmetric(i)) {
        b[1] = -b[0];
        beta[0] = beta[1] = betax(i-1,j,k);
    } else {
        beta[0] = betax(i-1,j,k);
        beta[1] = betax(i  ,j,k);
    }

    if (sinfo.ylo_is_symmetric(j)) {
        b[2] = -b[3];
        beta[2] = beta[3] = betay(i,j,k);
    } else if (sinfo.yhi_is_symmetric(j)) {
        b[3] = -b[2];
        beta[2] = beta[3] = betay(i,j-1,k);
    } else {
        beta[2] = betay(i,j-1,k);
        beta[3] = betay(i,j  ,k);
    }

    if constexpr (PCG) {
        Real diagInv[4] = {Real(1.0) / (dyy*Real(2.0) + beta[0]),
                           Real(1.0) / (dyy*Real(2.0) + beta[1]),
                           Real(1.0) / (dxx*Real(2.0) + beta[2]),
                           Real(1.0) / (dxx*Real(2.0) + beta[3])};
        auto precond = [&] (Real * AMREX_RESTRICT z,
                            Real const* AMREX_RESTRICT r)
        {
            for (int m = 0; m < 4; ++m) { z[m] = r[m] * diagInv[m]; }
        };
        auto mat = [&] (Real * AMREX_RESTRICT Av,
                        Real const* AMREX_RESTRICT v)
        {
            Av[0] = (dyy*Real(2.0) + beta[0]) * v[0] - dxy * v[2] + dxy * v[3];
            Av[1] = (dyy*Real(2.0) + beta[1]) * v[1] + dxy * v[2] - dxy * v[3];
            Av[2] = -dxy * v[0] + dxy * v[1] + (dxx*Real(2.0) + beta[2]) * v[2];
            Av[3] =  dxy * v[0] - dxy * v[1] + (dxx*Real(2.0) + beta[3]) * v[3];
        };
        Real sol[4] = {0, 0, 0, 0};
        pcg_solve<4>(sol, b.data(), mat, precond, 8, Real(1.e-8));
        ex(i-1,j  ,k  ) = sol[0];
        ex(i  ,j  ,k  ) = sol[1];
        ey(i  ,j-1,k  ) = sol[2];
        ey(i  ,j  ,k  ) = sol[3];
    } else {
        LUSolver<4,Real> lusolver
            ({dyy*Real(2.0) + beta[0],
              Real(0.0),
             -dxy,
              dxy,
              //
              Real(0.0),
              dyy*Real(2.0) + beta[1],
              dxy,
             -dxy,
              //
             -dxy,
              dxy,
              dxx*Real(2.0) + beta[2],
              Real(0.0),
              //
              dxy,
             -dxy,
              Real(0.0),
              dxx*Real(2.0) + beta[3]});
        lusolver(beta.data(), b.data());
        ex(i-1,j  ,k  ) = beta[0];
        ex(i  ,j  ,k  ) = beta[1];
        ey(i  ,j-1,k  ) = beta[2];
        ey(i  ,j  ,k  ) = beta[3];
    }

#elif (AMREX_SPACEDIM == 3)

    if (my_color != color) { return; }

    Real dxx = adxinv[0]*adxinv[0];
    Real dyy = adxinv[1]*adxinv[1];
    Real dzz = adxinv[2]*adxinv[2];
    Real dxy = adxinv[0]*adxinv[1];
    Real dxz = adxinv[0]*adxinv[2];
    Real dyz = adxinv[1]*adxinv[2];

    GpuArray<Real,6> b
        {rhsx(i-1,j,k) - (-dyy * ( ex(i-1,j-1,k  ) +
                                   ex(i-1,j+1,k  ))
                         - dzz * ( ex(i-1,j  ,k+1) +
                                   ex(i-1,j  ,k-1))
                         + dxy * ( ey(i-1,j-1,k  )
                                  -ey(i-1,j  ,k  ))
                         + dxz * ( ez(i-1,j  ,k-1)
                                  -ez(i-1,j  ,k  ))),
         rhsx(i  ,j,k) - (-dyy * ( ex(i  ,j-1,k  ) +
                                   ex(i  ,j+1,k  ))
                         - dzz * ( ex(i  ,j  ,k+1) +
                                   ex(i  ,j  ,k-1))
                         + dxy * (-ey(i+1,j-1,k  )
                                  +ey(i+1,j  ,k  ))
                         + dxz * (-ez(i+1,j  ,k-1)
                                  +ez(i+1,j  ,k  ))),
         rhsy(i,j-1,k) - (-dxx * ( ey(i-1,j-1,k  ) +
                                   ey(i+1,j-1,k  ))
                         - dzz * ( ey(i  ,j-1,k-1) +
                                   ey(i  ,j-1,k+1))
                         + dxy * ( ex(i-1,j-1,k  )
                                  -ex(i  ,j-1,k  ))
                         + dyz * ( ez(i  ,j-1,k-1)
                                  -ez(i  ,j-1,k  ))),
         rhsy(i,j  ,k) - (-dxx * ( ey(i-1,j  ,k  ) +
                                   ey(i+1,j  ,k  ))
                         - dzz * ( ey(i  ,j  ,k-1) +
                                   ey(i  ,j  ,k+1))
                         + dxy * (-ex(i-1,j+1,k  )
                                  +ex(i  ,j+1,k  ))
                         + dyz * (-ez(i  ,j+1,k-1)
                                  +ez(i  ,j+1,k  ))),
         rhsz(i,j,k-1) - (-dxx * ( ez(i-1,j  ,k-1) +
                                   ez(i+1,j  ,k-1))
                         - dyy * ( ez(i  ,j-1,k-1) +
                                   ez(i  ,j+1,k-1))
                         + dxz * ( ex(i-1,j  ,k-1)
                                  -ex(i  ,j  ,k-1))
                         + dyz * ( ey(i  ,j-1,k-1)
                                  -ey(i  ,j  ,k-1))),
         rhsz(i,j,k  ) - (-dxx * ( ez(i-1,j  ,k  ) +
                                   ez(i+1,j  ,k  ))
                         - dyy * ( ez(i  ,j-1,k  ) +
                                   ez(i  ,j+1,k  ))
                         + dxz * (-ex(i-1,j  ,k+1)
                                  +ex(i  ,j  ,k+1))
                         + dyz * (-ey(i  ,j-1,k+1)
                                  +ey(i  ,j  ,k+1)))};

    GpuArray<Real,6> beta;

    if (sinfo.xlo_is_symmetric(i)) {
        b[0] = -b[1];
        beta[0] = beta[1] = betax(i,j,k);
    } else if (sinfo.xhi_is_symmetric(i)) {
        b[1] = -b[0];
        beta[0] = beta[1] = betax(i-1,j,k);
    } else {
        beta[0] = betax(i-1,j,k);
        beta[1] = betax(i  ,j,k);
    }

    if (sinfo.ylo_is_symmetric(j)) {
        b[2] = -b[3];
        beta[2] = beta[3] = betay(i,j,k);
    } else if (sinfo.yhi_is_symmetric(j)) {
        b[3] = -b[2];
        beta[2] = beta[3] = betay(i,j-1,k);
    } else {
        beta[2] = betay(i,j-1,k);
        beta[3] = betay(i,j  ,k);
    }

    if (sinfo.zlo_is_symmetric(k)) {
        b[4] = -b[5];
        beta[4] = beta[5] = betaz(i,j,k);
    } else if (sinfo.zhi_is_symmetric(k)) {
        b[5] = -b[4];
        beta[4] = beta[5] = betaz(i,j,k-1);
    } else {
        beta[4] = betaz(i,j,k-1);
        beta[5] = betaz(i,j,k  );
    }

    if constexpr (PCG) {
        Real diagInv[6] = {Real(1.0) / ((dyy+dzz)*Real(2.0) + beta[0]),
                           Real(1.0) / ((dyy+dzz)*Real(2.0) + beta[1]),
                           Real(1.0) / ((dxx+dzz)*Real(2.0) + beta[2]),
                           Real(1.0) / ((dxx+dzz)*Real(2.0) + beta[3]),
                           Real(1.0) / ((dxx+dyy)*Real(2.0) + beta[4]),
                           Real(1.0) / ((dxx+dyy)*Real(2.0) + beta[5])};
        auto precond = [&] (Real * AMREX_RESTRICT z,
                            Real const* AMREX_RESTRICT r)
        {
            for (int m = 0; m < 6; ++m) { z[m] = r[m] * diagInv[m]; }
        };
        auto mat = [&] (Real * AMREX_RESTRICT Av,
                        Real const* AMREX_RESTRICT v)
        {
            Av[0] = ((dyy+dzz)*Real(2.0) + beta[0]) * v[0] - dxy * v[2]
                + dxy * v[3] - dxz * v[4] + dxz * v[5];
            Av[1] = ((dyy+dzz)*Real(2.0) + beta[1]) * v[1] + dxy * v[2]
                - dxy * v[3] + dxz * v[4] - dxz * v[5];
            Av[2] = -dxy * v[0] + dxy * v[1] + ((dxx+dzz)*Real(2.0) + beta[2]) * v[2]
                - dyz * v[4] + dyz * v[5];
            Av[3] = dxy * v[0] - dxy * v[1] + ((dxx+dzz)*Real(2.0) + beta[3]) * v[3]
                + dyz * v[4] - dyz * v[5];
            Av[4] = -dxz * v[0] + dxz * v[1] - dyz * v[2] + dyz * v[3]
                + ((dxx+dyy)*Real(2.0) + beta[4]) * v[4];
            Av[5] = dxz * v[0] - dxz * v[1] + dyz * v[2] - dyz * v[3]
                + ((dxx+dyy)*Real(2.0) + beta[5]) * v[5];
        };
        Real sol[6] = {0, 0, 0, 0, 0, 0};
        pcg_solve<6>(sol, b.data(), mat, precond, 8, Real(1.e-8));
        ex(i-1,j  ,k  ) = sol[0];
        ex(i  ,j  ,k  ) = sol[1];
        ey(i  ,j-1,k  ) = sol[2];
        ey(i  ,j  ,k  ) = sol[3];
        ez(i  ,j  ,k-1) = sol[4];
        ez(i  ,j  ,k  ) = sol[5];
    } else {
        LUSolver<6,Real> lusolver
            ({(dyy+dzz)*Real(2.0) + beta[0],
              Real(0.0),
             -dxy,
              dxy,
             -dxz,
              dxz,
              //
              Real(0.0),
              (dyy+dzz)*Real(2.0) + beta[1],
              dxy,
             -dxy,
              dxz,
             -dxz,
              //
             -dxy,
              dxy,
              (dxx+dzz)*Real(2.0) + beta[2],
              Real(0.0),
             -dyz,
              dyz,
              //
              dxy,
             -dxy,
              Real(0.0),
              (dxx+dzz)*Real(2.0) + beta[3],
              dyz,
             -dyz,
              //
             -dxz,
              dxz,
             -dyz,
              dyz,
              (dxx+dyy)*Real(2.0) + beta[4],
              Real(0.0),
              //
              dxz,
             -dxz,
              dyz,
             -dyz,
              Real(0.0),
              (dxx+dyy)*Real(2.0) + beta[5]});
        lusolver(beta.data(), b.data());
        ex(i-1,j  ,k  ) = beta[0];
        ex(i  ,j  ,k  ) = beta[1];
        ey(i  ,j-1,k  ) = beta[2];
        ey(i  ,j  ,k  ) = beta[3];
        ez(i  ,j  ,k-1) = beta[4];
        ez(i  ,j  ,k  ) = beta[5];
    }
#endif
}

//! Variable-alpha 4-color Gauss-Seidel smoother.
/**
 * Coupled solve for the 2*SPACEDIM unknowns at each node with
 * spatially-varying alpha on face centerings.
 *
 * \param i,j,k Node index
 * \param ex,ey,ez E-field edge components (modified in place)
 * \param rhsx,rhsy,rhsz Right-hand side
 * \param dxinv Raw inverse cell sizes (NOT scaled by sqrt(alpha))
 * \param color Gauss-Seidel color (0..3)
 * \param betax,betay,betaz Per-cell beta coefficients on edge centerings
 * \param alphax Alpha on x-faces
 * \param alphay Alpha on y-faces
 * \param alphaz Alpha on z-faces
 * \param dinfo Dirichlet boundary info
 * \param sinfo Symmetry boundary info
 */
template <bool PCG>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void mlcurlcurl_cns_gs4_varalpha (int i, int j, int k,
                     Array4<Real> const& ex,
                     Array4<Real> const& ey,
                     Array4<Real> const& ez,
                     Array4<Real const> const& rhsx,
                     Array4<Real const> const& rhsy,
                     Array4<Real const> const& rhsz,
                     GpuArray<Real,AMREX_SPACEDIM> const& dxinv,
                     int color,
                     Array4<Real const> const& betax,
                     Array4<Real const> const& betay,
                     Array4<Real const> const& betaz,
                     Array4<Real const> const& alphax,
                     Array4<Real const> const& alphay,
                     Array4<Real const> const& alphaz,
                     CurlCurlDirichletInfo const& dinfo,
                     CurlCurlSymmetryInfo const& sinfo)
{
    if (dinfo.is_dirichlet_node(i,j,k)) { return; }

    int my_color = i%2 + 2*(j%2);
    if (k%2 != 0) {
        my_color = 3 - my_color;
    }

#if (AMREX_SPACEDIM == 2)

    Real dxx = dxinv[0] * dxinv[0];
    Real dyy = dxinv[1] * dxinv[1];

    // --- Ez SOR sweep (uses alphax, alphay) ---
    if (((my_color == 0 || my_color == 3) && (color == 0 || color == 3)) ||
        ((my_color == 1 || my_color == 2) && (color == 1 || color == 2)))
    {
        Real ay_hi = alphay(i  ,j,k);
        Real ay_lo = alphay(i-1,j,k);
        Real ax_hi = alphax(i,j  ,k);
        Real ax_lo = alphax(i,j-1,k);
        Real gamma = (ay_hi + ay_lo) * dxx + (ax_hi + ax_lo) * dyy
                   + betaz(i,j,k);
        Real ccez = - ay_hi * dxx * ez(i+1,j,k) - ay_lo * dxx * ez(i-1,j,k)
                    - ax_hi * dyy * ez(i,j+1,k) - ax_lo * dyy * ez(i,j-1,k);
        Real res = rhsz(i,j,k) - (gamma*ez(i,j,k) + ccez);
        constexpr Real omega = Real(1.15);
        ez(i,j,k) += omega/gamma * res;
    }

    if (my_color != color) { return; }

    Real dxy = dxinv[0]*dxinv[1];

    // Local alpha values at the 4 cells surrounding node (i,j)
    Real az00 = alphaz(i-1,j-1,k);
    Real az10 = alphaz(i  ,j-1,k);
    Real az01 = alphaz(i-1,j  ,k);
    Real az11 = alphaz(i  ,j  ,k);

    // b vector: rhs minus non-system neighbor contributions
    GpuArray<Real,6> b
        {rhsx(i-1,j,k) - (- az01*dyy * ex(i-1,j+1,k)
                           - az00*dyy * ex(i-1,j-1,k)
                           - az01*dxy * ey(i-1,j  ,k)
                           + az00*dxy * ey(i-1,j-1,k)),
         rhsx(i  ,j,k) - (- az11*dyy * ex(i  ,j+1,k)
                           - az10*dyy * ex(i  ,j-1,k)
                           + az11*dxy * ey(i+1,j  ,k)
                           - az10*dxy * ey(i+1,j-1,k)),
         rhsy(i,j-1,k) - (- az10*dxx * ey(i+1,j-1,k)
                           - az00*dxx * ey(i-1,j-1,k)
                           - az10*dxy * ex(i  ,j-1,k)
                           + az00*dxy * ex(i-1,j-1,k)),
         rhsy(i,j  ,k) - (- az11*dxx * ey(i+1,j  ,k)
                           - az01*dxx * ey(i-1,j  ,k)
                           + az11*dxy * ex(i  ,j+1,k)
                           - az01*dxy * ex(i-1,j+1,k))};

    GpuArray<Real,4> beta;

    if (sinfo.xlo_is_symmetric(i)) {
        b[0] = -b[1];
        beta[0] = beta[1] = betax(i,j,k);
    } else if (sinfo.xhi_is_symmetric(i)) {
        b[1] = -b[0];
        beta[0] = beta[1] = betax(i-1,j,k);
    } else {
        beta[0] = betax(i-1,j,k);
        beta[1] = betax(i  ,j,k);
    }

    if (sinfo.ylo_is_symmetric(j)) {
        b[2] = -b[3];
        beta[2] = beta[3] = betay(i,j,k);
    } else if (sinfo.yhi_is_symmetric(j)) {
        b[3] = -b[2];
        beta[2] = beta[3] = betay(i,j-1,k);
    } else {
        beta[2] = betay(i,j-1,k);
        beta[3] = betay(i,j  ,k);
    }

    // 4x4 matrix with variable alpha
    // A(0,0) = (az01+az00)*dyy + beta[0]
    // A(0,2) = -az00*dxy, A(0,3) = +az01*dxy
    // A(1,1) = (az11+az10)*dyy + beta[1]
    // A(1,2) = +az10*dxy, A(1,3) = -az11*dxy
    // A(2,2) = (az10+az00)*dxx + beta[2]
    // A(2,0) = -az00*dxy, A(2,1) = +az10*dxy
    // A(3,3) = (az11+az01)*dxx + beta[3]
    // A(3,0) = +az01*dxy, A(3,1) = -az11*dxy

    if constexpr (PCG) {
        Real diagInv[4] = {Real(1.0) / ((az01+az00)*dyy + beta[0]),
                           Real(1.0) / ((az11+az10)*dyy + beta[1]),
                           Real(1.0) / ((az10+az00)*dxx + beta[2]),
                           Real(1.0) / ((az11+az01)*dxx + beta[3])};
        auto precond = [&] (Real * AMREX_RESTRICT z,
                            Real const* AMREX_RESTRICT r)
        {
            for (int m = 0; m < 4; ++m) { z[m] = r[m] * diagInv[m]; }
        };
        auto mat = [&] (Real * AMREX_RESTRICT Av,
                        Real const* AMREX_RESTRICT v)
        {
            Av[0] = ((az01+az00)*dyy + beta[0]) * v[0] - az00*dxy * v[2] + az01*dxy * v[3];
            Av[1] = ((az11+az10)*dyy + beta[1]) * v[1] + az10*dxy * v[2] - az11*dxy * v[3];
            Av[2] = -az00*dxy * v[0] + az10*dxy * v[1] + ((az10+az00)*dxx + beta[2]) * v[2];
            Av[3] =  az01*dxy * v[0] - az11*dxy * v[1] + ((az11+az01)*dxx + beta[3]) * v[3];
        };
        Real sol[4] = {0, 0, 0, 0};
        pcg_solve<4>(sol, b.data(), mat, precond, 8, Real(1.e-8));
        ex(i-1,j  ,k  ) = sol[0];
        ex(i  ,j  ,k  ) = sol[1];
        ey(i  ,j-1,k  ) = sol[2];
        ey(i  ,j  ,k  ) = sol[3];
    } else {
        LUSolver<4,Real> lusolver
            ({(az01+az00)*dyy + beta[0],
              Real(0.0),
             -az00*dxy,
              az01*dxy,
              //
              Real(0.0),
              (az11+az10)*dyy + beta[1],
              az10*dxy,
             -az11*dxy,
              //
             -az00*dxy,
              az10*dxy,
              (az10+az00)*dxx + beta[2],
              Real(0.0),
              //
              az01*dxy,
             -az11*dxy,
              Real(0.0),
              (az11+az01)*dxx + beta[3]});
        lusolver(beta.data(), b.data());
        ex(i-1,j  ,k  ) = beta[0];
        ex(i  ,j  ,k  ) = beta[1];
        ey(i  ,j-1,k  ) = beta[2];
        ey(i  ,j  ,k  ) = beta[3];
    }

#elif (AMREX_SPACEDIM == 3)

    if (my_color != color) { return; }

    Real dxx = dxinv[0]*dxinv[0];
    Real dyy = dxinv[1]*dxinv[1];
    Real dzz = dxinv[2]*dxinv[2];
    Real dxy = dxinv[0]*dxinv[1];
    Real dxz = dxinv[0]*dxinv[2];
    Real dyz = dxinv[1]*dxinv[2];

    // Alpha values at the 4 cells surrounding node (i,j,k) for each face type
    // alphaz at (0,0,1): 4 values straddling in x,y
    Real az00 = alphaz(i-1,j-1,k);
    Real az10 = alphaz(i  ,j-1,k);
    Real az01 = alphaz(i-1,j  ,k);
    Real az11 = alphaz(i  ,j  ,k);
    // alphay at (0,1,0): 4 values straddling in x,z
    Real ay00 = alphay(i-1,j,k-1);
    Real ay10 = alphay(i  ,j,k-1);
    Real ay01 = alphay(i-1,j,k  );
    Real ay11 = alphay(i  ,j,k  );
    // alphax at (1,0,0): 4 values straddling in y,z
    Real ax00 = alphax(i,j-1,k-1);
    Real ax10 = alphax(i,j  ,k-1);
    Real ax01 = alphax(i,j-1,k  );
    Real ax11 = alphax(i,j  ,k  );

    // b vector: rhs minus non-system neighbor contributions
    GpuArray<Real,6> b
        {rhsx(i-1,j,k) - (- az01*dyy * ex(i-1,j+1,k)
                           - az00*dyy * ex(i-1,j-1,k)
                           - ay01*dzz * ex(i-1,j,k+1)
                           - ay00*dzz * ex(i-1,j,k-1)
                           - az01*dxy * ey(i-1,j  ,k)
                           + az00*dxy * ey(i-1,j-1,k)
                           - ay01*dxz * ez(i-1,j,k  )
                           + ay00*dxz * ez(i-1,j,k-1)),
         rhsx(i  ,j,k) - (- az11*dyy * ex(i  ,j+1,k)
                           - az10*dyy * ex(i  ,j-1,k)
                           - ay11*dzz * ex(i  ,j,k+1)
                           - ay10*dzz * ex(i  ,j,k-1)
                           + az11*dxy * ey(i+1,j  ,k)
                           - az10*dxy * ey(i+1,j-1,k)
                           + ay11*dxz * ez(i+1,j,k  )
                           - ay10*dxz * ez(i+1,j,k-1)),
         rhsy(i,j-1,k) - (- az10*dxx * ey(i+1,j-1,k)
                           - az00*dxx * ey(i-1,j-1,k)
                           - ax01*dzz * ey(i,j-1,k+1)
                           - ax00*dzz * ey(i,j-1,k-1)
                           - az10*dxy * ex(i  ,j-1,k)
                           + az00*dxy * ex(i-1,j-1,k)
                           - ax01*dyz * ez(i,j-1,k  )
                           + ax00*dyz * ez(i,j-1,k-1)),
         rhsy(i,j  ,k) - (- az11*dxx * ey(i+1,j  ,k)
                           - az01*dxx * ey(i-1,j  ,k)
                           - ax11*dzz * ey(i,j  ,k+1)
                           - ax10*dzz * ey(i,j  ,k-1)
                           + az11*dxy * ex(i  ,j+1,k)
                           - az01*dxy * ex(i-1,j+1,k)
                           + ax11*dyz * ez(i,j+1,k  )
                           - ax10*dyz * ez(i,j+1,k-1)),
         rhsz(i,j,k-1) - (- ay10*dxx * ez(i+1,j,k-1)
                           - ay00*dxx * ez(i-1,j,k-1)
                           - ax10*dyy * ez(i,j+1,k-1)
                           - ax00*dyy * ez(i,j-1,k-1)
                           - ay10*dxz * ex(i  ,j,k-1)
                           + ay00*dxz * ex(i-1,j,k-1)
                           - ax10*dyz * ey(i,j  ,k-1)
                           + ax00*dyz * ey(i,j-1,k-1)),
         rhsz(i,j,k  ) - (- ay11*dxx * ez(i+1,j,k  )
                           - ay01*dxx * ez(i-1,j,k  )
                           - ax11*dyy * ez(i,j+1,k  )
                           - ax01*dyy * ez(i,j-1,k  )
                           + ay11*dxz * ex(i  ,j,k+1)
                           - ay01*dxz * ex(i-1,j,k+1)
                           + ax11*dyz * ey(i,j  ,k+1)
                           - ax01*dyz * ey(i,j-1,k+1))};

    GpuArray<Real,6> beta;

    if (sinfo.xlo_is_symmetric(i)) {
        b[0] = -b[1];
        beta[0] = beta[1] = betax(i,j,k);
    } else if (sinfo.xhi_is_symmetric(i)) {
        b[1] = -b[0];
        beta[0] = beta[1] = betax(i-1,j,k);
    } else {
        beta[0] = betax(i-1,j,k);
        beta[1] = betax(i  ,j,k);
    }

    if (sinfo.ylo_is_symmetric(j)) {
        b[2] = -b[3];
        beta[2] = beta[3] = betay(i,j,k);
    } else if (sinfo.yhi_is_symmetric(j)) {
        b[3] = -b[2];
        beta[2] = beta[3] = betay(i,j-1,k);
    } else {
        beta[2] = betay(i,j-1,k);
        beta[3] = betay(i,j  ,k);
    }

    if (sinfo.zlo_is_symmetric(k)) {
        b[4] = -b[5];
        beta[4] = beta[5] = betaz(i,j,k);
    } else if (sinfo.zhi_is_symmetric(k)) {
        b[5] = -b[4];
        beta[4] = beta[5] = betaz(i,j,k-1);
    } else {
        beta[4] = betaz(i,j,k-1);
        beta[5] = betaz(i,j,k  );
    }

    // 6x6 matrix with variable alpha
    // Diagonals:
    // A(0,0) = (az01+az00)*dyy + (ay01+ay00)*dzz + beta[0]
    // A(1,1) = (az11+az10)*dyy + (ay11+ay10)*dzz + beta[1]
    // A(2,2) = (az10+az00)*dxx + (ax01+ax00)*dzz + beta[2]
    // A(3,3) = (az11+az01)*dxx + (ax11+ax10)*dzz + beta[3]
    // A(4,4) = (ay10+ay00)*dxx + (ax10+ax00)*dyy + beta[4]
    // A(5,5) = (ay11+ay01)*dxx + (ax11+ax01)*dyy + beta[5]

    Real d0 = (az01+az00)*dyy + (ay01+ay00)*dzz + beta[0];
    Real d1 = (az11+az10)*dyy + (ay11+ay10)*dzz + beta[1];
    Real d2 = (az10+az00)*dxx + (ax01+ax00)*dzz + beta[2];
    Real d3 = (az11+az01)*dxx + (ax11+ax10)*dzz + beta[3];
    Real d4 = (ay10+ay00)*dxx + (ax10+ax00)*dyy + beta[4];
    Real d5 = (ay11+ay01)*dxx + (ax11+ax01)*dyy + beta[5];

    // Off-diagonals (symmetric matrix):
    // A(0,2)=A(2,0) = -az00*dxy
    // A(0,3)=A(3,0) = +az01*dxy
    // A(0,4)=A(4,0) = -ay00*dxz
    // A(0,5)=A(5,0) = +ay01*dxz
    // A(1,2)=A(2,1) = +az10*dxy
    // A(1,3)=A(3,1) = -az11*dxy
    // A(1,4)=A(4,1) = +ay10*dxz
    // A(1,5)=A(5,1) = -ay11*dxz
    // A(2,4)=A(4,2) = -ax00*dyz
    // A(2,5)=A(5,2) = +ax01*dyz
    // A(3,4)=A(4,3) = +ax10*dyz
    // A(3,5)=A(5,3) = -ax11*dyz

    if constexpr (PCG) {
        Real diagInv[6] = {Real(1.0)/d0, Real(1.0)/d1, Real(1.0)/d2,
                           Real(1.0)/d3, Real(1.0)/d4, Real(1.0)/d5};
        auto precond = [&] (Real * AMREX_RESTRICT z,
                            Real const* AMREX_RESTRICT r)
        {
            for (int m = 0; m < 6; ++m) { z[m] = r[m] * diagInv[m]; }
        };
        auto mat = [&] (Real * AMREX_RESTRICT Av,
                        Real const* AMREX_RESTRICT v)
        {
            Av[0] = d0 * v[0] - az00*dxy * v[2] + az01*dxy * v[3]
                  - ay00*dxz * v[4] + ay01*dxz * v[5];
            Av[1] = d1 * v[1] + az10*dxy * v[2] - az11*dxy * v[3]
                  + ay10*dxz * v[4] - ay11*dxz * v[5];
            Av[2] = -az00*dxy * v[0] + az10*dxy * v[1] + d2 * v[2]
                  - ax00*dyz * v[4] + ax01*dyz * v[5];
            Av[3] = az01*dxy * v[0] - az11*dxy * v[1] + d3 * v[3]
                  + ax10*dyz * v[4] - ax11*dyz * v[5];
            Av[4] = -ay00*dxz * v[0] + ay10*dxz * v[1] - ax00*dyz * v[2]
                  + ax10*dyz * v[3] + d4 * v[4];
            Av[5] = ay01*dxz * v[0] - ay11*dxz * v[1] + ax01*dyz * v[2]
                  - ax11*dyz * v[3] + d5 * v[5];
        };
        Real sol[6] = {0, 0, 0, 0, 0, 0};
        pcg_solve<6>(sol, b.data(), mat, precond, 8, Real(1.e-8));
        ex(i-1,j  ,k  ) = sol[0];
        ex(i  ,j  ,k  ) = sol[1];
        ey(i  ,j-1,k  ) = sol[2];
        ey(i  ,j  ,k  ) = sol[3];
        ez(i  ,j  ,k-1) = sol[4];
        ez(i  ,j  ,k  ) = sol[5];
    } else {
        LUSolver<6,Real> lusolver
            ({d0,
              Real(0.0),
             -az00*dxy,
              az01*dxy,
             -ay00*dxz,
              ay01*dxz,
              //
              Real(0.0),
              d1,
              az10*dxy,
             -az11*dxy,
              ay10*dxz,
             -ay11*dxz,
              //
             -az00*dxy,
              az10*dxy,
              d2,
              Real(0.0),
             -ax00*dyz,
              ax01*dyz,
              //
              az01*dxy,
             -az11*dxy,
              Real(0.0),
              d3,
              ax10*dyz,
             -ax11*dyz,
              //
             -ay00*dxz,
              ay10*dxz,
             -ax00*dyz,
              ax10*dyz,
              d4,
              Real(0.0),
              //
              ay01*dxz,
             -ay11*dxz,
              ax01*dyz,
             -ax11*dyz,
              Real(0.0),
              d5});
        lusolver(beta.data(), b.data());
        ex(i-1,j  ,k  ) = beta[0];
        ex(i  ,j  ,k  ) = beta[1];
        ey(i  ,j-1,k  ) = beta[2];
        ey(i  ,j  ,k  ) = beta[3];
        ez(i  ,j  ,k-1) = beta[4];
        ez(i  ,j  ,k  ) = beta[5];
    }
#endif
}

#endif

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void mlcurlcurl_cns_interpadd (int dir, int i, int j, int k,
                           Array4<Real> const& fine,
                           Array4<Real const> const& crse)
{
    int ic = amrex::coarsen(i,2);
    int jc = amrex::coarsen(j,2);
    int kc = amrex::coarsen(k,2);
    if (dir == 0) {
        bool j_is_odd = (jc*2 != j);
        bool k_is_odd = (kc*2 != k);
        if (j_is_odd && k_is_odd) {
            fine(i,j,k) += Real(0.25) *
                (crse(ic,jc,kc  ) + crse(ic,jc+1,kc   ) +
                 crse(ic,jc,kc+1) + crse(ic,jc+1,kc+1));
        } else if (j_is_odd) {
            fine(i,j,k) += Real(0.5) * (crse(ic,jc,kc) + crse(ic,jc+1,kc));
        } else if (k_is_odd) {
            fine(i,j,k) += Real(0.5) * (crse(ic,jc,kc) + crse(ic,jc,kc+1));
        } else {
            fine(i,j,k) += crse(ic,jc,kc);
        }
    } else if (dir == 1) {
        bool i_is_odd = (ic*2 != i);
        bool k_is_odd = (kc*2 != k);
        if (i_is_odd && k_is_odd) {
            fine(i,j,k) += Real(0.25) *
                (crse(ic  ,jc,kc  ) + crse(ic+1,jc,kc  ) +
                 crse(ic  ,jc,kc+1) + crse(ic+1,jc,kc+1));
        } else if (i_is_odd) {
            fine(i,j,k) += Real(0.5) * (crse(ic,jc,kc) + crse(ic+1,jc,kc));
        } else if (k_is_odd) {
            fine(i,j,k) += Real(0.5) * (crse(ic,jc,kc) + crse(ic,jc,kc+1));
        } else {
            fine(i,j,k) += crse(ic,jc,kc);
        }
    } else {
        bool i_is_odd = (ic*2 != i);
        bool j_is_odd = (jc*2 != j);
        if (i_is_odd && j_is_odd) {
            fine(i,j,k) += Real(0.25) *
                (crse(ic  ,jc  ,kc) + crse(ic+1,jc  ,kc) +
                 crse(ic  ,jc+1,kc) + crse(ic+1,jc+1,kc));
        } else if (i_is_odd) {
            fine(i,j,k) += Real(0.5) * (crse(ic,jc,kc) + crse(ic+1,jc,kc));
        } else if (j_is_odd) {
            fine(i,j,k) += Real(0.5) * (crse(ic,jc,kc) + crse(ic,jc+1,kc));
        } else {
            fine(i,j,k) += crse(ic,jc,kc);
        }
    }
}

//! Beta-weighted AMR interpolation for operator-dependent prolongation.
/**
 * Same as mlcurlcurl_cns_interpadd but uses the fine-level beta
 * coefficient as interpolation weight.  At coefficient discontinuities
 * (e.g. beta = 1000 on one side, 0.001 on the other), this prevents the
 * coarse correction from overshooting into the low-beta region by
 * weighting toward the high-beta side.  Follows MLNodeLaplacian's
 * mlndlap_interpadd_aa pattern.
 *
 * For nodal directions that need interpolation (odd indices), the weight
 * between coarse nodes jc and jc+1 is:
 *   w_lo = beta(fine even pos below) / (w_lo + w_hi)
 *   w_hi = beta(fine even pos above) / (w_lo + w_hi)
 * At even positions (coincident with coarse), injection (weight = 1).
 */
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void mlcurlcurl_cns_interpadd_beta (int dir, int i, int j, int k,
                                    Array4<Real> const& fine,
                                    Array4<Real const> const& crse,
                                    Array4<Real const> const& beta)
{
    int ic = amrex::coarsen(i,2);
    int jc = amrex::coarsen(j,2);
    int kc = amrex::coarsen(k,2);

    // Helper: beta-weighted average of two values.
    auto bwt = [] (Real v0, Real b0, Real v1, Real b1) -> Real
    {
        Real w = b0 + b1;
        return (w > Real(0.0)) ? (b0*v0 + b1*v1) / w
                               : Real(0.5) * (v0 + v1);
    };

    if (dir == 0) { // Ex: cell-x, nodal-y, nodal-z
        bool j_is_odd = (jc*2 != j);
        bool k_is_odd = (kc*2 != k);
        if (j_is_odd && k_is_odd) {
            // Bilinear in y,z using beta at the 4 coarse node positions
            Real b00 = beta(i, jc*2,   kc*2  );
            Real b10 = beta(i,(jc+1)*2,kc*2  );
            Real b01 = beta(i, jc*2,  (kc+1)*2);
            Real b11 = beta(i,(jc+1)*2,(kc+1)*2);
            Real w = b00 + b10 + b01 + b11;
            fine(i,j,k) += (w > Real(0.0))
                ? (b00*crse(ic,jc,kc  ) + b10*crse(ic,jc+1,kc  )
                 + b01*crse(ic,jc,kc+1) + b11*crse(ic,jc+1,kc+1)) / w
                : Real(0.25) * (crse(ic,jc,kc) + crse(ic,jc+1,kc)
                              + crse(ic,jc,kc+1) + crse(ic,jc+1,kc+1));
        } else if (j_is_odd) {
            fine(i,j,k) += bwt(crse(ic,jc,kc), beta(i,jc*2,kc*2),
                               crse(ic,jc+1,kc), beta(i,(jc+1)*2,kc*2));
        } else if (k_is_odd) {
            fine(i,j,k) += bwt(crse(ic,jc,kc), beta(i,jc*2,kc*2),
                               crse(ic,jc,kc+1), beta(i,jc*2,(kc+1)*2));
        } else {
            fine(i,j,k) += crse(ic,jc,kc);
        }
    } else if (dir == 1) { // Ey: nodal-x, cell-y, nodal-z
        bool i_is_odd = (ic*2 != i);
        bool k_is_odd = (kc*2 != k);
        if (i_is_odd && k_is_odd) {
            Real b00 = beta(ic*2,   j, kc*2  );
            Real b10 = beta((ic+1)*2,j, kc*2  );
            Real b01 = beta(ic*2,   j,(kc+1)*2);
            Real b11 = beta((ic+1)*2,j,(kc+1)*2);
            Real w = b00 + b10 + b01 + b11;
            fine(i,j,k) += (w > Real(0.0))
                ? (b00*crse(ic,jc,kc  ) + b10*crse(ic+1,jc,kc  )
                 + b01*crse(ic,jc,kc+1) + b11*crse(ic+1,jc,kc+1)) / w
                : Real(0.25) * (crse(ic,jc,kc) + crse(ic+1,jc,kc)
                              + crse(ic,jc,kc+1) + crse(ic+1,jc,kc+1));
        } else if (i_is_odd) {
            fine(i,j,k) += bwt(crse(ic,jc,kc), beta(ic*2,j,kc*2),
                               crse(ic+1,jc,kc), beta((ic+1)*2,j,kc*2));
        } else if (k_is_odd) {
            fine(i,j,k) += bwt(crse(ic,jc,kc), beta(ic*2,j,kc*2),
                               crse(ic,jc,kc+1), beta(ic*2,j,(kc+1)*2));
        } else {
            fine(i,j,k) += crse(ic,jc,kc);
        }
    } else { // Ez: nodal-x, nodal-y, cell-z
        bool i_is_odd = (ic*2 != i);
        bool j_is_odd = (jc*2 != j);
        if (i_is_odd && j_is_odd) {
            Real b00 = beta(ic*2,    jc*2,   k);
            Real b10 = beta((ic+1)*2,jc*2,   k);
            Real b01 = beta(ic*2,   (jc+1)*2,k);
            Real b11 = beta((ic+1)*2,(jc+1)*2,k);
            Real w = b00 + b10 + b01 + b11;
            fine(i,j,k) += (w > Real(0.0))
                ? (b00*crse(ic,jc,kc) + b10*crse(ic+1,jc,kc)
                 + b01*crse(ic,jc+1,kc) + b11*crse(ic+1,jc+1,kc)) / w
                : Real(0.25) * (crse(ic,jc,kc) + crse(ic+1,jc,kc)
                              + crse(ic,jc+1,kc) + crse(ic+1,jc+1,kc));
        } else if (i_is_odd) {
            fine(i,j,k) += bwt(crse(ic,jc,kc), beta(ic*2,jc*2,k),
                               crse(ic+1,jc,kc), beta((ic+1)*2,jc*2,k));
        } else if (j_is_odd) {
            fine(i,j,k) += bwt(crse(ic,jc,kc), beta(ic*2,jc*2,k),
                               crse(ic,jc+1,kc), beta(ic*2,(jc+1)*2,k));
        } else {
            fine(i,j,k) += crse(ic,jc,kc);
        }
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void mlcurlcurl_cns_restriction (int dir, int i, int j, int k,
                             Array4<Real> const& crse,
                             Array4<Real const> const& fine,
                             CurlCurlDirichletInfo const& dinfo)
{
    int ii = i*2;
    int jj = j*2;
    int kk = k*2;
    if (dinfo.is_dirichlet_edge(dir,ii,jj,kk)) {
        crse(i,j,k) = Real(0.0);
    }
    else
    {
#if (AMREX_SPACEDIM == 3)
        if (dir == 0) {
            crse(i,j,k) = Real(1./32.) * (fine(ii  ,jj-1,kk-1)             +
                                          fine(ii  ,jj  ,kk-1) * Real(2.0) +
                                          fine(ii  ,jj+1,kk-1)             +
                                          fine(ii  ,jj-1,kk  ) * Real(2.0) +
                                          fine(ii  ,jj  ,kk  ) * Real(4.0) +
                                          fine(ii  ,jj+1,kk  ) * Real(2.0) +
                                          fine(ii  ,jj-1,kk+1)             +
                                          fine(ii  ,jj  ,kk+1) * Real(2.0) +
                                          fine(ii  ,jj+1,kk+1)             +
                                          fine(ii+1,jj-1,kk-1)             +
                                          fine(ii+1,jj  ,kk-1) * Real(2.0) +
                                          fine(ii+1,jj+1,kk-1)             +
                                          fine(ii+1,jj-1,kk  ) * Real(2.0) +
                                          fine(ii+1,jj  ,kk  ) * Real(4.0) +
                                          fine(ii+1,jj+1,kk  ) * Real(2.0) +
                                          fine(ii+1,jj-1,kk+1)             +
                                          fine(ii+1,jj  ,kk+1) * Real(2.0) +
                                          fine(ii+1,jj+1,kk+1)             );
        } else if (dir == 1) {
            crse(i,j,k) = Real(1./32.) * (fine(ii-1,jj  ,kk-1)             +
                                          fine(ii  ,jj  ,kk-1) * Real(2.0) +
                                          fine(ii+1,jj  ,kk-1)             +
                                          fine(ii-1,jj  ,kk  ) * Real(2.0) +
                                          fine(ii  ,jj  ,kk  ) * Real(4.0) +
                                          fine(ii+1,jj  ,kk  ) * Real(2.0) +
                                          fine(ii-1,jj  ,kk+1)             +
                                          fine(ii  ,jj  ,kk+1) * Real(2.0) +
                                          fine(ii+1,jj  ,kk+1)             +
                                          fine(ii-1,jj+1,kk-1)             +
                                          fine(ii  ,jj+1,kk-1) * Real(2.0) +
                                          fine(ii+1,jj+1,kk-1)             +
                                          fine(ii-1,jj+1,kk  ) * Real(2.0) +
                                          fine(ii  ,jj+1,kk  ) * Real(4.0) +
                                          fine(ii+1,jj+1,kk  ) * Real(2.0) +
                                          fine(ii-1,jj+1,kk+1)             +
                                          fine(ii  ,jj+1,kk+1) * Real(2.0) +
                                          fine(ii+1,jj+1,kk+1)             );
        } else {
            crse(i,j,k) = Real(1./32.) * (fine(ii-1,jj-1,kk  )             +
                                          fine(ii  ,jj-1,kk  ) * Real(2.0) +
                                          fine(ii+1,jj-1,kk  )             +
                                          fine(ii-1,jj  ,kk  ) * Real(2.0) +
                                          fine(ii  ,jj  ,kk  ) * Real(4.0) +
                                          fine(ii+1,jj  ,kk  ) * Real(2.0) +
                                          fine(ii-1,jj+1,kk  )             +
                                          fine(ii  ,jj+1,kk  ) * Real(2.0) +
                                          fine(ii+1,jj+1,kk  )             +
                                          fine(ii-1,jj-1,kk+1)             +
                                          fine(ii  ,jj-1,kk+1) * Real(2.0) +
                                          fine(ii+1,jj-1,kk+1)             +
                                          fine(ii-1,jj  ,kk+1) * Real(2.0) +
                                          fine(ii  ,jj  ,kk+1) * Real(4.0) +
                                          fine(ii+1,jj  ,kk+1) * Real(2.0) +
                                          fine(ii-1,jj+1,kk+1)             +
                                          fine(ii  ,jj+1,kk+1) * Real(2.0) +
                                          fine(ii+1,jj+1,kk+1)             );
        }
#elif (AMREX_SPACEDIM == 2)
        if (dir == 0) {
            crse(i,j,0) = Real(0.125) * (fine(ii  ,jj-1,0)             +
                                         fine(ii  ,jj  ,0) * Real(2.0) +
                                         fine(ii  ,jj+1,0)             +
                                         fine(ii+1,jj-1,0)             +
                                         fine(ii+1,jj  ,0) * Real(2.0) +
                                         fine(ii+1,jj+1,0)             );
        } else if (dir == 1) {
            crse(i,j,0) = Real(0.125) * (fine(ii-1,jj  ,0)             +
                                         fine(ii  ,jj  ,0) * Real(2.0) +
                                         fine(ii+1,jj  ,0)             +
                                         fine(ii-1,jj+1,0)             +
                                         fine(ii  ,jj+1,0) * Real(2.0) +
                                         fine(ii+1,jj+1,0)             );
        } else {
            crse(i,j,0) = Real(1./16.)*(fine(ii-1,jj-1,0) + Real(2.)*fine(ii  ,jj-1,0) +          fine(ii+1,jj-1,0)
                             + Real(2.)*fine(ii-1,jj  ,0) + Real(4.)*fine(ii  ,jj  ,0) + Real(2.)*fine(ii+1,jj  ,0)
                                      + fine(ii-1,jj+1,0) + Real(2.)*fine(ii  ,jj+1,0) +          fine(ii+1,jj+1,0));

        }

#elif (AMREX_SPACEDIM == 1)
        if (dir == 0) {
            crse(i,0,0) = Real(0.5) * (fine(ii,0,0) + fine(ii+1,0,0));
        } else {
            crse(i,0,0) = Real(0.25) * (fine(ii-1,0,0)             +
                                        fine(ii  ,0,0) * Real(2.0) +
                                        fine(ii+1,0,0));
        }
#endif
    }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void mlcurlcurl_cns_bc_symmetry (int i, int j, int k, Orientation face, IndexType it,
                             Array4<Real> const& a)
{
    int const idir = face.coordDir();
    int offset = face.isLow() ? 1 : -1;
    Real sign;
    if (it.cellCentered(idir)) {
        sign = Real(-1.0);
    } else {
        sign = Real(1.0);
        offset *= 2;
    }

    if (idir == 0) {
        a(i,j,k) = sign * a(i+offset,j,k);
    } else if (idir == 1) {
        a(i,j,k) = sign * a(i,j+offset,k);
    } else {
        a(i,j,k) = sign * a(i,j,k+offset);
    }
}

}

#endif
