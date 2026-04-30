#include "initProb_K.H"

#include "MyTest.H"

#include <AMReX_iMultiFab.H>

using namespace amrex;

enum class CoordID { Cartesian, Cyl1D, Cyl2D, Sph1D };

void
MyTest::initProb ()
{
    const auto prob_lo = geom.ProbLoArray();
    const auto dx      = geom.CellSizeArray();
    const auto a = alpha;
    const auto b = beta;
    const auto ndhi = geom.Domain().bigEnd()+1;

    auto cid = CoordID::Cartesian;
#if (AMREX_SPACEDIM == 1)
    if (this->coord == 1) {
        cid = CoordID::Cyl1D;
        AMREX_ALWAYS_ASSERT(prob_lo[0] == 0);
    } else if (this->coord == 2) {
        cid = CoordID::Sph1D;
        AMREX_ALWAYS_ASSERT(prob_lo[0] == 0);
    }
#elif (AMREX_SPACEDIM == 2)
    if (this->coord == 1) {
        amrex::Abort("2D Cylindrical support will be added later");
    }
#endif

    amrex::ignore_unused(cid,ndhi);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(rhs[0], TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& gbx = mfi.tilebox(IntVect(1),IntVect(1));
        GpuArray<Array4<Real>,3> rhsfab{rhs[0].array(mfi),
                                        rhs[1].array(mfi),
                                        rhs[2].array(mfi)};
        GpuArray<Array4<Real>,3> solfab{solution[0].array(mfi),
                                        solution[1].array(mfi),
                                        solution[2].array(mfi)};
        Array4<Real> alphafab;
        if (variable_alpha) {
            alphafab = alpha_node.array(mfi);
        }
        amrex::ParallelFor(gbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
#if (AMREX_SPACEDIM == 1)
            if (cid == CoordID::Sph1D) {
                actual_init_prob_sph1d(i,j,k,rhsfab,solfab,prob_lo,dx,a,b,ndhi,alphafab);
            } else if (cid == CoordID::Cyl1D) {
                actual_init_prob_cyl1d(i,j,k,rhsfab,solfab,prob_lo,dx,a,b,ndhi,alphafab);
            } else
#endif
            {
                actual_init_prob(i,j,k,rhsfab,solfab,prob_lo,dx,a,b,alphafab);
            }
        });
    }
}

void
MyTest::populateOversetMask ()
{
    const auto prob_lo = geom.ProbLoArray();
    const auto prob_hi = geom.ProbHiArray();
    const auto dx      = geom.CellSizeArray();
    const Box& domain  = geom.Domain();

    for (int idim = 0; idim < 3; ++idim) {
        overset_mask_mf[idim].setVal(1);
    }

    if (overset_pattern == OversetPattern::None) {
        return;
    }

    if (overset_pattern == OversetPattern::Point) {
        IntVect p(0);
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            p[d] = domain.smallEnd(d) + n_cell/2;
        }
        for (int idim = 0; idim < 3; ++idim) {
            for (MFIter mfi(overset_mask_mf[idim]); mfi.isValid(); ++mfi) {
                if (mfi.validbox().contains(p)) {
                    auto const& m = overset_mask_mf[idim].array(mfi);
                    const Box pbx(p, p);
                    ParallelFor(pbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        m(i,j,k) = 0;
                    });
                }
            }
        }
    } else if (overset_pattern == OversetPattern::Box) {
        const Box mask_region = amrex::grow(domain, -(n_cell/4));
        for (int idim = 0; idim < 3; ++idim) {
            const Box mask_region_conv = amrex::convert(mask_region, overset_mask_mf[idim].ixType());
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(overset_mask_mf[idim], TilingIfNotGPU());
                 mfi.isValid(); ++mfi)
            {
                const Box& bx = mfi.tilebox();
                auto const& m = overset_mask_mf[idim].array(mfi);
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if (mask_region_conv.contains(IntVect(AMREX_D_DECL(i,j,k)))) {
                        m(i,j,k) = 0;
                    }
                });
            }
        }
    } else { // Blob
        const Real Lx = prob_hi[0] - prob_lo[0];
        const Real cx = prob_lo[0] + Real(0.4)*Lx;
        const Real r  = Lx/Real(8.0);
        const Real r2 = r*r;
#if (AMREX_SPACEDIM > 1)
        const Real Ly = prob_hi[1] - prob_lo[1];
        const Real cy = prob_lo[1] + Real(0.6)*Ly;
#endif
#if (AMREX_SPACEDIM == 3)
        const Real Lz = prob_hi[2] - prob_lo[2];
        const Real cz = prob_lo[2] + Real(0.45)*Lz;
#endif
        for (int idim = 0; idim < 3; ++idim) {
            const Real ox = (idim == 0 && 0 < AMREX_SPACEDIM) ? Real(0.5) : Real(0);
#if (AMREX_SPACEDIM > 1)
            const Real oy = (idim == 1) ? Real(0.5) : Real(0);
#endif
#if (AMREX_SPACEDIM == 3)
            const Real oz = (idim == 2) ? Real(0.5) : Real(0);
#endif
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(overset_mask_mf[idim], TilingIfNotGPU());
                 mfi.isValid(); ++mfi)
            {
                const Box& bx = mfi.tilebox();
                auto const& m = overset_mask_mf[idim].array(mfi);
                ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    Real x = prob_lo[0] + (Real(i) + ox)*dx[0];
                    Real d2 = (x - cx)*(x - cx);
#if (AMREX_SPACEDIM > 1)
                    Real y = prob_lo[1] + (Real(j) + oy)*dx[1];
                    d2 += (y - cy)*(y - cy);
#endif
#if (AMREX_SPACEDIM == 3)
                    Real z = prob_lo[2] + (Real(k) + oz)*dx[2];
                    d2 += (z - cz)*(z - cz);
#endif
                    if (d2 < r2) { m(i,j,k) = 0; }
                });
            }
        }
    }

    // Safety: never mask domain-boundary DOFs (BC code owns those).
    // Re-set mask=1 on any DOF that lies on a nodal domain face.
    for (int idim = 0; idim < 3; ++idim) {
        IntVect itype(1);
#if (AMREX_SPACEDIM < 3)
        if (idim < AMREX_SPACEDIM)
#endif
        {
            itype[idim] = 0;
        }
        const Box ndom = amrex::convert(domain, itype);
        for (int d = 0; d < AMREX_SPACEDIM; ++d) {
            if (itype[d] != 1) { continue; }   // only nodal directions own BC
            for (int side = 0; side < 2; ++side) {
                Box face = ndom;
                if (side == 0) {
                    face.setBig(d, ndom.smallEnd(d));
                } else {
                    face.setSmall(d, ndom.bigEnd(d));
                }
                for (MFIter mfi(overset_mask_mf[idim]); mfi.isValid(); ++mfi) {
                    Box isect = mfi.validbox() & face;
                    if (isect.ok()) {
                        auto const& m = overset_mask_mf[idim].array(mfi);
                        ParallelFor(isect, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                        {
                            m(i,j,k) = 1;
                        });
                    }
                }
            }
        }
    }
}
