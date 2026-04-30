#include <AMReX_MLCurlCurl.H>
#include <AMReX_Arena.H>
#include <AMReX_Math.H>
#include <AMReX_MLNodeLinOp_K.H>

namespace amrex {

MLCurlCurl::MLCurlCurl (const Vector<Geometry>& a_geom,
                        const Vector<BoxArray>& a_grids,
                        const Vector<DistributionMapping>& a_dmap,
                        const LPInfo& a_info, int a_coord)
{
    define(a_geom, a_grids, a_dmap, a_info, a_coord);
}

void MLCurlCurl::define (const Vector<Geometry>& a_geom,
                         const Vector<BoxArray>& a_grids,
                         const Vector<DistributionMapping>& a_dmap,
                         const LPInfo& a_info, int a_coord)
{
    MLLinOpT<MF>::define(a_geom, a_grids, a_dmap, a_info, {});

    m_coord = a_coord;
#if (AMREX_SPACEDIM == 2)
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_coord == 0,
                                     "CurlCurl: In 2D, only Cartesian is supported.");
#elif (AMREX_SPACEDIM == 3)
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_coord == 0,
                                     "CurlCurl: In 3D, only Cartesian is supported.");
#endif
    if (m_coord == 1 || m_coord == 2) {
        AMREX_ALWAYS_ASSERT(a_geom[0].ProbLo(0) == 0);
    }

    m_dotmask.resize(this->m_num_amr_levels);
    for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev) {
        m_dotmask[amrlev].resize(this->m_num_mg_levels[amrlev]);
    }

    m_bcoefs.resize(this->m_num_amr_levels);
    for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev) {
        m_bcoefs[amrlev].resize(this->m_num_mg_levels[amrlev]);
    }

    m_acoefs.resize(this->m_num_amr_levels);
    for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev) {
        m_acoefs[amrlev].resize(this->m_num_mg_levels[amrlev]);
    }

    m_lusolver.resize(this->m_num_amr_levels);
    for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev) {
        m_lusolver[amrlev].resize(this->m_num_mg_levels[amrlev]);
    }

    m_overset_mask.resize(this->m_num_amr_levels);
    for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev) {
        m_overset_mask[amrlev].resize(this->m_num_mg_levels[amrlev]);
    }

    m_nodal_overset_mask.resize(this->m_num_amr_levels);
    for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev) {
        m_nodal_overset_mask[amrlev].resize(this->m_num_mg_levels[amrlev]);
    }
}

void MLCurlCurl::define (const Vector<Geometry>& a_geom,
                         const Vector<BoxArray>& a_grids,
                         const Vector<DistributionMapping>& a_dmap,
                         const Vector<Array<const iMultiFab*, 3>>& a_overset_mask,
                         const LPInfo& a_info, int a_coord)
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(a_overset_mask.size() == 1,
                                     "AMR not currently supported with overset mask");

    // Define overset mask multifabs and copy over. ng=1 so the smoother's
    // per-box loop over nodal indices can safely read the cell-edge mask at
    // i=box_hi+1: domain bndry ghost set to 1 (active), internal ghosts filled
    // from neighbour fabs (periodic via geom).
    auto namrlevs = static_cast<int>(a_geom.size());
    m_overset_mask.resize(namrlevs);
    for (int amrlev = 0; amrlev < namrlevs; ++amrlev) {
        constexpr int ng_osm = 1;
        m_overset_mask[amrlev].push_back({std::make_unique<iMultiFab>(amrex::convert(a_grids[amrlev], m_etype[0]),
                                                                         a_dmap[amrlev], 1, ng_osm),
                                            std::make_unique<iMultiFab>(amrex::convert(a_grids[amrlev], m_etype[1]),
                                                                         a_dmap[amrlev], 1, ng_osm),
                                                std::make_unique<iMultiFab>(amrex::convert(a_grids[amrlev], m_etype[2]),
                                                                         a_dmap[amrlev], 1, ng_osm),});
        for (int idim = 0; idim < 3; ++idim) {
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
                a_overset_mask[amrlev][idim]->ixType() == IndexType(m_etype[idim]),
                "MLCurlCurl: overset mask index type must match edge centering");
            iMultiFab::Copy(*(m_overset_mask[amrlev][0][idim]), *a_overset_mask[amrlev][idim], 0, 0, 1, 0);
            m_overset_mask[amrlev][0][idim]->setDomainBndry(1, a_geom[amrlev]);
            m_overset_mask[amrlev][0][idim]->FillBoundary(a_geom[amrlev].periodicity());
        }
        if (amrlev > 1) {
            AMREX_ALWAYS_ASSERT(amrex::refine(a_geom[amrlev-1].Domain(),2)
                                == a_geom[amrlev].Domain());
        }
    }

    // Determine max MG coarsening level by attempting to coarsen the supplied
    // edge/face overset masks. A coarse cell is allowed only if all underlying
    // fine entries agree (all 1 -> 1, all 0 -> 0); any mixed coarse cell makes
    // the level uncoarsenable.
    {
        const int amrlev = 0;
        Box dom = a_geom[0].Domain();
        Geometry cgeom = a_geom[0];
        for (int mglev = 1; mglev <= a_info.max_coarsening_level; ++mglev) {
            AMREX_ALWAYS_ASSERT(this->mg_coarsen_ratio == 2);
            if (!dom.coarsenable(2)) { break; }

            Array<std::unique_ptr<iMultiFab>, 3> crse;
            bool ok = true;
            for (int idim = 0; idim < 3; ++idim) {
                iMultiFab const& fine = *(m_overset_mask[amrlev][mglev-1][idim]);
                if (!fine.boxArray().coarsenable(2)) { ok = false; break; }
                crse[idim] = std::make_unique<iMultiFab>(
                    amrex::coarsen(fine.boxArray(), 2),
                    fine.DistributionMap(), 1, 1);

                if (m_etype[idim] == IntVect(1)) {
                    // Pure-nodal: coarse node value = corresponding fine node value.
                    amrex::average_down_nodal(fine, *crse[idim], IntVect(2));
                } else {
                    ReduceOps<ReduceOpSum> reduce_op;
                    ReduceData<int> reduce_data(reduce_op);
                    using ReduceTuple = typename decltype(reduce_data)::Type;
                    const IntVect et = m_etype[idim];
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
                    for (MFIter mfi(*crse[idim], TilingIfNotGPU()); mfi.isValid(); ++mfi)
                    {
                        const Box& bx = mfi.tilebox();
                        Array4<int const> const& fmsk = fine.const_array(mfi);
                        Array4<int> const& cmsk = crse[idim]->array(mfi);
                        reduce_op.eval(bx, reduce_data,
                        [=] AMREX_GPU_HOST_DEVICE (Box const& b) -> ReduceTuple
                        {
                            return { coarsen_overset_mask_etype(b, et, cmsk, fmsk) };
                        });
                    }
                    int mixed = amrex::get<0>(reduce_data.value(reduce_op));
                    ParallelAllReduce::Max(mixed, ParallelContext::CommunicatorSub());
                    if (mixed > 0) { ok = false; break; }
                }
            }
            if (!ok) { break; }
            dom.coarsen(2);
            cgeom.coarsen(IntVect(2));
            for (int idim = 0; idim < 3; ++idim) {
                crse[idim]->setDomainBndry(1, cgeom);
                crse[idim]->FillBoundary(cgeom.periodicity());
            }
            m_overset_mask[amrlev].push_back({std::move(crse[0]),
                                              std::move(crse[1]),
                                              std::move(crse[2])});
        }
        int max_overset_mask_coarsening_level =
            static_cast<int>(m_overset_mask[amrlev].size()) - 1;
        ParallelAllReduce::Min(max_overset_mask_coarsening_level,
                               ParallelContext::CommunicatorSub());
        m_overset_mask[amrlev].resize(max_overset_mask_coarsening_level + 1);

        LPInfo linfo = a_info;
        linfo.max_coarsening_level = std::min(a_info.max_coarsening_level,
                                              max_overset_mask_coarsening_level);
        MLLinOpT<MF>::define(a_geom, a_grids, a_dmap, linfo, {});
    }

    // Initialise everything else normally
    m_coord = a_coord;
#if (AMREX_SPACEDIM == 2)
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_coord == 0,
                                     "CurlCurl: In 2D, only Cartesian is supported.");
#elif (AMREX_SPACEDIM == 3)
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_coord == 0,
                                     "CurlCurl: In 3D, only Cartesian is supported.");
#endif
    if (m_coord == 1 || m_coord == 2) {
        AMREX_ALWAYS_ASSERT(a_geom[0].ProbLo(0) == 0);
    }

    m_dotmask.resize(this->m_num_amr_levels);
    for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev) {
        m_dotmask[amrlev].resize(this->m_num_mg_levels[amrlev]);
    }

    m_bcoefs.resize(this->m_num_amr_levels);
    for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev) {
        m_bcoefs[amrlev].resize(this->m_num_mg_levels[amrlev]);
    }

    m_acoefs.resize(this->m_num_amr_levels);
    for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev) {
        m_acoefs[amrlev].resize(this->m_num_mg_levels[amrlev]);
    }

    m_lusolver.resize(this->m_num_amr_levels);
    for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev) {
        m_lusolver[amrlev].resize(this->m_num_mg_levels[amrlev]);
    }

    // Ensure the coarsened overset masks live on the same DistributionMap as
    // the linop's grids at each mglev. If the layouts differ (MFIter-unsafe),
    // ParallelCopy onto a freshly-allocated mask using the linop's DM, then
    // re-establish ng=1 ghost values for the new layout.
    for (int amrlev = 0; amrlev < namrlevs; ++amrlev) {
        for (int mglev = 1; mglev < this->m_num_mg_levels[amrlev]; ++mglev) {
            for (int idim = 0; idim < 3; ++idim) {
                BoxArray ba = amrex::convert(this->m_grids[amrlev][mglev], m_etype[idim]);
                iMultiFab foo(ba, this->m_dmap[amrlev][mglev], 1, 0,
                              MFInfo().SetAlloc(false));
                if (! amrex::isMFIterSafe(*(m_overset_mask[amrlev][mglev][idim]), foo)) {
                    auto osm = std::make_unique<iMultiFab>(
                        ba, this->m_dmap[amrlev][mglev], 1, 1);
                    osm->ParallelCopy(*(m_overset_mask[amrlev][mglev][idim]));
                    osm->setDomainBndry(1, this->m_geom[amrlev][mglev]);
                    osm->FillBoundary(this->m_geom[amrlev][mglev].periodicity());
                    std::swap(osm, m_overset_mask[amrlev][mglev][idim]);
                }
            }
        }
    }

    // Build truly-nodal overset mask from the user-supplied edge/face masks.
    // node_mask = 1 iff every DOF the smoother stencil at (i,j,k) touches is
    // unmasked (osm = 1); 0 if any of those DOFs is masked.
    m_nodal_overset_mask.resize(namrlevs);
    for (int amrlev = 0; amrlev < namrlevs; ++amrlev) {
        m_nodal_overset_mask[amrlev].resize(this->m_num_mg_levels[amrlev]);

        for (int mglev = 0; mglev < this->m_num_mg_levels[amrlev]; ++mglev) {
            BoxArray nba = amrex::convert(this->m_grids[amrlev][mglev], IntVect(1));
            m_nodal_overset_mask[amrlev][mglev]
                = std::make_unique<iMultiFab>(nba, this->m_dmap[amrlev][mglev], 1, 0);

            auto const& xosm = m_overset_mask[amrlev][mglev][0]->const_arrays();
            auto const& yosm = m_overset_mask[amrlev][mglev][1]->const_arrays();
            auto const& zosm = m_overset_mask[amrlev][mglev][2]->const_arrays();
            auto const& nosm = m_nodal_overset_mask[amrlev][mglev]->arrays();

            ParallelFor(*m_nodal_overset_mask[amrlev][mglev],
                [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
            {
                // Edge masks have ng=1 with out-of-domain ghosts set to 1
                // (active) and internal ghosts FillBoundary'd, so direct reads
                // at i-1 / j-1 / k-1 are in bounds with the correct values.
#if (AMREX_SPACEDIM == 2)
                bool active = xosm[bno](i-1, j  , k) && xosm[bno](i, j, k)
                           && yosm[bno](i  , j-1, k) && yosm[bno](i, j, k)
                           && zosm[bno](i  , j  , k);
#elif (AMREX_SPACEDIM == 3)
                bool active = xosm[bno](i-1, j  , k  ) && xosm[bno](i, j, k)
                           && yosm[bno](i  , j-1, k  ) && yosm[bno](i, j, k)
                           && zosm[bno](i  , j  , k-1) && zosm[bno](i, j, k);
#else
                bool active = xosm[bno](i, j, k) && yosm[bno](i, j, k)
                           && zosm[bno](i, j, k);
#endif
                nosm[bno](i,j,k) = active ? 1 : 0;
            });
            Gpu::streamSynchronize();
        }
    }
}

void MLCurlCurl::setScalars (RT a_alpha, RT a_beta) noexcept
{
    m_needs_update = true;
    m_alpha = a_alpha;
    m_beta = a_beta;
    clearAlphaMultiFab();
    clearBetaMultiFab();
    AMREX_ASSERT(m_beta > RT(0));
}

void MLCurlCurl::setBeta (const Vector<Array<MultiFab const*,3>>& a_bcoefs)
{
    m_needs_update = true;

    Array<IntVect,3> ng;
    for (int idim = 0; idim < 3; ++idim) {
        ng[idim] = IntVect(1) - m_etype[idim]; // 1 ghost for cell direction, 0 for node
    }

    for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev) {
        for (int idim = 0; idim < 3; ++idim) {
            if (m_bcoefs[amrlev][0][idim] == nullptr) {
                m_bcoefs[amrlev][0][idim] = std::make_unique<MultiFab>
                    (a_bcoefs[amrlev][idim]->boxArray(),
                     a_bcoefs[amrlev][idim]->DistributionMap(), 1, ng[idim]);
            }
            MultiFab::Copy(*m_bcoefs[amrlev][0][idim], *a_bcoefs[amrlev][idim], 0, 0, 1, 0);
            m_bcoefs[amrlev][0][idim]->FillBoundary(m_geom[amrlev][0].periodicity());
        }
    }

    // Need to average down from fine AMR level to coarse level and we need
    // to support periodic boundary
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_num_amr_levels == 1,
                                     "MLCurlCurl: multi-level not supported yet");

    for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev) {
        for (int mglev = 1; mglev < m_num_mg_levels[amrlev]; ++mglev) {
            IntVect ratio = (amrlev > 0) ? IntVect(2)
                : mg_coarsen_ratio_vec[mglev-1];
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                if (m_bcoefs[amrlev][mglev][idim] == nullptr) {
                    m_bcoefs[amrlev][mglev][idim] = std::make_unique<MultiFab>
                        (amrex::convert(m_grids[amrlev][mglev], m_etype[idim]),
                         m_dmap[amrlev][mglev], 1, ng[idim]);
                }
                average_down_edges(*m_bcoefs[amrlev][mglev-1][idim],
                                   *m_bcoefs[amrlev][mglev  ][idim], ratio);
                m_bcoefs[amrlev][mglev][idim]->FillBoundary(m_geom[amrlev][mglev].periodicity());
            }
#if (AMREX_SPACEDIM < 3)
            if (m_bcoefs[amrlev][mglev][2] == nullptr) {
                m_bcoefs[amrlev][mglev][2] = std::make_unique<MultiFab>
                    (amrex::convert(m_grids[amrlev][mglev], m_etype[2]),
                     m_dmap[amrlev][mglev], 1, 0);
            }
            average_down_nodal(*m_bcoefs[amrlev][mglev-1][2],
                               *m_bcoefs[amrlev][mglev  ][2], ratio);
#endif
#if (AMREX_SPACEDIM == 1)
            if (m_bcoefs[amrlev][mglev][1] == nullptr) {
                m_bcoefs[amrlev][mglev][1] = std::make_unique<MultiFab>
                    (amrex::convert(m_grids[amrlev][mglev], m_etype[1]),
                     m_dmap[amrlev][mglev], 1, 0);
            }
            average_down_nodal(*m_bcoefs[amrlev][mglev-1][1],
                               *m_bcoefs[amrlev][mglev  ][1], ratio);
#endif
        }
    }

    for (auto& amrvec : m_lusolver) {
        for (auto& mgptr : amrvec) {
            mgptr.reset();
        }
    }
}

void MLCurlCurl::setAlpha (const Vector<MultiFab const*>& a_acoeffs)
{
    AMREX_ALWAYS_ASSERT(static_cast<int>(a_acoeffs.size()) == m_num_amr_levels);

    m_needs_update = true;

    auto lobc = LoBC();
    auto hibc = HiBC();
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        if (lobc[idim] != LinOpBCType::Periodic) { lobc[idim] = LinOpBCType::Neumann; }
        if (hibc[idim] != LinOpBCType::Periodic) { hibc[idim] = LinOpBCType::Neumann; }
    }

    for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev) {
        AMREX_ALWAYS_ASSERT(a_acoeffs[amrlev]->is_nodal());
        MultiFab nodal_alpha(a_acoeffs[amrlev]->boxArray(),
                             a_acoeffs[amrlev]->DistributionMap(), 1, 1);
        MultiFab::Copy(nodal_alpha, *a_acoeffs[amrlev], 0, 0, 1, 0);
        nodal_alpha.FillBoundaryAndSync(m_geom[amrlev][0].periodicity());

        for (int mglev = 0; mglev < m_num_mg_levels[amrlev]; ++mglev) {

            Box nd_domain = amrex::surroundingNodes(m_geom[amrlev][mglev].Domain());
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(nodal_alpha); mfi.isValid(); ++mfi) {
                auto const& afab = nodal_alpha.array(mfi);
                Box const& box = mfi.validbox();
                mlndlap_applybc(box, afab, nd_domain, lobc, hibc);
            }

            auto const& anode = nodal_alpha.const_arrays();

            GpuArray<MultiArray4<Real>,3> aface;
            for (int idim = 0; idim < 3; ++idim) {
                IntVect typ(0);
                if (idim < AMREX_SPACEDIM) { typ[idim] = 1; }
                m_acoefs[amrlev][mglev][idim] = std::make_unique<MultiFab>
                    (amrex::convert(m_grids[amrlev][mglev], typ),
                     m_dmap[amrlev][mglev], 1, 1);
                aface[idim] = m_acoefs[amrlev][mglev][idim]->arrays();
            }

            amrex::ParallelFor(nodal_alpha, IntVect(1),
                               [=] AMREX_GPU_DEVICE (int b, int i, int j, int k)
            {
                auto const& an = anode[b];
                auto const& ax = aface[0][b];
                auto const& ay = aface[1][b];
                auto const& az = aface[2][b];
                if (ax.contains(i,j,k)) {
#if (AMREX_SPACEDIM == 1)
                    ax(i,0,0) = an(i,0,0);
#elif (AMREX_SPACEDIM == 2)
                    ax(i,j,0) = Real(0.5)*(an(i,j,0)+an(i,j+1,0));
#else
                    ax(i,j,k) = Real(0.25)*(an(i,j,k)+an(i,j+1,k)+an(i,j,k+1)+an(i,j+1,k+1));
#endif
                }
                if (ay.contains(i,j,k)) {
#if (AMREX_SPACEDIM == 1)
                    ay(i,0,0) = Real(0.5)*(an(i,0,0)+an(i+1,0,0));
#elif (AMREX_SPACEDIM == 2)
                    ay(i,j,0) = Real(0.5)*(an(i,j,0)+an(i+1,j,0));
#else
                    ay(i,j,k) = Real(0.25)*(an(i,j,k)+an(i+1,j,k)+an(i,j,k+1)+an(i+1,j,k+1));
#endif
                }
                if (az.contains(i,j,k)) {
#if (AMREX_SPACEDIM == 1)
                    az(i,0,0) = Real(0.5)*(an(i,0,0)+an(i+1,0,0));
#elif (AMREX_SPACEDIM >= 2)
                    az(i,j,k) = Real(0.25)*(an(i,j,k)+an(i+1,j,k)+an(i,j+1,k)+an(i+1,j+1,k));
#endif
                }
            });

            if (mglev+1 < m_num_mg_levels[amrlev]) {
                MultiFab tmp(amrex::convert(m_grids[amrlev][mglev+1], IntVect(1)),
                             m_dmap[amrlev][mglev+1], 1, 1);
                IntVect ratio = (amrlev > 0) ? IntVect(2) : mg_coarsen_ratio_vec[mglev];
                average_down_nodal(nodal_alpha, tmp, ratio);
                std::swap(nodal_alpha, tmp);
                nodal_alpha.FillBoundary(m_geom[amrlev][mglev+1].periodicity());
            }
        }
    }

    for (auto& amrvec : m_lusolver) {
        for (auto& mgptr : amrvec) {
            mgptr.reset();
        }
    }
}

void MLCurlCurl::prepareRHS (Vector<MF*> const& rhs) const
{
    for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev) {
        for (auto& mf : *rhs[amrlev]) {
            mf.OverrideSync(m_geom[amrlev][0].periodicity());
        }
    }
}

void MLCurlCurl::setDirichletNodesToZero (int amrlev, int mglev, MF& a_mf) const
{
    MFItInfo mfi_info{};
#ifdef AMREX_USE_GPU
    Vector<Array4BoxTag<RT>> tags;
    mfi_info.DisableDeviceSync();
#endif

    for (int imf = 0; imf < 3; ++imf)
    {
        auto& mf = a_mf[imf];
        auto const idxtype = mf.ixType();
        Box const domain = amrex::convert(m_geom[amrlev][mglev].Domain(), idxtype);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(mf,mfi_info); mfi.isValid(); ++mfi) {
            auto const& vbx = mfi.validbox();
            auto const& a = mf.array(mfi);
            for (OrientationIter oit; oit; ++oit) {
                Orientation const face = oit();
                int const idim = face.coordDir();
                bool is_dirichlet = face.isLow()
                    ? m_lobc[0][idim] == LinOpBCType::Dirichlet
                    : m_hibc[0][idim] == LinOpBCType::Dirichlet;
#if (AMREX_SPACEDIM == 1)
                if (m_coord == 1 && imf == 2 && face.isLow()) {
                    is_dirichlet = false; // Ez in 1d cyl is not Dirichlet at r=0.
                }
#endif
                if (is_dirichlet && domain[face] == vbx[face] &&
                    idxtype.nodeCentered(idim))
                {
                    Box b = vbx;
                    b.setRange(idim, vbx[face], 1);
#ifdef AMREX_USE_GPU
                    tags.emplace_back(Array4BoxTag<RT>{a,b});
#else
                    amrex::LoopOnCpu(b, [&] (int i, int j, int k)
                    {
                        a(i,j,k) = RT(0.0);
                    });
#endif
                }
            }
        }
    }

#ifdef AMREX_USE_GPU
    ParallelFor(tags,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, Array4BoxTag<RT> const& tag) noexcept
    {
        tag.dfab(i,j,k) = RT(0.0);
    });
#endif
    // Zero dirichlet values set by overset mask
    if (m_overset_mask[amrlev][mglev][0] != nullptr)
    {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        AMREX_ASSERT(a_mf[0].nGrowVect() == IntVect(0));
        AMREX_ASSERT(a_mf[1].nGrowVect() == IntVect(0));
        AMREX_ASSERT(a_mf[2].nGrowVect() == IntVect(0));
        for (MFIter mfi(a_mf[0], mfi_info); mfi.isValid(); ++mfi) {
            Box const& xbx = mfi.tilebox(a_mf[0].ixType().toIntVect());
            Box const& ybx = mfi.tilebox(a_mf[1].ixType().toIntVect());
            Box const& zbx = mfi.tilebox(a_mf[2].ixType().toIntVect());
            const auto& xosm = m_overset_mask[amrlev][mglev][0]->const_array(mfi);
            const auto& yosm = m_overset_mask[amrlev][mglev][1]->const_array(mfi);
            const auto& zosm = m_overset_mask[amrlev][mglev][2]->const_array(mfi);
            const auto& xout = a_mf[0].array(mfi);
            const auto& yout = a_mf[1].array(mfi);
            const auto& zout = a_mf[2].array(mfi);
            ParallelFor(xbx,ybx,zbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (xosm(i,j,k) == 0) {xout(i,j,k) = Real(0.0); }
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (yosm(i,j,k) == 0) {yout(i,j,k) = Real(0.0); }
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (zosm(i,j,k) == 0) {zout(i,j,k) = Real(0.0); }
            });
        }
    }
}

void MLCurlCurl::setLevelBC (int amrlev, const MF* levelbcdata, // TODO
                             const MF* robinbc_a, const MF* robinbc_b,
                             const MF* robinbc_f)
{
    amrex::ignore_unused(amrlev, levelbcdata, robinbc_a, robinbc_b, robinbc_f);
}

void MLCurlCurl::restriction (int amrlev, int cmglev, MF& crse, MF& fine) const
{
    IntVect ratio = (amrlev > 0) ? IntVect(2) : this->mg_coarsen_ratio_vec[cmglev-1];
    AMREX_ALWAYS_ASSERT(ratio == 2);

    applyBC(amrlev, cmglev-1, fine, CurlCurlStateType::r);

    auto dinfo = getDirichletInfo(amrlev,cmglev-1);

    for (int idim = 0; idim < 3; ++idim) {
        bool need_parallel_copy = !amrex::isMFIterSafe(crse[idim], fine[idim]);
        MultiFab cfine;
        if (need_parallel_copy) {
            BoxArray const& ba = amrex::coarsen(fine[idim].boxArray(), 2);
            cfine.define(ba, fine[idim].DistributionMap(), 1, 0,
                         MFInfo().SetArena(The_Async_Arena()));
        }

        MultiFab* pcrse = (need_parallel_copy) ? &cfine : &(crse[idim]);

        // Coarse overset mask for this idim's edge centering. When mask layout
        // is MFIter-incompatible with cfine, fall back to no mask in-kernel and
        // post-zero via ParallelCopy below.
        bool have_osm = (m_overset_mask[amrlev][cmglev][idim] != nullptr);
        bool osm_iter_safe = have_osm
            && amrex::isMFIterSafe(*pcrse, *m_overset_mask[amrlev][cmglev][idim]);

        auto const& crsema = pcrse->arrays();
        auto const& finema = fine[idim].const_arrays();
        if (have_osm && osm_iter_safe) {
            auto const& osma = m_overset_mask[amrlev][cmglev][idim]->const_arrays();
            ParallelFor(*pcrse, [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
            {
                mlcurlcurl_restriction(idim,i,j,k,crsema[bno],finema[bno],dinfo,
                                       osma[bno]);
            });
        } else {
            ParallelFor(*pcrse, [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
            {
                Array4<int const> empty;
                mlcurlcurl_restriction(idim,i,j,k,crsema[bno],finema[bno],dinfo,
                                       empty);
            });
        }
        if (!Gpu::inNoSyncRegion()) {
            Gpu::streamSynchronize();
        }

        if (need_parallel_copy) {
            crse[idim].ParallelCopy(cfine);
        }

        // Zero coarse-masked edges on crse[idim] in case the
        // in-kernel mask was unavailable or
        // ParallelCopy brought in nonzero values from a foreign layout.
        if (have_osm && !osm_iter_safe) {
            // ParallelCopy mask onto crse's layout so we can zero per-bno.
            iMultiFab osm_local(crse[idim].boxArray(),
                                crse[idim].DistributionMap(), 1, 0);
            osm_local.setVal(1);
            osm_local.ParallelCopy(*m_overset_mask[amrlev][cmglev][idim]);
            auto const& cma = crse[idim].arrays();
            auto const& lma = osm_local.const_arrays();
            ParallelFor(crse[idim], [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
            {
                if (lma[bno](i,j,k) == 0) { cma[bno](i,j,k) = Real(0.0); }
            });
            if (!Gpu::inNoSyncRegion()) { Gpu::streamSynchronize(); }
        }
    }
}

void MLCurlCurl::interpolation (int amrlev, int fmglev, MF& fine,
                                const MF& crse) const
{
    IntVect ratio = (amrlev > 0) ? IntVect(2) : this->mg_coarsen_ratio_vec[fmglev];
    AMREX_ALWAYS_ASSERT(ratio == 2);

    auto dinfo = getDirichletInfo(amrlev,fmglev);

    int const cmglev = fmglev + 1;
    bool const have_cosm = (m_overset_mask[amrlev][cmglev][0] != nullptr);
    bool const have_fosm = (m_overset_mask[amrlev][fmglev][0] != nullptr);

    for (int idim = 0; idim < 3; ++idim) {
        bool need_parallel_copy = !amrex::isMFIterSafe(crse[idim], fine[idim]);
        bool need_local = need_parallel_copy || have_cosm;

        MultiFab cwork;
        MultiFab const* cmf = &(crse[idim]);
        if (need_local) {
            BoxArray ba = need_parallel_copy
                          ? amrex::coarsen(fine[idim].boxArray(), 2)
                          : crse[idim].boxArray();
            DistributionMapping dm = need_parallel_copy
                                     ? fine[idim].DistributionMap()
                                     : crse[idim].DistributionMap();
            cwork.define(ba, dm, 1, 0, MFInfo().SetArena(The_Async_Arena()));
            cwork.ParallelCopy(crse[idim]);
            cmf = &cwork;
        }

        // Pre-zero coarse correction at coarse-masked edges so interp does
        // not leak coarse_cor into fine masked DOFs.
        if (have_cosm) {
            iMultiFab osm_local(cwork.boxArray(), cwork.DistributionMap(), 1, 0);
            osm_local.setVal(1);
            osm_local.ParallelCopy(*m_overset_mask[amrlev][cmglev][idim]);
            auto const& zma = cwork.arrays();
            auto const& osma = osm_local.const_arrays();
            ParallelFor(cwork, [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
            {
                if (osma[bno](i,j,k) == 0) { zma[bno](i,j,k) = Real(0.0); }
            });
            if (!Gpu::inNoSyncRegion()) { Gpu::streamSynchronize(); }
        }

        auto const& finema = fine[idim].arrays();
        auto const& crsema = cmf->const_arrays();
        if (have_fosm) {
            auto const& fosma = m_overset_mask[amrlev][fmglev][idim]->const_arrays();
            ParallelFor(fine[idim], [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
            {
                if (!dinfo.is_dirichlet_edge(idim,i,j,k) && fosma[bno](i,j,k) != 0) {
                    mlcurlcurl_interpadd(idim,i,j,k,finema[bno],crsema[bno]);
                }
            });
        } else {
            ParallelFor(fine[idim], [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
            {
                if (!dinfo.is_dirichlet_edge(idim,i,j,k)) {
                    mlcurlcurl_interpadd(idim,i,j,k,finema[bno],crsema[bno]);
                }
            });
        }
        if (!Gpu::inNoSyncRegion()) {
            Gpu::streamSynchronize();
        }
    }
}

void
MLCurlCurl::apply (int amrlev, int mglev, MF& out, MF& in, BCMode /*bc_mode*/,
                   StateMode /*s_mode*/, const MLMGBndryT<MF>* /*bndry*/) const
{
    applyBC(amrlev, mglev, in, CurlCurlStateType::x);

    auto dxinv = this->m_geom[amrlev][mglev].InvCellSizeArray();
    auto adxinv = dxinv;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        adxinv[idim] *= std::sqrt(m_alpha);
    }
    auto const b = m_beta;
    bool const has_beta = (m_bcoefs[amrlev][mglev][0] != nullptr);
    bool const has_alpha = (m_acoefs[amrlev][mglev][0] != nullptr);

    auto dinfo = getDirichletInfo(amrlev,mglev);
    auto coord = m_coord;
    amrex::ignore_unused(coord);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(out[0],TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Box const& xbx = mfi.tilebox(out[0].ixType().toIntVect());
        Box const& ybx = mfi.tilebox(out[1].ixType().toIntVect());
        Box const& zbx = mfi.tilebox(out[2].ixType().toIntVect());
        auto const& xout = out[0].array(mfi);
        auto const& yout = out[1].array(mfi);
        auto const& zout = out[2].array(mfi);
        auto const& xin = in[0].array(mfi);
        auto const& yin = in[1].array(mfi);
        auto const& zin = in[2].array(mfi);

        if (has_alpha) {
            Array4<Real const> bcx, bcy, bcz;
            if (has_beta) {
                bcx = m_bcoefs[amrlev][mglev][0]->const_array(mfi);
                bcy = m_bcoefs[amrlev][mglev][1]->const_array(mfi);
                bcz = m_bcoefs[amrlev][mglev][2]->const_array(mfi);
            }
            Array4<int const> xosm, yosm, zosm;
            if (m_overset_mask[amrlev][mglev][0])
            {
                AMREX_ASSERT(m_overset_mask[amrlev][mglev][1]);
                AMREX_ASSERT(m_overset_mask[amrlev][mglev][2]);
                xosm = m_overset_mask[amrlev][mglev][0]->const_array(mfi);
                yosm = m_overset_mask[amrlev][mglev][1]->const_array(mfi);
                zosm = m_overset_mask[amrlev][mglev][2]->const_array(mfi);
            }
            auto const afx = m_acoefs[amrlev][mglev][0]->const_array(mfi);
            auto const afy = m_acoefs[amrlev][mglev][1]->const_array(mfi);
            auto const afz = m_acoefs[amrlev][mglev][2]->const_array(mfi);
            amrex::ParallelFor(xbx, ybx, zbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (dinfo.is_dirichlet_x_edge(i,j,k) || (xosm && xosm(i,j,k) == 0)) {
                    xout(i,j,k) = Real(0.0);
                } else {
                    Real beta = bcx ? bcx(i,j,k) : b;
                    mlcurlcurl_adotx_x_alpha(i,j,k,xout,xin,yin,zin,
                                             afy,afz,beta,dxinv);
                }
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (dinfo.is_dirichlet_y_edge(i,j,k) || (yosm && yosm(i,j,k) == 0)) {
                    yout(i,j,k) = Real(0.0);
                } else {
                    Real beta = bcy ? bcy(i,j,k) : b;
                    mlcurlcurl_adotx_y_alpha(i,j,k,yout,xin,yin,zin,
                                             afx,afz,beta,dxinv
#if (AMREX_SPACEDIM < 3)
                                             ,coord
#endif
                                             );
                }
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (dinfo.is_dirichlet_z_edge(i,j,k) || (zosm && zosm(i,j,k) == 0)) {
                    zout(i,j,k) = Real(0.0);
                } else {
                    Real beta = bcz ? bcz(i,j,k) : b;
                    mlcurlcurl_adotx_z_alpha(i,j,k,zout,xin,yin,zin,
                                             afx,afy,beta,dxinv
#if (AMREX_SPACEDIM < 3)
                                             ,coord
#endif
                                             );
                }
            });
        } else if (has_beta) {
            auto const& bcx = m_bcoefs[amrlev][mglev][0]->const_array(mfi);
            auto const& bcy = m_bcoefs[amrlev][mglev][1]->const_array(mfi);
            auto const& bcz = m_bcoefs[amrlev][mglev][2]->const_array(mfi);
            Array4<int const> xosm, yosm, zosm;
            if (m_overset_mask[amrlev][mglev][0])
            {
                AMREX_ASSERT(m_overset_mask[amrlev][mglev][1]);
                AMREX_ASSERT(m_overset_mask[amrlev][mglev][2]);
                xosm = m_overset_mask[amrlev][mglev][0]->const_array(mfi);
                yosm = m_overset_mask[amrlev][mglev][1]->const_array(mfi);
                zosm = m_overset_mask[amrlev][mglev][2]->const_array(mfi);
            }
            amrex::ParallelFor(xbx, ybx, zbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (dinfo.is_dirichlet_x_edge(i,j,k) || (xosm && xosm(i,j,k) == 0)) {
                    xout(i,j,k) = Real(0.0);
                } else {
                    mlcurlcurl_adotx_x(i,j,k,xout,xin,yin,zin,bcx(i,j,k),adxinv);
                }
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (dinfo.is_dirichlet_y_edge(i,j,k) || (yosm && yosm(i,j,k) == 0)) {
                    yout(i,j,k) = Real(0.0);
                } else {
                    mlcurlcurl_adotx_y(i,j,k,yout,xin,yin,zin,bcy(i,j,k),adxinv
#if (AMREX_SPACEDIM < 3)
                                       ,coord
#endif
                                      );
                }
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (dinfo.is_dirichlet_z_edge(i,j,k) || (zosm && zosm(i,j,k) == 0)) {
                    zout(i,j,k) = Real(0.0);
                } else {
                    mlcurlcurl_adotx_z(i,j,k,zout,xin,yin,zin,bcz(i,j,k),adxinv
#if (AMREX_SPACEDIM < 3)
                                       ,coord
#endif
                                      );
                }
            });
        } else {
            Array4<int const> xosm, yosm, zosm;
            if (m_overset_mask[amrlev][mglev][0])
            {
                AMREX_ASSERT(m_overset_mask[amrlev][mglev][1]);
                AMREX_ASSERT(m_overset_mask[amrlev][mglev][2]);
                xosm = m_overset_mask[amrlev][mglev][0]->const_array(mfi);
                yosm = m_overset_mask[amrlev][mglev][1]->const_array(mfi);
                zosm = m_overset_mask[amrlev][mglev][2]->const_array(mfi);
            }
            amrex::ParallelFor(xbx, ybx, zbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (dinfo.is_dirichlet_x_edge(i,j,k) || (xosm && xosm(i,j,k) == 0)) {
                    xout(i,j,k) = Real(0.0);
                } else {
                    mlcurlcurl_adotx_x(i,j,k,xout,xin,yin,zin,b,adxinv);
                }
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (dinfo.is_dirichlet_y_edge(i,j,k) || (yosm && yosm(i,j,k) == 0)) {
                    yout(i,j,k) = Real(0.0);
                } else {
                    mlcurlcurl_adotx_y(i,j,k,yout,xin,yin,zin,b,adxinv
#if (AMREX_SPACEDIM < 3)
                                       ,coord
#endif
                                      );
                }
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (dinfo.is_dirichlet_z_edge(i,j,k) || (zosm && zosm(i,j,k) == 0)) {
                    zout(i,j,k) = Real(0.0);
                } else {
                    mlcurlcurl_adotx_z(i,j,k,zout,xin,yin,zin,b,adxinv
#if (AMREX_SPACEDIM < 3)
                                       ,coord
#endif
                                      );
                }
            });
        }
    }
}

void MLCurlCurl::smooth (int amrlev, int mglev, MF& sol, const MF& rhs,
                         bool skip_fillboundary, int niter) const
{
    AMREX_ASSERT(rhs[0].nGrowVect().allGE(1));

    applyBC(amrlev, mglev, const_cast<MF&>(rhs), CurlCurlStateType::b);
#if (AMREX_SPACEDIM == 1)
    int ncolors = 2;
#else
    int ncolors = 4;
#endif

    for (int i = 0; i < niter; ++i) {
        for (int color = 0; color < ncolors; ++color) {
            if (!skip_fillboundary) {
                applyBC(amrlev, mglev, sol, CurlCurlStateType::x);
            }
            skip_fillboundary = false;
#if (AMREX_SPACEDIM == 1)
            smooth1D(amrlev, mglev, sol, rhs, color);
#else
            smooth4(amrlev, mglev, sol, rhs, color);
#endif
        }
    }
}

#if (AMREX_SPACEDIM == 1)
void MLCurlCurl::smooth1D (int amrlev, int mglev, MF& sol, MF const& rhs,
                           int color) const
{
    auto const& ex = sol[0].arrays();
    auto const& ey = sol[1].arrays();
    auto const& ez = sol[2].arrays();
    auto const& rhsx = rhs[0].const_arrays();
    auto const& rhsy = rhs[1].const_arrays();
    auto const& rhsz = rhs[2].const_arrays();

    auto b = m_beta;

    auto dinfo = getDirichletInfo(amrlev,mglev);
    auto dxinv = this->m_geom[amrlev][mglev].InvCellSizeArray();
    auto adxinv = dxinv;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        adxinv[idim] *= std::sqrt(m_alpha);
    }

    int xhi = this->m_geom[amrlev][mglev].Domain().bigEnd(0);

    auto coord = m_coord;

    MultiFab nmf(amrex::convert(rhs[0].boxArray(),IntVect(1)),
                 rhs[0].DistributionMap(), 1, 0, MFInfo().SetAlloc(false));

    bool const has_beta = (m_bcoefs[amrlev][mglev][0] != nullptr);
    bool const has_alpha = (m_acoefs[amrlev][mglev][0] != nullptr);
    bool const has_osm = (m_overset_mask[amrlev][mglev][0] != nullptr);

    if (has_alpha && has_beta) {
        auto const& acy = m_acoefs[amrlev][mglev][1]->const_arrays();
        auto const& acz = m_acoefs[amrlev][mglev][2]->const_arrays();
        auto const& bcx = m_bcoefs[amrlev][mglev][0]->const_arrays();
        auto const& bcy = m_bcoefs[amrlev][mglev][1]->const_arrays();
        auto const& bcz = m_bcoefs[amrlev][mglev][2]->const_arrays();
        if (has_osm) {
            auto const& xosm = m_overset_mask[amrlev][mglev][0]->const_arrays();
            auto const& yosm = m_overset_mask[amrlev][mglev][1]->const_arrays();
            auto const& zosm = m_overset_mask[amrlev][mglev][2]->const_arrays();
            ParallelFor( nmf, [=] AMREX_GPU_DEVICE(int bno, int i, int j, int k)
            {
                bool valid_x = i <= xhi; // x is cell-centered, not nodal
                mlcurlcurl_smooth_1d_alpha_beta(i,j,k,ex[bno],ey[bno],ez[bno],
                                                rhsx[bno],rhsy[bno],rhsz[bno],
                                                bcx[bno],bcy[bno],bcz[bno],
                                                dxinv,color,dinfo,valid_x,coord,
                                                acy[bno],acz[bno],
                                                xosm[bno],yosm[bno],zosm[bno]);
            });
        } else {
            ParallelFor( nmf, [=] AMREX_GPU_DEVICE(int bno, int i, int j, int k)
            {
                bool valid_x = i <= xhi; // x is cell-centered, not nodal
                Array4<int const> empty;
                mlcurlcurl_smooth_1d_alpha_beta(i,j,k,ex[bno],ey[bno],ez[bno],
                                                rhsx[bno],rhsy[bno],rhsz[bno],
                                                bcx[bno],bcy[bno],bcz[bno],
                                                dxinv,color,dinfo,valid_x,coord,
                                                acy[bno],acz[bno],
                                                empty,empty,empty);
            });
        }
    } else if (has_alpha && !has_beta) {
        auto const& acy = m_acoefs[amrlev][mglev][1]->const_arrays();
        auto const& acz = m_acoefs[amrlev][mglev][2]->const_arrays();
        if (has_osm) {
            auto const& xosm = m_overset_mask[amrlev][mglev][0]->const_arrays();
            auto const& yosm = m_overset_mask[amrlev][mglev][1]->const_arrays();
            auto const& zosm = m_overset_mask[amrlev][mglev][2]->const_arrays();
            ParallelFor( nmf, [=] AMREX_GPU_DEVICE(int bno, int i, int j, int k)
            {
                bool valid_x = i <= xhi; // x is cell-centered, not nodal
                mlcurlcurl_smooth_1d_alpha(i,j,k,ex[bno],ey[bno],ez[bno],
                                           rhsx[bno],rhsy[bno],rhsz[bno],
                                           b,
                                           dxinv,color,dinfo,valid_x,coord,
                                           acy[bno],acz[bno],
                                           xosm[bno],yosm[bno],zosm[bno]);
            });
        } else {
            ParallelFor( nmf, [=] AMREX_GPU_DEVICE(int bno, int i, int j, int k)
            {
                bool valid_x = i <= xhi; // x is cell-centered, not nodal
                Array4<int const> empty;
                mlcurlcurl_smooth_1d_alpha(i,j,k,ex[bno],ey[bno],ez[bno],
                                           rhsx[bno],rhsy[bno],rhsz[bno],
                                           b,
                                           dxinv,color,dinfo,valid_x,coord,
                                           acy[bno],acz[bno],
                                           empty,empty,empty);
            });
        }
    } else if (!has_alpha && has_beta) {
        auto const& bcx = m_bcoefs[amrlev][mglev][0]->const_arrays();
        auto const& bcy = m_bcoefs[amrlev][mglev][1]->const_arrays();
        auto const& bcz = m_bcoefs[amrlev][mglev][2]->const_arrays();
        if (has_osm) {
            auto const& xosm = m_overset_mask[amrlev][mglev][0]->const_arrays();
            auto const& yosm = m_overset_mask[amrlev][mglev][1]->const_arrays();
            auto const& zosm = m_overset_mask[amrlev][mglev][2]->const_arrays();
            ParallelFor( nmf, [=] AMREX_GPU_DEVICE(int bno, int i, int j, int k)
            {
                bool valid_x = i <= xhi; // x is cell-centered, not nodal
                mlcurlcurl_smooth_1d(i,j,k,ex[bno],ey[bno],ez[bno],
                                     rhsx[bno],rhsy[bno],rhsz[bno],
                                     bcx[bno],bcy[bno],bcz[bno],
                                     adxinv,color,dinfo,valid_x,coord,
                                     xosm[bno],yosm[bno],zosm[bno]);
            });
        } else {
            ParallelFor( nmf, [=] AMREX_GPU_DEVICE(int bno, int i, int j, int k)
            {
                bool valid_x = i <= xhi; // x is cell-centered, not nodal
                Array4<int const> empty;
                mlcurlcurl_smooth_1d(i,j,k,ex[bno],ey[bno],ez[bno],
                                     rhsx[bno],rhsy[bno],rhsz[bno],
                                     bcx[bno],bcy[bno],bcz[bno],
                                     adxinv,color,dinfo,valid_x,coord,
                                     empty,empty,empty);
            });
        }
    } else {
        if (has_osm) {
            auto const& xosm = m_overset_mask[amrlev][mglev][0]->const_arrays();
            auto const& yosm = m_overset_mask[amrlev][mglev][1]->const_arrays();
            auto const& zosm = m_overset_mask[amrlev][mglev][2]->const_arrays();
            ParallelFor( nmf, [=] AMREX_GPU_DEVICE(int bno, int i, int j, int k)
            {
                amrex::Print() << "Looping over " << i << " w/ bno = " << bno << std::endl;
                bool valid_x = i <= xhi; // x is cell-centered, not nodal
                mlcurlcurl_smooth_1d(i,j,k,ex[bno],ey[bno],ez[bno],
                                     rhsx[bno],rhsy[bno],rhsz[bno],
                                     b,adxinv,color,dinfo,valid_x,coord,
                                     xosm[bno],yosm[bno],zosm[bno]);
            });
        } else {
            ParallelFor( nmf, [=] AMREX_GPU_DEVICE(int bno, int i, int j, int k)
            {
                bool valid_x = i <= xhi; // x is cell-centered, not nodal
                Array4<int const> empty;
                mlcurlcurl_smooth_1d(i,j,k,ex[bno],ey[bno],ez[bno],
                                     rhsx[bno],rhsy[bno],rhsz[bno],
                                     b,adxinv,color,dinfo,valid_x,coord,
                                     empty,empty,empty);
            });
        }
    }
    if (!Gpu::inNoSyncRegion()) {
        Gpu::streamSynchronize();
    }
}
#endif

#if (AMREX_SPACEDIM > 1)
void MLCurlCurl::smooth4 (int amrlev, int mglev, MF& sol, MF const& rhs,
                          int color) const
{
    auto const& ex = sol[0].arrays();
    auto const& ey = sol[1].arrays();
    auto const& ez = sol[2].arrays();
    auto const& rhsx = rhs[0].const_arrays();
    auto const& rhsy = rhs[1].const_arrays();
    auto const& rhsz = rhs[2].const_arrays();

    auto dxinv = this->m_geom[amrlev][mglev].InvCellSizeArray();
    auto adxinv = dxinv;
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
        adxinv[idim] *= std::sqrt(m_alpha);
    }

    bool const has_beta = (m_bcoefs[amrlev][mglev][0] != nullptr);
    bool const has_alpha = (m_acoefs[amrlev][mglev][0] != nullptr);
    bool const has_osm = (m_overset_mask[amrlev][mglev][0] != nullptr);
    bool const use_pcg = m_use_pcg || has_alpha;
    // We support LU solver with variable beta and scalar alpha.

    auto dinfo = getDirichletInfo(amrlev,mglev);
    auto sinfo = getSymmetryInfo(amrlev,mglev);

    MultiFab nmf(amrex::convert(rhs[0].boxArray(),IntVect(1)),
                 rhs[0].DistributionMap(), 1, 0, MFInfo().SetAlloc(false));
    if (m_lusolver[amrlev][mglev] && !has_alpha && !has_beta && !has_osm) {
#if (AMREX_SPACEDIM == 2)
        auto b = m_beta;
#endif
        auto* plusolver = m_lusolver[amrlev][mglev]->dataPtr();
        ParallelFor(nmf, [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
        {
            mlcurlcurl_gs4_lu(i,j,k,ex[bno],ey[bno],ez[bno],
                              rhsx[bno],rhsy[bno],rhsz[bno],
#if (AMREX_SPACEDIM == 2)
                              b,
#endif
                              adxinv,color,*plusolver,dinfo,sinfo);
        });
    } else if (has_alpha && has_beta) {
        auto const& acx = m_acoefs[amrlev][mglev][0]->const_arrays();
        auto const& acy = m_acoefs[amrlev][mglev][1]->const_arrays();
        auto const& acz = m_acoefs[amrlev][mglev][2]->const_arrays();
        auto b = m_beta;
        auto const& bcx = m_bcoefs[amrlev][mglev][0]->const_arrays();
        auto const& bcy = m_bcoefs[amrlev][mglev][1]->const_arrays();
        auto const& bcz = m_bcoefs[amrlev][mglev][2]->const_arrays();
        if (has_osm)
        {
            auto const& nosm = m_nodal_overset_mask[amrlev][mglev]->const_arrays();
            ParallelFor(nmf, [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
            {
                mlcurlcurl_gs4_alpha_os(i,j,k,ex[bno],ey[bno],ez[bno],
                                    rhsx[bno],rhsy[bno],rhsz[bno],
                                    dxinv,color,
                                    acx[bno],acy[bno],acz[bno],
                                    bcx[bno],bcy[bno],bcz[bno],
                                    b,nosm[bno],dinfo,sinfo);
            });
        } else {
            ParallelFor(nmf, [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
            {
                mlcurlcurl_gs4_alpha(i,j,k,ex[bno],ey[bno],ez[bno],
                                    rhsx[bno],rhsy[bno],rhsz[bno],
                                    dxinv,color,
                                    acx[bno],acy[bno],acz[bno],
                                    bcx[bno],bcy[bno],bcz[bno],
                                    b,dinfo,sinfo);
            });
        }
    } else if (has_alpha && !has_beta) {
        auto const& acx = m_acoefs[amrlev][mglev][0]->const_arrays();
        auto const& acy = m_acoefs[amrlev][mglev][1]->const_arrays();
        auto const& acz = m_acoefs[amrlev][mglev][2]->const_arrays();
        auto b = m_beta;
        if (has_osm)
        {
            auto const& nosm = m_nodal_overset_mask[amrlev][mglev]->const_arrays();
            ParallelFor(nmf, [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
            {
                Array4<Real const> empty;
                mlcurlcurl_gs4_alpha_os(i,j,k,ex[bno],ey[bno],ez[bno],
                                    rhsx[bno],rhsy[bno],rhsz[bno],
                                    dxinv,color,
                                    acx[bno],acy[bno],acz[bno],
                                    empty,empty,empty,
                                    b,nosm[bno],dinfo,sinfo);
            });
        } else {
            ParallelFor(nmf, [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
            {
                Array4<Real const> empty;
                mlcurlcurl_gs4_alpha(i,j,k,ex[bno],ey[bno],ez[bno],
                                    rhsx[bno],rhsy[bno],rhsz[bno],
                                    dxinv,color,
                                    acx[bno],acy[bno],acz[bno],
                                    empty,empty,empty,
                                    b,dinfo,sinfo);
            });
        }
    } else if (!has_alpha && has_beta) {
        // This branch covers scalar alpha and variable beta.
        // If LU is used, we will build local solvers.
        auto const& bcx = m_bcoefs[amrlev][mglev][0]->const_arrays();
        auto const& bcy = m_bcoefs[amrlev][mglev][1]->const_arrays();
        auto const& bcz = m_bcoefs[amrlev][mglev][2]->const_arrays();
        if (use_pcg) {
            ParallelFor(nmf, [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
            {
                mlcurlcurl_gs4<true>(i,j,k,ex[bno],ey[bno],ez[bno],
                                     rhsx[bno],rhsy[bno],rhsz[bno],
                                     adxinv,color,bcx[bno],bcy[bno],bcz[bno],
                                     dinfo,sinfo);
            });
        } else {
            ParallelFor(nmf, [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
            {
                mlcurlcurl_gs4<false>(i,j,k,ex[bno],ey[bno],ez[bno],
                                      rhsx[bno],rhsy[bno],rhsz[bno],
                                      adxinv,color,bcx[bno],bcy[bno],bcz[bno],
                                      dinfo,sinfo);
            });
        }
    } else {
        AMREX_ASSERT(!has_alpha && !has_beta && has_osm);
        auto b = m_beta;
        auto const& nosm = m_nodal_overset_mask[amrlev][mglev]->const_arrays();
        if (use_pcg) {
            ParallelFor(nmf, [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
            {
                mlcurlcurl_gs4_os<true>(i,j,k,ex[bno],ey[bno],ez[bno],
                                        rhsx[bno],rhsy[bno],rhsz[bno],
                                        adxinv,color,b,nosm[bno],dinfo,sinfo);
            });
        } else {
            ParallelFor(nmf, [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
            {
                mlcurlcurl_gs4_os<false>(i,j,k,ex[bno],ey[bno],ez[bno],
                                         rhsx[bno],rhsy[bno],rhsz[bno],
                                         adxinv,color,b,nosm[bno],dinfo,sinfo);
            });
        }
    }
    if (!Gpu::inNoSyncRegion()) {
        Gpu::streamSynchronize();
    }
}
#endif

void MLCurlCurl::solutionResidual (int amrlev, MF& resid, MF& x, const MF& b,
                                   const MF* /*crse_bcdata*/)
{
    BL_PROFILE("MLCurlCurl::solutionResidual()");
    const int mglev = 0;
    apply(amrlev, mglev, resid, x, BCMode::Inhomogeneous, StateMode::Solution);
    compresid(amrlev, mglev, resid, b);
}

void MLCurlCurl::correctionResidual (int amrlev, int mglev, MF& resid, MF& x,
                                     const MF& b, BCMode bc_mode,
                                     const MF* crse_bcdata)
{
    AMREX_ALWAYS_ASSERT(bc_mode != BCMode::Inhomogeneous && crse_bcdata == nullptr);
    apply(amrlev, mglev, resid, x, BCMode::Homogeneous, StateMode::Correction);
    compresid(amrlev, mglev, resid, b);
}

void MLCurlCurl::compresid (int amrlev, int mglev, MF& resid, MF const& b) const
{
    auto dinfo = getDirichletInfo(amrlev,mglev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(resid[0],TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Box const& xbx = mfi.tilebox(resid[0].ixType().toIntVect());
        Box const& ybx = mfi.tilebox(resid[1].ixType().toIntVect());
        Box const& zbx = mfi.tilebox(resid[2].ixType().toIntVect());
        auto const& resx = resid[0].array(mfi);
        auto const& resy = resid[1].array(mfi);
        auto const& resz = resid[2].array(mfi);
        auto const& bx = b[0].array(mfi);
        auto const& by = b[1].array(mfi);
        auto const& bz = b[2].array(mfi);
        Array4<int const> xosm, yosm, zosm;
        if (m_overset_mask[amrlev][mglev][0])
        {
            AMREX_ASSERT(m_overset_mask[amrlev][mglev][1]);
            AMREX_ASSERT(m_overset_mask[amrlev][mglev][2]);
            xosm = m_overset_mask[amrlev][mglev][0]->const_array(mfi);
            yosm = m_overset_mask[amrlev][mglev][1]->const_array(mfi);
            zosm = m_overset_mask[amrlev][mglev][2]->const_array(mfi);
        }
        amrex::ParallelFor(xbx, ybx, zbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            if (dinfo.is_dirichlet_x_edge(i,j,k)) {
                resx(i,j,k) = Real(0.0);
            } else if (xosm && xosm(i,j,k) == 0) {
                resx(i,j,k) = Real(0.0);
            } else {
                resx(i,j,k) = bx(i,j,k) - resx(i,j,k);
            }
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            if (dinfo.is_dirichlet_y_edge(i,j,k)) {
                resy(i,j,k) = Real(0.0);
            } else if (yosm && yosm(i,j,k) == 0) {
                resy(i,j,k) = Real(0.0);
            } else {
                resy(i,j,k) = by(i,j,k) - resy(i,j,k);
            }
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            if (dinfo.is_dirichlet_z_edge(i,j,k)) {
                resz(i,j,k) = Real(0.0);
            } else if (zosm && zosm(i,j,k) == 0) {
                resz(i,j,k) = Real(0.0);
            } else {
                resz(i,j,k) = bz(i,j,k) - resz(i,j,k);
            }
        });
    }
}

void MLCurlCurl::update_lusolver ()
{
#if (AMREX_SPACEDIM > 1)
    // There is no global LU Solver that can be built for variable alpha or beta.
    // Overset mask zeroes per-node rows/cols, so the uniform factor is invalid too.
    if (m_bcoefs[0][0][0] == nullptr && m_acoefs[0][0][0] == nullptr
        && m_overset_mask[0][0][0] == nullptr) {
        for (int amrlev = 0;  amrlev < m_num_amr_levels; ++amrlev) {
            for (int mglev = 0; mglev < m_num_mg_levels[amrlev]; ++mglev) {
                auto const& dxinv = this->m_geom[amrlev][mglev].InvCellSizeArray();
                Real dxx = dxinv[0]*dxinv[0];
                Real dyy = dxinv[1]*dxinv[1];
                Real dxy = dxinv[0]*dxinv[1];
#if (AMREX_SPACEDIM == 2)
                Array2D<Real,0,3,0,3,Order::C> A
                    {m_alpha*dyy*Real(2.0) + m_beta,
                     Real(0.0),
                    -m_alpha*dxy,
                     m_alpha*dxy,
                     //
                     Real(0.0),
                     m_alpha*dyy*Real(2.0) + m_beta,
                     m_alpha*dxy,
                    -m_alpha*dxy,
                     //
                    -m_alpha*dxy,
                     m_alpha*dxy,
                     m_alpha*dxx*Real(2.0) + m_beta,
                     Real(0.0),
                     //
                     m_alpha*dxy,
                    -m_alpha*dxy,
                     Real(0.0),
                     m_alpha*dxx*Real(2.0) + m_beta};
#else
                Real dzz = dxinv[2]*dxinv[2];
                Real dxz = dxinv[0]*dxinv[2];
                Real dyz = dxinv[1]*dxinv[2];

                Array2D<Real,0,5,0,5,Order::C> A
                    {m_alpha*(dyy+dzz)*Real(2.0) + m_beta,
                     Real(0.0),
                    -m_alpha*dxy,
                     m_alpha*dxy,
                    -m_alpha*dxz,
                     m_alpha*dxz,
                     //
                     Real(0.0),
                     m_alpha*(dyy+dzz)*Real(2.0) + m_beta,
                     m_alpha*dxy,
                    -m_alpha*dxy,
                     m_alpha*dxz,
                    -m_alpha*dxz,
                     //
                    -m_alpha*dxy,
                     m_alpha*dxy,
                     m_alpha*(dxx+dzz)*Real(2.0) + m_beta,
                     Real(0.0),
                    -m_alpha*dyz,
                     m_alpha*dyz,
                     //
                     m_alpha*dxy,
                    -m_alpha*dxy,
                     Real(0.0),
                     m_alpha*(dxx+dzz)*Real(2.0) + m_beta,
                     m_alpha*dyz,
                    -m_alpha*dyz,
                     //
                    -m_alpha*dxz,
                     m_alpha*dxz,
                    -m_alpha*dyz,
                     m_alpha*dyz,
                     m_alpha*(dxx+dyy)*Real(2.0) + m_beta,
                     Real(0.0),
                     //
                     m_alpha*dxz,
                    -m_alpha*dxz,
                     m_alpha*dyz,
                    -m_alpha*dyz,
                     Real(0.0),
                     m_alpha*(dxx+dyy)*Real(2.0) + m_beta};
#endif

                m_lusolver[amrlev][mglev]
                    = std::make_unique<Gpu::DeviceScalar
                                       <LUSolver<AMREX_SPACEDIM*2,RT>>>(A);
            }
        }
    }
#endif
}

void MLCurlCurl::prepareForSolve ()
{
    set_curvilinear_domain_bc();
    update_lusolver();
}

void MLCurlCurl::preparePrecond ()
{
    set_curvilinear_domain_bc();
}

void MLCurlCurl::set_curvilinear_domain_bc ()
{
#if (AMREX_SPACEDIM == 1)
    if (m_coord > 0) {
        // Even though it's not exactly Dirichlet, setting this to Dirichlet
        // will skip ghost cell filling at the axis.
        m_lobc[0][0] = LinOpBCType::Dirichlet;
    }
#endif
}

void MLCurlCurl::clearAlphaMultiFab ()
{
    for (auto& amrvec : m_acoefs) {
        for (auto& arr : amrvec) {
            for (auto& mf : arr) {
                mf.reset();
            }
        }
    }
}

void MLCurlCurl::clearBetaMultiFab ()
{
    for (auto& amrvec : m_bcoefs) {
        for (auto& arr : amrvec) {
            for (auto& mf : arr) {
                mf.reset();
            }
        }
    }
}

Real MLCurlCurl::xdoty (int amrlev, int mglev, const MF& x, const MF& y,
                        bool local) const
{
    auto result = Real(0.0);
    for (int idim = 0; idim < 3; ++idim) {
        auto rtmp = MultiFab::Dot(getDotMask(amrlev,mglev,idim),
                                  x[idim], 0, y[idim], 0, 1, 0, true);
        result += rtmp;
    }
    if (!local) {
        ParallelAllReduce::Sum(result, ParallelContext::CommunicatorSub());
    }
    return result;
}

Real MLCurlCurl::normInf (int amrlev, MF const& mf, bool local) const
{
    constexpr int mglev = 0;
    Real r = Real(0.0);
    for (int idim = 0; idim < 3; ++idim) {
        Real ridim;
        if (m_overset_mask[amrlev][mglev][idim]) {
            ReduceOps<ReduceOpMax> reduce_op;
            ReduceData<Real> reduce_data(reduce_op);
            using ReduceTuple = typename decltype(reduce_data)::Type;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(mf[idim], TilingIfNotGPU()); mfi.isValid(); ++mfi) {
                Box const& bx = mfi.tilebox();
                auto const& a = mf[idim].const_array(mfi);
                auto const& osm = m_overset_mask[amrlev][mglev][idim]->const_array(mfi);
                reduce_op.eval(bx, reduce_data,
                [=] AMREX_GPU_HOST_DEVICE (int i, int j, int k) -> ReduceTuple
                {
                    return { (osm(i,j,k) == 0) ? Real(0.0)
                                               : std::abs(a(i,j,k)) };
                });
            }
            ridim = amrex::get<0>(reduce_data.value(reduce_op));
        } else {
            ridim = amrex::norminf(mf[idim], 0, m_ncomp, IntVect(0), true);
        }
        r = std::max(r, ridim);
    }
    if (!local) {
        ParallelAllReduce::Max(r, ParallelContext::CommunicatorSub());
    }
    return r;
}

void MLCurlCurl::averageDownAndSync (Vector<MF>& sol) const
{
    BL_PROFILE("MLCurlCurl::averageDownAndSync()");
    AMREX_ALWAYS_ASSERT(sol.size() == 1);
    const int amrlev = 0;
    const int mglev = 0;
    for (int idim = 0; idim < 3; ++idim) {
        amrex::OverrideSync(sol[amrlev][idim], getDotMask(amrlev,mglev,idim),
                            this->m_geom[amrlev][mglev].periodicity());
    }
}

void MLCurlCurl::make (Vector<Vector<MF> >& mf, IntVect const& ng) const
{
    MLLinOpT<MF>::make(mf, ng);
}

Array<MultiFab,3>
MLCurlCurl::make (int amrlev, int mglev, IntVect const& ng) const
{
    MF r;
    for (int idim = 0; idim < 3; ++idim) {
        r[idim].define(amrex::convert(this->m_grids[amrlev][mglev], m_etype[idim]),
                       this->m_dmap[amrlev][mglev], m_ncomp, ng, MFInfo(),
                       *(this->m_factory)[amrlev][mglev]);
    }
    return r;
}

Array<MultiFab,3>
MLCurlCurl::makeAlias (MF const& mf) const
{
    MF r;
    for (int idim = 0; idim < 3; ++idim) {
        r[idim] = MultiFab(mf[idim], amrex::make_alias, 0, mf[idim].nComp());
    }
    return r;
}

Array<MultiFab,3>
MLCurlCurl::makeCoarseMG (int amrlev, int mglev, IntVect const& ng) const
{
    BoxArray cba = this->m_grids[amrlev][mglev];
    IntVect ratio = (amrlev > 0) ? IntVect(2) : this->mg_coarsen_ratio_vec[mglev];
    cba.coarsen(ratio);

    MF r;
    for (int idim = 0; idim < 3; ++idim) {
        r[idim].define(amrex::convert(cba, m_etype[idim]),
                       this->m_dmap[amrlev][mglev], m_ncomp, ng);
    }
    return r;
}

Array<MultiFab,3>
MLCurlCurl::makeCoarseAmr (int famrlev, IntVect const& ng) const
{
    BoxArray cba = this->m_grids[famrlev][0];
    IntVect ratio(this->AMRRefRatio(famrlev-1));
    cba.coarsen(ratio);

    MF r;
    for (int idim = 0; idim < 3; ++idim) {
        r[idim].define(amrex::convert(cba, m_etype[idim]),
                       this->m_dmap[famrlev][0], m_ncomp, ng);
    }
    return r;
}

void MLCurlCurl::applyBC (int amrlev, int mglev, MF& in, CurlCurlStateType type) const
{
    int nmfs = 3;
#if (AMREX_SPACEDIM == 2)
    if (CurlCurlStateType::b == type) {
        nmfs = 2; // no need to applyBC on Ez
    }
#elif (AMREX_SPACEDIM == 1)
    if (CurlCurlStateType::b == type) {
        nmfs = 1; // no need to applyBC on Ey and Ez
    }
#endif
    Vector<MultiFab*> mfs(nmfs);
    for (int imf = 0; imf < nmfs; ++imf) {
        mfs[imf] = in.data() + imf;
    }
    FillBoundary(mfs, this->m_geom[amrlev][mglev].periodicity());
    for (auto* mf : mfs) {
        applyPhysBC(amrlev, mglev, *mf, type);
    }
}

void MLCurlCurl::applyPhysBC (int amrlev, int mglev, MultiFab& mf, CurlCurlStateType type) const
{
    if (CurlCurlStateType::b == type) { return; }

    auto const idxtype = mf.ixType();
    Box const domain = amrex::convert(this->m_geom[amrlev][mglev].Domain(), idxtype);
    Box const gdomain = amrex::convert
        (this->m_geom[amrlev][mglev].growPeriodicDomain(1), idxtype);

    MFItInfo mfi_info{};

#ifdef AMREX_USE_GPU
    Vector<Array4BoxOrientationTag<RT>> tags;
    mfi_info.DisableDeviceSync();
#endif

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(mf,mfi_info); mfi.isValid(); ++mfi) {
        auto const& vbx = mfi.validbox();
        auto const& a = mf.array(mfi);
        for (OrientationIter oit; oit; ++oit) {
            Orientation const face = oit();
            int const idim = face.coordDir();
            bool is_symmetric = face.isLow()
                ? m_lobc[0][idim] == LinOpBCType::symmetry
                : m_hibc[0][idim] == LinOpBCType::symmetry;
            if (domain[face] == vbx[face] && is_symmetric &&
                ((type == CurlCurlStateType::x) ||
                 (type == CurlCurlStateType::r && idxtype.nodeCentered(idim)))) // transverse direction only
            {
                Box b = vbx;
                for (int jdim = 0; jdim < AMREX_SPACEDIM; ++jdim) {
                    if (jdim == idim) {
                        int shift = face.isLow() ? -1 : 1;
                        b.setRange(jdim, domain[face] + shift, 1);
                    } else {
                        if (b.smallEnd(jdim) > gdomain.smallEnd(jdim)) {
                            b.growLo(jdim);
                        }
                        if (b.bigEnd(jdim) < gdomain.bigEnd(jdim)) {
                            b.growHi(jdim);
                        }
                    }
                }
#ifdef AMREX_USE_GPU
                tags.emplace_back(Array4BoxOrientationTag<RT>{a,b,face});
#else
                amrex::LoopOnCpu(b, [&] (int i, int j, int k)
                {
                    mlcurlcurl_bc_symmetry(i, j, k, face, idxtype, a);
                });
#endif
            }
        }
    }

#ifdef AMREX_USE_GPU
    ParallelFor(tags,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, Array4BoxOrientationTag<RT> const& tag) noexcept
    {
        mlcurlcurl_bc_symmetry(i, j, k, tag.face, idxtype, tag.fab);
    });
#endif

    if (CurlCurlStateType::r == type) { // fix domain edges
        auto sinfo = getSymmetryInfo(amrlev,mglev);

#ifdef AMREX_USE_GPU
        Vector<Array4BoxOffsetTag<RT>> tags2;
#endif

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(mf,mfi_info); mfi.isValid(); ++mfi) {
            auto const& vbx = mfi.validbox();
            auto const& a = mf.array(mfi);
            for (int idim = 0; idim < AMREX_SPACEDIM-1; ++idim) {
                for (int jdim = idim+1; jdim < AMREX_SPACEDIM; ++jdim) {
                    if (idxtype.nodeCentered(idim) &&
                        idxtype.nodeCentered(jdim))
                    {
                        for (int iside = 0; iside < 2; ++iside) {
                            int ii = (iside == 0) ? vbx.smallEnd(idim) : vbx.bigEnd(idim);
                            for (int jside = 0; jside < 2; ++jside) {
                                int jj = (jside == 0) ? vbx.smallEnd(jdim) : vbx.bigEnd(jdim);
                                if (sinfo.is_symmetric(idim,iside,ii) &&
                                    sinfo.is_symmetric(jdim,jside,jj))
                                {
                                    IntVect oiv(0);
                                    oiv[idim] = (iside == 0) ? 2 : -2;
                                    oiv[jdim] = (jside == 0) ? 2 : -2;
                                    Dim3 offset = oiv.dim3();

                                    Box b = vbx;
                                    if (iside == 0) {
                                        b.setRange(idim,vbx.smallEnd(idim)-1);
                                    } else {
                                        b.setRange(idim,vbx.bigEnd(idim)+1);
                                    }
                                    if (jside == 0) {
                                        b.setRange(jdim,vbx.smallEnd(jdim)-1);
                                    } else {
                                        b.setRange(jdim,vbx.bigEnd(jdim)+1);
                                    }
#ifdef AMREX_USE_GPU
                                    tags2.emplace_back(Array4BoxOffsetTag<RT>{a,b,offset});
#else
                                    amrex::LoopOnCpu(b, [&] (int i, int j, int k)
                                    {
                                        a(i,j,k) = a(i+offset.x,j+offset.y,k+offset.z);
                                    });
#endif
                                }
                            }
                        }
                    }
                }
            }
        }

#ifdef AMREX_USE_GPU
        ParallelFor(tags2,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, Array4BoxOffsetTag<RT> const& tag)
        {
            tag.fab(i,j,k) = tag.fab(i+tag.offset.x,j+tag.offset.y,k+tag.offset.z);
        });
#endif
    }
}

iMultiFab const& MLCurlCurl::getDotMask (int amrlev, int mglev, int idim) const
{
    if (m_dotmask[amrlev][mglev][idim] == nullptr) {
        MultiFab tmp(amrex::convert(this->m_grids[amrlev][mglev], m_etype[idim]),
                     this->m_dmap[amrlev][mglev], 1, 0, MFInfo().SetAlloc(false));
        m_dotmask[amrlev][mglev][idim] =
            tmp.OwnerMask(this->m_geom[amrlev][mglev].periodicity());

        // AND in overset mask so masked DOFs are not counted in dot products.
        if (m_overset_mask[amrlev][mglev][idim]) {
            auto& dm = *m_dotmask[amrlev][mglev][idim];
            auto const& dma = dm.arrays();
            auto const& osma = m_overset_mask[amrlev][mglev][idim]->const_arrays();
            ParallelFor(dm, [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
            {
                if (osma[bno](i,j,k) == 0) { dma[bno](i,j,k) = 0; }
            });
            if (!Gpu::inNoSyncRegion()) { Gpu::streamSynchronize(); }
        }
    }
    return *m_dotmask[amrlev][mglev][idim];
}

CurlCurlDirichletInfo MLCurlCurl::getDirichletInfo (int amrlev, int mglev) const
{
    auto helper = [&] (int idim, int face) -> int
    {
#if (AMREX_SPACEDIM == 2)
        if (idim == 2) {
            return std::numeric_limits<int>::lowest();
        }
#elif (AMREX_SPACEDIM == 1)
        if (idim > 0) {
            return std::numeric_limits<int>::lowest();
        }
#endif
        // The code above is to avoid compiler warnings. It has no meanning.

        if (face == 0) {
            if (m_lobc[0][idim] == LinOpBCType::Dirichlet) {
                return m_geom[amrlev][mglev].Domain().smallEnd(idim);
            } else {
                return std::numeric_limits<int>::lowest();
            }
        } else {
            if (m_hibc[0][idim] == LinOpBCType::Dirichlet) {
                return m_geom[amrlev][mglev].Domain().bigEnd(idim) + 1;
            } else {
                return std::numeric_limits<int>::max();
            }
        }
    };

    return CurlCurlDirichletInfo{IntVect(AMREX_D_DECL(helper(0,0),
                                                      helper(1,0),
                                                      helper(2,0))),
                                 IntVect(AMREX_D_DECL(helper(0,1),
                                                      helper(1,1),
                                                      helper(2,1)))
#if (AMREX_SPACEDIM < 3)
                                 ,m_coord
#endif
                                };
}

CurlCurlSymmetryInfo MLCurlCurl::getSymmetryInfo (int amrlev, int mglev) const
{
    auto helper = [&] (int idim, int face) -> int
    {
#if (AMREX_SPACEDIM == 2)
        if (idim == 2) {
            return std::numeric_limits<int>::lowest();
        }
#elif (AMREX_SPACEDIM == 1)
        if (idim > 0) {
            return std::numeric_limits<int>::lowest();
        }
#endif
        // The code above is to avoid compiler warnings. It has no meaning.

        if (face == 0) {
            if (m_lobc[0][idim] == LinOpBCType::symmetry) {
                return m_geom[amrlev][mglev].Domain().smallEnd(idim);
            } else {
                return std::numeric_limits<int>::lowest();
            }
        } else {
            if (m_hibc[0][idim] == LinOpBCType::symmetry) {
                return m_geom[amrlev][mglev].Domain().bigEnd(idim) + 1;
            } else {
                return std::numeric_limits<int>::max();
            }
        }
    };

    return CurlCurlSymmetryInfo{IntVect(AMREX_D_DECL(helper(0,0),
                                                     helper(1,0),
                                                     helper(2,0))),
                                IntVect(AMREX_D_DECL(helper(0,1),
                                                     helper(1,1),
                                                     helper(2,1)))};
}

void MLCurlCurl::update ()
{
    if (MLLinOpT<Array<MultiFab,3>>::needsUpdate()) {
        MLLinOpT<Array<MultiFab,3>>::update();
    }

    if (m_needs_update) {
        update_lusolver();
        m_needs_update = false;
    }
}

void MLCurlCurl::applyOverset (int amrlev, Array<MultiFab,3>& rhs) const
{
    // Called once on finest level (mglev=0) by MLMG::prepareForSolve.
    if (m_overset_mask[amrlev][0][0]) {
        for (int idim = 0; idim < 3; ++idim)
        {
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
            for (MFIter mfi(*m_overset_mask[amrlev][0][idim],TilingIfNotGPU()); mfi.isValid(); ++mfi)
            {
                const Box& bx = mfi.tilebox();
                auto const& rfab = rhs[idim].array(mfi);
                auto const& osm = m_overset_mask[amrlev][0][idim]->const_array(mfi);
                // todo: make this a version that doesn't specify n
                AMREX_HOST_DEVICE_PARALLEL_FOR_4D(bx, 1, i, j, k, n,
                {
                    if (osm(i,j,k) == 0) { rfab(i,j,k,n) = RT(0.0); }
                });
            }
        }
    }
}

}
