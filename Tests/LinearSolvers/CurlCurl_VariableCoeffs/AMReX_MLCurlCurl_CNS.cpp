#include <AMReX_MLCurlCurl_CNS.H>

namespace {

//! Fill ghost cells by zero-order extrapolation from the nearest valid cell.
/**
 * After average_down + FillBoundary, ghost cells not covered by any
 * neighbouring box (physical-boundary and coarse/fine-boundary ghosts)
 * may still hold uninitialised values.  This function copies the nearest
 * valid-cell value into every ghost cell so that the MG smoother
 * stencil always sees reasonable coefficients.
 *
 * Call order: average_down → fillGhostByExtrapolation → FillBoundary.
 * FillBoundary overwrites inter-box ghosts with the correct neighbour
 * values; physical/C-F boundary ghosts keep the extrapolated values.
 *
 * \param mf MultiFab whose ghost cells are to be filled
 */
void fillGhostByExtrapolation (amrex::MultiFab& mf)
{
  using namespace amrex;
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
  for (MFIter mfi(mf); mfi.isValid(); ++mfi)
  {
    Box const& vbx = mfi.validbox();
    Box const& gbx = mfi.fabbox();
    if (vbx == gbx) { continue; }
    auto const& arr = mf.array(mfi);
    auto const lo = vbx.smallEnd();
    auto const hi = vbx.bigEnd();
    auto const vb = vbx;
    ParallelFor(gbx, [=] AMREX_GPU_DEVICE (int i, int j, int k)
    {
      if (!vb.contains(IntVect(AMREX_D_DECL(i,j,k))))
      {
        int ci = amrex::max(lo[0], amrex::min(i, hi[0]));
        int cj = amrex::max(lo[1], amrex::min(j, hi[1]));
#if (AMREX_SPACEDIM == 3)
        int ck = amrex::max(lo[2], amrex::min(k, hi[2]));
#else
        int ck = k;
#endif
        arr(i,j,k) = arr(ci,cj,ck);
      }
    });
  }
}

} // anonymous namespace

namespace amrex {

MLCurlCurl_CNS::MLCurlCurl_CNS (const Vector<Geometry>& a_geom,
                        const Vector<BoxArray>& a_grids,
                        const Vector<DistributionMapping>& a_dmap,
                        const LPInfo& a_info)
{
    define(a_geom, a_grids, a_dmap, a_info);
}

void MLCurlCurl_CNS::define (const Vector<Geometry>& a_geom,
                         const Vector<BoxArray>& a_grids,
                         const Vector<DistributionMapping>& a_dmap,
                         const LPInfo& a_info)
{
    MLLinOpT<MF>::define(a_geom, a_grids, a_dmap, a_info, {});

    m_dotmask.resize(this->m_num_amr_levels);
    for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev) {
        m_dotmask[amrlev].resize(this->m_num_mg_levels[amrlev]);
    }

    m_cfmask.resize(this->m_num_amr_levels);
    for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev) {
        m_cfmask[amrlev].resize(this->m_num_mg_levels[amrlev]);
    }

    m_bcoefs.resize(this->m_num_amr_levels);
    for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev) {
        m_bcoefs[amrlev].resize(this->m_num_mg_levels[amrlev]);
    }

    m_lusolver.resize(this->m_num_amr_levels);
    for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev) {
        m_lusolver[amrlev].resize(this->m_num_mg_levels[amrlev]);
    }

    m_acoefs.resize(this->m_num_amr_levels);
    for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev) {
        m_acoefs[amrlev].resize(this->m_num_mg_levels[amrlev]);
    }

    m_cf_data.resize(this->m_num_amr_levels);
    m_has_cf_data.resize(this->m_num_amr_levels, 0);
}

void MLCurlCurl_CNS::setScalars (RT a_alpha, RT a_beta) noexcept
{
    m_needs_update = true;
    m_alpha = a_alpha;
    m_beta = a_beta;
    AMREX_ASSERT(m_beta > RT(0));
}

void MLCurlCurl_CNS::setBeta (const Vector<Array<MultiFab const*,3>>& a_bcoefs)
{
    m_needs_update = true;

    Array<IntVect,3> ng;
    for (int idim = 0; idim < 3; ++idim) {
        ng[idim] = IntVect(2) - m_etype[idim]; // 2 ghost for cell direction, 1 for node
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

    //     for (int idim = 0; idim < 3; ++idim) {
    // for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev) {
    // for (int mglev = 0; mglev < m_num_mg_levels[amrlev]; ++mglev) {
    // m_bcoefs[amrlev][mglev][idim]->setVal(0.0001) ; }}}
}

void MLCurlCurl_CNS::setAlpha (const Vector<Array<MultiFab const*,3>>& a_acoefs)
{
    m_needs_update = true;
    m_has_variable_alpha = true;

    // Ghost cells: 2 in cell-centred directions, 1 in nodal directions.
    // Extra ghosts needed for the ghost-nodes strategy where the smoother
    // iterates over layer-1 CF ghosts and the stencil reaches layer-2.
    Array<IntVect,3> ng;
    for (int idim = 0; idim < 3; ++idim) {
        ng[idim] = IntVect(2) - m_ftype[idim];
    }

    // Copy user data into internal storage at MG level 0
    for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev) {
        for (int idim = 0; idim < 3; ++idim) {
            if (m_acoefs[amrlev][0][idim] == nullptr) {
                m_acoefs[amrlev][0][idim] = std::make_unique<MultiFab>
                    (a_acoefs[amrlev][idim]->boxArray(),
                     a_acoefs[amrlev][idim]->DistributionMap(), 1, ng[idim]);
            }
            MultiFab::Copy(*m_acoefs[amrlev][0][idim],
                           *a_acoefs[amrlev][idim], 0, 0, 1, ng[idim]);
            fillGhostByExtrapolation(*m_acoefs[amrlev][0][idim]);
            m_acoefs[amrlev][0][idim]->FillBoundary(
                m_geom[amrlev][0].periodicity());
        }
    }

    // Coarsen alpha for multigrid levels
    for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev) {
        for (int mglev = 1; mglev < m_num_mg_levels[amrlev]; ++mglev) {
            IntVect ratio = (amrlev > 0) ? IntVect(2)
                : mg_coarsen_ratio_vec[mglev-1];

            // Face-centred components within SPACEDIM: use average_down_faces
            for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
                if (m_acoefs[amrlev][mglev][idim] == nullptr) {
                    m_acoefs[amrlev][mglev][idim] = std::make_unique<MultiFab>
                        (amrex::convert(m_grids[amrlev][mglev], m_ftype[idim]),
                         m_dmap[amrlev][mglev], 1, ng[idim]);
                }
                average_down_faces(*m_acoefs[amrlev][mglev-1][idim],
                                   *m_acoefs[amrlev][mglev  ][idim], ratio);
                fillGhostByExtrapolation(*m_acoefs[amrlev][mglev][idim]);
                m_acoefs[amrlev][mglev][idim]->FillBoundary(
                    m_geom[amrlev][mglev].periodicity());
            }

            // Extra components beyond SPACEDIM are cell-centred or nodal
#if (AMREX_SPACEDIM < 3)
            // Component 2 (z-face): cell-centred in 2D, use average_down
            if (m_acoefs[amrlev][mglev][2] == nullptr) {
                m_acoefs[amrlev][mglev][2] = std::make_unique<MultiFab>
                    (m_grids[amrlev][mglev],m_dmap[amrlev][mglev], 1, ng[2]);
            }
            amrex::average_down(*m_acoefs[amrlev][mglev-1][2],
                                *m_acoefs[amrlev][mglev  ][2],
                                0, 1, ratio);
            fillGhostByExtrapolation(*m_acoefs[amrlev][mglev][2]);
            m_acoefs[amrlev][mglev][2]->FillBoundary(
                m_geom[amrlev][mglev].periodicity());

#endif
#if (AMREX_SPACEDIM == 1)
            // Component 1 (y-face): also cell-centred in 1D
            if (m_acoefs[amrlev][mglev][1] == nullptr) {
                m_acoefs[amrlev][mglev][1] = std::make_unique<MultiFab>
                    (m_grids[amrlev][mglev],m_dmap[amrlev][mglev], 1, ng[1]);
            }
            amrex::average_down(*m_acoefs[amrlev][mglev-1][1],
                                *m_acoefs[amrlev][mglev  ][1],
                                0, 1, ratio);
            fillGhostByExtrapolation(*m_acoefs[amrlev][mglev][1]);
            m_acoefs[amrlev][mglev][1]->FillBoundary(
                m_geom[amrlev][mglev].periodicity());
#endif
        }
    }
}

void MLCurlCurl_CNS::prepareRHS (Vector<MF*> const& rhs) const
{
    for (int amrlev = 0; amrlev < m_num_amr_levels; ++amrlev) {
        for (auto& mf : *rhs[amrlev]) {
            mf.OverrideSync(m_geom[amrlev][0].periodicity());
        }
    }
}

void MLCurlCurl_CNS::setDirichletNodesToZero (int amrlev, int mglev, MF& a_mf) const
{
    MFItInfo mfi_info{};
#ifdef AMREX_USE_GPU
    Vector<Array4BoxTag<RT>> tags;
    mfi_info.DisableDeviceSync();
#endif

    for (auto& mf : a_mf)
    {
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
}

void MLCurlCurl_CNS::setLevelBC (int amrlev, const MF* levelbcdata,
                             const MF* robinbc_a, const MF* robinbc_b,
                             const MF* robinbc_f)
{
    amrex::ignore_unused(levelbcdata, robinbc_a, robinbc_b, robinbc_f);

    if (amrlev == 0 && this->needsCoarseDataForBC())
    {
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
            this->m_coarse_data_for_bc != nullptr,
            "MLCurlCurl_CNS::setLevelBC: setCoarseFineBC must be called before "
            "setLevelBC when the domain is not fully covered.");

        updateCFData(amrlev, *this->m_coarse_data_for_bc,
                     this->m_coarse_data_crse_ratio);
    }
}

void MLCurlCurl_CNS::updateCFData (int amrlev, const MF& crse, IntVect ratio)
{
    // Build the coarse-fine masks if not already built
    if (m_cfmask[amrlev][0][0] == nullptr)
    {
        buildCFMasks();
    }

    // Interpolate coarse E data to fine resolution and store.
    // The coarse data may live on a different BoxArray (the coarse level's),
    // so we first parallel-copy it onto a coarsened version of the fine
    // BoxArray, then inject from coarse to fine using matched iterators.
    auto const& fba = this->m_grids[amrlev][0];
    auto const& fdm = this->m_dmap[amrlev][0];

    for (int idim = 0; idim < 3; ++idim)
    {
        BoxArray fineBA = amrex::convert(fba, m_etype[idim]);
        m_cf_data[amrlev][idim].define(fineBA, fdm, 1, 2);
        m_cf_data[amrlev][idim].setVal(Real(0.0));

        // Build a coarsened version of the fine BA for the parallel copy.
        // Need 2 coarse ghost cells so the fine interpolation stencil
        // has data for both ghost layers.
        BoxArray crseBA = fineBA;
        crseBA.coarsen(ratio);
        MultiFab crseOnFineProc(crseBA, fdm, 1, 2);
        crseOnFineProc.setVal(Real(0.0));
        crseOnFineProc.ParallelCopy(crse[idim], 0, 0, 1,
                                    IntVect(1), IntVect(2));

        // Linear interpolation from coarse to fine edges.
        // In node directions the field is nodal, so fine nodes at
        // odd indices are linearly interpolated between adjacent
        // coarse nodes.  In cell-centred directions fine cells
        // share their enclosing coarse cell (injection).
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(m_cf_data[amrlev][idim], TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& fbx = mfi.growntilebox(2);
            auto const& fine_arr = m_cf_data[amrlev][idim].array(mfi);
            auto const& crse_arr = crseOnFineProc.const_array(mfi);
            auto const  r = ratio;
            auto const  et = m_etype[idim];
            ParallelFor(fbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                // Coarse index for each direction.
                // Node directions: ci = i / r  (integer division)
                // Cell directions: ci = floor(i / r) with negative handling
                int ci = i, cj = j, ck = k;
                AMREX_D_TERM(
                    ci = et[0] ? i / r[0] : (i < 0 ? (i-r[0]+1)/r[0] : i/r[0]);,
                    cj = et[1] ? j / r[1] : (j < 0 ? (j-r[1]+1)/r[1] : j/r[1]);,
                    ck = et[2] ? k / r[2] : (k < 0 ? (k-r[2]+1)/r[2] : k/r[2]);
                )

                // Determine which node directions need interpolation.
                // A fine node at an odd index in a nodal direction sits
                // between coarse nodes ci and ci+1.
                AMREX_D_TERM(
                    bool const x_interp = (et[0] == 1) && (i % r[0] != 0);,
                    bool const y_interp = (et[1] == 1) && (j % r[1] != 0);,
                    bool const z_interp = (et[2] == 1) && (k % r[2] != 0);
                )

#if (AMREX_SPACEDIM == 1)
                if (x_interp) {
                    fine_arr(i,j,k) = Real(0.5) *
                        (crse_arr(ci,cj,ck) + crse_arr(ci+1,cj,ck));
                } else {
                    fine_arr(i,j,k) = crse_arr(ci,cj,ck);
                }
#elif (AMREX_SPACEDIM == 2)
                if (x_interp && y_interp) {
                    fine_arr(i,j,k) = Real(0.25) *
                        (crse_arr(ci,  cj,  ck) + crse_arr(ci+1,cj,  ck) +
                         crse_arr(ci,  cj+1,ck) + crse_arr(ci+1,cj+1,ck));
                } else if (x_interp) {
                    fine_arr(i,j,k) = Real(0.5) *
                        (crse_arr(ci,cj,ck) + crse_arr(ci+1,cj,ck));
                } else if (y_interp) {
                    fine_arr(i,j,k) = Real(0.5) *
                        (crse_arr(ci,cj,ck) + crse_arr(ci,cj+1,ck));
                } else {
                    fine_arr(i,j,k) = crse_arr(ci,cj,ck);
                }
#elif (AMREX_SPACEDIM == 3)
                if (x_interp && y_interp && z_interp) {
                    fine_arr(i,j,k) = Real(0.125) *
                        (crse_arr(ci,  cj,  ck  ) + crse_arr(ci+1,cj,  ck  ) +
                         crse_arr(ci,  cj+1,ck  ) + crse_arr(ci+1,cj+1,ck  ) +
                         crse_arr(ci,  cj,  ck+1) + crse_arr(ci+1,cj,  ck+1) +
                         crse_arr(ci,  cj+1,ck+1) + crse_arr(ci+1,cj+1,ck+1));
                } else if (x_interp && y_interp) {
                    fine_arr(i,j,k) = Real(0.25) *
                        (crse_arr(ci,  cj,  ck) + crse_arr(ci+1,cj,  ck) +
                         crse_arr(ci,  cj+1,ck) + crse_arr(ci+1,cj+1,ck));
                } else if (x_interp && z_interp) {
                    fine_arr(i,j,k) = Real(0.25) *
                        (crse_arr(ci,  cj,ck  ) + crse_arr(ci+1,cj,ck  ) +
                         crse_arr(ci,  cj,ck+1) + crse_arr(ci+1,cj,ck+1));
                } else if (y_interp && z_interp) {
                    fine_arr(i,j,k) = Real(0.25) *
                        (crse_arr(ci,cj,  ck  ) + crse_arr(ci,cj+1,ck  ) +
                         crse_arr(ci,cj,  ck+1) + crse_arr(ci,cj+1,ck+1));
                } else if (x_interp) {
                    fine_arr(i,j,k) = Real(0.5) *
                        (crse_arr(ci,cj,ck) + crse_arr(ci+1,cj,ck));
                } else if (y_interp) {
                    fine_arr(i,j,k) = Real(0.5) *
                        (crse_arr(ci,cj,ck) + crse_arr(ci,cj+1,ck));
                } else if (z_interp) {
                    fine_arr(i,j,k) = Real(0.5) *
                        (crse_arr(ci,cj,ck) + crse_arr(ci,cj,ck+1));
                } else {
                    fine_arr(i,j,k) = crse_arr(ci,cj,ck);
                }
#endif
            });
        }
        m_cf_data[amrlev][idim].FillBoundary(
            this->m_geom[amrlev][0].periodicity());
    }

    m_has_cf_data[amrlev] = 1;
}

void MLCurlCurl_CNS::buildCFMasks ()
{
    // Build 2-layer masks for the ghost-nodes CF strategy.
    // 0 = valid cell or interior ghost (covered by same-level neighbor)
    // 1 = layer-1 CF ghost (adjacent to valid region — relaxed by smoother)
    // 2 = layer-2 CF ghost (outer ring — frozen Dirichlet)
    // Physical-boundary ghosts are always 0 (not treated as CF).
    for (int amrlev = 0; amrlev < this->m_num_amr_levels; ++amrlev)
    {
        for (int mglev = 0; mglev < this->m_num_mg_levels[amrlev]; ++mglev)
        {
            auto const& geom = this->m_geom[amrlev][mglev];
            auto const& period = geom.periodicity();

            for (int idim = 0; idim < 3; ++idim)
            {
                BoxArray const edgeBA = amrex::convert(
                    this->m_grids[amrlev][mglev], m_etype[idim]);
                m_cfmask[amrlev][mglev][idim] = std::make_unique<iMultiFab>(
                    edgeBA, this->m_dmap[amrlev][mglev], 1, 2);

                // Start: all cells = 2 (outer ghost).
                m_cfmask[amrlev][mglev][idim]->setVal(2);

                // Valid cells = 0.
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
                for (MFIter mfi(*m_cfmask[amrlev][mglev][idim],
                                TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    Box const& vbx = mfi.validbox();
                    auto const& mask = m_cfmask[amrlev][mglev][idim]->array(mfi);
                    ParallelFor(vbx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        mask(i,j,k) = 0;
                    });
                }

                // FillBoundary: ghost cells covered by neighbor valid
                // cells become 0.
                m_cfmask[amrlev][mglev][idim]->FillBoundary(period);

                // Now non-zero cells are CF ghosts.  Distinguish layers:
                // layer-1 = ghost cell adjacent (in ±x,±y,±z) to a valid
                //           (mask==0) cell; layer-2 = everything else.
                // We do this in two passes to avoid race conditions.

                // Pass 1: build a temporary copy of the current mask so
                // reads are consistent while we write.
                iMultiFab tmp(edgeBA, this->m_dmap[amrlev][mglev], 1, 2);
                iMultiFab::Copy(tmp, *m_cfmask[amrlev][mglev][idim],
                                0, 0, 1, 2);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
                for (MFIter mfi(*m_cfmask[amrlev][mglev][idim],
                                TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    Box const& gbx = amrex::grow(mfi.validbox(), 2);
                    auto const& mask = m_cfmask[amrlev][mglev][idim]->array(mfi);
                    auto const& old_mask = tmp.const_array(mfi);

                    ParallelFor(gbx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        if (old_mask(i,j,k) == 0) { return; } // valid cell
                        // Check if any face-neighbor is a valid cell
                        bool adj_valid = false;
#if (AMREX_SPACEDIM >= 2)
                        adj_valid = adj_valid
                            || old_mask(i-1,j,k) == 0
                            || old_mask(i+1,j,k) == 0
                            || old_mask(i,j-1,k) == 0
                            || old_mask(i,j+1,k) == 0;
#endif
#if (AMREX_SPACEDIM == 3)
                        adj_valid = adj_valid
                            || old_mask(i,j,k-1) == 0
                            || old_mask(i,j,k+1) == 0;
#endif
                        mask(i,j,k) = adj_valid ? 1 : 2;
                    });
                }

                // Zero out physical-boundary ghosts (outside domain).
                auto const idxtype = m_etype[idim];
                Box const domain = amrex::convert(geom.Domain(), idxtype);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
                for (MFIter mfi(*m_cfmask[amrlev][mglev][idim],
                                TilingIfNotGPU()); mfi.isValid(); ++mfi)
                {
                    Box const& gbx = amrex::grow(mfi.validbox(), 2);
                    auto const& mask = m_cfmask[amrlev][mglev][idim]->array(mfi);
                    auto const dom = domain;

                    ParallelFor(gbx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        if (!dom.contains(IntVect(AMREX_D_DECL(i,j,k))))
                        {
                            mask(i,j,k) = 0;
                        }
                    });
                }
            }
        }
    }
}

void MLCurlCurl_CNS::applyCFBC (int amrlev, int mglev, MF& in,
                            bool homogeneous) const
{
    if (!m_has_cf_data[amrlev]) { return; }

    for (int idim = 0; idim < 3; ++idim)
    {
        if (m_cfmask[amrlev][mglev][idim] == nullptr) { continue; }

        // Use stored inhomogeneous coarse data only at MG level 0
        // and only when not in homogeneous (correction) mode.
        // V-cycle corrections need zero c/f BCs at all MG levels.
        bool const use_data = (mglev == 0 && !homogeneous);

        // Ghost-nodes CF boundary treatment:
        //
        // Inhomogeneous (solution) mode — fill BOTH layer-1 and layer-2
        //   from coarse-interpolated data.  The operator stencil at valid
        //   cells near CF reads these ghosts, so they must hold correct
        //   values for the residual r = rhs - A*x to be meaningful.
        //
        // Homogeneous (correction) mode — only set layer-2 to zero
        //   (frozen Dirichlet).  Layer-1 ghosts are relaxed by the
        //   smoother and must NOT be zeroed between sweeps.
        int const nghost = in[idim].nGrow();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(in[idim], TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& gbx = mfi.growntilebox(nghost);
            auto const& sol = in[idim].array(mfi);
            auto const& mask = m_cfmask[amrlev][mglev][idim]->const_array(mfi);
            if (use_data)
            {
                // Inhomogeneous: fill both layers from coarse data.
                auto const& cfdata = m_cf_data[amrlev][idim].const_array(mfi);
                ParallelFor(gbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if (mask(i,j,k) >= 1)
                    {
                        sol(i,j,k) = cfdata(i,j,k);
                    }
                });
            }
            else
            {
                // Homogeneous: only freeze layer-2.
                ParallelFor(gbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if (mask(i,j,k) == 2)
                    {
                        sol(i,j,k) = Real(0.0);
                    }
                });
            }
        }
    }
}


void MLCurlCurl_CNS::restriction (int amrlev, int cmglev, MF& crse, MF& fine) const
{
    IntVect ratio = (amrlev > 0) ? IntVect(2) : this->mg_coarsen_ratio_vec[cmglev-1];
    AMREX_ALWAYS_ASSERT(ratio == 2);

    // Restriction always operates on correction residuals, which need
    // homogeneous (zero) c/f ghost values.
    applyBC(amrlev, cmglev-1, fine, CurlCurlStateType::r,
            /*homogeneous=*/true);

    auto dinfo = getDirichletInfo(amrlev,cmglev-1);

    for (int idim = 0; idim < 3; ++idim) {
        bool need_parallel_copy = !amrex::isMFIterSafe(crse[idim], fine[idim]);
        MultiFab cfine;
        if (need_parallel_copy) {
            BoxArray const& ba = amrex::coarsen(fine[idim].boxArray(), 2);
            cfine.define(ba, fine[idim].DistributionMap(), 1, 0);
        }

        MultiFab* pcrse = (need_parallel_copy) ? &cfine : &(crse[idim]);

        auto const& crsema = pcrse->arrays();
        auto const& finema = fine[idim].const_arrays();
        ParallelFor(*pcrse, [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
        {
            mlcurlcurl_cns_restriction(idim,i,j,k,crsema[bno],finema[bno],dinfo);
        });
        Gpu::streamSynchronize();

        if (need_parallel_copy) {
            crse[idim].ParallelCopy(cfine);
        }
    }
}

void MLCurlCurl_CNS::interpolation (int amrlev, int fmglev, MF& fine,
                                const MF& crse) const
{
    IntVect ratio = (amrlev > 0) ? IntVect(2) : this->mg_coarsen_ratio_vec[fmglev];
    AMREX_ALWAYS_ASSERT(ratio == 2);

    auto dinfo = getDirichletInfo(amrlev,fmglev);

    for (int idim = 0; idim < 3; ++idim) {
        bool need_parallel_copy = !amrex::isMFIterSafe(crse[idim], fine[idim]);
        MultiFab cfine;
        MultiFab const* cmf = &(crse[idim]);
        if (need_parallel_copy) {
            BoxArray const& ba = amrex::coarsen(fine[idim].boxArray(), 2);
            cfine.define(ba, fine[idim].DistributionMap(), 1, 0);
            cfine.ParallelCopy(crse[idim]);
            cmf = &cfine;
        }
        auto const& finema = fine[idim].arrays();
        auto const& crsema = cmf->const_arrays();
        ParallelFor(fine[idim], [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
        {
            if (!dinfo.is_dirichlet_edge(idim,i,j,k)) {
                mlcurlcurl_cns_interpadd(idim,i,j,k,finema[bno],crsema[bno]);
            }
        });
        Gpu::streamSynchronize();
    }
}

void MLCurlCurl_CNS::interpolationAmr (int famrlev, MF& fine, MF const& crse,
                                  IntVect const& nghost) const
{
    AMREX_ALWAYS_ASSERT(famrlev > 0);
    AMREX_ALWAYS_ASSERT(famrlev < this->m_num_amr_levels);

    IntVect const ratio(this->AMRRefRatio(famrlev-1));
    AMREX_ALWAYS_ASSERT(ratio == IntVect(2));

    // AMR interpolation is always at MG level 0 on the fine AMR level.
    auto const dinfo = getDirichletInfo(famrlev, 0);

    for (int idim = 0; idim < 3; ++idim)
    {
        bool const need_parallel_copy = !amrex::isMFIterSafe(crse[idim], fine[idim]);

        MultiFab cfine;
        MultiFab const* cmf = &(crse[idim]);

        if (need_parallel_copy)
        {
            // Build a coarse MultiFab on the coarsened fine BoxArray/DM so that
            // MFIter matches fine[idim]. Include enough region to cover the
            // requested fine ghost interpolation region.
            BoxArray cba = fine[idim].boxArray();
            if (nghost != IntVect(0)) {
                cba.grow(nghost);
            }
            cba.coarsen(ratio);

            cfine.define(cba, fine[idim].DistributionMap(), 1, 0);
            cfine.ParallelCopy(crse[idim]);
            cmf = &cfine;
        }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(fine[idim], TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box bx = mfi.validbox();
            if (nghost != IntVect(0)) {
                bx.grow(nghost);
                bx &= fine[idim][mfi].box();
            }

            auto const& finearr = fine[idim].array(mfi);
            auto const& crsearr = cmf->const_array(mfi);

            // Zero fine before interpolation so this is an ASSIGNMENT,
            // not an accumulation.  Without this, cor[fine] from the
            // down-sweep (not swapped for the finest level) gets
            // double-counted when sol += cor follows in the up-sweep.
            ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                finearr(i,j,k) = Real(0.0);
            });

            // Use beta-weighted (operator-dependent) interpolation when
            // variable beta is available.  At coefficient discontinuities
            // the geometric (fixed-weight) prolongation causes the coarse
            // correction to overshoot into the low-beta region, leading
            // to divergence.  Beta-weighting pulls the interpolated
            // correction toward the high-beta side where the solution is
            // more strongly constrained, following MLNodeLaplacian's
            // mlndlap_interpadd_aa pattern.
            if (m_bcoefs[famrlev][0][idim] != nullptr)
            {
                auto const& betaarr =
                    m_bcoefs[famrlev][0][idim]->const_array(mfi);
                int dim = idim;
                ParallelFor(bx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if (!dinfo.is_dirichlet_edge(dim, i, j, k)) {
                        mlcurlcurl_cns_interpadd_beta(dim, i, j, k,
                                                      finearr, crsearr,
                                                      betaarr);
                    }
                });
            }
            else
            {
                int dim = idim;
                ParallelFor(bx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    if (!dinfo.is_dirichlet_edge(dim, i, j, k)) {
                        mlcurlcurl_cns_interpadd(dim, i, j, k,
                                                 finearr, crsearr);
                    }
                });
            }
        }

        Gpu::streamSynchronize();
    }
}

void
MLCurlCurl_CNS::apply (int amrlev, int mglev, MF& out, MF& in, BCMode bc_mode,
                   StateMode /*s_mode*/, const MLMGBndryT<MF>* /*bndry*/) const
{
    bool const homogeneous = (bc_mode == BCMode::Homogeneous);
    applyBC(amrlev, mglev, in, CurlCurlStateType::x, homogeneous);

    auto dinfo = getDirichletInfo(amrlev,mglev);

    // Ghost-nodes: in correction (homogeneous) mode, compute A*x on
    // layer-1 CF ghost cells so the residual there is valid for
    // restriction.  In solution (inhomogeneous) mode, the RHS ghost
    // cells are not populated, so we must NOT grow.
    bool const grow_for_cf = (amrlev > 0 && mglev == 0 && homogeneous
                              && m_cfmask[amrlev][mglev][0] != nullptr);

    if (m_has_variable_alpha)
    {
        // Variable alpha path: use raw dxinv (not scaled by sqrt(alpha))
        auto dxinv = this->m_geom[amrlev][mglev].InvCellSizeArray();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(out[0],TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box xbx = mfi.tilebox(out[0].ixType().toIntVect());
            Box ybx = mfi.tilebox(out[1].ixType().toIntVect());
            Box zbx = mfi.tilebox(out[2].ixType().toIntVect());
            if (grow_for_cf)
            {
                xbx.grow(1);
                ybx.grow(1);
                zbx.grow(1);
            }
            auto const& xout = out[0].array(mfi);
            auto const& yout = out[1].array(mfi);
            auto const& zout = out[2].array(mfi);
            auto const& xin = in[0].array(mfi);
            auto const& yin = in[1].array(mfi);
            auto const& zin = in[2].array(mfi);
            auto const& acx = m_acoefs[amrlev][mglev][0]->const_array(mfi);
            auto const& acy = m_acoefs[amrlev][mglev][1]->const_array(mfi);
            auto const& acz = m_acoefs[amrlev][mglev][2]->const_array(mfi);
            if (m_bcoefs[amrlev][mglev][0]) {
                auto const& bcx = m_bcoefs[amrlev][mglev][0]->const_array(mfi);
                auto const& bcy = m_bcoefs[amrlev][mglev][1]->const_array(mfi);
                auto const& bcz = m_bcoefs[amrlev][mglev][2]->const_array(mfi);
                amrex::ParallelFor(xbx, ybx, zbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    if (dinfo.is_dirichlet_x_edge(i,j,k)) {
                        xout(i,j,k) = Real(0.0);
                    } else {
                        mlcurlcurl_cns_adotx_x(i,j,k,xout,xin,yin,zin,
                                               bcx(i,j,k),acy,acz,dxinv);
                    }
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    if (dinfo.is_dirichlet_y_edge(i,j,k)) {
                        yout(i,j,k) = Real(0.0);
                    } else {
                        mlcurlcurl_cns_adotx_y(i,j,k,yout,xin,yin,zin,
                                               bcy(i,j,k),acx,acz,dxinv);
                    }
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    if (dinfo.is_dirichlet_z_edge(i,j,k)) {
                        zout(i,j,k) = Real(0.0);
                    } else {
                        mlcurlcurl_cns_adotx_z(i,j,k,zout,xin,yin,zin,
                                               bcz(i,j,k),acx,acy,dxinv);
                    }
                });
            } else {
                auto const b = m_beta;
                amrex::ParallelFor(xbx, ybx, zbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    if (dinfo.is_dirichlet_x_edge(i,j,k)) {
                        xout(i,j,k) = Real(0.0);
                    } else {
                        mlcurlcurl_cns_adotx_x(i,j,k,xout,xin,yin,zin,
                                               b,acy,acz,dxinv);
                    }
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    if (dinfo.is_dirichlet_y_edge(i,j,k)) {
                        yout(i,j,k) = Real(0.0);
                    } else {
                        mlcurlcurl_cns_adotx_y(i,j,k,yout,xin,yin,zin,
                                               b,acx,acz,dxinv);
                    }
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    if (dinfo.is_dirichlet_z_edge(i,j,k)) {
                        zout(i,j,k) = Real(0.0);
                    } else {
                        mlcurlcurl_cns_adotx_z(i,j,k,zout,xin,yin,zin,
                                               b,acx,acy,dxinv);
                    }
                });
            }

            // Galerkin diagonal correction at CF boundary edges (variable alpha).
            // Compensates for zeroed ghost neighbors in correction mode.
            // Ghost-nodes: only fires for layer-2 ghost neighbors (frozen).
            // Layer-1 ghosts are relaxed and do not need compensation.
#if (AMREX_SPACEDIM >= 2)
            if ((mglev > 0 || (amrlev > 0 && homogeneous)) && m_cfmask[amrlev][mglev][0] != nullptr)
            {
                auto const& exm = m_cfmask[amrlev][mglev][0]->const_array(mfi);
                auto const& eym = m_cfmask[amrlev][mglev][1]->const_array(mfi);
                Real dyy = dxinv[1]*dxinv[1];
                Real dxx = dxinv[0]*dxinv[0];
#if (AMREX_SPACEDIM == 3)
                auto const& ezm = m_cfmask[amrlev][mglev][2]->const_array(mfi);
                Real dzz = dxinv[2]*dxinv[2];
#endif
                // Ex: nodal in y (and z in 3D)
                // y-coupling through curl_z → acz; z-coupling through curl_y → acy
                ParallelFor(xbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    if (dinfo.is_dirichlet_x_edge(i,j,k)) { return; }
                    Real extra = Real(0.0);
                    if (exm(i,j+1,k) == 2) { extra += acz(i,j,k) * dyy; }
                    if (exm(i,j-1,k) == 2) { extra += acz(i,j-1,k) * dyy; }
#if (AMREX_SPACEDIM == 3)
                    if (exm(i,j,k+1) == 2) { extra += acy(i,j,k) * dzz; }
                    if (exm(i,j,k-1) == 2) { extra += acy(i,j,k-1) * dzz; }
#endif
                    xout(i,j,k) += extra * xin(i,j,k);
                });
                // Ey: nodal in x (and z in 3D)
                // x-coupling through curl_z → acz; z-coupling through curl_x → acx
                ParallelFor(ybx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    if (dinfo.is_dirichlet_y_edge(i,j,k)) { return; }
                    Real extra = Real(0.0);
                    if (eym(i+1,j,k) == 2) { extra += acz(i,j,k) * dxx; }
                    if (eym(i-1,j,k) == 2) { extra += acz(i-1,j,k) * dxx; }
#if (AMREX_SPACEDIM == 3)
                    if (eym(i,j,k+1) == 2) { extra += acx(i,j,k) * dzz; }
                    if (eym(i,j,k-1) == 2) { extra += acx(i,j,k-1) * dzz; }
#endif
                    yout(i,j,k) += extra * yin(i,j,k);
                });
#if (AMREX_SPACEDIM == 3)
                // Ez: nodal in x,y
                // x-coupling through curl_y → acy; y-coupling through curl_x → acx
                ParallelFor(zbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    if (dinfo.is_dirichlet_z_edge(i,j,k)) { return; }
                    Real extra = Real(0.0);
                    if (ezm(i+1,j,k) == 2) { extra += acy(i,j,k) * dxx; }
                    if (ezm(i-1,j,k) == 2) { extra += acy(i-1,j,k) * dxx; }
                    if (ezm(i,j+1,k) == 2) { extra += acx(i,j,k) * dyy; }
                    if (ezm(i,j-1,k) == 2) { extra += acx(i,j-1,k) * dyy; }
                    zout(i,j,k) += extra * zin(i,j,k);
                });
#endif
            }
#endif

        }
    }
    else
    {
        // Scalar alpha path: bake alpha into adxinv
        auto adxinv = this->m_geom[amrlev][mglev].InvCellSizeArray();
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            adxinv[idim] *= std::sqrt(m_alpha);
        }
        auto const b = m_beta;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(out[0],TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box xbx = mfi.tilebox(out[0].ixType().toIntVect());
            Box ybx = mfi.tilebox(out[1].ixType().toIntVect());
            Box zbx = mfi.tilebox(out[2].ixType().toIntVect());
            if (grow_for_cf)
            {
                xbx.grow(1);
                ybx.grow(1);
                zbx.grow(1);
            }
            auto const& xout = out[0].array(mfi);
            auto const& yout = out[1].array(mfi);
            auto const& zout = out[2].array(mfi);
            auto const& xin = in[0].array(mfi);
            auto const& yin = in[1].array(mfi);
            auto const& zin = in[2].array(mfi);
            if (m_bcoefs[amrlev][mglev][0]) {
                auto const& bcx = m_bcoefs[amrlev][mglev][0]->const_array(mfi);
                auto const& bcy = m_bcoefs[amrlev][mglev][1]->const_array(mfi);
                auto const& bcz = m_bcoefs[amrlev][mglev][2]->const_array(mfi);
                amrex::ParallelFor(xbx, ybx, zbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    if (dinfo.is_dirichlet_x_edge(i,j,k)) {
                        xout(i,j,k) = Real(0.0);
                    } else {
                        mlcurlcurl_cns_adotx_x(i,j,k,xout,xin,yin,zin,bcx(i,j,k),adxinv);
                    }
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    if (dinfo.is_dirichlet_y_edge(i,j,k)) {
                        yout(i,j,k) = Real(0.0);
                    } else {
                        mlcurlcurl_cns_adotx_y(i,j,k,yout,xin,yin,zin,bcy(i,j,k),adxinv);
                    }
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    if (dinfo.is_dirichlet_z_edge(i,j,k)) {
                        zout(i,j,k) = Real(0.0);
                    } else {
                        mlcurlcurl_cns_adotx_z(i,j,k,zout,xin,yin,zin,bcz(i,j,k),adxinv);
                    }
                });
            } else {
                amrex::ParallelFor(xbx, ybx, zbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    if (dinfo.is_dirichlet_x_edge(i,j,k)) {
                        xout(i,j,k) = Real(0.0);
                    } else {
                        mlcurlcurl_cns_adotx_x(i,j,k,xout,xin,yin,zin,b,adxinv);
                    }
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    if (dinfo.is_dirichlet_y_edge(i,j,k)) {
                        yout(i,j,k) = Real(0.0);
                    } else {
                        mlcurlcurl_cns_adotx_y(i,j,k,yout,xin,yin,zin,b,adxinv);
                    }
                },
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    if (dinfo.is_dirichlet_z_edge(i,j,k)) {
                        zout(i,j,k) = Real(0.0);
                    } else {
                        mlcurlcurl_cns_adotx_z(i,j,k,zout,xin,yin,zin,b,adxinv);
                    }
                });
            }

            // Galerkin diagonal correction at CF boundary edges.
            // Ghost-nodes: only for layer-2 ghost neighbors (frozen).
#if (AMREX_SPACEDIM >= 2)
            if ((mglev > 0 || (amrlev > 0 && homogeneous)) && m_cfmask[amrlev][mglev][0] != nullptr)
            {
                auto const& exm = m_cfmask[amrlev][mglev][0]->const_array(mfi);
                auto const& eym = m_cfmask[amrlev][mglev][1]->const_array(mfi);
                Real dyy = adxinv[1]*adxinv[1];
                Real dxx = adxinv[0]*adxinv[0];
#if (AMREX_SPACEDIM == 3)
                auto const& ezm = m_cfmask[amrlev][mglev][2]->const_array(mfi);
                Real dzz = adxinv[2]*adxinv[2];
#endif
                ParallelFor(xbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    if (dinfo.is_dirichlet_x_edge(i,j,k)) { return; }
                    Real extra = Real(0.0);
                    if (exm(i,j+1,k) == 2) { extra += dyy; }
                    if (exm(i,j-1,k) == 2) { extra += dyy; }
#if (AMREX_SPACEDIM == 3)
                    if (exm(i,j,k+1) == 2) { extra += dzz; }
                    if (exm(i,j,k-1) == 2) { extra += dzz; }
#endif
                    xout(i,j,k) += extra * xin(i,j,k);
                });
                ParallelFor(ybx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    if (dinfo.is_dirichlet_y_edge(i,j,k)) { return; }
                    Real extra = Real(0.0);
                    if (eym(i+1,j,k) == 2) { extra += dxx; }
                    if (eym(i-1,j,k) == 2) { extra += dxx; }
#if (AMREX_SPACEDIM == 3)
                    if (eym(i,j,k+1) == 2) { extra += dzz; }
                    if (eym(i,j,k-1) == 2) { extra += dzz; }
#endif
                    yout(i,j,k) += extra * yin(i,j,k);
                });
#if (AMREX_SPACEDIM == 3)
                ParallelFor(zbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    if (dinfo.is_dirichlet_z_edge(i,j,k)) { return; }
                    Real extra = Real(0.0);
                    if (ezm(i+1,j,k) == 2) { extra += dxx; }
                    if (ezm(i-1,j,k) == 2) { extra += dxx; }
                    if (ezm(i,j+1,k) == 2) { extra += dyy; }
                    if (ezm(i,j-1,k) == 2) { extra += dyy; }
                    zout(i,j,k) += extra * zin(i,j,k);
                });
#endif
            }
#endif
        }
    }

}

void MLCurlCurl_CNS::smooth (int amrlev, int mglev, MF& sol, const MF& rhs,
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
                // Smoothing always operates on corrections, so use
                // homogeneous (zero) c/f boundary conditions.
                applyBC(amrlev, mglev, sol, CurlCurlStateType::x,
                        /*homogeneous=*/true);
            }
            skip_fillboundary = false;
            // c/f ghost cells of sol are restored by applyBC above.
            // The smoother handles constraints at c/f boundary nodes
            // via constraint elimination in the coupled LU solve.
#if (AMREX_SPACEDIM == 1)
            smooth1D(amrlev, mglev, sol, rhs, color);
#else
            smooth4(amrlev, mglev, sol, rhs, color);
#endif
        }
    }

}

#if (AMREX_SPACEDIM == 1)
void MLCurlCurl_CNS::smooth1D (int amrlev, int mglev, MF& sol, MF const& rhs,
                           int color) const
{
    auto const& ex = sol[0].arrays();
    auto const& ey = sol[1].arrays();
    auto const& ez = sol[2].arrays();
    auto const& rhsx = rhs[0].const_arrays();
    auto const& rhsy = rhs[1].const_arrays();
    auto const& rhsz = rhs[2].const_arrays();

    auto dinfo = getDirichletInfo(amrlev,mglev);

    int xhi = this->m_geom[amrlev][mglev].Domain().bigEnd(0);

    MultiFab nmf(amrex::convert(rhs[0].boxArray(),IntVect(1)),
                 rhs[0].DistributionMap(), 1, 0, MFInfo().SetAlloc(false));

    if (m_has_variable_alpha) {
        auto dxinv = this->m_geom[amrlev][mglev].InvCellSizeArray();
        auto const& acy = m_acoefs[amrlev][mglev][1]->const_arrays();
        auto const& acz = m_acoefs[amrlev][mglev][2]->const_arrays();
        if (m_bcoefs[amrlev][mglev][0]) {
            auto const& bcx = m_bcoefs[amrlev][mglev][0]->const_arrays();
            auto const& bcy = m_bcoefs[amrlev][mglev][1]->const_arrays();
            auto const& bcz = m_bcoefs[amrlev][mglev][2]->const_arrays();
            ParallelFor( nmf, [=] AMREX_GPU_DEVICE(int bno, int i, int j, int k)
            {
                bool valid_x = i <= xhi;
                mlcurlcurl_cns_1D(i,j,k,ex[bno],ey[bno],ez[bno],
                                  rhsx[bno],rhsy[bno],rhsz[bno],
                                  bcx[bno],bcy[bno],bcz[bno],
                                  acy[bno],acz[bno],
                                  dxinv,color,dinfo,valid_x);
            });
        } else {
            auto b = m_beta;
            ParallelFor( nmf, [=] AMREX_GPU_DEVICE(int bno, int i, int j, int k)
            {
                bool valid_x = i <= xhi;
                mlcurlcurl_cns_1D(i,j,k,ex[bno],ey[bno],ez[bno],
                                  rhsx[bno],rhsy[bno],rhsz[bno],
                                  b,acy[bno],acz[bno],
                                  dxinv,color,dinfo,valid_x);
            });
        }
    } else {
        auto adxinv = this->m_geom[amrlev][mglev].InvCellSizeArray();
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            adxinv[idim] *= std::sqrt(m_alpha);
        }
        auto b = m_beta;
        if (m_bcoefs[amrlev][mglev][0]) {
            auto const& bcx = m_bcoefs[amrlev][mglev][0]->const_arrays();
            auto const& bcy = m_bcoefs[amrlev][mglev][1]->const_arrays();
            auto const& bcz = m_bcoefs[amrlev][mglev][2]->const_arrays();
            ParallelFor( nmf, [=] AMREX_GPU_DEVICE(int bno, int i, int j, int k)
            {
                bool valid_x = i <= xhi;
                mlcurlcurl_1D(i,j,k,ex[bno],ey[bno],ez[bno],
                              rhsx[bno],rhsy[bno],rhsz[bno],
                              bcx[bno],bcy[bno],bcz[bno],
                              adxinv,color,dinfo,valid_x);
            });
        } else {
            ParallelFor( nmf, [=] AMREX_GPU_DEVICE(int bno, int i, int j, int k)
            {
                bool valid_x = i <= xhi;
                mlcurlcurl_1D(i,j,k,ex[bno],ey[bno],ez[bno],
                              rhsx[bno],rhsy[bno],rhsz[bno],
                              b,adxinv,color,dinfo,valid_x);
            });
        }
    }
    Gpu::streamSynchronize();
}
#endif

#if (AMREX_SPACEDIM > 1)
void MLCurlCurl_CNS::smooth4 (int amrlev, int mglev, MF& sol, MF const& rhs,
                          int color) const
{
    auto const& ex = sol[0].arrays();
    auto const& ey = sol[1].arrays();
    auto const& ez = sol[2].arrays();
    auto const& rhsx = rhs[0].const_arrays();
    auto const& rhsy = rhs[1].const_arrays();
    auto const& rhsz = rhs[2].const_arrays();

    auto dinfo = getDirichletInfo(amrlev,mglev);
    auto sinfo = getSymmetryInfo(amrlev,mglev);

    // Ghost-nodes: grow iteration box to include layer-1 CF ghosts
    // so they are relaxed by the smoother.
    bool const grow_for_cf = (amrlev > 0 && mglev == 0
                              && m_cfmask[amrlev][mglev][0] != nullptr);
    BoxArray smooth_ba = rhs[0].boxArray();
    if (grow_for_cf) { smooth_ba.grow(1); }
    MultiFab nmf(amrex::convert(smooth_ba, IntVect(1)),
                 rhs[0].DistributionMap(), 1, 0, MFInfo().SetAlloc(false));

    if (m_has_variable_alpha)
    {
        // Variable alpha path: raw dxinv, variable alpha arrays
        auto dxinv = this->m_geom[amrlev][mglev].InvCellSizeArray();
        AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_bcoefs[amrlev][mglev][0] != nullptr,
            "MLCurlCurl_CNS: variable alpha requires variable beta (setBeta)");
        auto const& acx = m_acoefs[amrlev][mglev][0]->const_arrays();
        auto const& acy = m_acoefs[amrlev][mglev][1]->const_arrays();
        auto const& acz = m_acoefs[amrlev][mglev][2]->const_arrays();
        auto const& bcx = m_bcoefs[amrlev][mglev][0]->const_arrays();
        auto const& bcy = m_bcoefs[amrlev][mglev][1]->const_arrays();
        auto const& bcz = m_bcoefs[amrlev][mglev][2]->const_arrays();

        if (m_has_cf_data[amrlev] && m_cfmask[amrlev][mglev][0] != nullptr)
        {
            // Variable alpha with coarse/fine boundary constraint elimination
            auto const& exm = m_cfmask[amrlev][mglev][0]->const_arrays();
            auto const& eym = m_cfmask[amrlev][mglev][1]->const_arrays();
#if (AMREX_SPACEDIM == 3)
            auto const& ezm = m_cfmask[amrlev][mglev][2]->const_arrays();
#endif
            int const mg = mglev;   // for Galerkin correction guard
            int const alev = amrlev; // for AMR CF boundary guard
            ParallelFor(nmf, [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
            {
#if (AMREX_SPACEDIM == 2)
                if (dinfo.is_dirichlet_node(i,j,k)) { return; }

                int my_color = i%2 + 2*(j%2);

                Real dxx = dxinv[0] * dxinv[0];
                Real dyy = dxinv[1] * dxinv[1];

                // --- Ez SOR sweep with variable alpha ---
                if (((my_color == 0 || my_color == 3) && (color == 0 || color == 3)) ||
                    ((my_color == 1 || my_color == 2) && (color == 1 || color == 2)))
                {
                    Real ay_hi = acy[bno](i  ,j,k);
                    Real ay_lo = acy[bno](i-1,j,k);
                    Real ax_hi = acx[bno](i,j  ,k);
                    Real ax_lo = acx[bno](i,j-1,k);
                    Real gamma = (ay_hi + ay_lo) * dxx + (ax_hi + ax_lo) * dyy
                               + bcz[bno](i,j,k);
                    Real ccez = - ay_hi * dxx * ez[bno](i+1,j,k)
                                - ay_lo * dxx * ez[bno](i-1,j,k)
                                - ax_hi * dyy * ez[bno](i,j+1,k)
                                - ax_lo * dyy * ez[bno](i,j-1,k);

                    Real res = rhsz[bno](i,j,k) - (gamma*ez[bno](i,j,k) + ccez);
                    constexpr Real omega = Real(1.15);
                    ez[bno](i,j,k) += omega/gamma * res;
                }

                if (my_color != color) { return; }

                // Ghost-nodes: only layer-2 (mask==2) edges are frozen.
                // Layer-1 (mask==1) edges are relaxed normally.
                bool is_ghost[4] = {exm[bno](i-1,j,k) == 2,
                                    exm[bno](i  ,j,k) == 2,
                                    eym[bno](i,j-1,k) == 2,
                                    eym[bno](i,j  ,k) == 2};
                bool any_ghost = is_ghost[0] || is_ghost[1] ||
                                 is_ghost[2] || is_ghost[3];

                bool near_cf = any_ghost;
                if (mg > 0 || alev > 0) {
                    near_cf = near_cf
                        || (exm[bno](i-1,j+1,k)==2) || (exm[bno](i-1,j-1,k)==2)
                        || (exm[bno](i  ,j+1,k)==2) || (exm[bno](i  ,j-1,k)==2)
                        || (eym[bno](i+1,j-1,k)==2) || (eym[bno](i-1,j-1,k)==2)
                        || (eym[bno](i+1,j  ,k)==2) || (eym[bno](i-1,j  ,k)==2);
                }

                if (!near_cf)
                {
                    // No c/f ghosts — use standard variable alpha kernel
                    mlcurlcurl_cns_gs4_varalpha<false>(i,j,k,
                        ex[bno],ey[bno],ez[bno],
                        rhsx[bno],rhsy[bno],rhsz[bno],
                        dxinv,color,bcx[bno],bcy[bno],bcz[bno],
                        acx[bno],acy[bno],acz[bno],
                        dinfo,sinfo);
                    return;
                }

                // --- Coupled LU solve with constraint elimination ---
                Real dxy = dxinv[0]*dxinv[1];

                // Local alpha values at the 4 cells surrounding node (i,j)
                Real az00 = acz[bno](i-1,j-1,k);
                Real az10 = acz[bno](i  ,j-1,k);
                Real az01 = acz[bno](i-1,j  ,k);
                Real az11 = acz[bno](i  ,j  ,k);

                // Build b vector with variable alpha
                GpuArray<Real,4> bv
                    {rhsx[bno](i-1,j,k) - (- az01*dyy * ex[bno](i-1,j+1,k)
                                            - az00*dyy * ex[bno](i-1,j-1,k)
                                            - az01*dxy * ey[bno](i-1,j  ,k)
                                            + az00*dxy * ey[bno](i-1,j-1,k)),
                     rhsx[bno](i  ,j,k) - (- az11*dyy * ex[bno](i  ,j+1,k)
                                            - az10*dyy * ex[bno](i  ,j-1,k)
                                            + az11*dxy * ey[bno](i+1,j  ,k)
                                            - az10*dxy * ey[bno](i+1,j-1,k)),
                     rhsy[bno](i,j-1,k) - (- az10*dxx * ey[bno](i+1,j-1,k)
                                            - az00*dxx * ey[bno](i-1,j-1,k)
                                            - az10*dxy * ex[bno](i  ,j-1,k)
                                            + az00*dxy * ex[bno](i-1,j-1,k)),
                     rhsy[bno](i,j  ,k) - (- az11*dxx * ey[bno](i+1,j  ,k)
                                            - az01*dxx * ey[bno](i-1,j  ,k)
                                            + az11*dxy * ex[bno](i  ,j+1,k)
                                            - az01*dxy * ex[bno](i-1,j+1,k))};

                // Build beta values (handle symmetry)
                GpuArray<Real,4> beta;
                if (sinfo.xlo_is_symmetric(i)) {
                    bv[0] = -bv[1];
                    beta[0] = beta[1] = bcx[bno](i,j,k);
                } else if (sinfo.xhi_is_symmetric(i)) {
                    bv[1] = -bv[0];
                    beta[0] = beta[1] = bcx[bno](i-1,j,k);
                } else {
                    beta[0] = bcx[bno](i-1,j,k);
                    beta[1] = bcx[bno](i  ,j,k);
                }

                if (sinfo.ylo_is_symmetric(j)) {
                    bv[2] = -bv[3];
                    beta[2] = beta[3] = bcy[bno](i,j,k);
                } else if (sinfo.yhi_is_symmetric(j)) {
                    bv[3] = -bv[2];
                    beta[2] = beta[3] = bcy[bno](i,j-1,k);
                } else {
                    beta[2] = bcy[bno](i,j-1,k);
                    beta[3] = bcy[bno](i,j  ,k);
                }

                // Build 4x4 matrix with variable alpha
                Array2D<Real,0,3,0,3,Order::C> A;
                A(0,0) = (az01+az00)*dyy+beta[0]; A(0,1) = Real(0);
                A(0,2) = -az00*dxy;                A(0,3) = az01*dxy;
                A(1,0) = Real(0);                  A(1,1) = (az11+az10)*dyy+beta[1];
                A(1,2) = az10*dxy;                 A(1,3) = -az11*dxy;
                A(2,0) = -az00*dxy;                A(2,1) = az10*dxy;
                A(2,2) = (az10+az00)*dxx+beta[2];  A(2,3) = Real(0);
                A(3,0) = az01*dxy;                 A(3,1) = -az11*dxy;
                A(3,2) = Real(0);                  A(3,3) = (az11+az01)*dxx+beta[3];

                // Galerkin diagonal correction: only for layer-2 ghost neighbors.
                if (mg > 0 || alev > 0)
                {
                    if (exm[bno](i-1,j+1,k)==2) { A(0,0) += az01*dyy; }
                    if (exm[bno](i-1,j-1,k)==2) { A(0,0) += az00*dyy; }
                    if (exm[bno](i  ,j+1,k)==2) { A(1,1) += az11*dyy; }
                    if (exm[bno](i  ,j-1,k)==2) { A(1,1) += az10*dyy; }
                    if (eym[bno](i+1,j-1,k)==2) { A(2,2) += az10*dxx; }
                    if (eym[bno](i-1,j-1,k)==2) { A(2,2) += az00*dxx; }
                    if (eym[bno](i+1,j  ,k)==2) { A(3,3) += az11*dxx; }
                    if (eym[bno](i-1,j  ,k)==2) { A(3,3) += az01*dxx; }
                }

                Real gval[4] = {ex[bno](i-1,j,k), ex[bno](i,j,k),
                                ey[bno](i,j-1,k), ey[bno](i,j,k)};

                // Constraint elimination: freeze layer-2 ghost edges.
                // Layer-1 edges are relaxed, so only freeze when the
                // edge itself is a layer-2 ghost or its stencil neighbor
                // is a layer-2 ghost.
                bool is_frozen[4] = {is_ghost[0], is_ghost[1],
                                     is_ghost[2], is_ghost[3]};
                if (alev > 0 && mg == 0)
                {
                    is_frozen[0] = is_frozen[0] || exm[bno](i-1,j+1,k)==2 || exm[bno](i-1,j-1,k)==2;
                    is_frozen[1] = is_frozen[1] || exm[bno](i  ,j+1,k)==2 || exm[bno](i  ,j-1,k)==2;
                    is_frozen[2] = is_frozen[2] || eym[bno](i+1,j-1,k)==2 || eym[bno](i-1,j-1,k)==2;
                    is_frozen[3] = is_frozen[3] || eym[bno](i+1,j  ,k)==2 || eym[bno](i-1,j  ,k)==2;
                }
                for (int m = 0; m < 4; ++m)
                {
                    if (!is_frozen[m]) { continue; }
                    for (int n = 0; n < 4; ++n)
                    {
                        if (n != m) {
                            bv[n] -= A(n,m) * gval[m];
                            A(n,m) = Real(0);
                        }
                        A(m,n) = Real(0);
                    }
                    A(m,m) = Real(1);
                    bv[m] = gval[m];
                }

                // Solve modified system
                LUSolver<4,Real> lusolver(A);
                lusolver(beta.data(), bv.data());

                ex[bno](i-1,j  ,k  ) = beta[0];
                ex[bno](i  ,j  ,k  ) = beta[1];
                ey[bno](i  ,j-1,k  ) = beta[2];
                ey[bno](i  ,j  ,k  ) = beta[3];
#elif (AMREX_SPACEDIM == 3)
                if (dinfo.is_dirichlet_node(i,j,k)) { return; }

                int my_color = i%2 + 2*(j%2);
                if (k%2 != 0) { my_color = 3 - my_color; }
                if (my_color != color) { return; }

                Real dxx = dxinv[0]*dxinv[0];
                Real dyy = dxinv[1]*dxinv[1];
                Real dzz = dxinv[2]*dxinv[2];
                Real dxy = dxinv[0]*dxinv[1];
                Real dxz = dxinv[0]*dxinv[2];
                Real dyz = dxinv[1]*dxinv[2];

                // Ghost-nodes: only layer-2 (mask==2) edges are frozen.
                bool is_ghost[6] = {exm[bno](i-1,j,k) == 2,
                                    exm[bno](i  ,j,k) == 2,
                                    eym[bno](i,j-1,k) == 2,
                                    eym[bno](i,j  ,k) == 2,
                                    ezm[bno](i,j,k-1) == 2,
                                    ezm[bno](i,j,k  ) == 2};
                bool any_ghost = is_ghost[0] || is_ghost[1] || is_ghost[2] ||
                                 is_ghost[3] || is_ghost[4] || is_ghost[5];

                bool near_cf = any_ghost;
                if (mg > 0 || alev > 0) {
                    near_cf = near_cf
                        || (exm[bno](i-1,j+1,k)==2) || (exm[bno](i-1,j-1,k)==2)
                        || (exm[bno](i-1,j,k+1)==2) || (exm[bno](i-1,j,k-1)==2)
                        || (exm[bno](i  ,j+1,k)==2) || (exm[bno](i  ,j-1,k)==2)
                        || (exm[bno](i  ,j,k+1)==2) || (exm[bno](i  ,j,k-1)==2)
                        || (eym[bno](i+1,j-1,k)==2) || (eym[bno](i-1,j-1,k)==2)
                        || (eym[bno](i,j-1,k+1)==2) || (eym[bno](i,j-1,k-1)==2)
                        || (eym[bno](i+1,j  ,k)==2) || (eym[bno](i-1,j  ,k)==2)
                        || (eym[bno](i,j  ,k+1)==2) || (eym[bno](i,j  ,k-1)==2)
                        || (ezm[bno](i+1,j,k-1)==2) || (ezm[bno](i-1,j,k-1)==2)
                        || (ezm[bno](i,j+1,k-1)==2) || (ezm[bno](i,j-1,k-1)==2)
                        || (ezm[bno](i+1,j,k  )==2) || (ezm[bno](i-1,j,k  )==2)
                        || (ezm[bno](i,j+1,k  )==2) || (ezm[bno](i,j-1,k  )==2);
                }

                if (!near_cf)
                {
                    mlcurlcurl_cns_gs4_varalpha<false>(i,j,k,
                        ex[bno],ey[bno],ez[bno],
                        rhsx[bno],rhsy[bno],rhsz[bno],
                        dxinv,color,bcx[bno],bcy[bno],bcz[bno],
                        acx[bno],acy[bno],acz[bno],
                        dinfo,sinfo);
                    return;
                }

                // Alpha values at surrounding cells
                Real az00 = acz[bno](i-1,j-1,k);
                Real az10 = acz[bno](i  ,j-1,k);
                Real az01 = acz[bno](i-1,j  ,k);
                Real az11 = acz[bno](i  ,j  ,k);
                Real ay00 = acy[bno](i-1,j,k-1);
                Real ay10 = acy[bno](i  ,j,k-1);
                Real ay01 = acy[bno](i-1,j,k  );
                Real ay11 = acy[bno](i  ,j,k  );
                Real ax00 = acx[bno](i,j-1,k-1);
                Real ax10 = acx[bno](i,j  ,k-1);
                Real ax01 = acx[bno](i,j-1,k  );
                Real ax11 = acx[bno](i,j  ,k  );

                // b vector
                GpuArray<Real,6> bv
                    {rhsx[bno](i-1,j,k) - (- az01*dyy * ex[bno](i-1,j+1,k)
                                            - az00*dyy * ex[bno](i-1,j-1,k)
                                            - ay01*dzz * ex[bno](i-1,j,k+1)
                                            - ay00*dzz * ex[bno](i-1,j,k-1)
                                            - az01*dxy * ey[bno](i-1,j  ,k)
                                            + az00*dxy * ey[bno](i-1,j-1,k)
                                            - ay01*dxz * ez[bno](i-1,j,k  )
                                            + ay00*dxz * ez[bno](i-1,j,k-1)),
                     rhsx[bno](i  ,j,k) - (- az11*dyy * ex[bno](i  ,j+1,k)
                                            - az10*dyy * ex[bno](i  ,j-1,k)
                                            - ay11*dzz * ex[bno](i  ,j,k+1)
                                            - ay10*dzz * ex[bno](i  ,j,k-1)
                                            + az11*dxy * ey[bno](i+1,j  ,k)
                                            - az10*dxy * ey[bno](i+1,j-1,k)
                                            + ay11*dxz * ez[bno](i+1,j,k  )
                                            - ay10*dxz * ez[bno](i+1,j,k-1)),
                     rhsy[bno](i,j-1,k) - (- az10*dxx * ey[bno](i+1,j-1,k)
                                            - az00*dxx * ey[bno](i-1,j-1,k)
                                            - ax01*dzz * ey[bno](i,j-1,k+1)
                                            - ax00*dzz * ey[bno](i,j-1,k-1)
                                            - az10*dxy * ex[bno](i  ,j-1,k)
                                            + az00*dxy * ex[bno](i-1,j-1,k)
                                            - ax01*dyz * ez[bno](i,j-1,k  )
                                            + ax00*dyz * ez[bno](i,j-1,k-1)),
                     rhsy[bno](i,j  ,k) - (- az11*dxx * ey[bno](i+1,j  ,k)
                                            - az01*dxx * ey[bno](i-1,j  ,k)
                                            - ax11*dzz * ey[bno](i,j  ,k+1)
                                            - ax10*dzz * ey[bno](i,j  ,k-1)
                                            + az11*dxy * ex[bno](i  ,j+1,k)
                                            - az01*dxy * ex[bno](i-1,j+1,k)
                                            + ax11*dyz * ez[bno](i,j+1,k  )
                                            - ax10*dyz * ez[bno](i,j+1,k-1)),
                     rhsz[bno](i,j,k-1) - (- ay10*dxx * ez[bno](i+1,j,k-1)
                                            - ay00*dxx * ez[bno](i-1,j,k-1)
                                            - ax10*dyy * ez[bno](i,j+1,k-1)
                                            - ax00*dyy * ez[bno](i,j-1,k-1)
                                            - ay10*dxz * ex[bno](i  ,j,k-1)
                                            + ay00*dxz * ex[bno](i-1,j,k-1)
                                            - ax10*dyz * ey[bno](i,j  ,k-1)
                                            + ax00*dyz * ey[bno](i,j-1,k-1)),
                     rhsz[bno](i,j,k  ) - (- ay11*dxx * ez[bno](i+1,j,k  )
                                            - ay01*dxx * ez[bno](i-1,j,k  )
                                            - ax11*dyy * ez[bno](i,j+1,k  )
                                            - ax01*dyy * ez[bno](i,j-1,k  )
                                            + ay11*dxz * ex[bno](i  ,j,k+1)
                                            - ay01*dxz * ex[bno](i-1,j,k+1)
                                            + ax11*dyz * ey[bno](i,j  ,k+1)
                                            - ax01*dyz * ey[bno](i,j-1,k+1))};

                // Beta values with symmetry
                GpuArray<Real,6> beta;
                if (sinfo.xlo_is_symmetric(i)) {
                    bv[0] = -bv[1];
                    beta[0] = beta[1] = bcx[bno](i,j,k);
                } else if (sinfo.xhi_is_symmetric(i)) {
                    bv[1] = -bv[0];
                    beta[0] = beta[1] = bcx[bno](i-1,j,k);
                } else {
                    beta[0] = bcx[bno](i-1,j,k);
                    beta[1] = bcx[bno](i  ,j,k);
                }
                if (sinfo.ylo_is_symmetric(j)) {
                    bv[2] = -bv[3];
                    beta[2] = beta[3] = bcy[bno](i,j,k);
                } else if (sinfo.yhi_is_symmetric(j)) {
                    bv[3] = -bv[2];
                    beta[2] = beta[3] = bcy[bno](i,j-1,k);
                } else {
                    beta[2] = bcy[bno](i,j-1,k);
                    beta[3] = bcy[bno](i,j  ,k);
                }
                if (sinfo.zlo_is_symmetric(k)) {
                    bv[4] = -bv[5];
                    beta[4] = beta[5] = bcz[bno](i,j,k);
                } else if (sinfo.zhi_is_symmetric(k)) {
                    bv[5] = -bv[4];
                    beta[4] = beta[5] = bcz[bno](i,j,k-1);
                } else {
                    beta[4] = bcz[bno](i,j,k-1);
                    beta[5] = bcz[bno](i,j,k  );
                }

                // 6x6 matrix with variable alpha
                Array2D<Real,0,5,0,5,Order::C> A;
                A(0,0) = (az01+az00)*dyy+(ay01+ay00)*dzz+beta[0];
                A(0,1) = Real(0);    A(0,2) = -az00*dxy; A(0,3) = az01*dxy;
                A(0,4) = -ay00*dxz;  A(0,5) = ay01*dxz;
                A(1,0) = Real(0);
                A(1,1) = (az11+az10)*dyy+(ay11+ay10)*dzz+beta[1];
                A(1,2) = az10*dxy;   A(1,3) = -az11*dxy;
                A(1,4) = ay10*dxz;   A(1,5) = -ay11*dxz;
                A(2,0) = -az00*dxy;  A(2,1) = az10*dxy;
                A(2,2) = (az10+az00)*dxx+(ax01+ax00)*dzz+beta[2];
                A(2,3) = Real(0);    A(2,4) = -ax00*dyz; A(2,5) = ax01*dyz;
                A(3,0) = az01*dxy;   A(3,1) = -az11*dxy; A(3,2) = Real(0);
                A(3,3) = (az11+az01)*dxx+(ax11+ax10)*dzz+beta[3];
                A(3,4) = ax10*dyz;   A(3,5) = -ax11*dyz;
                A(4,0) = -ay00*dxz;  A(4,1) = ay10*dxz;
                A(4,2) = -ax00*dyz;  A(4,3) = ax10*dyz;
                A(4,4) = (ay10+ay00)*dxx+(ax10+ax00)*dyy+beta[4];
                A(4,5) = Real(0);
                A(5,0) = ay01*dxz;   A(5,1) = -ay11*dxz;
                A(5,2) = ax01*dyz;   A(5,3) = -ax11*dyz; A(5,4) = Real(0);
                A(5,5) = (ay11+ay01)*dxx+(ax11+ax01)*dyy+beta[5];

                // Galerkin diagonal correction: only for layer-2 ghost neighbors.
                if (mg > 0 || alev > 0)
                {
                    if (exm[bno](i-1,j+1,k)==2) { A(0,0) += az01*dyy; }
                    if (exm[bno](i-1,j-1,k)==2) { A(0,0) += az00*dyy; }
                    if (exm[bno](i-1,j,k+1)==2) { A(0,0) += ay01*dzz; }
                    if (exm[bno](i-1,j,k-1)==2) { A(0,0) += ay00*dzz; }
                    if (exm[bno](i  ,j+1,k)==2) { A(1,1) += az11*dyy; }
                    if (exm[bno](i  ,j-1,k)==2) { A(1,1) += az10*dyy; }
                    if (exm[bno](i  ,j,k+1)==2) { A(1,1) += ay11*dzz; }
                    if (exm[bno](i  ,j,k-1)==2) { A(1,1) += ay10*dzz; }
                    if (eym[bno](i+1,j-1,k)==2) { A(2,2) += az10*dxx; }
                    if (eym[bno](i-1,j-1,k)==2) { A(2,2) += az00*dxx; }
                    if (eym[bno](i,j-1,k+1)==2) { A(2,2) += ax01*dzz; }
                    if (eym[bno](i,j-1,k-1)==2) { A(2,2) += ax00*dzz; }
                    if (eym[bno](i+1,j  ,k)==2) { A(3,3) += az11*dxx; }
                    if (eym[bno](i-1,j  ,k)==2) { A(3,3) += az01*dxx; }
                    if (eym[bno](i,j  ,k+1)==2) { A(3,3) += ax11*dzz; }
                    if (eym[bno](i,j  ,k-1)==2) { A(3,3) += ax10*dzz; }
                    if (ezm[bno](i+1,j,k-1)==2) { A(4,4) += ay10*dxx; }
                    if (ezm[bno](i-1,j,k-1)==2) { A(4,4) += ay00*dxx; }
                    if (ezm[bno](i,j+1,k-1)==2) { A(4,4) += ax10*dyy; }
                    if (ezm[bno](i,j-1,k-1)==2) { A(4,4) += ax00*dyy; }
                    if (ezm[bno](i+1,j,k  )==2) { A(5,5) += ay11*dxx; }
                    if (ezm[bno](i-1,j,k  )==2) { A(5,5) += ay01*dxx; }
                    if (ezm[bno](i,j+1,k  )==2) { A(5,5) += ax11*dyy; }
                    if (ezm[bno](i,j-1,k  )==2) { A(5,5) += ax01*dyy; }
                }

                Real gval[6] = {ex[bno](i-1,j,k), ex[bno](i,j,k),
                                ey[bno](i,j-1,k), ey[bno](i,j,k),
                                ez[bno](i,j,k-1), ez[bno](i,j,k)};

                // Constraint elimination: freeze layer-2 ghost edges.
                bool is_frozen[6] = {is_ghost[0], is_ghost[1],
                                     is_ghost[2], is_ghost[3],
                                     is_ghost[4], is_ghost[5]};
                if (alev > 0 && mg == 0)
                {
                    is_frozen[0] = is_frozen[0] || exm[bno](i-1,j+1,k)==2 || exm[bno](i-1,j-1,k)==2
                                                 || exm[bno](i-1,j,k+1)==2 || exm[bno](i-1,j,k-1)==2;
                    is_frozen[1] = is_frozen[1] || exm[bno](i  ,j+1,k)==2 || exm[bno](i  ,j-1,k)==2
                                                 || exm[bno](i  ,j,k+1)==2 || exm[bno](i  ,j,k-1)==2;
                    is_frozen[2] = is_frozen[2] || eym[bno](i+1,j-1,k)==2 || eym[bno](i-1,j-1,k)==2
                                                 || eym[bno](i,j-1,k+1)==2 || eym[bno](i,j-1,k-1)==2;
                    is_frozen[3] = is_frozen[3] || eym[bno](i+1,j  ,k)==2 || eym[bno](i-1,j  ,k)==2
                                                 || eym[bno](i,j  ,k+1)==2 || eym[bno](i,j  ,k-1)==2;
                    is_frozen[4] = is_frozen[4] || ezm[bno](i+1,j,k-1)==2 || ezm[bno](i-1,j,k-1)==2
                                                 || ezm[bno](i,j+1,k-1)==2 || ezm[bno](i,j-1,k-1)==2;
                    is_frozen[5] = is_frozen[5] || ezm[bno](i+1,j,k  )==2 || ezm[bno](i-1,j,k  )==2
                                                 || ezm[bno](i,j+1,k  )==2 || ezm[bno](i,j-1,k  )==2;
                }
                for (int m = 0; m < 6; ++m)
                {
                    if (!is_frozen[m]) { continue; }
                    for (int n = 0; n < 6; ++n)
                    {
                        if (n != m) {
                            bv[n] -= A(n,m) * gval[m];
                            A(n,m) = Real(0);
                        }
                        A(m,n) = Real(0);
                    }
                    A(m,m) = Real(1);
                    bv[m] = gval[m];
                }

                LUSolver<6,Real> lusolver(A);
                lusolver(beta.data(), bv.data());

                ex[bno](i-1,j  ,k  ) = beta[0];
                ex[bno](i  ,j  ,k  ) = beta[1];
                ey[bno](i  ,j-1,k  ) = beta[2];
                ey[bno](i  ,j  ,k  ) = beta[3];
                ez[bno](i  ,j  ,k-1) = beta[4];
                ez[bno](i  ,j  ,k  ) = beta[5];
#endif
            });
        }
        else if (m_use_pcg) {
            ParallelFor(nmf, [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
            {
                mlcurlcurl_cns_gs4_varalpha<true>(i,j,k,ex[bno],ey[bno],ez[bno],
                                     rhsx[bno],rhsy[bno],rhsz[bno],
                                     dxinv,color,bcx[bno],bcy[bno],bcz[bno],
                                     acx[bno],acy[bno],acz[bno],
                                     dinfo,sinfo);
            });
        } else {
            ParallelFor(nmf, [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
            {
                mlcurlcurl_cns_gs4_varalpha<false>(i,j,k,ex[bno],ey[bno],ez[bno],
                                      rhsx[bno],rhsy[bno],rhsz[bno],
                                      dxinv,color,bcx[bno],bcy[bno],bcz[bno],
                                      acx[bno],acy[bno],acz[bno],
                                      dinfo,sinfo);
            });
        }
    }
    else
    {
        // Scalar alpha path
        auto adxinv = this->m_geom[amrlev][mglev].InvCellSizeArray();
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim) {
            adxinv[idim] *= std::sqrt(m_alpha);
        }
#if (AMREX_SPACEDIM == 2)
        auto b = m_beta;
#endif

    if (m_lusolver[amrlev][mglev]) {
        auto* plusolver = m_lusolver[amrlev][mglev]->dataPtr();
        ParallelFor(nmf, [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
        {
            mlcurlcurl_cns_gs4_lu(i,j,k,ex[bno],ey[bno],ez[bno],
                              rhsx[bno],rhsy[bno],rhsz[bno],
#if (AMREX_SPACEDIM == 2)
                              b,
#endif
                              adxinv,color,*plusolver,dinfo,sinfo);
        });
    } else {
        auto const& bcx = m_bcoefs[amrlev][mglev][0]->const_arrays();
        auto const& bcy = m_bcoefs[amrlev][mglev][1]->const_arrays();
        auto const& bcz = m_bcoefs[amrlev][mglev][2]->const_arrays();
        if (m_use_pcg) {
            ParallelFor(nmf, [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
            {
                mlcurlcurl_cns_gs4<true>(i,j,k,ex[bno],ey[bno],ez[bno],
                                     rhsx[bno],rhsy[bno],rhsz[bno],
                                     adxinv,color,bcx[bno],bcy[bno],bcz[bno],
                                     dinfo,sinfo);
            });
        } else if (m_has_cf_data[amrlev] && m_cfmask[amrlev][mglev][0] != nullptr) {
            // Variable-beta path with c/f boundary constraint elimination.
            // At boundary nodes where one or more edges are c/f ghost cells,
            // the coupled LU solve is modified to enforce the ghost values
            // as constraints, ensuring valid edges get correct corrections.
            auto const& exm = m_cfmask[amrlev][mglev][0]->const_arrays();
            auto const& eym = m_cfmask[amrlev][mglev][1]->const_arrays();
#if (AMREX_SPACEDIM == 3)
            auto const& ezm = m_cfmask[amrlev][mglev][2]->const_arrays();
#endif
            int const mg = mglev;   // for Galerkin correction guard
            int const alev = amrlev; // for AMR CF boundary guard
            ParallelFor(nmf, [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
            {
#if (AMREX_SPACEDIM == 2)
                if (dinfo.is_dirichlet_node(i,j,k)) { return; }

                int my_color = i%2 + 2*(j%2);

                Real dxx = adxinv[0] * adxinv[0];
                Real dyy = adxinv[1] * adxinv[1];

                // --- Ez GS sweep (same as standard, unaffected by c/f) ---
                if (((my_color == 0 || my_color == 3) && (color == 0 || color == 3)) ||
                    ((my_color == 1 || my_color == 2) && (color == 1 || color == 2)))
                {
                    Real gamma = (dxx+dyy)*Real(2.0) + bcz[bno](i,j,k);
                    Real ccez = - dxx * (ez[bno](i-1,j  ,k  ) +
                                         ez[bno](i+1,j  ,k  ))
                                - dyy * (ez[bno](i  ,j-1,k  ) +
                                         ez[bno](i  ,j+1,k  ));
                    Real res = rhsz[bno](i,j,k) - (gamma*ez[bno](i,j,k) + ccez);
                    constexpr Real omega = Real(1.15);
                    ez[bno](i,j,k) += omega/gamma * res;
                }

                if (my_color != color) { return; }

                // Check if any edge at this node is a c/f ghost
                bool is_ghost[4] = {exm[bno](i-1,j,k) != 0,
                                    exm[bno](i  ,j,k) != 0,
                                    eym[bno](i,j-1,k) != 0,
                                    eym[bno](i,j  ,k) != 0};
                bool any_ghost = is_ghost[0] || is_ghost[1] ||
                                 is_ghost[2] || is_ghost[3];

                // Check if any edge NEIGHBOR is a CF ghost (for
                // Galerkin diagonal correction) in addition to
                // the edge itself being a ghost (for constraint elimination).
                // At mglev>0, also enter the CF path for edges with
                // CF ghost NEIGHBORS (for Galerkin diagonal correction).
                // At mglev=0 the physical operator is correct as-is.
                bool near_cf = any_ghost;
                if (mg > 0 || alev > 0) {
                    near_cf = near_cf
                        || (exm[bno](i-1,j+1,k)==1) || (exm[bno](i-1,j-1,k)==1)
                        || (exm[bno](i  ,j+1,k)==1) || (exm[bno](i  ,j-1,k)==1)
                        || (eym[bno](i+1,j-1,k)==1) || (eym[bno](i-1,j-1,k)==1)
                        || (eym[bno](i+1,j  ,k)==1) || (eym[bno](i-1,j  ,k)==1);
                }

                if (!near_cf)
                {
                    // Far from c/f boundary — use standard kernel
                    mlcurlcurl_cns_gs4<false>(i,j,k,ex[bno],ey[bno],ez[bno],
                                          rhsx[bno],rhsy[bno],rhsz[bno],
                                          adxinv,color,bcx[bno],bcy[bno],bcz[bno],
                                          dinfo,sinfo);
                    return;
                }

                // --- Coupled LU solve with constraint elimination ---
                Real dxy = adxinv[0]*adxinv[1];

                // Build b vector
                GpuArray<Real,4> bv
                    {rhsx[bno](i-1,j,k) - (-dyy * ( ex[bno](i-1,j-1,k  ) +
                                                     ex[bno](i-1,j+1,k  ))
                                           + dxy * ( ey[bno](i-1,j-1,k  )
                                                    -ey[bno](i-1,j  ,k  ))),
                     rhsx[bno](i  ,j,k) - (-dyy * ( ex[bno](i  ,j-1,k  ) +
                                                     ex[bno](i  ,j+1,k  ))
                                           + dxy * (-ey[bno](i+1,j-1,k  )
                                                    +ey[bno](i+1,j  ,k  ))),
                     rhsy[bno](i,j-1,k) - (-dxx * ( ey[bno](i-1,j-1,k  ) +
                                                     ey[bno](i+1,j-1,k  ))
                                           + dxy * ( ex[bno](i-1,j-1,k  )
                                                    -ex[bno](i  ,j-1,k  ))),
                     rhsy[bno](i,j  ,k) - (-dxx * ( ey[bno](i-1,j  ,k  ) +
                                                     ey[bno](i+1,j  ,k  ))
                                           + dxy * (-ex[bno](i-1,j+1,k  )
                                                    +ex[bno](i  ,j+1,k  )))};

                // Build beta values (handle symmetry)
                GpuArray<Real,4> beta;
                if (sinfo.xlo_is_symmetric(i)) {
                    bv[0] = -bv[1];
                    beta[0] = beta[1] = bcx[bno](i,j,k);
                } else if (sinfo.xhi_is_symmetric(i)) {
                    bv[1] = -bv[0];
                    beta[0] = beta[1] = bcx[bno](i-1,j,k);
                } else {
                    beta[0] = bcx[bno](i-1,j,k);
                    beta[1] = bcx[bno](i  ,j,k);
                }
                if (sinfo.ylo_is_symmetric(j)) {
                    bv[2] = -bv[3];
                    beta[2] = beta[3] = bcy[bno](i,j,k);
                } else if (sinfo.yhi_is_symmetric(j)) {
                    bv[3] = -bv[2];
                    beta[2] = beta[3] = bcy[bno](i,j-1,k);
                } else {
                    beta[2] = bcy[bno](i,j-1,k);
                    beta[3] = bcy[bno](i,j  ,k);
                }

                // Build 4x4 matrix
                Array2D<Real,0,3,0,3,Order::C> A;
                A(0,0) = dyy*Real(2.0)+beta[0]; A(0,1) = Real(0); A(0,2) = -dxy;    A(0,3) = dxy;
                A(1,0) = Real(0); A(1,1) = dyy*Real(2.0)+beta[1]; A(1,2) = dxy;     A(1,3) = -dxy;
                A(2,0) = -dxy;    A(2,1) = dxy;     A(2,2) = dxx*Real(2.0)+beta[2]; A(2,3) = Real(0);
                A(3,0) = dxy;     A(3,1) = -dxy;    A(3,2) = Real(0); A(3,3) = dxx*Real(2.0)+beta[3];

                // Galerkin diagonal correction at MG CF boundary (mglev>0)
                // and AMR CF boundary (amrlev>0, mglev=0).
                if (mg > 0 || alev > 0)
                {
                    if (exm[bno](i-1,j+1,k)==1) { A(0,0) += dyy; }
                    if (exm[bno](i-1,j-1,k)==1) { A(0,0) += dyy; }
                    if (exm[bno](i  ,j+1,k)==1) { A(1,1) += dyy; }
                    if (exm[bno](i  ,j-1,k)==1) { A(1,1) += dyy; }
                    if (eym[bno](i+1,j-1,k)==1) { A(2,2) += dxx; }
                    if (eym[bno](i-1,j-1,k)==1) { A(2,2) += dxx; }
                    if (eym[bno](i+1,j  ,k)==1) { A(3,3) += dxx; }
                    if (eym[bno](i-1,j  ,k)==1) { A(3,3) += dxx; }
                }

                Real gval[4] = {ex[bno](i-1,j,k), ex[bno](i,j,k),
                                ey[bno](i,j-1,k), ey[bno](i,j,k)};

                // Constraint elimination: freeze ghost + CF-boundary edges.
                bool is_frozen[4] = {is_ghost[0], is_ghost[1],
                                     is_ghost[2], is_ghost[3]};
                if (alev > 0 && mg == 0)
                {
                    is_frozen[0] = is_frozen[0] || exm[bno](i-1,j+1,k)==1 || exm[bno](i-1,j-1,k)==1;
                    is_frozen[1] = is_frozen[1] || exm[bno](i  ,j+1,k)==1 || exm[bno](i  ,j-1,k)==1;
                    is_frozen[2] = is_frozen[2] || eym[bno](i+1,j-1,k)==1 || eym[bno](i-1,j-1,k)==1;
                    is_frozen[3] = is_frozen[3] || eym[bno](i+1,j  ,k)==1 || eym[bno](i-1,j  ,k)==1;
                }
                for (int m = 0; m < 4; ++m)
                {
                    if (!is_frozen[m]) { continue; }
                    for (int n = 0; n < 4; ++n)
                    {
                        if (n != m) {
                            bv[n] -= A(n,m) * gval[m];
                            A(n,m) = Real(0);
                        }
                        A(m,n) = Real(0);
                    }
                    A(m,m) = Real(1);
                    bv[m] = gval[m];
                }

                // Solve modified system
                LUSolver<4,Real> lusolver(A);
                lusolver(beta.data(), bv.data());

                ex[bno](i-1,j  ,k  ) = beta[0];
                ex[bno](i  ,j  ,k  ) = beta[1];
                ey[bno](i  ,j-1,k  ) = beta[2];
                ey[bno](i  ,j  ,k  ) = beta[3];
#elif (AMREX_SPACEDIM == 3)
                if (dinfo.is_dirichlet_node(i,j,k)) { return; }

                int my_color = i%2 + 2*(j%2);
                if (k%2 != 0) { my_color = 3 - my_color; }
                if (my_color != color) { return; }

                Real dxx = adxinv[0]*adxinv[0];
                Real dyy = adxinv[1]*adxinv[1];
                Real dzz = adxinv[2]*adxinv[2];
                Real dxy = adxinv[0]*adxinv[1];
                Real dxz = adxinv[0]*adxinv[2];
                Real dyz = adxinv[1]*adxinv[2];

                // Check if any of the 6 edges at this node is a c/f ghost
                bool is_ghost[6] = {exm[bno](i-1,j,k) != 0,
                                    exm[bno](i  ,j,k) != 0,
                                    eym[bno](i,j-1,k) != 0,
                                    eym[bno](i,j  ,k) != 0,
                                    ezm[bno](i,j,k-1) != 0,
                                    ezm[bno](i,j,k  ) != 0};
                bool any_ghost = is_ghost[0] || is_ghost[1] || is_ghost[2] ||
                                 is_ghost[3] || is_ghost[4] || is_ghost[5];

                bool near_cf = any_ghost;
                if (mg > 0) {
                    near_cf = near_cf
                        || (exm[bno](i-1,j+1,k)==1) || (exm[bno](i-1,j-1,k)==1)
                        || (exm[bno](i-1,j,k+1)==1) || (exm[bno](i-1,j,k-1)==1)
                        || (exm[bno](i  ,j+1,k)==1) || (exm[bno](i  ,j-1,k)==1)
                        || (exm[bno](i  ,j,k+1)==1) || (exm[bno](i  ,j,k-1)==1)
                        || (eym[bno](i+1,j-1,k)==1) || (eym[bno](i-1,j-1,k)==1)
                        || (eym[bno](i,j-1,k+1)==1) || (eym[bno](i,j-1,k-1)==1)
                        || (eym[bno](i+1,j  ,k)==1) || (eym[bno](i-1,j  ,k)==1)
                        || (eym[bno](i,j  ,k+1)==1) || (eym[bno](i,j  ,k-1)==1)
                        || (ezm[bno](i+1,j,k-1)==1) || (ezm[bno](i-1,j,k-1)==1)
                        || (ezm[bno](i,j+1,k-1)==1) || (ezm[bno](i,j-1,k-1)==1)
                        || (ezm[bno](i+1,j,k  )==1) || (ezm[bno](i-1,j,k  )==1)
                        || (ezm[bno](i,j+1,k  )==1) || (ezm[bno](i,j-1,k  )==1);
                }

                if (!near_cf)
                {
                    mlcurlcurl_cns_gs4<false>(i,j,k,ex[bno],ey[bno],ez[bno],
                                          rhsx[bno],rhsy[bno],rhsz[bno],
                                          adxinv,color,bcx[bno],bcy[bno],bcz[bno],
                                          dinfo,sinfo);
                    return;
                }

                // b vector (constant alpha baked into adxinv)
                GpuArray<Real,6> bv
                    {rhsx[bno](i-1,j,k) - (-dyy * ( ex[bno](i-1,j-1,k  ) +
                                                      ex[bno](i-1,j+1,k  ))
                                            -dzz * ( ex[bno](i-1,j  ,k+1) +
                                                      ex[bno](i-1,j  ,k-1))
                                            +dxy * ( ey[bno](i-1,j-1,k  )
                                                     -ey[bno](i-1,j  ,k  ))
                                            +dxz * ( ez[bno](i-1,j  ,k-1)
                                                     -ez[bno](i-1,j  ,k  ))),
                     rhsx[bno](i  ,j,k) - (-dyy * ( ex[bno](i  ,j-1,k  ) +
                                                      ex[bno](i  ,j+1,k  ))
                                            -dzz * ( ex[bno](i  ,j  ,k+1) +
                                                      ex[bno](i  ,j  ,k-1))
                                            +dxy * (-ey[bno](i+1,j-1,k  )
                                                     +ey[bno](i+1,j  ,k  ))
                                            +dxz * (-ez[bno](i+1,j  ,k-1)
                                                     +ez[bno](i+1,j  ,k  ))),
                     rhsy[bno](i,j-1,k) - (-dxx * ( ey[bno](i-1,j-1,k  ) +
                                                      ey[bno](i+1,j-1,k  ))
                                            -dzz * ( ey[bno](i  ,j-1,k-1) +
                                                      ey[bno](i  ,j-1,k+1))
                                            +dxy * ( ex[bno](i-1,j-1,k  )
                                                     -ex[bno](i  ,j-1,k  ))
                                            +dyz * ( ez[bno](i  ,j-1,k-1)
                                                     -ez[bno](i  ,j-1,k  ))),
                     rhsy[bno](i,j  ,k) - (-dxx * ( ey[bno](i-1,j  ,k  ) +
                                                      ey[bno](i+1,j  ,k  ))
                                            -dzz * ( ey[bno](i  ,j  ,k-1) +
                                                      ey[bno](i  ,j  ,k+1))
                                            +dxy * (-ex[bno](i-1,j+1,k  )
                                                     +ex[bno](i  ,j+1,k  ))
                                            +dyz * (-ez[bno](i  ,j+1,k-1)
                                                     +ez[bno](i  ,j+1,k  ))),
                     rhsz[bno](i,j,k-1) - (-dxx * ( ez[bno](i-1,j  ,k-1) +
                                                      ez[bno](i+1,j  ,k-1))
                                            -dyy * ( ez[bno](i  ,j-1,k-1) +
                                                      ez[bno](i  ,j+1,k-1))
                                            +dxz * ( ex[bno](i-1,j  ,k-1)
                                                     -ex[bno](i  ,j  ,k-1))
                                            +dyz * ( ey[bno](i  ,j-1,k-1)
                                                     -ey[bno](i  ,j  ,k-1))),
                     rhsz[bno](i,j,k  ) - (-dxx * ( ez[bno](i-1,j  ,k  ) +
                                                      ez[bno](i+1,j  ,k  ))
                                            -dyy * ( ez[bno](i  ,j-1,k  ) +
                                                      ez[bno](i  ,j+1,k  ))
                                            +dxz * (-ex[bno](i-1,j  ,k+1)
                                                     +ex[bno](i  ,j  ,k+1))
                                            +dyz * (-ey[bno](i  ,j-1,k+1)
                                                     +ey[bno](i  ,j  ,k+1)))};

                // Beta values with symmetry
                GpuArray<Real,6> beta;
                if (sinfo.xlo_is_symmetric(i)) {
                    bv[0] = -bv[1];
                    beta[0] = beta[1] = bcx[bno](i,j,k);
                } else if (sinfo.xhi_is_symmetric(i)) {
                    bv[1] = -bv[0];
                    beta[0] = beta[1] = bcx[bno](i-1,j,k);
                } else {
                    beta[0] = bcx[bno](i-1,j,k);
                    beta[1] = bcx[bno](i  ,j,k);
                }
                if (sinfo.ylo_is_symmetric(j)) {
                    bv[2] = -bv[3];
                    beta[2] = beta[3] = bcy[bno](i,j,k);
                } else if (sinfo.yhi_is_symmetric(j)) {
                    bv[3] = -bv[2];
                    beta[2] = beta[3] = bcy[bno](i,j-1,k);
                } else {
                    beta[2] = bcy[bno](i,j-1,k);
                    beta[3] = bcy[bno](i,j  ,k);
                }
                if (sinfo.zlo_is_symmetric(k)) {
                    bv[4] = -bv[5];
                    beta[4] = beta[5] = bcz[bno](i,j,k);
                } else if (sinfo.zhi_is_symmetric(k)) {
                    bv[5] = -bv[4];
                    beta[4] = beta[5] = bcz[bno](i,j,k-1);
                } else {
                    beta[4] = bcz[bno](i,j,k-1);
                    beta[5] = bcz[bno](i,j,k  );
                }

                // 6x6 matrix (constant alpha baked into adxinv)
                Array2D<Real,0,5,0,5,Order::C> A;
                A(0,0) = (dyy+dzz)*Real(2.0)+beta[0];
                A(0,1) = Real(0);    A(0,2) = -dxy;  A(0,3) = dxy;
                A(0,4) = -dxz;       A(0,5) = dxz;
                A(1,0) = Real(0);
                A(1,1) = (dyy+dzz)*Real(2.0)+beta[1];
                A(1,2) = dxy;        A(1,3) = -dxy;
                A(1,4) = dxz;        A(1,5) = -dxz;
                A(2,0) = -dxy;       A(2,1) = dxy;
                A(2,2) = (dxx+dzz)*Real(2.0)+beta[2];
                A(2,3) = Real(0);    A(2,4) = -dyz;  A(2,5) = dyz;
                A(3,0) = dxy;        A(3,1) = -dxy;  A(3,2) = Real(0);
                A(3,3) = (dxx+dzz)*Real(2.0)+beta[3];
                A(3,4) = dyz;        A(3,5) = -dyz;
                A(4,0) = -dxz;       A(4,1) = dxz;
                A(4,2) = -dyz;       A(4,3) = dyz;
                A(4,4) = (dxx+dyy)*Real(2.0)+beta[4];
                A(4,5) = Real(0);
                A(5,0) = dxz;        A(5,1) = -dxz;
                A(5,2) = dyz;        A(5,3) = -dyz;  A(5,4) = Real(0);
                A(5,5) = (dxx+dyy)*Real(2.0)+beta[5];

                // Galerkin diagonal correction at mglev>0
                if (mg > 0)
                {
                    if (exm[bno](i-1,j+1,k)==1) { A(0,0) += dyy; }
                    if (exm[bno](i-1,j-1,k)==1) { A(0,0) += dyy; }
                    if (exm[bno](i-1,j,k+1)==1) { A(0,0) += dzz; }
                    if (exm[bno](i-1,j,k-1)==1) { A(0,0) += dzz; }
                    if (exm[bno](i  ,j+1,k)==1) { A(1,1) += dyy; }
                    if (exm[bno](i  ,j-1,k)==1) { A(1,1) += dyy; }
                    if (exm[bno](i  ,j,k+1)==1) { A(1,1) += dzz; }
                    if (exm[bno](i  ,j,k-1)==1) { A(1,1) += dzz; }
                    if (eym[bno](i+1,j-1,k)==1) { A(2,2) += dxx; }
                    if (eym[bno](i-1,j-1,k)==1) { A(2,2) += dxx; }
                    if (eym[bno](i,j-1,k+1)==1) { A(2,2) += dzz; }
                    if (eym[bno](i,j-1,k-1)==1) { A(2,2) += dzz; }
                    if (eym[bno](i+1,j  ,k)==1) { A(3,3) += dxx; }
                    if (eym[bno](i-1,j  ,k)==1) { A(3,3) += dxx; }
                    if (eym[bno](i,j  ,k+1)==1) { A(3,3) += dzz; }
                    if (eym[bno](i,j  ,k-1)==1) { A(3,3) += dzz; }
                    if (ezm[bno](i+1,j,k-1)==1) { A(4,4) += dxx; }
                    if (ezm[bno](i-1,j,k-1)==1) { A(4,4) += dxx; }
                    if (ezm[bno](i,j+1,k-1)==1) { A(4,4) += dyy; }
                    if (ezm[bno](i,j-1,k-1)==1) { A(4,4) += dyy; }
                    if (ezm[bno](i+1,j,k  )==1) { A(5,5) += dxx; }
                    if (ezm[bno](i-1,j,k  )==1) { A(5,5) += dxx; }
                    if (ezm[bno](i,j+1,k  )==1) { A(5,5) += dyy; }
                    if (ezm[bno](i,j-1,k  )==1) { A(5,5) += dyy; }
                }

                Real gval[6] = {ex[bno](i-1,j,k), ex[bno](i,j,k),
                                ey[bno](i,j-1,k), ey[bno](i,j,k),
                                ez[bno](i,j,k-1), ez[bno](i,j,k)};

                // Constraint elimination: freeze ghost + CF-boundary edges.
                bool is_frozen[6] = {is_ghost[0], is_ghost[1],
                                     is_ghost[2], is_ghost[3],
                                     is_ghost[4], is_ghost[5]};
                if (alev > 0 && mg == 0)
                {
                    is_frozen[0] = is_frozen[0] || exm[bno](i-1,j+1,k)==1 || exm[bno](i-1,j-1,k)==1
                                                 || exm[bno](i-1,j,k+1)==1 || exm[bno](i-1,j,k-1)==1;
                    is_frozen[1] = is_frozen[1] || exm[bno](i  ,j+1,k)==1 || exm[bno](i  ,j-1,k)==1
                                                 || exm[bno](i  ,j,k+1)==1 || exm[bno](i  ,j,k-1)==1;
                    is_frozen[2] = is_frozen[2] || eym[bno](i+1,j-1,k)==1 || eym[bno](i-1,j-1,k)==1
                                                 || eym[bno](i,j-1,k+1)==1 || eym[bno](i,j-1,k-1)==1;
                    is_frozen[3] = is_frozen[3] || eym[bno](i+1,j  ,k)==1 || eym[bno](i-1,j  ,k)==1
                                                 || eym[bno](i,j  ,k+1)==1 || eym[bno](i,j  ,k-1)==1;
                    is_frozen[4] = is_frozen[4] || ezm[bno](i+1,j,k-1)==1 || ezm[bno](i-1,j,k-1)==1
                                                 || ezm[bno](i,j+1,k-1)==1 || ezm[bno](i,j-1,k-1)==1;
                    is_frozen[5] = is_frozen[5] || ezm[bno](i+1,j,k  )==1 || ezm[bno](i-1,j,k  )==1
                                                 || ezm[bno](i,j+1,k  )==1 || ezm[bno](i,j-1,k  )==1;
                }
                for (int m = 0; m < 6; ++m)
                {
                    if (!is_frozen[m]) { continue; }
                    for (int n = 0; n < 6; ++n)
                    {
                        if (n != m) {
                            bv[n] -= A(n,m) * gval[m];
                            A(n,m) = Real(0);
                        }
                        A(m,n) = Real(0);
                    }
                    A(m,m) = Real(1);
                    bv[m] = gval[m];
                }

                LUSolver<6,Real> lusolver(A);
                lusolver(beta.data(), bv.data());

                ex[bno](i-1,j  ,k  ) = beta[0];
                ex[bno](i  ,j  ,k  ) = beta[1];
                ey[bno](i  ,j-1,k  ) = beta[2];
                ey[bno](i  ,j  ,k  ) = beta[3];
                ez[bno](i  ,j  ,k-1) = beta[4];
                ez[bno](i  ,j  ,k  ) = beta[5];
#endif
            });
        } else {
            ParallelFor(nmf, [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
            {
                mlcurlcurl_cns_gs4<false>(i,j,k,ex[bno],ey[bno],ez[bno],
                                      rhsx[bno],rhsy[bno],rhsz[bno],
                                      adxinv,color,bcx[bno],bcy[bno],bcz[bno],
                                      dinfo,sinfo);
            });
        }
    }
    } // end else (scalar alpha)
    Gpu::streamSynchronize();
}
#endif

void MLCurlCurl_CNS::solutionResidual (int amrlev, MF& resid, MF& x, const MF& b,
                                   const MF* crse_bcdata)
{
    BL_PROFILE("MLCurlCurl_CNS::solutionResidual()");
    if (crse_bcdata != nullptr && amrlev > 0)
    {
        updateCFData(amrlev, *crse_bcdata, IntVect(this->AMRRefRatio(amrlev-1)));
    }
    const int mglev = 0;
    apply(amrlev, mglev, resid, x, BCMode::Inhomogeneous, StateMode::Solution);
    compresid(amrlev, mglev, resid, b);

    // Ghost-nodes strategy: the layer-1 CF ghosts are relaxed by the
    // smoother, so the residual near the CF boundary is well-behaved.
    // No residual zeroing is needed.
}

void MLCurlCurl_CNS::correctionResidual (int amrlev, int mglev, MF& resid, MF& x,
                                     const MF& b, BCMode bc_mode,
                                     const MF* crse_bcdata)
{
    if (crse_bcdata != nullptr && amrlev > 0)
    {
        updateCFData(amrlev, *crse_bcdata,
                     IntVect(this->AMRRefRatio(amrlev-1)));
    }
    bool const homogeneous = (bc_mode == BCMode::Homogeneous);
    apply(amrlev, mglev, resid, x,
          homogeneous ? BCMode::Homogeneous : BCMode::Inhomogeneous,
          StateMode::Correction);
    bool const grow_cf = (homogeneous && amrlev > 0 && mglev == 0
                          && m_cfmask[amrlev][mglev][0] != nullptr);
    compresid(amrlev, mglev, resid, b, grow_cf);
}

void MLCurlCurl_CNS::compresid (int amrlev, int mglev, MF& resid, MF const& b,
                                bool grow_for_cf) const
{
    auto dinfo = getDirichletInfo(amrlev,mglev);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(resid[0],TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        Box xbx = mfi.tilebox(resid[0].ixType().toIntVect());
        Box ybx = mfi.tilebox(resid[1].ixType().toIntVect());
        Box zbx = mfi.tilebox(resid[2].ixType().toIntVect());
        if (grow_for_cf)
        {
            xbx.grow(1);
            ybx.grow(1);
            zbx.grow(1);
        }
        auto const& resx = resid[0].array(mfi);
        auto const& resy = resid[1].array(mfi);
        auto const& resz = resid[2].array(mfi);
        auto const& bx = b[0].array(mfi);
        auto const& by = b[1].array(mfi);
        auto const& bz = b[2].array(mfi);
        amrex::ParallelFor(xbx, ybx, zbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            if (dinfo.is_dirichlet_x_edge(i,j,k)) {
                resx(i,j,k) = Real(0.0);
            } else {
                resx(i,j,k) = bx(i,j,k) - resx(i,j,k);
            }
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            if (dinfo.is_dirichlet_y_edge(i,j,k)) {
                resy(i,j,k) = Real(0.0);
            } else {
                resy(i,j,k) = by(i,j,k) - resy(i,j,k);
            }
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            if (dinfo.is_dirichlet_z_edge(i,j,k)) {
                resz(i,j,k) = Real(0.0);
            } else {
                resz(i,j,k) = bz(i,j,k) - resz(i,j,k);
            }
        });
    }
}

void MLCurlCurl_CNS::update_lusolver ()
{
#if (AMREX_SPACEDIM > 1)
    // LU precompute is only valid with constant alpha and constant beta.
    if (m_bcoefs[0][0][0] == nullptr && !m_has_variable_alpha) {
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

void MLCurlCurl_CNS::prepareForSolve ()
{
    update_lusolver();
}

Real MLCurlCurl_CNS::xdoty (int amrlev, int mglev, const MF& x, const MF& y,
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

Real MLCurlCurl_CNS::normInf (int /*amrlev*/, MF const& mf, bool local) const
{
    return amrex::norminf(mf, 0, m_ncomp, IntVect(0), local);
}

void MLCurlCurl_CNS::averageDownAndSync (Vector<MF>& sol) const
{
    BL_PROFILE("MLCurlCurl_CNS::averageDownAndSync()");

    const int mglev = 0;

    // Average down from fine to coarse AMR levels.
    for (int falev = int(sol.size()) - 1; falev > 0; --falev)
    {
        IntVect ratio(this->AMRRefRatio(falev-1));
        for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
        {
            average_down_edges(sol[falev][idim], sol[falev-1][idim], ratio);
        }
#if (AMREX_SPACEDIM < 3)
        average_down_nodal(sol[falev][2], sol[falev-1][2], ratio);
#endif
#if (AMREX_SPACEDIM == 1)
        average_down_nodal(sol[falev][1], sol[falev-1][1], ratio);
#endif
    }

    // Sync each level (ensures shared nodal data is consistent).
    for (int amrlev = 0; amrlev < int(sol.size()); ++amrlev)
    {
        for (int idim = 0; idim < 3; ++idim)
        {
            amrex::OverrideSync(sol[amrlev][idim],
                                getDotMask(amrlev,mglev,idim),
                                this->m_geom[amrlev][mglev].periodicity());
        }
    }

    // FillBoundary on all levels so that ghost cells across MPI
    // boundaries are consistent with the newly averaged-down valid
    // cells.  Without this, updateCFData's ParallelCopy from sol[0]
    // can read stale ghost values at process boundaries.
    for (int amrlev = 0; amrlev < int(sol.size()); ++amrlev)
    {
        auto const& period = this->m_geom[amrlev][mglev].periodicity();
        for (int idim = 0; idim < 3; ++idim)
        {
            sol[amrlev][idim].FillBoundary(period);
        }
    }
}

//! Average down fine AMR solution and RHS to coarse level for composite solves.
/**
 * \param camrlev  Coarse AMR level index
 * \param crse_sol Coarse-level solution to be overwritten with averaged-down fine data
 * \param crse_rhs Coarse-level RHS to be overwritten with averaged-down fine data
 * \param fine_sol Fine-level solution to average down
 * \param fine_rhs Fine-level RHS to average down
 */
void MLCurlCurl_CNS::averageDownSolutionRHS (int camrlev, MF& crse_sol, MF& crse_rhs,
                                              const MF& fine_sol, const MF& fine_rhs)
{
    const IntVect ratio(this->AMRRefRatio(camrlev));

    // Average down each component using the appropriate stencil for its staggering.
    // In 3D all 3 components are edge-centred.
    // In 2D components 0 and 1 are edge-centred; component 2 is nodal.
    // In 1D component 0 is edge-centred; components 1 and 2 are nodal.
    for (int idim = 0; idim < AMREX_SPACEDIM; ++idim)
    {
        average_down_edges(fine_sol[idim], crse_sol[idim], ratio);
        average_down_edges(fine_rhs[idim], crse_rhs[idim], ratio);
    }
#if (AMREX_SPACEDIM < 3)
    average_down_nodal(fine_sol[2], crse_sol[2], ratio);
    average_down_nodal(fine_rhs[2], crse_rhs[2], ratio);
#endif
#if (AMREX_SPACEDIM == 1)
    average_down_nodal(fine_sol[1], crse_sol[1], ratio);
    average_down_nodal(fine_rhs[1], crse_rhs[1], ratio);
#endif

    // Shared nodal data at box boundaries may be inconsistent after
    // average_down.  OverrideSync resolves ownership so every process
    // holds the same value at shared nodes.
    auto const& period = this->m_geom[camrlev][0].periodicity();
    for (int idim = 0; idim < 3; ++idim)
    {
        amrex::OverrideSync(crse_sol[idim],
                            getDotMask(camrlev, 0, idim), period);
        amrex::OverrideSync(crse_rhs[idim],
                            getDotMask(camrlev, 0, idim), period);
    }

    // FillBoundary so that ghost cells across MPI boundaries are
    // consistent with the newly averaged-down valid cells.  Without
    // this, updateCFData's ParallelCopy from the coarse solution can
    // read stale ghost values at process boundaries.
    for (int idim = 0; idim < 3; ++idim)
    {
        crse_sol[idim].FillBoundary(period);
        crse_rhs[idim].FillBoundary(period);
    }
}

//! Average fine AMR residual down to coarse level for composite solves.
/**
 * For staggered operators, this is a no-op.  All inter-AMR-level residual
 * transfer is handled by reflux() instead, following the same pattern as
 * AMReX's MLNodeLinOp.  The reason is that staggered (edge/nodal) degrees
 * of freedom are shared between coarse and fine levels at the coarse-fine
 * boundary, so a simple average_down would create a discontinuity in the
 * composite residual.
 *
 * \param clev  Coarse AMR level index
 * \param cres  Coarse residual (unmodified)
 * \param fres  Fine residual (unused)
 */
void MLCurlCurl_CNS::avgDownResAmr (int clev, MF& cres, MF const& fres) const
{
    amrex::ignore_unused(clev, cres, fres);
}

//! Restrict fine residual to coarse and correct the coarse-fine boundary.
/**
 * For staggered operators (edges/nodes), avgDownResAmr is a no-op, so this
 * function handles all inter-AMR-level residual transfer.  It uses the
 * full-weighting MG restriction operator (mlcurlcurl_cns_restriction) to
 * restrict the fine residual to the coarse grid, then ParallelCopy to
 * overwrite the coarse residual in the fine-covered region.
 *
 * Before restriction, C/F ghost cells of the fine residual are filled
 * with values linearly interpolated from the coarse residual (rather
 * than set to zero).  This ensures the restriction stencil at C/F
 * boundary nodes produces a physically meaningful composite residual
 * that smoothly transitions between the fine and coarse levels.
 *
 * \param crse_amrlev Coarse AMR level index
 * \param res         Coarse residual; fine-covered region is overwritten
 * \param crse_sol    Coarse solution
 * \param crse_rhs    Coarse right-hand side
 * \param fine_res    Fine residual to restrict
 * \param fine_sol    Fine solution
 * \param fine_rhs    Fine right-hand side
 */
void MLCurlCurl_CNS::reflux (int crse_amrlev, MF& res, const MF& crse_sol,
                              const MF& crse_rhs, MF& fine_res, MF& fine_sol,
                              const MF& fine_rhs) const
{
    BL_PROFILE("MLCurlCurl_CNS::reflux()");

    amrex::ignore_unused(crse_sol, crse_rhs, fine_sol, fine_rhs);

    const int fine_amrlev = crse_amrlev + 1;
    const int mglev = 0;
    const IntVect ratio(this->AMRRefRatio(crse_amrlev));
    AMREX_ALWAYS_ASSERT(ratio == IntVect(2));

    auto const& crse_period = this->m_geom[crse_amrlev][0].periodicity();

    // Ghost-nodes strategy: the fine residual already has valid data
    // in layer-1 ghost cells (relaxed by the smoother).  Apply BCs
    // so that FillBoundary + physical BCs are up to date.
    applyBC(fine_amrlev, mglev, fine_res, CurlCurlStateType::r,
            /*homogeneous=*/true);

    // Restrict fine residual (including layer-1 ghost data) to coarse
    // and overwrite the fine-covered region of the coarse residual.
    // The restriction stencil at CF boundary coarse nodes naturally
    // picks up the relaxed layer-1 ghost values, producing the correct
    // composite residual without any syncCFResidual correction.
    auto const dinfo = getDirichletInfo(fine_amrlev, mglev);

    for (int idim = 0; idim < 3; ++idim)
    {
        BoxArray cba = fine_res[idim].boxArray();
        cba.coarsen(ratio);

        MultiFab ctmp(cba, fine_res[idim].DistributionMap(), 1, 0);

        auto const& crsema = ctmp.arrays();
        auto const& finema = fine_res[idim].const_arrays();
        ParallelFor(ctmp,
        [=] AMREX_GPU_DEVICE (int bno, int i, int j, int k)
        {
            mlcurlcurl_cns_restriction(idim, i, j, k, crsema[bno], finema[bno], dinfo);
        });
        Gpu::streamSynchronize();

        res[idim].ParallelCopy(ctmp, 0, 0, 1, IntVect(0), IntVect(0),
                               crse_period);
    }
}

void MLCurlCurl_CNS::syncCFResidual (
    int crse_amrlev, MF& res,
    const MF& crse_sol, const MF& crse_rhs,
    const MF& fine_sol,
    const MF& fine_res, const MF& fine_rhs) const
{
    BL_PROFILE("MLCurlCurl_CNS::syncCFResidual()");

    amrex::ignore_unused(fine_res, fine_rhs);

    const int fine_amrlev = crse_amrlev + 1;
    const int mglev = 0;
    const IntVect ratio(this->AMRRefRatio(crse_amrlev));
    auto const& crse_period = this->m_geom[crse_amrlev][0].periodicity();

    auto const cdxinv = this->m_geom[crse_amrlev][0].InvCellSizeArray();
    auto const fdxinv = this->m_geom[fine_amrlev][0].InvCellSizeArray();

    auto const dinfo = getDirichletInfo(crse_amrlev, mglev);

#if (AMREX_SPACEDIM == 2)

    // Process each E-field component.
    for (int idim = 0; idim < 3; ++idim)
    {
        auto const& fdm = fine_res[idim].DistributionMap();
        BoxArray cba = fine_res[idim].boxArray();
        cba.coarsen(ratio);

        // Covered mask on the coarsened-fine layout: 1 = fine-covered.
        iMultiFab covered(cba, fdm, 1, 1);
        covered.setVal(0);
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(covered, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            auto const& c = covered.array(mfi);
            Box const& vbx = mfi.validbox();
            ParallelFor(vbx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) { c(i,j,k) = 1; });
        }
        covered.FillBoundary(crse_period);

        // --- Coarse solution on coarsened-fine layout (1 ghost) -----
        MultiFab crse_sol_cf(cba, fdm, 1, 1);
        crse_sol_cf.setVal(Real(0.0));
        crse_sol_cf.ParallelCopy(crse_sol[idim], 0, 0, 1,
                                 IntVect(0), IntVect(1), crse_period);

        // Coarse RHS
        MultiFab crse_rhs_cf(cba, fdm, 1, 0);
        crse_rhs_cf.setVal(Real(0.0));
        crse_rhs_cf.ParallelCopy(crse_rhs[idim], 0, 0, 1,
                                 IntVect(0), IntVect(0), crse_period);

        // --- Coarse cross-term solution (Ex→ey, Ey→ex) on c-f layout
        int cross_comp = -1;
        if (idim == 0) { cross_comp = 1; } // Ex cross uses ey
        else if (idim == 1) { cross_comp = 0; } // Ey cross uses ex

        MultiFab crse_cross_cf;
        if (cross_comp >= 0)
        {
            BoxArray cross_cba = fine_sol[cross_comp].boxArray();
            cross_cba.coarsen(ratio);
            crse_cross_cf.define(cross_cba, fdm, 1, 1);
            crse_cross_cf.setVal(Real(0.0));
            crse_cross_cf.ParallelCopy(crse_sol[cross_comp], 0, 0, 1,
                                       IntVect(0), IntVect(1), crse_period);
        }

        // --- Coarse alpha on coarsened-fine layout ------------------
        //   Ez: ±x uses acy (idx 1), ±y uses acx (idx 0)
        //   Ex: ±y uses acz (idx 2)
        //   Ey: ±x uses acz (idx 2)
        int alpha_for_xflux = -1;
        int alpha_for_yflux = -1;
        if (idim == 2) { alpha_for_xflux = 1; alpha_for_yflux = 0; }
        else if (idim == 0) { alpha_for_yflux = 2; }
        else if (idim == 1) { alpha_for_xflux = 2; }

        BoxArray cba_cell = this->m_grids[fine_amrlev][0];
        cba_cell.coarsen(ratio);

        auto build_alpha_cf = [&](int aidx) -> std::unique_ptr<MultiFab>
        {
            BoxArray fba = amrex::convert(cba_cell, m_ftype[aidx]);
            auto p = std::make_unique<MultiFab>(fba, fdm, 1, 1);
            p->setVal(Real(0.0));
            p->ParallelCopy(*m_acoefs[crse_amrlev][0][aidx], 0, 0, 1,
                            IntVect(0), IntVect(1), crse_period);
            return p;
        };

        std::unique_ptr<MultiFab> crse_a_xflux_cf, crse_a_yflux_cf;
        if (alpha_for_xflux >= 0) { crse_a_xflux_cf = build_alpha_cf(alpha_for_xflux); }
        if (alpha_for_yflux >= 0) { crse_a_yflux_cf = build_alpha_cf(alpha_for_yflux); }

        // --- Coarse beta ----
        Real const_beta = m_beta;
        bool has_var_beta = (m_bcoefs[crse_amrlev][mglev][idim] != nullptr);
        MultiFab crse_beta_cf;
        if (has_var_beta)
        {
            crse_beta_cf.define(cba, fdm, 1, 0);
            crse_beta_cf.setVal(Real(0.0));
            crse_beta_cf.ParallelCopy(*m_bcoefs[crse_amrlev][0][idim],
                                      0, 0, 1, IntVect(0), IntVect(0),
                                      crse_period);
        }

        // --- Direct composite stencil kernel -----------------------
        MultiFab sync_corr(cba, fdm, 1, 0);
        sync_corr.setVal(Real(0.0));

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(sync_corr, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& bx = mfi.tilebox();
            auto const& corr = sync_corr.array(mfi);
            auto const& cov  = covered.const_array(mfi);
            auto const& c_sol = crse_sol_cf.const_array(mfi);
            auto const& c_rhs = crse_rhs_cf.const_array(mfi);

            // Fine solution (all 3 components, fine-level indices)
            auto const& f_ex = fine_sol[0].const_array(mfi);
            auto const& f_ey = fine_sol[1].const_array(mfi);
            auto const& f_ez = fine_sol[2].const_array(mfi);

            // Coarse cross-term solution
            Array4<Real const> c_cross;
            if (cross_comp >= 0) {
                c_cross = crse_cross_cf.const_array(mfi);
            }

            // Coarse alpha
            Array4<Real const> c_a_xflux, c_a_yflux;
            if (alpha_for_xflux >= 0) {
                c_a_xflux = crse_a_xflux_cf->const_array(mfi);
            }
            if (alpha_for_yflux >= 0) {
                c_a_yflux = crse_a_yflux_cf->const_array(mfi);
            }

            // Fine alpha (fine-level indexing)
            Array4<Real const> f_a_xflux, f_a_yflux;
            if (alpha_for_xflux >= 0) {
                f_a_xflux = m_acoefs[fine_amrlev][0][alpha_for_xflux]->const_array(mfi);
            }
            if (alpha_for_yflux >= 0) {
                f_a_yflux = m_acoefs[fine_amrlev][0][alpha_for_yflux]->const_array(mfi);
            }

            // Beta
            Array4<Real const> c_beta_arr;
            if (has_var_beta) { c_beta_arr = crse_beta_cf.const_array(mfi); }

            auto const cdxi = cdxinv;
            auto const fdxi = fdxinv;
            int dim = idim;
            Real c_beta = const_beta;
            bool var_beta = has_var_beta;

            ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int I, int J, int K)
            {
                if (cov(I,J,K) != 1) { return; }
                bool on_cf = cov(I-1,J,K)==0 || cov(I+1,J,K)==0
                          || cov(I,J-1,K)==0 || cov(I,J+1,K)==0;
                if (!on_cf) { return; }
                if (dinfo.is_dirichlet_edge(dim,I,J,K)) { return; }

                int ii = 2*I, jj = 2*J;
                Real CDXX = cdxi[0]*cdxi[0];
                Real CDYY = cdxi[1]*cdxi[1];
                Real FDXX = fdxi[0]*fdxi[0];
                Real FDYY = fdxi[1]*fdxi[1];

                Real A_comp = Real(0.0);

                // -------------------------------------------------
                // Ez (nodal x, nodal y): coincident fine node (ii,jj)
                //   ±x flux uses alphay (idx 1, stag (0,1))
                //   ±y flux uses alphax (idx 0, stag (1,0))
                //   No cross-terms in 2D.
                //
                //   Covered directions use fine data but the fine
                //   stencil at fine spacing overestimates the coarse
                //   composite flux by 2× (fine Δ ≈ coarse Δ / 2,
                //   fine dx² = coarse dx² / 4, ratio = 4×0.5 = 2).
                //   The Real(0.5) factor corrects this.
                // -------------------------------------------------
                if (dim == 2)
                {
                    // +x
                    if (cov(I+1,J,K) == 1) {
                        A_comp += Real(0.5) * f_a_xflux(ii,jj,K) * FDXX
                                * (f_ez(ii,jj,K) - f_ez(ii+1,jj,K));
                    } else {
                        A_comp += c_a_xflux(I,J,K) * CDXX
                                * (c_sol(I,J,K) - c_sol(I+1,J,K));
                    }
                    // -x
                    if (cov(I-1,J,K) == 1) {
                        A_comp += Real(0.5) * f_a_xflux(ii-1,jj,K) * FDXX
                                * (f_ez(ii,jj,K) - f_ez(ii-1,jj,K));
                    } else {
                        A_comp += c_a_xflux(I-1,J,K) * CDXX
                                * (c_sol(I,J,K) - c_sol(I-1,J,K));
                    }
                    // +y
                    if (cov(I,J+1,K) == 1) {
                        A_comp += Real(0.5) * f_a_yflux(ii,jj,K) * FDYY
                                * (f_ez(ii,jj,K) - f_ez(ii,jj+1,K));
                    } else {
                        A_comp += c_a_yflux(I,J,K) * CDYY
                                * (c_sol(I,J,K) - c_sol(I,J+1,K));
                    }
                    // -y
                    if (cov(I,J-1,K) == 1) {
                        A_comp += Real(0.5) * f_a_yflux(ii,jj-1,K) * FDYY
                                * (f_ez(ii,jj,K) - f_ez(ii,jj-1,K));
                    } else {
                        A_comp += c_a_yflux(I,J-1,K) * CDYY
                                * (c_sol(I,J,K) - c_sol(I,J-1,K));
                    }
                    // beta
                    Real bv = var_beta ? c_beta_arr(I,J,K) : c_beta;
                    A_comp += bv * c_sol(I,J,K);
                }
                // -------------------------------------------------
                // Ex (cell x, nodal y): average 2 fine cells (ii,jj)
                //   and (ii+1,jj).
                //   ±y flux + cross uses alphaz (idx 2, stag (0,0))
                //   Cross-term involves ey.
                //
                //   The 2-cell average with fine-spacing stencil
                //   overestimates by 2×.  Factor changes from 0.5
                //   to 0.25 to correct.
                // -------------------------------------------------
                else if (dim == 0)
                {
                    Real CDXY = cdxi[0]*cdxi[1];
                    Real FDXY = fdxi[0]*fdxi[1];
                    // +y
                    if (cov(I,J+1,K) == 1) {
                        Real flux = Real(0.0);
                        for (int s = 0; s < 2; ++s) {
                            int fi = ii + s;
                            flux += f_a_yflux(fi,jj,K) * FDYY
                                    * (f_ex(fi,jj,K) - f_ex(fi,jj+1,K))
                                  + f_a_yflux(fi,jj,K) * FDXY
                                    * (f_ey(fi+1,jj,K) - f_ey(fi,jj,K));
                        }
                        A_comp += Real(0.25) * flux;
                    } else {
                        A_comp += c_a_yflux(I,J,K) * CDYY
                                * (c_sol(I,J,K) - c_sol(I,J+1,K))
                                + c_a_yflux(I,J,K) * CDXY
                                * (c_cross(I+1,J,K) - c_cross(I,J,K));
                    }
                    // -y
                    if (cov(I,J-1,K) == 1) {
                        Real flux = Real(0.0);
                        for (int s = 0; s < 2; ++s) {
                            int fi = ii + s;
                            flux += f_a_yflux(fi,jj-1,K) * FDYY
                                    * (f_ex(fi,jj,K) - f_ex(fi,jj-1,K))
                                  + f_a_yflux(fi,jj-1,K) * FDXY
                                    * (f_ey(fi,jj-1,K) - f_ey(fi+1,jj-1,K));
                        }
                        A_comp += Real(0.25) * flux;
                    } else {
                        A_comp += c_a_yflux(I,J-1,K) * CDYY
                                * (c_sol(I,J,K) - c_sol(I,J-1,K))
                                + c_a_yflux(I,J-1,K) * CDXY
                                * (c_cross(I,J-1,K) - c_cross(I+1,J-1,K));
                    }
                    // beta
                    Real bv = var_beta ? c_beta_arr(I,J,K) : c_beta;
                    A_comp += bv * c_sol(I,J,K);
                }
                // -------------------------------------------------
                // Ey (nodal x, cell y): average 2 fine cells (ii,jj)
                //   and (ii,jj+1).
                //   ±x flux + cross uses alphaz (idx 2, stag (0,0))
                //   Cross-term involves ex.
                //
                //   Same 2× correction as Ex.
                // -------------------------------------------------
                else // dim == 1
                {
                    Real CDXY = cdxi[0]*cdxi[1];
                    Real FDXY = fdxi[0]*fdxi[1];
                    // +x
                    if (cov(I+1,J,K) == 1) {
                        Real flux = Real(0.0);
                        for (int s = 0; s < 2; ++s) {
                            int fj = jj + s;
                            flux += f_a_xflux(ii,fj,K) * FDXX
                                    * (f_ey(ii,fj,K) - f_ey(ii+1,fj,K))
                                  + f_a_xflux(ii,fj,K) * FDXY
                                    * (f_ex(ii,fj+1,K) - f_ex(ii,fj,K));
                        }
                        A_comp += Real(0.25) * flux;
                    } else {
                        A_comp += c_a_xflux(I,J,K) * CDXX
                                * (c_sol(I,J,K) - c_sol(I+1,J,K))
                                + c_a_xflux(I,J,K) * CDXY
                                * (c_cross(I,J+1,K) - c_cross(I,J,K));
                    }
                    // -x
                    if (cov(I-1,J,K) == 1) {
                        Real flux = Real(0.0);
                        for (int s = 0; s < 2; ++s) {
                            int fj = jj + s;
                            flux += f_a_xflux(ii-1,fj,K) * FDXX
                                    * (f_ey(ii,fj,K) - f_ey(ii-1,fj,K))
                                  + f_a_xflux(ii-1,fj,K) * FDXY
                                    * (f_ex(ii-1,fj,K) - f_ex(ii-1,fj+1,K));
                        }
                        A_comp += Real(0.25) * flux;
                    } else {
                        A_comp += c_a_xflux(I-1,J,K) * CDXX
                                * (c_sol(I,J,K) - c_sol(I-1,J,K))
                                + c_a_xflux(I-1,J,K) * CDXY
                                * (c_cross(I-1,J,K) - c_cross(I-1,J+1,K));
                    }
                    // beta
                    Real bv = var_beta ? c_beta_arr(I,J,K) : c_beta;
                    A_comp += bv * c_sol(I,J,K);
                }

                corr(I,J,K) = c_rhs(I,J,K) - A_comp;
            });
        }

        // --- Merge composite residual to coarse grid ---------------
        iMultiFab crse_covered(res[idim].boxArray(),
                               res[idim].DistributionMap(), 1, 1);
        crse_covered.setVal(0);
        {
            iMultiFab ones(cba, fdm, 1, 0);
            ones.setVal(1);
            crse_covered.ParallelCopy(ones, 0, 0, 1,
                                      IntVect(0), IntVect(0), crse_period);
        }
        crse_covered.FillBoundary(crse_period);

        MultiFab sync_on_crse(res[idim].boxArray(),
                              res[idim].DistributionMap(), 1, 0);
        sync_on_crse.setVal(Real(0.0));
        sync_on_crse.ParallelCopy(sync_corr, 0, 0, 1,
                                  IntVect(0), IntVect(0), crse_period);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(res[idim], TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            Box const& bx = mfi.tilebox();
            auto const& r   = res[idim].array(mfi);
            auto const& sc  = sync_on_crse.const_array(mfi);
            auto const& cov = crse_covered.const_array(mfi);
            ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k)
            {
                if (cov(i,j,k) != 1) { return; }
                bool on_cf = cov(i-1,j,k)==0 || cov(i+1,j,k)==0
                          || cov(i,j-1,k)==0 || cov(i,j+1,k)==0;
                if (on_cf) {
                    r(i,j,k) = sc(i,j,k);
                }
            });
        }
    }

#elif (AMREX_SPACEDIM == 3)
    amrex::ignore_unused(res, crse_sol, crse_rhs, fine_sol, fine_res, fine_rhs,
                         cdxinv, fdxinv, dinfo, crse_period, ratio);
    amrex::Abort("MLCurlCurl_CNS::syncCFResidual: 3D composite stencil not yet implemented");
#else
    amrex::ignore_unused(res, crse_sol, crse_rhs, fine_sol, fine_res, fine_rhs,
                         cdxinv, fdxinv, dinfo, crse_period, ratio);
#endif
}

void MLCurlCurl_CNS::make (Vector<Vector<MF> >& mf, IntVect const& ng) const
{
    MLLinOpT<MF>::make(mf, ng);
}

Array<MultiFab,3>
MLCurlCurl_CNS::make (int amrlev, int mglev, IntVect const& ng) const
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
MLCurlCurl_CNS::makeAlias (MF const& mf) const
{
    MF r;
    for (int idim = 0; idim < 3; ++idim) {
        r[idim] = MultiFab(mf[idim], amrex::make_alias, 0, mf[idim].nComp());
    }
    return r;
}

Array<MultiFab,3>
MLCurlCurl_CNS::makeCoarseMG (int amrlev, int mglev, IntVect const& ng) const
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
MLCurlCurl_CNS::makeCoarseAmr (int famrlev, IntVect const& ng) const
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

void MLCurlCurl_CNS::applyBC (int amrlev, int mglev, MF& in,
                          CurlCurlStateType type, bool homogeneous) const
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
    // Fill coarse-fine boundary ghost cells (no-op if no c/f data)
    if (CurlCurlStateType::b != type) {
        applyCFBC(amrlev, mglev, in, homogeneous);
    }
}

void MLCurlCurl_CNS::applyPhysBC (int amrlev, int mglev, MultiFab& mf, CurlCurlStateType type) const
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
                    mlcurlcurl_cns_bc_symmetry(i, j, k, face, idxtype, a);
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

iMultiFab const& MLCurlCurl_CNS::getDotMask (int amrlev, int mglev, int idim) const
{
    if (m_dotmask[amrlev][mglev][idim] == nullptr) {
        MultiFab tmp(amrex::convert(this->m_grids[amrlev][mglev], m_etype[idim]),
                     this->m_dmap[amrlev][mglev], 1, 0, MFInfo().SetAlloc(false));
        m_dotmask[amrlev][mglev][idim] =
            tmp.OwnerMask(this->m_geom[amrlev][mglev].periodicity());
    }
    return *m_dotmask[amrlev][mglev][idim];
}

CurlCurlDirichletInfo MLCurlCurl_CNS::getDirichletInfo (int amrlev, int mglev) const
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
                                                      helper(2,1)))};
}

CurlCurlSymmetryInfo MLCurlCurl_CNS::getSymmetryInfo (int amrlev, int mglev) const
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

void MLCurlCurl_CNS::update ()
{
    if (MLLinOpT<Array<MultiFab,3>>::needsUpdate()) {
        MLLinOpT<Array<MultiFab,3>>::update();
    }

    if (m_needs_update) {
        update_lusolver();
        m_needs_update = false;
    }
}

}
