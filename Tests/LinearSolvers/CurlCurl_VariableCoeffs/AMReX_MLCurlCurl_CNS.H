#ifndef AMREX_ML_CURL_CURL_CNS_H_
#define AMREX_ML_CURL_CURL_CNS_H_
#include <AMReX_Config.H>

#include <AMReX_MLLinOp.H>
#include <MLCurlCurl_CNS_K.H>

namespace amrex {

/**
 * \brief curl (alpha curl E) + beta E = rhs
 *
 * Here E is an Array of 3 MultiFabs on staggered grid.  Alpha is either a
 * positive scalar (setScalars) or a spatially varying field stored on face
 * centerings (setAlpha).  Beta is either a non-negative scalar or a MultiFab
 * on edge centerings (setBeta).
 *
 * When alpha is spatially varying the operator becomes
 *   curl( alpha(x) curl E ) + beta(x) E = rhs
 * with alpha provided as 3 face-centred MultiFabs whose centerings match
 * the components of curl E.
 *
 * It's the caller's responsibility to make sure rhs has consistent nodal
 * data. If needed, one could call prepareRHS for this.
 *
 * The smoother is based on the 4-color Gauss-Seidel smoother of Li
 * et. al. 2020.  "An Efficient Preconditioner for 3-D Finite Difference
 * Modeling of the Electromagnetic Diffusion Process in the Frequency
 * Domain", IEEE Transactions on Geoscience and Remote Sensing, 58, 500-509.
 */
class MLCurlCurl_CNS
    : public MLLinOpT<Array<MultiFab,3> >
{
public:
    using MF        = Array<MultiFab,3>;
    using RT        = typename MLLinOpT<MF>::RT;
    using BCType    = typename MLLinOpT<MF>::BCType;
    using BCMode    = typename MLLinOpT<MF>::BCMode;
    using StateMode = typename MLLinOpT<MF>::StateMode;
    using Location  = typename MLLinOpT<MF>::Location;

    MLCurlCurl_CNS () = default;
    MLCurlCurl_CNS (const Vector<Geometry>& a_geom,
                const Vector<BoxArray>& a_grids,
                const Vector<DistributionMapping>& a_dmap,
                const LPInfo& a_info = LPInfo());

    void define (const Vector<Geometry>& a_geom,
                 const Vector<BoxArray>& a_grids,
                 const Vector<DistributionMapping>& a_dmap,
                 const LPInfo& a_info = LPInfo());

    void setScalars (RT a_alpha, RT a_beta) noexcept;

    //! This is needed only if there is variable beta coefficient.
    void setBeta (const Vector<Array<MultiFab const*,3>>& a_bcoefs);

    //! Set spatially varying alpha on face centerings.
    /**
     * Each entry a_acoefs[amrlev] is an Array of 3 MultiFab pointers with
     * face centerings: (1,0,0), (0,1,0), (0,0,1) in 3D.
     * When set, the scalar alpha from setScalars is ignored for the
     * curl-curl term.
     */
    void setAlpha (const Vector<Array<MultiFab const*,3>>& a_acoefs);

    //! Synchronize RHS on nodal points. If the user can guarantee it, this
    //! function does not need to be called.
    void prepareRHS (Vector<MF*> const& rhs) const;

    void setDirichletNodesToZero (int amrlev, int mglev, MF& a_mf) const override;

    [[nodiscard]] std::string name () const override {
        return std::string("curl of curl");
    }

    bool setUsePCG (bool flag) { return std::exchange(m_use_pcg, flag); }

    void setLevelBC (int amrlev, const MF* levelbcdata,
                     const MF* robinbc_a = nullptr,
                     const MF* robinbc_b = nullptr,
                     const MF* robinbc_f = nullptr) override;

    void restriction (int amrlev, int cmglev, MF& crse, MF& fine) const override;

    void interpolation (int amrlev, int fmglev, MF& fine, const MF& crse) const override;

    void apply (int amrlev, int mglev, MF& out, MF& in, BCMode bc_mode,
                StateMode s_mode, const MLMGBndryT<MF>* bndry=nullptr) const override;

    void smooth (int amrlev, int mglev, MF& sol, const MF& rhs,
                 bool skip_fillboundary, int niter) const override;

    void solutionResidual (int amrlev, MF& resid, MF& x, const MF& b,
                           const MF* crse_bcdata=nullptr) override;

    void correctionResidual (int amrlev, int mglev, MF& resid, MF& x,
                             const MF& b, BCMode bc_mode,
                             const MF* crse_bcdata=nullptr) override;

    [[nodiscard]] bool needsUpdate () const override {
        return (m_needs_update || MLLinOpT<Array<MultiFab,3>>::needsUpdate());
    }
    void update () override;

    void prepareForSolve () override;

    [[nodiscard]] bool isSingular (int /*amrlev*/) const override { return false; }
    [[nodiscard]] bool isBottomSingular () const override { return false; }

    RT xdoty (int amrlev, int mglev, const MF& x, const MF& y, bool local) const override;

    [[nodiscard]] RT normInf (int amrlev, MF const& mf, bool local) const override;

    void averageDownAndSync (Vector<MF>& sol) const override;

    void averageDownSolutionRHS (int camrlev, MF& crse_sol, MF& crse_rhs,
                                 const MF& fine_sol, const MF& fine_rhs) override;

    void avgDownResAmr (int clev, MF& cres, MF const& fres) const override;

    void interpolationAmr (int famrlev, MF& fine, MF const& crse,
                                  IntVect const& nghost) const override ;

    void reflux (int crse_amrlev,
                 MF& res, const MF& crse_sol, const MF& crse_rhs,
                 MF& fine_res, MF& fine_sol, const MF& fine_rhs) const override;

    //! Sync-residual correction at C/F boundary coarse nodes.
    /**
     * Uses the direct composite stencil: for each flux direction at a
     * CF boundary coarse node, fine-covered neighbours use the fine-level
     * flux (fine alpha, fine E, fine dx) while non-covered neighbours use
     * the coarse-level flux (coarse alpha, coarse E, coarse dx).
     */
    void syncCFResidual (int crse_amrlev, MF& res,
                         const MF& crse_sol, const MF& crse_rhs,
                         const MF& fine_sol,
                         const MF& fine_res, const MF& fine_rhs) const;

    //! Number of ghost-cell layers for the ghost-nodes CF strategy.
    [[nodiscard]] int getNGrow (int = 0, int = 0) const override { return 2; }

    [[nodiscard]] IntVect getNGrowVectRestriction () const override {
        return IntVect(2);
    }

    void make (Vector<Vector<MF> >& mf, IntVect const& ng) const override;

    [[nodiscard]] MF make (int amrlev, int mglev, IntVect const& ng) const override;

    [[nodiscard]] MF makeAlias (MF const& mf) const override;

    [[nodiscard]] MF makeCoarseMG (int amrlev, int mglev, IntVect const& ng) const override;

    [[nodiscard]] MF makeCoarseAmr (int famrlev, IntVect const& ng) const override;

// public for cuda

#if (AMREX_SPACEDIM > 1)
    void smooth4 (int amrlev, int mglev, MF& sol, MF const& rhs, int color) const;
#else
    void smooth1D (int amrlev, int mglev, MF& sol, MF const& rhs, int color) const;
#endif

    void compresid (int amrlev, int mglev, MF& resid, MF const& b,
                    bool grow_for_cf = false) const;

    void applyPhysBC (int amrlev, int mglev, MultiFab& mf, CurlCurlStateType type) const;

private:

    void applyBC (int amrlev, int mglev, MF& in, CurlCurlStateType type,
                  bool homogeneous = false) const;

    //! Fill coarse-fine boundary ghost cells from stored coarse data.
    //! When homogeneous is true, zero is used even at mglev 0.
    void applyCFBC (int amrlev, int mglev, MF& in,
                    bool homogeneous = false) const;

    //! Build masks identifying coarse-fine boundary ghost cells for each
    //! edge type and MG level.
    void buildCFMasks ();

    [[nodiscard]] iMultiFab const& getDotMask (int amrlev, int mglev, int idim) const;

    [[nodiscard]] CurlCurlDirichletInfo getDirichletInfo (int amrlev, int mglev) const;
    [[nodiscard]] CurlCurlSymmetryInfo getSymmetryInfo (int amrlev, int mglev) const;

    void update_lusolver ();

    RT m_alpha = std::numeric_limits<RT>::lowest();
    RT m_beta  = std::numeric_limits<RT>::lowest();

    //! Edge centerings where E lives.
    Array<IntVect,3> m_etype
#if (AMREX_SPACEDIM == 3)
         {IntVect(0,1,1), IntVect(1,0,1), IntVect(1,1,0)};
#elif (AMREX_SPACEDIM == 2)
         {IntVect(0,1), IntVect(1,0), IntVect(1,1)};
#else
         {IntVect(0), IntVect(1), IntVect(1)};
#endif

    //! Face centerings where curl(E) lives — complement of m_etype.
    Array<IntVect,3> m_ftype
#if (AMREX_SPACEDIM == 3)
         {IntVect(1,0,0), IntVect(0,1,0), IntVect(0,0,1)};
#elif (AMREX_SPACEDIM == 2)
         {IntVect(1,0), IntVect(0,1), IntVect(0,0)};
#else
         {IntVect(1), IntVect(0), IntVect(0)};
#endif

    mutable Vector<Vector<Array<std::unique_ptr<iMultiFab>,3>>> m_dotmask;
    static constexpr int m_ncomp = 1;
    Vector<Vector<std::unique_ptr<Gpu::DeviceScalar
                                  <LUSolver<AMREX_SPACEDIM*2,RT>>>>> m_lusolver;
    //! Per-AMR-level, per-MG-level beta coefficients on edge centerings.
    Vector<Vector<Array<std::unique_ptr<MultiFab>,3>>> m_bcoefs;
    //! Per-AMR-level, per-MG-level alpha coefficients on face centerings.
    Vector<Vector<Array<std::unique_ptr<MultiFab>,3>>> m_acoefs;
    //! True when setAlpha has been called with spatially varying alpha.
    bool m_has_variable_alpha = false;
    bool m_use_pcg = false;
    bool m_needs_update = true;

    //! Per-AMR-level, per-MG-level masks for each edge type.
    //! 0 = valid or interior ghost, 1 = layer-1 CF ghost (relaxed),
    //! 2 = layer-2 CF ghost (frozen Dirichlet).
    Vector<Vector<Array<std::unique_ptr<iMultiFab>,3>>> m_cfmask;

    //! Per-AMR-level coarse E data interpolated to fine resolution for C/F BCs.
    //! m_cf_data[amrlev] is valid only when m_has_cf_data[amrlev] is true.
    Vector<MF> m_cf_data;
    //! 1 = C/F data valid, 0 = not valid.  Uses int instead of bool
    //! to avoid the std::vector<bool> proxy-reference issue with amrex::Vector.
    Vector<int> m_has_cf_data;

    //! Interpolate coarse E data to fine resolution and store in m_cf_data.
    /**
     * \param amrlev AMR level whose C/F data is updated
     * \param crse   Coarse-level E field (3-component Array<MultiFab,3>)
     * \param ratio  Coarsening ratio between the coarse and fine levels
     */
    void updateCFData (int amrlev, const MF& crse, IntVect ratio);
};

}

#endif
