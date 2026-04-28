#ifndef MY_TEST_H_
#define MY_TEST_H_

#include <AMReX_MultiFab.H>
#include <AMReX_iMultiFab.H>

class MyTest
{
public:

    enum class OversetPattern { None, Point, Box, Blob };

    MyTest ();

    void solve ();

// public for cuda
    void initProb ();
    void populateOversetMask ();

private:

    void readParameters ();
    void initData ();

    int coord = 0;
    int n_cell = 128;
    int max_grid_size = 64;

    // For MLMG solver
    int verbose = 1;
    int bottom_verbose = 0;
    int max_iter = 300;
    bool agglomeration = true;
    bool consolidation = true;
    int max_coarsening_level = 30;

    bool use_pcg = false;
    bool use_gmres = false;
    bool gmres_use_precond = true;
    int gmres_precond_niters = 1;

    amrex::Geometry geom;
    amrex::BoxArray grids;
    amrex::DistributionMapping dmap;

    amrex::Array<amrex::MultiFab,3> solution;
    amrex::Array<amrex::MultiFab,3> exact;
    amrex::Array<amrex::MultiFab,3> rhs;
    amrex::MultiFab alpha_node;
    amrex::Array<amrex::iMultiFab,3> overset_mask_mf;

    amrex::Real beta_factor = 1.e-2;
    amrex::Real alpha = 1.0;
    amrex::Real beta = 1.0;
    bool variable_beta = false;
    bool variable_alpha = false;
    bool overset_mask = false;
    OversetPattern overset_pattern = OversetPattern::Blob;
};

#endif
