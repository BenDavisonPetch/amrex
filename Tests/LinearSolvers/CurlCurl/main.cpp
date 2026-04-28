#include "MyTest.H"

#include <AMReX.H>
#include <AMReX_BLProfiler.H>
#include <AMReX_ParmParse.H>

using namespace amrex;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv);

    {
        BL_PROFILE("main");
        {
            ParmParse pp;
            Vector<int> variable_alpha;
            Vector<int> variable_beta;
            Vector<int> overset_mask;
            if (pp.contains("variable_alpha")) {
                int t;
                pp.get("variable_alpha", t);
                variable_alpha.push_back(t);
            } else {
                variable_alpha.push_back(0);
                variable_alpha.push_back(1);
            }
            if (pp.contains("variable_beta")) {
                int t;
                pp.get("variable_beta", t);
                variable_beta.push_back(t);
            } else {
                variable_beta.push_back(0);
                variable_beta.push_back(1);
            }
            if (pp.contains("overset_mask")) {
                int t;
                pp.get("overset_mask", t);
                overset_mask.push_back(t);
            } else {
                overset_mask.push_back(0);
                overset_mask.push_back(1);
            }
            for (auto alpha : variable_alpha) {
                for (auto beta : variable_beta) {
                    for (auto om : overset_mask) {
                        pp.add("variable_alpha", alpha);
                        pp.add("variable_beta", beta);
                        pp.add("overset_mask", om);
                        amrex::Print() << "\nTesting variable_alpha = " << alpha
                                       << " variable_beta = " << beta
                                       << " overset_mask = " << om << "\n";
                        MyTest mytest;
                        mytest.solve();
                    }
                }
            }
        }
    }

    amrex::Finalize();
}
