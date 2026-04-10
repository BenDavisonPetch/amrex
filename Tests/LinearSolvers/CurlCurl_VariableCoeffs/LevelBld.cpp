#include <AMReX_LevelBld.H>
#include "AmrLevelCurlCurl.H"

using namespace amrex;

class LevelBldCurlCurl
  : public LevelBld
{
  void variableSetUp () override;
  void variableCleanUp () override;
  AmrLevel* operator() () override;
  AmrLevel* operator() (Amr& papa,
                        int lev,
                        const Geometry& level_geom,
                        const BoxArray& ba,
                        const DistributionMapping& dm,
                        Real time) override;
};

LevelBldCurlCurl the_bld;

LevelBld*
getLevelBld ()
{
  return &the_bld;
}

void
LevelBldCurlCurl::variableSetUp ()
{
  AmrLevelCurlCurl::variableSetUp();
}

void
LevelBldCurlCurl::variableCleanUp ()
{
  AmrLevelCurlCurl::variableCleanUp();
}

AmrLevel*
LevelBldCurlCurl::operator() ()
{
  return new AmrLevelCurlCurl;
}

AmrLevel*
LevelBldCurlCurl::operator() (Amr& papa,
                              int lev,
                              const Geometry& level_geom,
                              const BoxArray& ba,
                              const DistributionMapping& dm,
                              Real time)
{
  return new AmrLevelCurlCurl(papa, lev, level_geom, ba, dm, time);
}
