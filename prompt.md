The code found in @DiffusiveMethod/ImplicitFD/CurlCurl/ is an adaptation of AMReX's nodal CurlCurl solver, which solves equations of the form `curl(alpha*curl(E)) + beta*E = f` for unknown 3-dim vector E, known beta and alpha, and known rhs f. It is modified from the amrex version to support *spatially varying* alpha and beta and basic AMR.

E and f always have three components and are stored in a 3-dimensional array of MultiFabs. In 3D, component i is cell-centred in direction i and nodal in the other two. In 2D, the z component is nodal, the x component lives on y faces, and the y component lives on x faces. In 1D the x component is cell-centred and the y and z components are face-centred. This logically determines the index type of alpha and beta.

I want to add support for an overset mask, so that `E` can be specified as known in certain locations. This is already supported in some other amrex operators, namely those derived from `MLNodeLinOp` (grep for `setOversetMask`).

I want to understand:
1. How dirichlet boundaries are currently treated in the curl-curl solver.
2. How the existing amrex overset mask implementation works and how it relates to this solver.

For each of the above things explain to me in steps:
A. Unigrid (no AMR, no MG)
B. no AMR, but MG
C. MG with AMR

After each step, we're going to add to a knowledge document @oversetmask_findings.md in the local directory.

# Important
Do NOT start planning or researching any code changes. I want to understand the process in maths/pseudocode ONLY.

When looking for amrex documentation, use the /find-docs skill with library ID /amrex-codes/amrex