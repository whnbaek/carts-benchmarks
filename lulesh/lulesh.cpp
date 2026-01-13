/*
  This is a Version 2.0 MPI + OpenMP implementation of LULESH

                 Copyright (c) 2010-2013.
      Lawrence Livermore National Security, LLC.
Produced at the Lawrence Livermore National Laboratory.
                  LLNL-CODE-461231
                All rights reserved.

This file is part of LULESH, Version 2.0.
Please also read this link -- http://www.opensource.org/licenses/index.php

//////////////
DIFFERENCES BETWEEN THIS VERSION (2.x) AND EARLIER VERSIONS:
 * Addition of regions to make work more representative of multi-material codes
 * Default size of each domain is 30^3 (27000 elem) instead of 45^3. This is
  more representative of our actual working set sizes
 * Single source distribution supports pure serial, pure OpenMP, MPI-only,
  and MPI+OpenMP
 * Addition of ability to visualize the mesh using VisIt
  https://wci.llnl.gov/codes/visit/download.html
 * Various command line options (see ./lulesh2.0 -h)
 -q              : quiet mode - suppress stdout
 -i <iterations> : number of cycles to run
 -s <size>       : length of cube mesh along side
 -r <numregions> : Number of distinct regions (def: 11)
 -b <balance>    : Load balance between regions of a domain (def: 1)
 -c <cost>       : Extra cost of more expensive regions (def: 1)
 -f <filepieces> : Number of file parts for viz output (def: np/9)
 -p              : Print out progress
 -v              : Output viz file (requires compiling with -DVIZ_MESH
 -h              : This message

 printf("Usage: %s [opts]\n", execname);
      printf(" where [opts] is one or more of:\n");
      printf(" -q              : quiet mode - suppress all stdout\n");
      printf(" -i <iterations> : number of cycles to run\n");
      printf(" -s <size>       : length of cube mesh along side\n");
      printf(" -r <numregions> : Number of distinct regions (def: 11)\n");
      printf(" -b <balance>    : Load balance between regions of a domain (def:
1)\n"); printf(" -c <cost>       : Extra cost of more expensive regions (def:
1)\n"); printf(" -f <numfiles>   : Number of files to split viz dump into (def:
(np+10)/9)\n"); printf(" -p              : Print out progress\n"); printf(" -v
: Output viz file (requires compiling with -DVIZ_MESH\n"); printf(" -h : This
message\n"); printf("\n\n");

 *Notable changes in LULESH 2.0

 * Split functionality into different files
lulesh.cc - where most (all?) of the timed functionality lies
lulesh-comm.cc - MPI functionality
lulesh-init.cc - Setup code
lulesh-viz.cc  - Support for visualization option
lulesh-util.cc - Non-timed functions
 *
 * The concept of "regions" was added, although every region is the same ideal
 *    gas material, and the same sedov blast wave problem is still the only
 *    problem its hardcoded to solve.
 * Regions allow two things important to making this proxy app more
representative:
 *   Four of the LULESH routines are now performed on a region-by-region basis,
 *     making the memory access patterns non-unit stride
 *   Artificial load imbalances can be easily introduced that could impact
 *     parallelization strategies.
 * The load balance flag changes region assignment.  Region number is raised to
 *   the power entered for assignment probability.  Most likely regions changes
 *   with MPI process id.
 * The cost flag raises the cost of ~45% of the regions to evaluate EOS by the
 *   entered multiple. The cost of 5% is 10x the entered multiple.
 * MPI and OpenMP were added, and coalesced into a single version of the source
 *   that can support serial builds, MPI-only, OpenMP-only, and MPI+OpenMP
 * Added support to write plot files using "poor mans parallel I/O" when linked
 *   with the silo library, which in turn can be read by VisIt.
 * Enabled variable timestep calculation by default (courant condition), which
 *   results in an additional reduction.
 * Default domain (mesh) size reduced from 45^3 to 30^3
 * Command line options to allow numerous test cases without needing to
recompile
 * Performance optimizations and code cleanup beyond LULESH 1.0
 * Added a "Figure of Merit" calculation (elements solved per microsecond) and
 *   output in support of using LULESH 2.0 for the 2017 CORAL procurement
 *
 * Possible Differences in Final Release (other changes possible)
 *
 * High Level mesh structure to allow data structure transformations
 * Different default parameters
 * Minor code performance changes and cleanup

TODO in future versions
 * Add reader for (truly) unstructured meshes, probably serial only
 * CMake based build system

//////////////

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

 * Redistributions of source code must retain the above copyright
     notice, this list of conditions and the disclaimer below.

 * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the disclaimer (as noted below)
     in the documentation and/or other materials provided with the
     distribution.

 * Neither the name of the LLNS/LLNL nor the names of its contributors
     may be used to endorse or promote products derived from this software
     without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC,
THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


Additional BSD Notice

1. This notice is required to be provided under our contract with the U.S.
   Department of Energy (DOE). This work was produced at Lawrence Livermore
   National Laboratory under Contract No. DE-AC52-07NA27344 with the DOE.

2. Neither the United States Government nor Lawrence Livermore National
   Security, LLC nor any of their employees, makes any warranty, express
   or implied, or assumes any liability or responsibility for the accuracy,
   completeness, or usefulness of any information, apparatus, product, or
   process disclosed, or represents that its use would not infringe
   privately-owned rights.

3. Also, reference herein to any specific commercial products, process, or
   services by trade name, trademark, manufacturer or otherwise does not
   necessarily constitute or imply its endorsement, recommendation, or
   favoring by the United States Government or Lawrence Livermore National
   Security, LLC. The views and opinions of authors expressed herein do not
   necessarily state or reflect those of the United States Government or
   Lawrence Livermore National Security, LLC, and shall not be used for
   advertising or product endorsement purposes.

 */

 #include <limits.h>
 #include <math.h>
 #ifdef _OPENMP
 #include <omp.h>
 #endif
 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <sys/time.h>
 #include <unistd.h>
 
 //**************************************************
 // Allow flexibility for arithmetic representations
 //**************************************************
 
 #define MAX(a, b) (((a) > (b)) ? (a) : (b))
 
 // Precision specification
 typedef float real4;
 typedef double real8;
 typedef long double real10; // 10 bytes on x86
 
 typedef int Index_t;  // array subscript and loop index
 typedef real8 Real_t; // floating point representation
 typedef int Int_t;    // integer representation
 
 enum { VolumeError = -1, QStopError = -2 };
 
 inline real4 SQRT(real4 arg) { return sqrtf(arg); }
 inline real8 SQRT(real8 arg) { return sqrt(arg); }
 inline real10 SQRT(real10 arg) { return sqrtl(arg); }
 
 inline real4 CBRT(real4 arg) { return cbrtf(arg); }
 inline real8 CBRT(real8 arg) { return cbrt(arg); }
 inline real10 CBRT(real10 arg) { return cbrtl(arg); }
 
 inline real4 FABS(real4 arg) { return fabsf(arg); }
 inline real8 FABS(real8 arg) { return fabs(arg); }
 inline real10 FABS(real10 arg) { return fabsl(arg); }
 
 inline Int_t IPOW(Int_t base, Int_t exp) {
   Int_t result = 1;
   while (exp > 0) {
     if (exp & 1)
       result *= base;
     base *= base;
     exp >>= 1;
   }
   return result;
 }
 
 // Stuff needed for boundary conditions
 // 2 BCs on each of 6 hexahedral faces (12 bits)
 #define XI_M 0x00007
 #define XI_M_SYMM 0x00001
 #define XI_M_FREE 0x00002
 #define XI_M_COMM 0x00004
 
 #define XI_P 0x00038
 #define XI_P_SYMM 0x00008
 #define XI_P_FREE 0x00010
 #define XI_P_COMM 0x00020
 
 #define ETA_M 0x001c0
 #define ETA_M_SYMM 0x00040
 #define ETA_M_FREE 0x00080
 #define ETA_M_COMM 0x00100
 
 #define ETA_P 0x00e00
 #define ETA_P_SYMM 0x00200
 #define ETA_P_FREE 0x00400
 #define ETA_P_COMM 0x00800
 
 #define ZETA_M 0x07000
 #define ZETA_M_SYMM 0x01000
 #define ZETA_M_FREE 0x02000
 #define ZETA_M_COMM 0x04000
 
 #define ZETA_P 0x38000
 #define ZETA_P_SYMM 0x08000
 #define ZETA_P_FREE 0x10000
 #define ZETA_P_COMM 0x20000
 
 // MPI Message Tags
 #define MSG_COMM_SBN 1024
 #define MSG_SYNC_POS_VEL 2048
 #define MSG_MONOQ 3072
 
 #define MAX_FIELDS_PER_MPI_COMM 6
 
 // Assume 128 byte coherence
 // Assume Real_t is an "integral power of 2" bytes wide
 #define CACHE_COHERENCE_PAD_REAL (128 / sizeof(Real_t))
 
 #define CACHE_ALIGN_REAL(n)                                                    \
   (((n) + (CACHE_COHERENCE_PAD_REAL - 1)) & ~(CACHE_COHERENCE_PAD_REAL - 1))
 
 //////////////////////////////////////////////////////
 // Primary data structure
 //////////////////////////////////////////////////////
 
 /*
  * The implementation of the data abstraction used for lulesh
  * resides entirely in the Domain class below.  You can change
  * grouping and interleaving of fields here to maximize data layout
  * efficiency for your underlying architecture or compiler.
  *
  * For example, fields can be implemented as STL objects or
  * raw array pointers.  As another example, individual fields
  * m_x, m_y, m_z could be budled into
  *
  *    struct { Real_t x, y, z ; } *m_coord ;
  *
  * allowing accessor functions such as
  *
  *  "Real_t &x(Index_t idx) { return m_coord[idx].x ; }"
  *  "Real_t &y(Index_t idx) { return m_coord[idx].y ; }"
  *  "Real_t &z(Index_t idx) { return m_coord[idx].z ; }"
  */
 
 /******************************************/
 
 /* Work Routines */
 
 static inline void TimeIncrement(Real_t *stoptime, Real_t *time,
                                  Real_t *dtfixed, Int_t *cycle,
                                  Real_t *deltatime, Real_t *dtcourant,
                                  Real_t *dthydro, Real_t deltatimemultlb,
                                  Real_t deltatimemultub, Real_t dtmax) {
   Real_t targetdt = *stoptime - *time;
 
   if ((*dtfixed <= Real_t(0.0)) && (*cycle != Int_t(0))) {
     Real_t ratio;
     Real_t olddt = *deltatime;
 
     /* This will require a reduction in parallel */
     Real_t gnewdt = Real_t(1.0e+20);
     Real_t newdt;
     if (*dtcourant < gnewdt) {
       gnewdt = *dtcourant / Real_t(2.0);
     }
     if (*dthydro < gnewdt) {
       gnewdt = *dthydro * Real_t(2.0) / Real_t(3.0);
     }
 
     newdt = gnewdt;
 
     ratio = newdt / olddt;
     if (ratio >= Real_t(1.0)) {
       if (ratio < deltatimemultlb) {
         newdt = olddt;
       } else if (ratio > deltatimemultub) {
         newdt = olddt * deltatimemultub;
       }
     }
 
     if (newdt > dtmax) {
       newdt = dtmax;
     }
     *deltatime = newdt;
   }
 
   /* TRY TO PREVENT VERY SMALL SCALING ON THE NEXT CYCLE */
   if ((targetdt > *deltatime) &&
       (targetdt < (Real_t(4.0) * (*deltatime) / Real_t(3.0)))) {
     targetdt = Real_t(2.0) * (*deltatime) / Real_t(3.0);
   }
 
   if (targetdt < *deltatime) {
     *deltatime = targetdt;
   }
 
   *time += *deltatime;
 
   ++(*cycle);
 }
 
 /******************************************/
 
 static inline void CollectDomainNodesToElemNodes(Real_t *x, Real_t *y,
                                                  Real_t *z,
                                                  Index_t *nodelist,
                                                  Index_t offset,
                                                  Real_t *elemX, Real_t *elemY,
                                                  Real_t *elemZ) {
   Index_t nd0i = nodelist[offset + 0];
   Index_t nd1i = nodelist[offset + 1];
   Index_t nd2i = nodelist[offset + 2];
   Index_t nd3i = nodelist[offset + 3];
   Index_t nd4i = nodelist[offset + 4];
   Index_t nd5i = nodelist[offset + 5];
   Index_t nd6i = nodelist[offset + 6];
   Index_t nd7i = nodelist[offset + 7];
 
   elemX[0] = x[nd0i];
   elemX[1] = x[nd1i];
   elemX[2] = x[nd2i];
   elemX[3] = x[nd3i];
   elemX[4] = x[nd4i];
   elemX[5] = x[nd5i];
   elemX[6] = x[nd6i];
   elemX[7] = x[nd7i];
 
   elemY[0] = y[nd0i];
   elemY[1] = y[nd1i];
   elemY[2] = y[nd2i];
   elemY[3] = y[nd3i];
   elemY[4] = y[nd4i];
   elemY[5] = y[nd5i];
   elemY[6] = y[nd6i];
   elemY[7] = y[nd7i];
 
   elemZ[0] = z[nd0i];
   elemZ[1] = z[nd1i];
   elemZ[2] = z[nd2i];
   elemZ[3] = z[nd3i];
   elemZ[4] = z[nd4i];
   elemZ[5] = z[nd5i];
   elemZ[6] = z[nd6i];
   elemZ[7] = z[nd7i];
 }
 
 /******************************************/
 
 static inline void InitStressTermsForElems(Real_t *p, Real_t *q, Real_t *sigxx,
                                            Real_t *sigyy, Real_t *sigzz,
                                            Index_t numElem) {
   //
   // pull in the stresses appropriate to the hydro integration
   //
 
 #pragma omp parallel for firstprivate(numElem)
   for (Index_t i = 0; i < numElem; ++i) {
     sigxx[i] = sigyy[i] = sigzz[i] = -p[i] - q[i];
   }
 }
 
 /******************************************/
 
 static inline void CalcElemShapeFunctionDerivatives(Real_t const *x,
                                                     Real_t const *y,
                                                     Real_t const *z, Real_t **b,
                                                     Real_t *const volume) {
   const Real_t x0 = x[0];
   const Real_t x1 = x[1];
   const Real_t x2 = x[2];
   const Real_t x3 = x[3];
   const Real_t x4 = x[4];
   const Real_t x5 = x[5];
   const Real_t x6 = x[6];
   const Real_t x7 = x[7];
 
   const Real_t y0 = y[0];
   const Real_t y1 = y[1];
   const Real_t y2 = y[2];
   const Real_t y3 = y[3];
   const Real_t y4 = y[4];
   const Real_t y5 = y[5];
   const Real_t y6 = y[6];
   const Real_t y7 = y[7];
 
   const Real_t z0 = z[0];
   const Real_t z1 = z[1];
   const Real_t z2 = z[2];
   const Real_t z3 = z[3];
   const Real_t z4 = z[4];
   const Real_t z5 = z[5];
   const Real_t z6 = z[6];
   const Real_t z7 = z[7];
 
   Real_t fjxxi, fjxet, fjxze;
   Real_t fjyxi, fjyet, fjyze;
   Real_t fjzxi, fjzet, fjzze;
   Real_t cjxxi, cjxet, cjxze;
   Real_t cjyxi, cjyet, cjyze;
   Real_t cjzxi, cjzet, cjzze;
 
   fjxxi = Real_t(.125) * ((x6 - x0) + (x5 - x3) - (x7 - x1) - (x4 - x2));
   fjxet = Real_t(.125) * ((x6 - x0) - (x5 - x3) + (x7 - x1) - (x4 - x2));
   fjxze = Real_t(.125) * ((x6 - x0) + (x5 - x3) + (x7 - x1) + (x4 - x2));
 
   fjyxi = Real_t(.125) * ((y6 - y0) + (y5 - y3) - (y7 - y1) - (y4 - y2));
   fjyet = Real_t(.125) * ((y6 - y0) - (y5 - y3) + (y7 - y1) - (y4 - y2));
   fjyze = Real_t(.125) * ((y6 - y0) + (y5 - y3) + (y7 - y1) + (y4 - y2));
 
   fjzxi = Real_t(.125) * ((z6 - z0) + (z5 - z3) - (z7 - z1) - (z4 - z2));
   fjzet = Real_t(.125) * ((z6 - z0) - (z5 - z3) + (z7 - z1) - (z4 - z2));
   fjzze = Real_t(.125) * ((z6 - z0) + (z5 - z3) + (z7 - z1) + (z4 - z2));
 
   /* compute cofactors */
   cjxxi = (fjyet * fjzze) - (fjzet * fjyze);
   cjxet = -(fjyxi * fjzze) + (fjzxi * fjyze);
   cjxze = (fjyxi * fjzet) - (fjzxi * fjyet);
 
   cjyxi = -(fjxet * fjzze) + (fjzet * fjxze);
   cjyet = (fjxxi * fjzze) - (fjzxi * fjxze);
   cjyze = -(fjxxi * fjzet) + (fjzxi * fjxet);
 
   cjzxi = (fjxet * fjyze) - (fjyet * fjxze);
   cjzet = -(fjxxi * fjyze) + (fjyxi * fjxze);
   cjzze = (fjxxi * fjyet) - (fjyxi * fjxet);
 
   /* calculate partials :
 this need only be done for l = 0,1,2,3   since , by symmetry ,
 (6,7,4,5) = - (0,1,2,3) .
    */
   b[0][0] = -cjxxi - cjxet - cjxze;
   b[0][1] = cjxxi - cjxet - cjxze;
   b[0][2] = cjxxi + cjxet - cjxze;
   b[0][3] = -cjxxi + cjxet - cjxze;
   b[0][4] = -b[0][2];
   b[0][5] = -b[0][3];
   b[0][6] = -b[0][0];
   b[0][7] = -b[0][1];
 
   b[1][0] = -cjyxi - cjyet - cjyze;
   b[1][1] = cjyxi - cjyet - cjyze;
   b[1][2] = cjyxi + cjyet - cjyze;
   b[1][3] = -cjyxi + cjyet - cjyze;
   b[1][4] = -b[1][2];
   b[1][5] = -b[1][3];
   b[1][6] = -b[1][0];
   b[1][7] = -b[1][1];
 
   b[2][0] = -cjzxi - cjzet - cjzze;
   b[2][1] = cjzxi - cjzet - cjzze;
   b[2][2] = cjzxi + cjzet - cjzze;
   b[2][3] = -cjzxi + cjzet - cjzze;
   b[2][4] = -b[2][2];
   b[2][5] = -b[2][3];
   b[2][6] = -b[2][0];
   b[2][7] = -b[2][1];
 
   /* calculate jacobian determinant (volume) */
   *volume = Real_t(8.) * (fjxet * cjxet + fjyet * cjyet + fjzet * cjzet);
 }
 
 /******************************************/
 
 static inline void SumElemFaceNormal(
     Real_t **B, Index_t i0, Index_t i1, Index_t i2, Index_t i3,
     const Real_t *x, const Real_t *y, const Real_t *z) {
   Real_t bisectX0 = Real_t(0.5) * (x[i3] + x[i2] - x[i1] - x[i0]);
   Real_t bisectY0 = Real_t(0.5) * (y[i3] + y[i2] - y[i1] - y[i0]);
   Real_t bisectZ0 = Real_t(0.5) * (z[i3] + z[i2] - z[i1] - z[i0]);
   Real_t bisectX1 = Real_t(0.5) * (x[i2] + x[i1] - x[i3] - x[i0]);
   Real_t bisectY1 = Real_t(0.5) * (y[i2] + y[i1] - y[i3] - y[i0]);
   Real_t bisectZ1 = Real_t(0.5) * (z[i2] + z[i1] - z[i3] - z[i0]);
   Real_t areaX = Real_t(0.25) * (bisectY0 * bisectZ1 - bisectZ0 * bisectY1);
   Real_t areaY = Real_t(0.25) * (bisectZ0 * bisectX1 - bisectX0 * bisectZ1);
   Real_t areaZ = Real_t(0.25) * (bisectX0 * bisectY1 - bisectY0 * bisectX1);
 
   B[0][i0] += areaX;
   B[0][i1] += areaX;
   B[0][i2] += areaX;
   B[0][i3] += areaX;
 
   B[1][i0] += areaY;
   B[1][i1] += areaY;
   B[1][i2] += areaY;
   B[1][i3] += areaY;
 
   B[2][i0] += areaZ;
   B[2][i1] += areaZ;
   B[2][i2] += areaZ;
   B[2][i3] += areaZ;
 }
 
 /******************************************/
 
 static inline void CalcElemNodeNormals(Real_t **B,
                                        const Real_t *x, const Real_t *y,
                                        const Real_t *z) {
   for (Index_t i = 0; i < 8; ++i) {
     B[0][i] = Real_t(0.0);
     B[1][i] = Real_t(0.0);
     B[2][i] = Real_t(0.0);
   }
   /* evaluate face one: nodes 0, 1, 2, 3 */
   SumElemFaceNormal(B, 0, 1, 2, 3, x, y, z);
   /* evaluate face two: nodes 0, 4, 5, 1 */
   SumElemFaceNormal(B, 0, 4, 5, 1, x, y, z);
   /* evaluate face three: nodes 1, 5, 6, 2 */
   SumElemFaceNormal(B, 1, 5, 6, 2, x, y, z);
   /* evaluate face four: nodes 2, 6, 7, 3 */
   SumElemFaceNormal(B, 2, 6, 7, 3, x, y, z);
   /* evaluate face five: nodes 3, 7, 4, 0 */
   SumElemFaceNormal(B, 3, 7, 4, 0, x, y, z);
   /* evaluate face six: nodes 4, 7, 6, 5 */
   SumElemFaceNormal(B, 4, 7, 6, 5, x, y, z);
 }
 
 /******************************************/
 
 static inline void
 SumElemStressesToNodeForces(Real_t **B, const Real_t stress_xx,
                             const Real_t stress_yy, const Real_t stress_zz,
                             Real_t *fx, Real_t *fy, Real_t *fz,
                             Index_t offset) {
   for (Index_t i = 0; i < 8; i++) {
     fx[offset + i] = -(stress_xx * B[0][i]);
     fy[offset + i] = -(stress_yy * B[1][i]);
     fz[offset + i] = -(stress_zz * B[2][i]);
   }
 }
 
 /******************************************/
 
 static inline void
 IntegrateStressForElems(Real_t *x, Real_t *y, Real_t *z, Real_t *fx, Real_t *fy,
                         Real_t *fz, Index_t *nodelist, Index_t *nodeElemStart,
                         Index_t *nodeElemCornerList, Real_t *sigxx,
                         Real_t *sigyy, Real_t *sigzz, Real_t *determ,
                         Index_t numElem, Index_t numNode) {
   Index_t numElem8 = numElem * 8;
   Real_t *fx_elem = (Real_t *)malloc(numElem8 * sizeof(Real_t));
   Real_t *fy_elem = (Real_t *)malloc(numElem8 * sizeof(Real_t));
   Real_t *fz_elem = (Real_t *)malloc(numElem8 * sizeof(Real_t));
 
 #pragma omp parallel for firstprivate(numElem)
   for (Index_t k = 0; k < numElem; ++k) {
     Real_t *B[3];
     B[0] = (Real_t *)malloc(8 * sizeof(Real_t));
     B[1] = (Real_t *)malloc(8 * sizeof(Real_t));
     B[2] = (Real_t *)malloc(8 * sizeof(Real_t));
     Real_t x_local[8];
     Real_t y_local[8];
     Real_t z_local[8];
 
     CollectDomainNodesToElemNodes(x, y, z, nodelist, k * 8, x_local, y_local,
                                   z_local);
     CalcElemShapeFunctionDerivatives(x_local, y_local, z_local, B, &determ[k]);
     CalcElemNodeNormals(B, x_local, y_local, z_local);
     SumElemStressesToNodeForces(B, sigxx[k], sigyy[k], sigzz[k],
                                 fx_elem, fy_elem, fz_elem, k * 8);
     free(B[0]);
     free(B[1]);
     free(B[2]);
   }
 
 #pragma omp parallel for firstprivate(numNode)
   for (Index_t gnode = 0; gnode < numNode; ++gnode) {
     Index_t start = nodeElemStart[gnode];
     Index_t count = nodeElemStart[gnode + 1] - start;
     Real_t fx_tmp = Real_t(0.0);
     Real_t fy_tmp = Real_t(0.0);
     Real_t fz_tmp = Real_t(0.0);
     for (Index_t i = 0; i < count; ++i) {
       Index_t elem = nodeElemCornerList[start + i];
       fx_tmp += fx_elem[elem];
       fy_tmp += fy_elem[elem];
       fz_tmp += fz_elem[elem];
     }
     fx[gnode] = fx_tmp;
     fy[gnode] = fy_tmp;
     fz[gnode] = fz_tmp;
   }
 
   free(fz_elem);
   free(fy_elem);
   free(fx_elem);
 }
 
 /******************************************/
 
 static inline void VoluDer(const Real_t *x, const Real_t *y, const Real_t *z,
                            Index_t i0, Index_t i1, Index_t i2,
                            Index_t i3, Index_t i4, Index_t i5,
                            Real_t *dvdx, Real_t *dvdy, Real_t *dvdz,
                            Index_t out_idx) {
   const Real_t twelfth = Real_t(1.0) / Real_t(12.0);
 
   dvdx[out_idx] = (y[i1] + y[i2]) * (z[i0] + z[i1]) - (y[i0] + y[i1]) * (z[i1] + z[i2]) +
           (y[i0] + y[i4]) * (z[i3] + z[i4]) - (y[i3] + y[i4]) * (z[i0] + z[i4]) -
           (y[i2] + y[i5]) * (z[i3] + z[i5]) + (y[i3] + y[i5]) * (z[i2] + z[i5]);
   dvdy[out_idx] = -(x[i1] + x[i2]) * (z[i0] + z[i1]) + (x[i0] + x[i1]) * (z[i1] + z[i2]) -
           (x[i0] + x[i4]) * (z[i3] + z[i4]) + (x[i3] + x[i4]) * (z[i0] + z[i4]) +
           (x[i2] + x[i5]) * (z[i3] + z[i5]) - (x[i3] + x[i5]) * (z[i2] + z[i5]);
 
   dvdz[out_idx] = -(y[i1] + y[i2]) * (x[i0] + x[i1]) + (y[i0] + y[i1]) * (x[i1] + x[i2]) -
           (y[i0] + y[i4]) * (x[i3] + x[i4]) + (y[i3] + y[i4]) * (x[i0] + x[i4]) +
           (y[i2] + y[i5]) * (x[i3] + x[i5]) - (y[i3] + y[i5]) * (x[i2] + x[i5]);
 
   dvdx[out_idx] *= twelfth;
   dvdy[out_idx] *= twelfth;
   dvdz[out_idx] *= twelfth;
 }
 
 /******************************************/
 
 static inline void CalcElemVolumeDerivative(Real_t *dvdx, Real_t *dvdy,
                                             Real_t *dvdz, const Real_t *x,
                                             const Real_t *y, const Real_t *z) {
   VoluDer(x, y, z, 1, 2, 3, 4, 5, 7, dvdx, dvdy, dvdz, 0);
   VoluDer(x, y, z, 0, 1, 2, 7, 4, 6, dvdx, dvdy, dvdz, 3);
   VoluDer(x, y, z, 3, 0, 1, 6, 7, 5, dvdx, dvdy, dvdz, 2);
   VoluDer(x, y, z, 2, 3, 0, 5, 6, 4, dvdx, dvdy, dvdz, 1);
   VoluDer(x, y, z, 7, 6, 5, 0, 3, 1, dvdx, dvdy, dvdz, 4);
   VoluDer(x, y, z, 4, 7, 6, 1, 0, 2, dvdx, dvdy, dvdz, 5);
   VoluDer(x, y, z, 5, 4, 7, 2, 1, 3, dvdx, dvdy, dvdz, 6);
   VoluDer(x, y, z, 6, 5, 4, 3, 2, 0, dvdx, dvdy, dvdz, 7);
 }
 
 /******************************************/
 
 static inline void CalcElemFBHourglassForce(Real_t *xd, Real_t *yd, Real_t *zd,
                                             Real_t **hourgam,
                                             Real_t coefficient, Real_t *hgfx,
                                             Real_t *hgfy, Real_t *hgfz) {
   Real_t hxx[4];
   for (Index_t i = 0; i < 4; i++) {
     hxx[i] = hourgam[0][i] * xd[0] + hourgam[1][i] * xd[1] +
              hourgam[2][i] * xd[2] + hourgam[3][i] * xd[3] +
              hourgam[4][i] * xd[4] + hourgam[5][i] * xd[5] +
              hourgam[6][i] * xd[6] + hourgam[7][i] * xd[7];
   }
   for (Index_t i = 0; i < 8; i++) {
     hgfx[i] = coefficient * (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
                              hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
   }
   for (Index_t i = 0; i < 4; i++) {
     hxx[i] = hourgam[0][i] * yd[0] + hourgam[1][i] * yd[1] +
              hourgam[2][i] * yd[2] + hourgam[3][i] * yd[3] +
              hourgam[4][i] * yd[4] + hourgam[5][i] * yd[5] +
              hourgam[6][i] * yd[6] + hourgam[7][i] * yd[7];
   }
   for (Index_t i = 0; i < 8; i++) {
     hgfy[i] = coefficient * (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
                              hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
   }
   for (Index_t i = 0; i < 4; i++) {
     hxx[i] = hourgam[0][i] * zd[0] + hourgam[1][i] * zd[1] +
              hourgam[2][i] * zd[2] + hourgam[3][i] * zd[3] +
              hourgam[4][i] * zd[4] + hourgam[5][i] * zd[5] +
              hourgam[6][i] * zd[6] + hourgam[7][i] * zd[7];
   }
   for (Index_t i = 0; i < 8; i++) {
     hgfz[i] = coefficient * (hourgam[i][0] * hxx[0] + hourgam[i][1] * hxx[1] +
                              hourgam[i][2] * hxx[2] + hourgam[i][3] * hxx[3]);
   }
 }
 
 /******************************************/
 
 static inline void CalcFBHourglassForceForElems(
     Index_t *nodelist, Real_t *ss, Real_t *elemMass, Real_t *xd, Real_t *yd,
     Real_t *zd, Real_t *fx, Real_t *fy, Real_t *fz, Index_t *nodeElemStart,
     Index_t *nodeElemCornerList, Real_t *determ, Real_t *x8n, Real_t *y8n,
     Real_t *z8n, Real_t *dvdx, Real_t *dvdy, Real_t *dvdz, Real_t hourg,
     Index_t numElem, Index_t numNode) {
   /*************************************************
    *
    *     FUNCTION: Calculates the Flanagan-Belytschko anti-hourglass
    *               force.
    *
    *************************************************/
 
   Index_t numElem8 = numElem * 8;
   Real_t *fx_elem = (Real_t *)malloc(numElem8 * sizeof(Real_t));
   Real_t *fy_elem = (Real_t *)malloc(numElem8 * sizeof(Real_t));
   Real_t *fz_elem = (Real_t *)malloc(numElem8 * sizeof(Real_t));
 
   Real_t gamma[4][8];
 
   gamma[0][0] = Real_t(1.);
   gamma[0][1] = Real_t(1.);
   gamma[0][2] = Real_t(-1.);
   gamma[0][3] = Real_t(-1.);
   gamma[0][4] = Real_t(-1.);
   gamma[0][5] = Real_t(-1.);
   gamma[0][6] = Real_t(1.);
   gamma[0][7] = Real_t(1.);
   gamma[1][0] = Real_t(1.);
   gamma[1][1] = Real_t(-1.);
   gamma[1][2] = Real_t(-1.);
   gamma[1][3] = Real_t(1.);
   gamma[1][4] = Real_t(-1.);
   gamma[1][5] = Real_t(1.);
   gamma[1][6] = Real_t(1.);
   gamma[1][7] = Real_t(-1.);
   gamma[2][0] = Real_t(1.);
   gamma[2][1] = Real_t(-1.);
   gamma[2][2] = Real_t(1.);
   gamma[2][3] = Real_t(-1.);
   gamma[2][4] = Real_t(1.);
   gamma[2][5] = Real_t(-1.);
   gamma[2][6] = Real_t(1.);
   gamma[2][7] = Real_t(-1.);
   gamma[3][0] = Real_t(-1.);
   gamma[3][1] = Real_t(1.);
   gamma[3][2] = Real_t(-1.);
   gamma[3][3] = Real_t(1.);
   gamma[3][4] = Real_t(1.);
   gamma[3][5] = Real_t(-1.);
   gamma[3][6] = Real_t(1.);
   gamma[3][7] = Real_t(-1.);
 
   /*************************************************/
   /*    compute the hourglass modes */
 
 #pragma omp parallel for firstprivate(numElem, hourg)
   for (Index_t i2 = 0; i2 < numElem; ++i2) {
     Real_t hgfx[8], hgfy[8], hgfz[8];
     Real_t coefficient;
     Real_t *hourgam[8];
     for (Index_t i = 0; i < 8; ++i) {
       hourgam[i] = (Real_t *)malloc(4 * sizeof(Real_t));
     }
 
     Real_t xd1[8], yd1[8], zd1[8];
 
     Index_t i3 = 8 * i2;
     Real_t volinv = Real_t(1.0) / determ[i2];
     Real_t ss1, mass1, volume13;
     for (Index_t i1 = 0; i1 < 4; ++i1) {
 
       Real_t hourmodx =
           x8n[i3] * gamma[i1][0] + x8n[i3 + 1] * gamma[i1][1] +
           x8n[i3 + 2] * gamma[i1][2] + x8n[i3 + 3] * gamma[i1][3] +
           x8n[i3 + 4] * gamma[i1][4] + x8n[i3 + 5] * gamma[i1][5] +
           x8n[i3 + 6] * gamma[i1][6] + x8n[i3 + 7] * gamma[i1][7];
 
       Real_t hourmody =
           y8n[i3] * gamma[i1][0] + y8n[i3 + 1] * gamma[i1][1] +
           y8n[i3 + 2] * gamma[i1][2] + y8n[i3 + 3] * gamma[i1][3] +
           y8n[i3 + 4] * gamma[i1][4] + y8n[i3 + 5] * gamma[i1][5] +
           y8n[i3 + 6] * gamma[i1][6] + y8n[i3 + 7] * gamma[i1][7];
 
       Real_t hourmodz =
           z8n[i3] * gamma[i1][0] + z8n[i3 + 1] * gamma[i1][1] +
           z8n[i3 + 2] * gamma[i1][2] + z8n[i3 + 3] * gamma[i1][3] +
           z8n[i3 + 4] * gamma[i1][4] + z8n[i3 + 5] * gamma[i1][5] +
           z8n[i3 + 6] * gamma[i1][6] + z8n[i3 + 7] * gamma[i1][7];
 
       hourgam[0][i1] =
           gamma[i1][0] - volinv * (dvdx[i3] * hourmodx + dvdy[i3] * hourmody +
                                    dvdz[i3] * hourmodz);
 
       hourgam[1][i1] = gamma[i1][1] - volinv * (dvdx[i3 + 1] * hourmodx +
                                                 dvdy[i3 + 1] * hourmody +
                                                 dvdz[i3 + 1] * hourmodz);
 
       hourgam[2][i1] = gamma[i1][2] - volinv * (dvdx[i3 + 2] * hourmodx +
                                                 dvdy[i3 + 2] * hourmody +
                                                 dvdz[i3 + 2] * hourmodz);
 
       hourgam[3][i1] = gamma[i1][3] - volinv * (dvdx[i3 + 3] * hourmodx +
                                                 dvdy[i3 + 3] * hourmody +
                                                 dvdz[i3 + 3] * hourmodz);
 
       hourgam[4][i1] = gamma[i1][4] - volinv * (dvdx[i3 + 4] * hourmodx +
                                                 dvdy[i3 + 4] * hourmody +
                                                 dvdz[i3 + 4] * hourmodz);
 
       hourgam[5][i1] = gamma[i1][5] - volinv * (dvdx[i3 + 5] * hourmodx +
                                                 dvdy[i3 + 5] * hourmody +
                                                 dvdz[i3 + 5] * hourmodz);
 
       hourgam[6][i1] = gamma[i1][6] - volinv * (dvdx[i3 + 6] * hourmodx +
                                                 dvdy[i3 + 6] * hourmody +
                                                 dvdz[i3 + 6] * hourmodz);
 
       hourgam[7][i1] = gamma[i1][7] - volinv * (dvdx[i3 + 7] * hourmodx +
                                                 dvdy[i3 + 7] * hourmody +
                                                 dvdz[i3 + 7] * hourmodz);
     }
 
     /* compute forces */
     /* store forces into h arrays (force arrays) */
 
     ss1 = ss[i2];
     mass1 = elemMass[i2];
     volume13 = CBRT(determ[i2]);
 
     Index_t n0si2 = nodelist[i3 + 0];
     Index_t n1si2 = nodelist[i3 + 1];
     Index_t n2si2 = nodelist[i3 + 2];
     Index_t n3si2 = nodelist[i3 + 3];
     Index_t n4si2 = nodelist[i3 + 4];
     Index_t n5si2 = nodelist[i3 + 5];
     Index_t n6si2 = nodelist[i3 + 6];
     Index_t n7si2 = nodelist[i3 + 7];
 
     xd1[0] = xd[n0si2];
     xd1[1] = xd[n1si2];
     xd1[2] = xd[n2si2];
     xd1[3] = xd[n3si2];
     xd1[4] = xd[n4si2];
     xd1[5] = xd[n5si2];
     xd1[6] = xd[n6si2];
     xd1[7] = xd[n7si2];
 
     yd1[0] = yd[n0si2];
     yd1[1] = yd[n1si2];
     yd1[2] = yd[n2si2];
     yd1[3] = yd[n3si2];
     yd1[4] = yd[n4si2];
     yd1[5] = yd[n5si2];
     yd1[6] = yd[n6si2];
     yd1[7] = yd[n7si2];
 
     zd1[0] = zd[n0si2];
     zd1[1] = zd[n1si2];
     zd1[2] = zd[n2si2];
     zd1[3] = zd[n3si2];
     zd1[4] = zd[n4si2];
     zd1[5] = zd[n5si2];
     zd1[6] = zd[n6si2];
     zd1[7] = zd[n7si2];
 
     coefficient = -hourg * Real_t(0.01) * ss1 * mass1 / volume13;
 
     CalcElemFBHourglassForce(xd1, yd1, zd1, hourgam, coefficient, hgfx, hgfy,
                              hgfz);
 
     // Write into per-element arrays using direct indexing
     fx_elem[i3 + 0] = hgfx[0];
     fx_elem[i3 + 1] = hgfx[1];
     fx_elem[i3 + 2] = hgfx[2];
     fx_elem[i3 + 3] = hgfx[3];
     fx_elem[i3 + 4] = hgfx[4];
     fx_elem[i3 + 5] = hgfx[5];
     fx_elem[i3 + 6] = hgfx[6];
     fx_elem[i3 + 7] = hgfx[7];
 
     fy_elem[i3 + 0] = hgfy[0];
     fy_elem[i3 + 1] = hgfy[1];
     fy_elem[i3 + 2] = hgfy[2];
     fy_elem[i3 + 3] = hgfy[3];
     fy_elem[i3 + 4] = hgfy[4];
     fy_elem[i3 + 5] = hgfy[5];
     fy_elem[i3 + 6] = hgfy[6];
     fy_elem[i3 + 7] = hgfy[7];
 
     fz_elem[i3 + 0] = hgfz[0];
     fz_elem[i3 + 1] = hgfz[1];
     fz_elem[i3 + 2] = hgfz[2];
     fz_elem[i3 + 3] = hgfz[3];
     fz_elem[i3 + 4] = hgfz[4];
     fz_elem[i3 + 5] = hgfz[5];
     fz_elem[i3 + 6] = hgfz[6];
     fz_elem[i3 + 7] = hgfz[7];
 
     for (Index_t i = 0; i < 8; ++i) {
       free(hourgam[i]);
     }
   }
 
   // Collect the data from the local arrays into the final force arrays
 #pragma omp parallel for firstprivate(numNode)
   for (Index_t gnode = 0; gnode < numNode; ++gnode) {
     Index_t start = nodeElemStart[gnode];
     Index_t count = nodeElemStart[gnode + 1] - start;
     Real_t fx_tmp = Real_t(0.0);
     Real_t fy_tmp = Real_t(0.0);
     Real_t fz_tmp = Real_t(0.0);
     for (Index_t i = 0; i < count; ++i) {
       Index_t elem = nodeElemCornerList[start + i];
       fx_tmp += fx_elem[elem];
       fy_tmp += fy_elem[elem];
       fz_tmp += fz_elem[elem];
     }
     fx[gnode] += fx_tmp;
     fy[gnode] += fy_tmp;
     fz[gnode] += fz_tmp;
   }
 
   free(fz_elem);
   free(fy_elem);
   free(fx_elem);
 }
 
 /******************************************/
 
 static inline void CalcHourglassControlForElems(
     Real_t *x, Real_t *y, Real_t *z, Real_t *fx, Real_t *fy, Real_t *fz,
     Real_t *xd, Real_t *yd, Real_t *zd, Index_t *nodelist, Real_t *volo,
     Real_t *v, Real_t *ss, Real_t *elemMass, Index_t *nodeElemStart,
     Index_t *nodeElemCornerList, Real_t *determ, Real_t hgcoef, Index_t numElem,
     Index_t numNode) {
   Index_t numElem8 = numElem * 8;
   Real_t *dvdx = (Real_t *)malloc(numElem8 * sizeof(Real_t));
   Real_t *dvdy = (Real_t *)malloc(numElem8 * sizeof(Real_t));
   Real_t *dvdz = (Real_t *)malloc(numElem8 * sizeof(Real_t));
   Real_t *x8n = (Real_t *)malloc(numElem8 * sizeof(Real_t));
   Real_t *y8n = (Real_t *)malloc(numElem8 * sizeof(Real_t));
   Real_t *z8n = (Real_t *)malloc(numElem8 * sizeof(Real_t));
 
   /* start loop over elements */
 #pragma omp parallel for firstprivate(numElem)
   for (Index_t i = 0; i < numElem; ++i) {
     Real_t x1[8], y1[8], z1[8];
     Real_t pfx[8], pfy[8], pfz[8];
 
     CollectDomainNodesToElemNodes(x, y, z, nodelist, i * 8, x1, y1, z1);
 
     CalcElemVolumeDerivative(pfx, pfy, pfz, x1, y1, z1);
 
     /* load into temporary storage for FB Hour Glass control */
     for (Index_t ii = 0; ii < 8; ++ii) {
       Index_t jj = 8 * i + ii;
 
       dvdx[jj] = pfx[ii];
       dvdy[jj] = pfy[ii];
       dvdz[jj] = pfz[ii];
       x8n[jj] = x1[ii];
       y8n[jj] = y1[ii];
       z8n[jj] = z1[ii];
     }
 
     determ[i] = volo[i] * v[i];
 
     /* Do a check for negative volumes */
     if (v[i] <= Real_t(0.0)) {
       exit(VolumeError);
     }
   }
 
   if (hgcoef > Real_t(0.)) {
     CalcFBHourglassForceForElems(nodelist, ss, elemMass, xd, yd, zd, fx, fy, fz,
                                  nodeElemStart, nodeElemCornerList, determ, x8n,
                                  y8n, z8n, dvdx, dvdy, dvdz, hgcoef, numElem,
                                  numNode);
   }
 
   free(z8n);
   free(y8n);
   free(x8n);
   free(dvdz);
   free(dvdy);
   free(dvdx);
 
   return;
 }
 
 /******************************************/
 
 static inline void
 CalcVolumeForceForElems(Real_t *x, Real_t *y, Real_t *z, Real_t *fx, Real_t *fy,
                         Real_t *fz, Real_t *xd, Real_t *yd, Real_t *zd,
                         Index_t *nodelist, Real_t *volo, Real_t *v, Real_t *p,
                         Real_t *q, Real_t *ss, Real_t *elemMass,
                         Index_t *nodeElemStart, Index_t *nodeElemCornerList,
                         Real_t hgcoef, Index_t numElem, Index_t numNode) {
   if (numElem != 0) {
     Real_t *sigxx = (Real_t *)malloc(numElem * sizeof(Real_t));
     Real_t *sigyy = (Real_t *)malloc(numElem * sizeof(Real_t));
     Real_t *sigzz = (Real_t *)malloc(numElem * sizeof(Real_t));
     Real_t *determ = (Real_t *)malloc(numElem * sizeof(Real_t));
 
     /* Sum contributions to total stress tensor */
     InitStressTermsForElems(p, q, sigxx, sigyy, sigzz, numElem);
 
     // call elemlib stress integration loop to produce nodal forces from
     // material stresses.
     IntegrateStressForElems(x, y, z, fx, fy, fz, nodelist, nodeElemStart,
                             nodeElemCornerList, sigxx, sigyy, sigzz, determ,
                             numElem, numNode);
 
     // check for negative element volume
 #pragma omp parallel for firstprivate(numElem)
     for (Index_t k = 0; k < numElem; ++k) {
       if (determ[k] <= Real_t(0.0)) {
         exit(VolumeError);
       }
     }
 
     CalcHourglassControlForElems(
         x, y, z, fx, fy, fz, xd, yd, zd, nodelist, volo, v, ss, elemMass,
         nodeElemStart, nodeElemCornerList, determ, hgcoef, numElem, numNode);
 
     free(determ);
     free(sigzz);
     free(sigyy);
     free(sigxx);
   }
 }
 
 /******************************************/
 
 static inline void CalcForceForNodes(Real_t *x, Real_t *y, Real_t *z,
                                      Real_t *fx, Real_t *fy, Real_t *fz,
                                      Real_t *xd, Real_t *yd, Real_t *zd,
                                      Index_t *nodelist, Real_t *volo, Real_t *v,
                                      Real_t *p, Real_t *q, Real_t *ss,
                                      Real_t *elemMass, Index_t *nodeElemStart,
                                      Index_t *nodeElemCornerList, Real_t hgcoef,
                                      Index_t numElem, Index_t numNode) {
 #pragma omp parallel for firstprivate(numNode)
   for (Index_t i = 0; i < numNode; ++i) {
     fx[i] = Real_t(0.0);
     fy[i] = Real_t(0.0);
     fz[i] = Real_t(0.0);
   }
 
   /* Calcforce calls partial, force, hourq */
   CalcVolumeForceForElems(x, y, z, fx, fy, fz, xd, yd, zd, nodelist, volo, v, p,
                           q, ss, elemMass, nodeElemStart, nodeElemCornerList,
                           hgcoef, numElem, numNode);
 }
 
 /******************************************/
 
 static inline void CalcAccelerationForNodes(Real_t *xdd, Real_t *ydd,
                                             Real_t *zdd, Real_t *fx, Real_t *fy,
                                             Real_t *fz, Real_t *nodalMass,
                                             Index_t numNode) {
 #pragma omp parallel for firstprivate(numNode)
   for (Index_t i = 0; i < numNode; ++i) {
     xdd[i] = fx[i] / nodalMass[i];
     ydd[i] = fy[i] / nodalMass[i];
     zdd[i] = fz[i] / nodalMass[i];
   }
 }
 
 /******************************************/
 
 static inline void ApplyAccelerationBoundaryConditionsForNodes(
     Real_t *xdd, Real_t *ydd, Real_t *zdd, Index_t *symmX, Index_t *symmY,
     Index_t *symmZ, Index_t symmXempty, Index_t symmYempty, Index_t symmZempty,
     Index_t sizeX) {
   Index_t numNodeBC = (sizeX + 1) * (sizeX + 1);
 
 #pragma omp parallel
   {
     if (!symmXempty) {
 #pragma omp for nowait firstprivate(numNodeBC)
       for (Index_t i = 0; i < numNodeBC; ++i)
         xdd[symmX[i]] = Real_t(0.0);
     }
 
     if (!symmYempty) {
 #pragma omp for nowait firstprivate(numNodeBC)
       for (Index_t i = 0; i < numNodeBC; ++i)
         ydd[symmY[i]] = Real_t(0.0);
     }
 
     if (!symmZempty) {
 #pragma omp for nowait firstprivate(numNodeBC)
       for (Index_t i = 0; i < numNodeBC; ++i)
         zdd[symmZ[i]] = Real_t(0.0);
     }
   }
 }
 
 /******************************************/
 
 static inline void CalcVelocityForNodes(Real_t *xd, Real_t *yd, Real_t *zd,
                                         Real_t *xdd, Real_t *ydd, Real_t *zdd,
                                         Real_t deltatime, Real_t u_cut,
                                         Index_t numNode) {
 #pragma omp parallel for firstprivate(numNode, deltatime, u_cut)
   for (Index_t i = 0; i < numNode; ++i) {
     Real_t xdtmp, ydtmp, zdtmp;
 
     xdtmp = xd[i] + xdd[i] * deltatime;
     if (FABS(xdtmp) < u_cut)
       xdtmp = Real_t(0.0);
     xd[i] = xdtmp;
 
     ydtmp = yd[i] + ydd[i] * deltatime;
     if (FABS(ydtmp) < u_cut)
       ydtmp = Real_t(0.0);
     yd[i] = ydtmp;
 
     zdtmp = zd[i] + zdd[i] * deltatime;
     if (FABS(zdtmp) < u_cut)
       zdtmp = Real_t(0.0);
     zd[i] = zdtmp;
   }
 }
 
 /******************************************/
 
 static inline void CalcPositionForNodes(Real_t *x, Real_t *y, Real_t *z,
                                         Real_t *xd, Real_t *yd, Real_t *zd,
                                         Real_t deltatime, Index_t numNode) {
 #pragma omp parallel for firstprivate(numNode, deltatime)
   for (Index_t i = 0; i < numNode; ++i) {
     x[i] += xd[i] * deltatime;
     y[i] += yd[i] * deltatime;
     z[i] += zd[i] * deltatime;
   }
 }
 
 /******************************************/
 
 static inline void
 LagrangeNodal(Real_t *x, Real_t *y, Real_t *z, Real_t *xd, Real_t *yd,
               Real_t *zd, Real_t *xdd, Real_t *ydd, Real_t *zdd, Real_t *fx,
               Real_t *fy, Real_t *fz, Real_t *nodalMass, Index_t *nodelist,
               Real_t *volo, Real_t *v, Real_t *p, Real_t *q, Real_t *ss,
               Real_t *elemMass, Index_t *nodeElemStart,
               Index_t *nodeElemCornerList, Index_t *symmX, Index_t *symmY,
               Index_t *symmZ, Index_t symmXempty, Index_t symmYempty,
               Index_t symmZempty, Real_t deltatime, Real_t u_cut, Real_t hgcoef,
               Index_t numElem, Index_t numNode, Index_t sizeX) {
 
   /* time of boundary condition evaluation is beginning of step for force and
    * acceleration boundary conditions. */
   CalcForceForNodes(x, y, z, fx, fy, fz, xd, yd, zd, nodelist, volo, v, p, q,
                     ss, elemMass, nodeElemStart, nodeElemCornerList, hgcoef,
                     numElem, numNode);
 
   CalcAccelerationForNodes(xdd, ydd, zdd, fx, fy, fz, nodalMass, numNode);
 
   ApplyAccelerationBoundaryConditionsForNodes(xdd, ydd, zdd, symmX, symmY,
                                               symmZ, symmXempty, symmYempty,
                                               symmZempty, sizeX);
 
   CalcVelocityForNodes(xd, yd, zd, xdd, ydd, zdd, deltatime, u_cut, numNode);
 
   CalcPositionForNodes(x, y, z, xd, yd, zd, deltatime, numNode);
 
   return;
 }
 
 /******************************************/
 
 static inline Real_t CalcElemVolume(
     const Real_t x0, const Real_t x1, const Real_t x2, const Real_t x3,
     const Real_t x4, const Real_t x5, const Real_t x6, const Real_t x7,
     const Real_t y0, const Real_t y1, const Real_t y2, const Real_t y3,
     const Real_t y4, const Real_t y5, const Real_t y6, const Real_t y7,
     const Real_t z0, const Real_t z1, const Real_t z2, const Real_t z3,
     const Real_t z4, const Real_t z5, const Real_t z6, const Real_t z7) {
   Real_t twelveth = Real_t(1.0) / Real_t(12.0);
 
   Real_t dx61 = x6 - x1;
   Real_t dy61 = y6 - y1;
   Real_t dz61 = z6 - z1;
 
   Real_t dx70 = x7 - x0;
   Real_t dy70 = y7 - y0;
   Real_t dz70 = z7 - z0;
 
   Real_t dx63 = x6 - x3;
   Real_t dy63 = y6 - y3;
   Real_t dz63 = z6 - z3;
 
   Real_t dx20 = x2 - x0;
   Real_t dy20 = y2 - y0;
   Real_t dz20 = z2 - z0;
 
   Real_t dx50 = x5 - x0;
   Real_t dy50 = y5 - y0;
   Real_t dz50 = z5 - z0;
 
   Real_t dx64 = x6 - x4;
   Real_t dy64 = y6 - y4;
   Real_t dz64 = z6 - z4;
 
   Real_t dx31 = x3 - x1;
   Real_t dy31 = y3 - y1;
   Real_t dz31 = z3 - z1;
 
   Real_t dx72 = x7 - x2;
   Real_t dy72 = y7 - y2;
   Real_t dz72 = z7 - z2;
 
   Real_t dx43 = x4 - x3;
   Real_t dy43 = y4 - y3;
   Real_t dz43 = z4 - z3;
 
   Real_t dx57 = x5 - x7;
   Real_t dy57 = y5 - y7;
   Real_t dz57 = z5 - z7;
 
   Real_t dx14 = x1 - x4;
   Real_t dy14 = y1 - y4;
   Real_t dz14 = z1 - z4;
 
   Real_t dx25 = x2 - x5;
   Real_t dy25 = y2 - y5;
   Real_t dz25 = z2 - z5;
 
 #define TRIPLE_PRODUCT(x1, y1, z1, x2, y2, z2, x3, y3, z3)                     \
   ((x1) * ((y2) * (z3) - (z2) * (y3)) + (x2) * ((z1) * (y3) - (y1) * (z3)) +   \
    (x3) * ((y1) * (z2) - (z1) * (y2)))
 
   Real_t volume = TRIPLE_PRODUCT(dx31 + dx72, dx63, dx20, dy31 + dy72, dy63,
                                  dy20, dz31 + dz72, dz63, dz20) +
                   TRIPLE_PRODUCT(dx43 + dx57, dx64, dx70, dy43 + dy57, dy64,
                                  dy70, dz43 + dz57, dz64, dz70) +
                   TRIPLE_PRODUCT(dx14 + dx25, dx61, dx50, dy14 + dy25, dy61,
                                  dy50, dz14 + dz25, dz61, dz50);
 
 #undef TRIPLE_PRODUCT
 
   volume *= twelveth;
 
   return volume;
 }
 
 /******************************************/
 
 // inline
 Real_t CalcElemVolume(const Real_t *x, const Real_t *y, const Real_t *z) {
   return CalcElemVolume(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], y[0],
                         y[1], y[2], y[3], y[4], y[5], y[6], y[7], z[0], z[1],
                         z[2], z[3], z[4], z[5], z[6], z[7]);
 }
 
 /******************************************/
 
 static inline Real_t AreaFace(const Real_t x0, const Real_t x1, const Real_t x2,
                               const Real_t x3, const Real_t y0, const Real_t y1,
                               const Real_t y2, const Real_t y3, const Real_t z0,
                               const Real_t z1, const Real_t z2,
                               const Real_t z3) {
   Real_t fx = (x2 - x0) - (x3 - x1);
   Real_t fy = (y2 - y0) - (y3 - y1);
   Real_t fz = (z2 - z0) - (z3 - z1);
   Real_t gx = (x2 - x0) + (x3 - x1);
   Real_t gy = (y2 - y0) + (y3 - y1);
   Real_t gz = (z2 - z0) + (z3 - z1);
   Real_t area = (fx * fx + fy * fy + fz * fz) * (gx * gx + gy * gy + gz * gz) -
                 (fx * gx + fy * gy + fz * gz) * (fx * gx + fy * gy + fz * gz);
   return area;
 }
 
 /******************************************/
 
 static inline Real_t CalcElemCharacteristicLength(const Real_t *x,
                                                   const Real_t *y,
                                                   const Real_t *z,
                                                   const Real_t volume) {
   Real_t a, charLength = Real_t(0.0);
 
   a = AreaFace(x[0], x[1], x[2], x[3], y[0], y[1], y[2], y[3], z[0], z[1], z[2],
                z[3]);
   charLength = MAX(a, charLength);
 
   a = AreaFace(x[4], x[5], x[6], x[7], y[4], y[5], y[6], y[7], z[4], z[5], z[6],
                z[7]);
   charLength = MAX(a, charLength);
 
   a = AreaFace(x[0], x[1], x[5], x[4], y[0], y[1], y[5], y[4], z[0], z[1], z[5],
                z[4]);
   charLength = MAX(a, charLength);
 
   a = AreaFace(x[1], x[2], x[6], x[5], y[1], y[2], y[6], y[5], z[1], z[2], z[6],
                z[5]);
   charLength = MAX(a, charLength);
 
   a = AreaFace(x[2], x[3], x[7], x[6], y[2], y[3], y[7], y[6], z[2], z[3], z[7],
                z[6]);
   charLength = MAX(a, charLength);
 
   a = AreaFace(x[3], x[0], x[4], x[7], y[3], y[0], y[4], y[7], z[3], z[0], z[4],
                z[7]);
   charLength = MAX(a, charLength);
 
   charLength = Real_t(4.0) * volume / SQRT(charLength);
 
   return charLength;
 }
 
 /******************************************/
 
 static inline void CalcElemVelocityGradient(Real_t *const xvel,
                                             Real_t *const yvel,
                                             Real_t *const zvel, Real_t **b,
                                             const Real_t detJ,
                                             Real_t *const d) {
   const Real_t inv_detJ = Real_t(1.0) / detJ;
   Real_t dyddx, dxddy, dzddx, dxddz, dzddy, dyddz;
   const Real_t *const pfx = b[0];
   const Real_t *const pfy = b[1];
   const Real_t *const pfz = b[2];
 
   d[0] =
       inv_detJ * (pfx[0] * (xvel[0] - xvel[6]) + pfx[1] * (xvel[1] - xvel[7]) +
                   pfx[2] * (xvel[2] - xvel[4]) + pfx[3] * (xvel[3] - xvel[5]));
 
   d[1] =
       inv_detJ * (pfy[0] * (yvel[0] - yvel[6]) + pfy[1] * (yvel[1] - yvel[7]) +
                   pfy[2] * (yvel[2] - yvel[4]) + pfy[3] * (yvel[3] - yvel[5]));
 
   d[2] =
       inv_detJ * (pfz[0] * (zvel[0] - zvel[6]) + pfz[1] * (zvel[1] - zvel[7]) +
                   pfz[2] * (zvel[2] - zvel[4]) + pfz[3] * (zvel[3] - zvel[5]));
 
   dyddx =
       inv_detJ * (pfx[0] * (yvel[0] - yvel[6]) + pfx[1] * (yvel[1] - yvel[7]) +
                   pfx[2] * (yvel[2] - yvel[4]) + pfx[3] * (yvel[3] - yvel[5]));
 
   dxddy =
       inv_detJ * (pfy[0] * (xvel[0] - xvel[6]) + pfy[1] * (xvel[1] - xvel[7]) +
                   pfy[2] * (xvel[2] - xvel[4]) + pfy[3] * (xvel[3] - xvel[5]));
 
   dzddx =
       inv_detJ * (pfx[0] * (zvel[0] - zvel[6]) + pfx[1] * (zvel[1] - zvel[7]) +
                   pfx[2] * (zvel[2] - zvel[4]) + pfx[3] * (zvel[3] - zvel[5]));
 
   dxddz =
       inv_detJ * (pfz[0] * (xvel[0] - xvel[6]) + pfz[1] * (xvel[1] - xvel[7]) +
                   pfz[2] * (xvel[2] - xvel[4]) + pfz[3] * (xvel[3] - xvel[5]));
 
   dzddy =
       inv_detJ * (pfy[0] * (zvel[0] - zvel[6]) + pfy[1] * (zvel[1] - zvel[7]) +
                   pfy[2] * (zvel[2] - zvel[4]) + pfy[3] * (zvel[3] - zvel[5]));
 
   dyddz =
       inv_detJ * (pfz[0] * (yvel[0] - yvel[6]) + pfz[1] * (yvel[1] - yvel[7]) +
                   pfz[2] * (yvel[2] - yvel[4]) + pfz[3] * (yvel[3] - yvel[5]));
   d[5] = Real_t(.5) * (dxddy + dyddx);
   d[4] = Real_t(.5) * (dxddz + dzddx);
   d[3] = Real_t(.5) * (dzddy + dyddz);
 }
 
 /******************************************/
 
 // static inline
 void CalcKinematicsForElems(Real_t *x, Real_t *y, Real_t *z, Real_t *xd,
                             Real_t *yd, Real_t *zd, Index_t *nodelist,
                             Real_t *volo, Real_t *v, Real_t *delv,
                             Real_t *arealg, Real_t *dxx, Real_t *dyy,
                             Real_t *dzz, Real_t *vnew, Real_t deltaTime,
                             Index_t numElem) {
 
   // loop over all elements
 #pragma omp parallel for firstprivate(numElem, deltaTime)
   for (Index_t k = 0; k < numElem; ++k) {
     Real_t *B[3]; /** shape function derivatives */
     B[0] = (Real_t *)malloc(8 * sizeof(Real_t));
     B[1] = (Real_t *)malloc(8 * sizeof(Real_t));
     B[2] = (Real_t *)malloc(8 * sizeof(Real_t));
     Real_t D[6];
     Real_t x_local[8];
     Real_t y_local[8];
     Real_t z_local[8];
     Real_t xd_local[8];
     Real_t yd_local[8];
     Real_t zd_local[8];
     Real_t detJ = Real_t(0.0);
 
     Real_t volume;
     Real_t relativeVolume;
 
     // get nodal coordinates from global arrays and copy into local arrays.
     CollectDomainNodesToElemNodes(x, y, z, nodelist, k * 8, x_local, y_local,
                                   z_local);
 
     // volume calculations
     volume = CalcElemVolume(x_local, y_local, z_local);
     relativeVolume = volume / volo[k];
     vnew[k] = relativeVolume;
     delv[k] = relativeVolume - v[k];
 
     // set characteristic length
     arealg[k] = CalcElemCharacteristicLength(x_local, y_local, z_local, volume);
 
     // get nodal velocities from global array and copy into local arrays.
     for (Index_t lnode = 0; lnode < 8; ++lnode) {
       Index_t gnode = nodelist[k * 8 + lnode];
       xd_local[lnode] = xd[gnode];
       yd_local[lnode] = yd[gnode];
       zd_local[lnode] = zd[gnode];
     }
 
     Real_t dt2 = Real_t(0.5) * deltaTime;
     for (Index_t j = 0; j < 8; ++j) {
       x_local[j] -= dt2 * xd_local[j];
       y_local[j] -= dt2 * yd_local[j];
       z_local[j] -= dt2 * zd_local[j];
     }
 
     CalcElemShapeFunctionDerivatives(x_local, y_local, z_local, B, &detJ);
 
     CalcElemVelocityGradient(xd_local, yd_local, zd_local, B, detJ, D);
 
     // put velocity gradient quantities into their global arrays.
     dxx[k] = D[0];
     dyy[k] = D[1];
     dzz[k] = D[2];
     free(B[0]);
     free(B[1]);
     free(B[2]);
   }
 }
 
 /******************************************/
 
 static inline void CalcLagrangeElements(Real_t *x, Real_t *y, Real_t *z,
                                         Real_t *xd, Real_t *yd, Real_t *zd,
                                         Index_t *nodelist, Real_t *volo,
                                         Real_t *v, Real_t *delv, Real_t *arealg,
                                         Real_t *vdov, Real_t *vnew,
                                         Real_t deltatime, Index_t numElem) {
   if (numElem > 0) {
     // AllocateStrains
     Real_t *dxx = (Real_t *)malloc(numElem * sizeof(Real_t));
     Real_t *dyy = (Real_t *)malloc(numElem * sizeof(Real_t));
     Real_t *dzz = (Real_t *)malloc(numElem * sizeof(Real_t));
 
     CalcKinematicsForElems(x, y, z, xd, yd, zd, nodelist, volo, v, delv, arealg,
                            dxx, dyy, dzz, vnew, deltatime, numElem);
 
     // element loop to do some stuff not included in the elemlib function.
 #pragma omp parallel for firstprivate(numElem)
     for (Index_t k = 0; k < numElem; ++k) {
       // calc strain rate and apply as constraint (only done in FB element)
       Real_t vdov_k = dxx[k] + dyy[k] + dzz[k];
       Real_t vdovthird = vdov_k / Real_t(3.0);
 
       // make the rate of deformation tensor deviatoric
       vdov[k] = vdov_k;
       dxx[k] -= vdovthird;
       dyy[k] -= vdovthird;
       dzz[k] -= vdovthird;
 
       // See if any volumes are negative, and take appropriate action.
       if (vnew[k] <= Real_t(0.0)) {
         exit(VolumeError);
       }
     }
     // DeallocateStrains
     free(dzz);
     free(dyy);
     free(dxx);
   }
 }
 
 /******************************************/
 
 static inline void CalcMonotonicQGradientsForElems(
     Real_t *x, Real_t *y, Real_t *z, Real_t *xd, Real_t *yd, Real_t *zd,
     Index_t *nodelist, Real_t *volo, Real_t *delx_zeta, Real_t *delv_zeta,
     Real_t *delx_xi, Real_t *delv_xi, Real_t *delx_eta, Real_t *delv_eta,
     Real_t *vnew, Index_t numElem) {
 
 #pragma omp parallel for firstprivate(numElem)
   for (Index_t i = 0; i < numElem; ++i) {
     const Real_t ptiny = Real_t(1.e-36);
     Real_t ax, ay, az;
     Real_t dxv, dyv, dzv;
 
     Index_t i8 = i * 8;
     Index_t n0 = nodelist[i8 + 0];
     Index_t n1 = nodelist[i8 + 1];
     Index_t n2 = nodelist[i8 + 2];
     Index_t n3 = nodelist[i8 + 3];
     Index_t n4 = nodelist[i8 + 4];
     Index_t n5 = nodelist[i8 + 5];
     Index_t n6 = nodelist[i8 + 6];
     Index_t n7 = nodelist[i8 + 7];
 
     Real_t x0 = x[n0];
     Real_t x1 = x[n1];
     Real_t x2 = x[n2];
     Real_t x3 = x[n3];
     Real_t x4 = x[n4];
     Real_t x5 = x[n5];
     Real_t x6 = x[n6];
     Real_t x7 = x[n7];
 
     Real_t y0 = y[n0];
     Real_t y1 = y[n1];
     Real_t y2 = y[n2];
     Real_t y3 = y[n3];
     Real_t y4 = y[n4];
     Real_t y5 = y[n5];
     Real_t y6 = y[n6];
     Real_t y7 = y[n7];
 
     Real_t z0 = z[n0];
     Real_t z1 = z[n1];
     Real_t z2 = z[n2];
     Real_t z3 = z[n3];
     Real_t z4 = z[n4];
     Real_t z5 = z[n5];
     Real_t z6 = z[n6];
     Real_t z7 = z[n7];
 
     Real_t xv0 = xd[n0];
     Real_t xv1 = xd[n1];
     Real_t xv2 = xd[n2];
     Real_t xv3 = xd[n3];
     Real_t xv4 = xd[n4];
     Real_t xv5 = xd[n5];
     Real_t xv6 = xd[n6];
     Real_t xv7 = xd[n7];
 
     Real_t yv0 = yd[n0];
     Real_t yv1 = yd[n1];
     Real_t yv2 = yd[n2];
     Real_t yv3 = yd[n3];
     Real_t yv4 = yd[n4];
     Real_t yv5 = yd[n5];
     Real_t yv6 = yd[n6];
     Real_t yv7 = yd[n7];
 
     Real_t zv0 = zd[n0];
     Real_t zv1 = zd[n1];
     Real_t zv2 = zd[n2];
     Real_t zv3 = zd[n3];
     Real_t zv4 = zd[n4];
     Real_t zv5 = zd[n5];
     Real_t zv6 = zd[n6];
     Real_t zv7 = zd[n7];
 
     Real_t vol = volo[i] * vnew[i];
     Real_t norm = Real_t(1.0) / (vol + ptiny);
 
     Real_t dxj = Real_t(-0.25) * ((x0 + x1 + x5 + x4) - (x3 + x2 + x6 + x7));
     Real_t dyj = Real_t(-0.25) * ((y0 + y1 + y5 + y4) - (y3 + y2 + y6 + y7));
     Real_t dzj = Real_t(-0.25) * ((z0 + z1 + z5 + z4) - (z3 + z2 + z6 + z7));
 
     Real_t dxi = Real_t(0.25) * ((x1 + x2 + x6 + x5) - (x0 + x3 + x7 + x4));
     Real_t dyi = Real_t(0.25) * ((y1 + y2 + y6 + y5) - (y0 + y3 + y7 + y4));
     Real_t dzi = Real_t(0.25) * ((z1 + z2 + z6 + z5) - (z0 + z3 + z7 + z4));
 
     Real_t dxk = Real_t(0.25) * ((x4 + x5 + x6 + x7) - (x0 + x1 + x2 + x3));
     Real_t dyk = Real_t(0.25) * ((y4 + y5 + y6 + y7) - (y0 + y1 + y2 + y3));
     Real_t dzk = Real_t(0.25) * ((z4 + z5 + z6 + z7) - (z0 + z1 + z2 + z3));
 
     /* find delvk and delxk ( i cross j ) */
 
     ax = dyi * dzj - dzi * dyj;
     ay = dzi * dxj - dxi * dzj;
     az = dxi * dyj - dyi * dxj;
 
     delx_zeta[i] = vol / SQRT(ax * ax + ay * ay + az * az + ptiny);
 
     ax *= norm;
     ay *= norm;
     az *= norm;
 
     dxv = Real_t(0.25) * ((xv4 + xv5 + xv6 + xv7) - (xv0 + xv1 + xv2 + xv3));
     dyv = Real_t(0.25) * ((yv4 + yv5 + yv6 + yv7) - (yv0 + yv1 + yv2 + yv3));
     dzv = Real_t(0.25) * ((zv4 + zv5 + zv6 + zv7) - (zv0 + zv1 + zv2 + zv3));
 
     delv_zeta[i] = ax * dxv + ay * dyv + az * dzv;
 
     /* find delxi and delvi ( j cross k ) */
 
     ax = dyj * dzk - dzj * dyk;
     ay = dzj * dxk - dxj * dzk;
     az = dxj * dyk - dyj * dxk;
 
     delx_xi[i] = vol / SQRT(ax * ax + ay * ay + az * az + ptiny);
 
     ax *= norm;
     ay *= norm;
     az *= norm;
 
     dxv = Real_t(0.25) * ((xv1 + xv2 + xv6 + xv5) - (xv0 + xv3 + xv7 + xv4));
     dyv = Real_t(0.25) * ((yv1 + yv2 + yv6 + yv5) - (yv0 + yv3 + yv7 + yv4));
     dzv = Real_t(0.25) * ((zv1 + zv2 + zv6 + zv5) - (zv0 + zv3 + zv7 + zv4));
 
     delv_xi[i] = ax * dxv + ay * dyv + az * dzv;
 
     /* find delxj and delvj ( k cross i ) */
 
     ax = dyk * dzi - dzk * dyi;
     ay = dzk * dxi - dxk * dzi;
     az = dxk * dyi - dyk * dxi;
 
     delx_eta[i] = vol / SQRT(ax * ax + ay * ay + az * az + ptiny);
 
     ax *= norm;
     ay *= norm;
     az *= norm;
 
     dxv = Real_t(-0.25) * ((xv0 + xv1 + xv5 + xv4) - (xv3 + xv2 + xv6 + xv7));
     dyv = Real_t(-0.25) * ((yv0 + yv1 + yv5 + yv4) - (yv3 + yv2 + yv6 + yv7));
     dzv = Real_t(-0.25) * ((zv0 + zv1 + zv5 + zv4) - (zv3 + zv2 + zv6 + zv7));
 
     delv_eta[i] = ax * dxv + ay * dyv + az * dzv;
   }
 }
 
 /******************************************/
 
 static inline void CalcMonotonicQRegionForElems(
     Index_t *regElemlist, Int_t *elemBC, Real_t *delv_xi, Real_t *delv_eta,
     Real_t *delv_zeta, Real_t *delx_xi, Real_t *delx_eta, Real_t *delx_zeta,
     Index_t *lxim, Index_t *lxip, Index_t *letam, Index_t *letap,
     Index_t *lzetam, Index_t *lzetap, Real_t *vdov, Real_t *elemMass,
     Real_t *volo, Real_t *qq, Real_t *ql, Real_t *vnew, Index_t regElemSize,
     Real_t monoq_limiter_mult, Real_t monoq_max_slope, Real_t qlc_monoq,
     Real_t qqc_monoq, Real_t ptiny) {
 
   // #pragma omp parallel for firstprivate(qlc_monoq, qqc_monoq,
   // monoq_limiter_mult, monoq_max_slope, ptiny)
   for (Index_t ielem = 0; ielem < regElemSize; ++ielem) {
     Index_t i = regElemlist[ielem];
     Real_t qlin, qquad;
     Real_t phixi, phieta, phizeta;
     Int_t bcMask = elemBC[i];
     Real_t delvm = 0.0, delvp = 0.0;
 
     /*  phixi     */
     Real_t norm = Real_t(1.) / (delv_xi[i] + ptiny);
 
     switch (bcMask & XI_M) {
     case XI_M_COMM: /* needs comm data */
     case 0:
       delvm = delv_xi[lxim[i]];
       break;
     case XI_M_SYMM:
       delvm = delv_xi[i];
       break;
     case XI_M_FREE:
       delvm = Real_t(0.0);
       break;
     default:
       fprintf(stderr, "Error in switch at %s line %d\n", __FILE__, __LINE__);
       delvm = 0; /* ERROR - but quiets the compiler */
       break;
     }
     switch (bcMask & XI_P) {
     case XI_P_COMM: /* needs comm data */
     case 0:
       delvp = delv_xi[lxip[i]];
       break;
     case XI_P_SYMM:
       delvp = delv_xi[i];
       break;
     case XI_P_FREE:
       delvp = Real_t(0.0);
       break;
     default:
       fprintf(stderr, "Error in switch at %s line %d\n", __FILE__, __LINE__);
       delvp = 0; /* ERROR - but quiets the compiler */
       break;
     }
 
     delvm = delvm * norm;
     delvp = delvp * norm;
 
     phixi = Real_t(.5) * (delvm + delvp);
 
     delvm *= monoq_limiter_mult;
     delvp *= monoq_limiter_mult;
 
     if (delvm < phixi)
       phixi = delvm;
     if (delvp < phixi)
       phixi = delvp;
     if (phixi < Real_t(0.))
       phixi = Real_t(0.);
     if (phixi > monoq_max_slope)
       phixi = monoq_max_slope;
 
     /*  phieta     */
     norm = Real_t(1.) / (delv_eta[i] + ptiny);
 
     switch (bcMask & ETA_M) {
     case ETA_M_COMM: /* needs comm data */
     case 0:
       delvm = delv_eta[letam[i]];
       break;
     case ETA_M_SYMM:
       delvm = delv_eta[i];
       break;
     case ETA_M_FREE:
       delvm = Real_t(0.0);
       break;
     default:
       fprintf(stderr, "Error in switch at %s line %d\n", __FILE__, __LINE__);
       delvm = 0; /* ERROR - but quiets the compiler */
       break;
     }
     switch (bcMask & ETA_P) {
     case ETA_P_COMM: /* needs comm data */
     case 0:
       delvp = delv_eta[letap[i]];
       break;
     case ETA_P_SYMM:
       delvp = delv_eta[i];
       break;
     case ETA_P_FREE:
       delvp = Real_t(0.0);
       break;
     default:
       fprintf(stderr, "Error in switch at %s line %d\n", __FILE__, __LINE__);
       delvp = 0; /* ERROR - but quiets the compiler */
       break;
     }
 
     delvm = delvm * norm;
     delvp = delvp * norm;
 
     phieta = Real_t(.5) * (delvm + delvp);
 
     delvm *= monoq_limiter_mult;
     delvp *= monoq_limiter_mult;
 
     if (delvm < phieta)
       phieta = delvm;
     if (delvp < phieta)
       phieta = delvp;
     if (phieta < Real_t(0.))
       phieta = Real_t(0.);
     if (phieta > monoq_max_slope)
       phieta = monoq_max_slope;
 
     /*  phizeta     */
     norm = Real_t(1.) / (delv_zeta[i] + ptiny);
 
     switch (bcMask & ZETA_M) {
     case ZETA_M_COMM: /* needs comm data */
     case 0:
       delvm = delv_zeta[lzetam[i]];
       break;
     case ZETA_M_SYMM:
       delvm = delv_zeta[i];
       break;
     case ZETA_M_FREE:
       delvm = Real_t(0.0);
       break;
     default:
       fprintf(stderr, "Error in switch at %s line %d\n", __FILE__, __LINE__);
       delvm = 0; /* ERROR - but quiets the compiler */
       break;
     }
     switch (bcMask & ZETA_P) {
     case ZETA_P_COMM: /* needs comm data */
     case 0:
       delvp = delv_zeta[lzetap[i]];
       break;
     case ZETA_P_SYMM:
       delvp = delv_zeta[i];
       break;
     case ZETA_P_FREE:
       delvp = Real_t(0.0);
       break;
     default:
       fprintf(stderr, "Error in switch at %s line %d\n", __FILE__, __LINE__);
       delvp = 0; /* ERROR - but quiets the compiler */
       break;
     }
 
     delvm = delvm * norm;
     delvp = delvp * norm;
 
     phizeta = Real_t(.5) * (delvm + delvp);
 
     delvm *= monoq_limiter_mult;
     delvp *= monoq_limiter_mult;
 
     if (delvm < phizeta)
       phizeta = delvm;
     if (delvp < phizeta)
       phizeta = delvp;
     if (phizeta < Real_t(0.))
       phizeta = Real_t(0.);
     if (phizeta > monoq_max_slope)
       phizeta = monoq_max_slope;
 
     /* Remove length scale */
 
     if (vdov[i] > Real_t(0.)) {
       qlin = Real_t(0.);
       qquad = Real_t(0.);
     } else {
       Real_t delvxxi = delv_xi[i] * delx_xi[i];
       Real_t delvxeta = delv_eta[i] * delx_eta[i];
       Real_t delvxzeta = delv_zeta[i] * delx_zeta[i];
 
       if (delvxxi > Real_t(0.))
         delvxxi = Real_t(0.);
       if (delvxeta > Real_t(0.))
         delvxeta = Real_t(0.);
       if (delvxzeta > Real_t(0.))
         delvxzeta = Real_t(0.);
 
       Real_t rho = elemMass[i] / (volo[i] * vnew[i]);
 
       qlin =
           -qlc_monoq * rho *
           (delvxxi * (Real_t(1.) - phixi) + delvxeta * (Real_t(1.) - phieta) +
            delvxzeta * (Real_t(1.) - phizeta));
 
       qquad = qqc_monoq * rho *
               (delvxxi * delvxxi * (Real_t(1.) - phixi * phixi) +
                delvxeta * delvxeta * (Real_t(1.) - phieta * phieta) +
                delvxzeta * delvxzeta * (Real_t(1.) - phizeta * phizeta));
     }
     qq[i] = qquad;
     ql[i] = qlin;
   }
 }
 
 /******************************************/
 
 static inline void CalcMonotonicQForElems(
     Index_t numReg, Index_t *regElemSize, Index_t **regElemlist, Int_t *elemBC,
     Real_t *delv_xi, Real_t *delv_eta, Real_t *delv_zeta, Real_t *delx_xi,
     Real_t *delx_eta, Real_t *delx_zeta, Index_t *lxim, Index_t *lxip,
     Index_t *letam, Index_t *letap, Index_t *lzetam, Index_t *lzetap,
     Real_t *vdov, Real_t *elemMass, Real_t *volo, Real_t *qq, Real_t *ql,
     Real_t *vnew, Real_t monoq_limiter_mult, Real_t monoq_max_slope,
     Real_t qlc_monoq, Real_t qqc_monoq) {
   //
   // initialize parameters
   //
   const Real_t ptiny = Real_t(1.e-36);
 
   //
   // calculate the monotonic q for all regions
   //
   for (Index_t r = numReg - 1; r >= 0; --r) {
 
     if (regElemSize[r] > 0) {
       CalcMonotonicQRegionForElems(
           regElemlist[r], elemBC, delv_xi, delv_eta, delv_zeta, delx_xi,
           delx_eta, delx_zeta, lxim, lxip, letam, letap, lzetam, lzetap, vdov,
           elemMass, volo, qq, ql, vnew, regElemSize[r], monoq_limiter_mult,
           monoq_max_slope, qlc_monoq, qqc_monoq, ptiny);
     }
   }
 }
 
 /******************************************/
 
 static inline void
 CalcQForElems(Real_t *x, Real_t *y, Real_t *z, Real_t *xd, Real_t *yd,
               Real_t *zd, Index_t *nodelist, Real_t *volo, Index_t numReg,
               Index_t *regElemSize, Index_t **regElemlist, Int_t *elemBC,
               Index_t *lxim, Index_t *lxip, Index_t *letam, Index_t *letap,
               Index_t *lzetam, Index_t *lzetap, Real_t *vdov, Real_t *elemMass,
               Real_t *qq, Real_t *ql, Real_t *q, Real_t *vnew, Index_t numElem,
               Index_t sizeX, Index_t sizeY, Index_t sizeZ,
               Real_t monoq_limiter_mult, Real_t monoq_max_slope,
               Real_t qlc_monoq, Real_t qqc_monoq, Real_t qstop) {
   //
   // MONOTONIC Q option
   //
 
   if (numElem != 0) {
     Int_t allElem = numElem +           /* local elem */
                     2 * sizeX * sizeY + /* plane ghosts */
                     2 * sizeX * sizeZ + /* row ghosts */
                     2 * sizeY * sizeZ;  /* col ghosts */
 
     // Allocate gradients
     Real_t *delx_xi = (Real_t *)malloc(numElem * sizeof(Real_t));
     Real_t *delx_eta = (Real_t *)malloc(numElem * sizeof(Real_t));
     Real_t *delx_zeta = (Real_t *)malloc(numElem * sizeof(Real_t));
     Real_t *delv_xi = (Real_t *)malloc(allElem * sizeof(Real_t));
     Real_t *delv_eta = (Real_t *)malloc(allElem * sizeof(Real_t));
     Real_t *delv_zeta = (Real_t *)malloc(allElem * sizeof(Real_t));
 
     /* Calculate velocity gradients */
     CalcMonotonicQGradientsForElems(x, y, z, xd, yd, zd, nodelist, volo,
                                     delx_zeta, delv_zeta, delx_xi, delv_xi,
                                     delx_eta, delv_eta, vnew, numElem);
 
     CalcMonotonicQForElems(numReg, regElemSize, regElemlist, elemBC, delv_xi,
                            delv_eta, delv_zeta, delx_xi, delx_eta, delx_zeta,
                            lxim, lxip, letam, letap, lzetam, lzetap, vdov,
                            elemMass, volo, qq, ql, vnew, monoq_limiter_mult,
                            monoq_max_slope, qlc_monoq, qqc_monoq);
 
     // Free up memory
     free(delv_zeta);
     free(delv_eta);
     free(delv_xi);
     free(delx_zeta);
     free(delx_eta);
     free(delx_xi);
 
     /* Don't allow excessive artificial viscosity */
     Index_t idx = -1;
     for (Index_t i = 0; i < numElem; ++i) {
       if (q[i] > qstop) {
         idx = i;
         break;
       }
     }
 
     if (idx >= 0) {
       exit(QStopError);
     }
   }
 }
 
 /******************************************/
 
 static inline void CalcPressureForElems(Real_t *p_new, Real_t *bvc,
                                         Real_t *pbvc, Real_t *e_old,
                                         Real_t *compression, Real_t *vnewc,
                                         Real_t pmin, Real_t p_cut,
                                         Real_t eosvmax, Index_t length,
                                         Index_t *regElemList) {
 #pragma omp parallel for firstprivate(length)
   for (Index_t i = 0; i < length; ++i) {
     Real_t c1s = Real_t(2.0) / Real_t(3.0);
     bvc[i] = c1s * (compression[i] + Real_t(1.));
     pbvc[i] = c1s;
   }
 
 #pragma omp parallel for firstprivate(length, pmin, p_cut, eosvmax)
   for (Index_t i = 0; i < length; ++i) {
     Index_t elem = regElemList[i];
 
     p_new[i] = bvc[i] * e_old[i];
 
     if (FABS(p_new[i]) < p_cut)
       p_new[i] = Real_t(0.0);
 
     if (vnewc[elem] >= eosvmax) /* impossible condition here? */
       p_new[i] = Real_t(0.0);
 
     if (p_new[i] < pmin)
       p_new[i] = pmin;
   }
 }
 
 /******************************************/
 
 static inline void
 CalcEnergyForElems(Real_t *p_new, Real_t *e_new, Real_t *q_new, Real_t *bvc,
                    Real_t *pbvc, Real_t *p_old, Real_t *e_old, Real_t *q_old,
                    Real_t *compression, Real_t *compHalfStep, Real_t *vnewc,
                    Real_t *work, Real_t *delvc, Real_t pmin, Real_t p_cut,
                    Real_t e_cut, Real_t q_cut, Real_t emin, Real_t *qq_old,
                    Real_t *ql_old, Real_t rho0, Real_t eosvmax, Index_t length,
                    Index_t *regElemList) {
   Real_t *pHalfStep = (Real_t *)malloc(length * sizeof(Real_t));
 
 #pragma omp parallel for firstprivate(length, emin)
   for (Index_t i = 0; i < length; ++i) {
     e_new[i] = e_old[i] - Real_t(0.5) * delvc[i] * (p_old[i] + q_old[i]) +
                Real_t(0.5) * work[i];
 
     if (e_new[i] < emin) {
       e_new[i] = emin;
     }
   }
 
   CalcPressureForElems(pHalfStep, bvc, pbvc, e_new, compHalfStep, vnewc, pmin,
                        p_cut, eosvmax, length, regElemList);
 
 #pragma omp parallel for firstprivate(length, rho0)
   for (Index_t i = 0; i < length; ++i) {
     Real_t vhalf = Real_t(1.) / (Real_t(1.) + compHalfStep[i]);
 
     if (delvc[i] > Real_t(0.)) {
       q_new[i] /* = qq_old[i] = ql_old[i] */ = Real_t(0.);
     } else {
       Real_t ssc =
           (pbvc[i] * e_new[i] + vhalf * vhalf * bvc[i] * pHalfStep[i]) / rho0;
 
       if (ssc <= Real_t(.1111111e-36)) {
         ssc = Real_t(.3333333e-18);
       } else {
         ssc = SQRT(ssc);
       }
 
       q_new[i] = (ssc * ql_old[i] + qq_old[i]);
     }
 
     e_new[i] = e_new[i] + Real_t(0.5) * delvc[i] *
                               (Real_t(3.0) * (p_old[i] + q_old[i]) -
                                Real_t(4.0) * (pHalfStep[i] + q_new[i]));
   }
 
 #pragma omp parallel for firstprivate(length, emin, e_cut)
   for (Index_t i = 0; i < length; ++i) {
 
     e_new[i] += Real_t(0.5) * work[i];
 
     if (FABS(e_new[i]) < e_cut) {
       e_new[i] = Real_t(0.);
     }
     if (e_new[i] < emin) {
       e_new[i] = emin;
     }
   }
 
   CalcPressureForElems(p_new, bvc, pbvc, e_new, compression, vnewc, pmin, p_cut,
                        eosvmax, length, regElemList);
 
 #pragma omp parallel for firstprivate(length, rho0, emin, e_cut)
   for (Index_t i = 0; i < length; ++i) {
     const Real_t sixth = Real_t(1.0) / Real_t(6.0);
     Index_t elem = regElemList[i];
     Real_t q_tilde;
 
     if (delvc[i] > Real_t(0.)) {
       q_tilde = Real_t(0.);
     } else {
       Real_t ssc =
           (pbvc[i] * e_new[i] + vnewc[elem] * vnewc[elem] * bvc[i] * p_new[i]) /
           rho0;
 
       if (ssc <= Real_t(.1111111e-36)) {
         ssc = Real_t(.3333333e-18);
       } else {
         ssc = SQRT(ssc);
       }
 
       q_tilde = (ssc * ql_old[i] + qq_old[i]);
     }
 
     e_new[i] = e_new[i] - (Real_t(7.0) * (p_old[i] + q_old[i]) -
                            Real_t(8.0) * (pHalfStep[i] + q_new[i]) +
                            (p_new[i] + q_tilde)) *
                               delvc[i] * sixth;
 
     if (FABS(e_new[i]) < e_cut) {
       e_new[i] = Real_t(0.);
     }
     if (e_new[i] < emin) {
       e_new[i] = emin;
     }
   }
 
   CalcPressureForElems(p_new, bvc, pbvc, e_new, compression, vnewc, pmin, p_cut,
                        eosvmax, length, regElemList);
 
 #pragma omp parallel for firstprivate(length, rho0, q_cut)
   for (Index_t i = 0; i < length; ++i) {
     Index_t elem = regElemList[i];
 
     if (delvc[i] <= Real_t(0.)) {
       Real_t ssc =
           (pbvc[i] * e_new[i] + vnewc[elem] * vnewc[elem] * bvc[i] * p_new[i]) /
           rho0;
 
       if (ssc <= Real_t(.1111111e-36)) {
         ssc = Real_t(.3333333e-18);
       } else {
         ssc = SQRT(ssc);
       }
 
       q_new[i] = (ssc * ql_old[i] + qq_old[i]);
 
       if (FABS(q_new[i]) < q_cut)
         q_new[i] = Real_t(0.);
     }
   }
 
   free(pHalfStep);
 
   return;
 }
 
 /******************************************/
 
 static inline void CalcSoundSpeedForElems(Real_t *ss, Real_t *vnewc,
                                           Real_t rho0, Real_t *enewc,
                                           Real_t *pnewc, Real_t *pbvc,
                                           Real_t *bvc, Real_t ss4o3,
                                           Index_t len, Index_t *regElemList) {
 #pragma omp parallel for firstprivate(len, rho0, ss4o3)
   for (Index_t i = 0; i < len; ++i) {
     Index_t elem = regElemList[i];
     Real_t ssTmp =
         (pbvc[i] * enewc[i] + vnewc[elem] * vnewc[elem] * bvc[i] * pnewc[i]) /
         rho0;
     if (ssTmp <= Real_t(.1111111e-36)) {
       ssTmp = Real_t(.3333333e-18);
     } else {
       ssTmp = SQRT(ssTmp);
     }
     ss[elem] = ssTmp;
   }
 }
 
 /******************************************/
 
 static inline void EvalEOSForElems(Real_t *e, Real_t *p, Real_t *q, Real_t *qq,
                                    Real_t *ql, Real_t *delv, Real_t *ss,
                                    Real_t *vnewc, Int_t numElemReg,
                                    Index_t *regElemList, Int_t rep,
                                    Real_t e_cut, Real_t p_cut, Real_t ss4o3,
                                    Real_t q_cut, Real_t eosvmax, Real_t eosvmin,
                                    Real_t pmin, Real_t emin, Real_t rho0) {
 
   // These temporaries will be of different size for
   // each call (due to different sized region element
   // lists)
   Real_t *e_old = (Real_t *)malloc(numElemReg * sizeof(Real_t));
   Real_t *delvc = (Real_t *)malloc(numElemReg * sizeof(Real_t));
   Real_t *p_old = (Real_t *)malloc(numElemReg * sizeof(Real_t));
   Real_t *q_old = (Real_t *)malloc(numElemReg * sizeof(Real_t));
   Real_t *compression = (Real_t *)malloc(numElemReg * sizeof(Real_t));
   Real_t *compHalfStep = (Real_t *)malloc(numElemReg * sizeof(Real_t));
   Real_t *qq_old = (Real_t *)malloc(numElemReg * sizeof(Real_t));
   Real_t *ql_old = (Real_t *)malloc(numElemReg * sizeof(Real_t));
   Real_t *work = (Real_t *)malloc(numElemReg * sizeof(Real_t));
   Real_t *p_new = (Real_t *)malloc(numElemReg * sizeof(Real_t));
   Real_t *e_new = (Real_t *)malloc(numElemReg * sizeof(Real_t));
   Real_t *q_new = (Real_t *)malloc(numElemReg * sizeof(Real_t));
   Real_t *bvc = (Real_t *)malloc(numElemReg * sizeof(Real_t));
   Real_t *pbvc = (Real_t *)malloc(numElemReg * sizeof(Real_t));
 
   // loop to add load imbalance based on region number
   for (Int_t j = 0; j < rep; j++) {
     /* compress data, minimal set */
 #pragma omp parallel
     {
 #pragma omp for nowait firstprivate(numElemReg)
       for (Index_t i = 0; i < numElemReg; ++i) {
         Index_t elem = regElemList[i];
         e_old[i] = e[elem];
         delvc[i] = delv[elem];
         p_old[i] = p[elem];
         q_old[i] = q[elem];
         qq_old[i] = qq[elem];
         ql_old[i] = ql[elem];
       }
 
 #pragma omp for firstprivate(numElemReg)
       for (Index_t i = 0; i < numElemReg; ++i) {
         Index_t elem = regElemList[i];
         Real_t vchalf;
         compression[i] = Real_t(1.) / vnewc[elem] - Real_t(1.);
         vchalf = vnewc[elem] - delvc[i] * Real_t(.5);
         compHalfStep[i] = Real_t(1.) / vchalf - Real_t(1.);
       }
 
       /* Check for v > eosvmax or v < eosvmin */
       if (eosvmin != Real_t(0.)) {
 #pragma omp for nowait firstprivate(numElemReg, eosvmin)
         for (Index_t i = 0; i < numElemReg; ++i) {
           Index_t elem = regElemList[i];
           if (vnewc[elem] <= eosvmin) { /* impossible due to calling func? */
             compHalfStep[i] = compression[i];
           }
         }
       }
       if (eosvmax != Real_t(0.)) {
 #pragma omp for nowait firstprivate(numElemReg, eosvmax)
         for (Index_t i = 0; i < numElemReg; ++i) {
           Index_t elem = regElemList[i];
           if (vnewc[elem] >= eosvmax) { /* impossible due to calling func? */
             p_old[i] = Real_t(0.);
             compression[i] = Real_t(0.);
             compHalfStep[i] = Real_t(0.);
           }
         }
       }
 
 #pragma omp for nowait firstprivate(numElemReg)
       for (Index_t i = 0; i < numElemReg; ++i) {
         work[i] = Real_t(0.);
       }
     }
     CalcEnergyForElems(p_new, e_new, q_new, bvc, pbvc, p_old, e_old, q_old,
                        compression, compHalfStep, vnewc, work, delvc, pmin,
                        p_cut, e_cut, q_cut, emin, qq_old, ql_old, rho0, eosvmax,
                        numElemReg, regElemList);
   }
 
 #pragma omp parallel for firstprivate(numElemReg)
   for (Index_t i = 0; i < numElemReg; ++i) {
     Index_t elem = regElemList[i];
     p[elem] = p_new[i];
     e[elem] = e_new[i];
     q[elem] = q_new[i];
   }
 
   CalcSoundSpeedForElems(ss, vnewc, rho0, e_new, p_new, pbvc, bvc, ss4o3,
                          numElemReg, regElemList);
 
   free(pbvc);
   free(bvc);
   free(q_new);
   free(e_new);
   free(p_new);
   free(work);
   free(ql_old);
   free(qq_old);
   free(compHalfStep);
   free(compression);
   free(q_old);
   free(p_old);
   free(delvc);
   free(e_old);
 }
 
 /******************************************/
 
 static inline void ApplyMaterialPropertiesForElems(
     Real_t *v, Real_t *e, Real_t *p, Real_t *q, Real_t *qq, Real_t *ql,
     Real_t *delv, Real_t *ss, Real_t *vnew, Index_t numElem, Index_t numReg,
     Index_t *regElemSize, Index_t **regElemlist, Int_t cost, Real_t eosvmin,
     Real_t eosvmax, Real_t e_cut, Real_t p_cut, Real_t ss4o3, Real_t q_cut,
     Real_t pmin, Real_t emin, Real_t rho0) {
   if (numElem != 0) {
     /* Expose all of the variables needed for material evaluation */
 
 #pragma omp parallel
     {
       // Bound the updated relative volumes with eosvmin/max
       if (eosvmin != Real_t(0.)) {
 #pragma omp for firstprivate(numElem)
         for (Index_t i = 0; i < numElem; ++i) {
 
           if (vnew[i] < eosvmin)
             vnew[i] = eosvmin;
         }
       }
 
       if (eosvmax != Real_t(0.)) {
 #pragma omp for nowait firstprivate(numElem)
         for (Index_t i = 0; i < numElem; ++i) {
           if (vnew[i] > eosvmax)
             vnew[i] = eosvmax;
         }
       }
 
       // This check may not make perfect sense in LULESH, but
       // it's representative of something in the full code -
       // just leave it in, please
 #pragma omp for nowait firstprivate(numElem)
       for (Index_t i = 0; i < numElem; ++i) {
         Real_t vc = v[i];
         if (eosvmin != Real_t(0.)) {
           if (vc < eosvmin)
             vc = eosvmin;
         }
         if (eosvmax != Real_t(0.)) {
           if (vc > eosvmax)
             vc = eosvmax;
         }
         if (vc <= 0.) {
           exit(VolumeError);
         }
       }
     }
 
     for (Int_t r = 0; r < numReg; r++) {
       Index_t numElemReg = regElemSize[r];
       Index_t *regElemList = regElemlist[r];
       Int_t rep;
       // Determine load imbalance for this region
       // round down the number with lowest cost
       if (r < numReg / 2)
         rep = 1;
       // you don't get an expensive region unless you at least have 5 regions
       else if (r < (numReg - (numReg + 15) / 20))
         rep = 1 + cost;
       // very expensive regions
       else
         rep = 10 * (1 + cost);
       EvalEOSForElems(e, p, q, qq, ql, delv, ss, vnew, numElemReg, regElemList,
                       rep, e_cut, p_cut, ss4o3, q_cut, eosvmax, eosvmin, pmin,
                       emin, rho0);
     }
   }
 }
 
 /******************************************/
 
 static inline void UpdateVolumesForElems(Real_t *v, Real_t *vnew, Real_t v_cut,
                                          Index_t length) {
   if (length != 0) {
 #pragma omp parallel for firstprivate(length, v_cut)
     for (Index_t i = 0; i < length; ++i) {
       Real_t tmpV = vnew[i];
 
       if (FABS(tmpV - Real_t(1.0)) < v_cut)
         tmpV = Real_t(1.0);
 
       v[i] = tmpV;
     }
   }
 
   return;
 }
 
 /******************************************/
 
 static inline void LagrangeElements(
     Real_t *x, Real_t *y, Real_t *z, Real_t *xd, Real_t *yd, Real_t *zd,
     Index_t *nodelist, Real_t *volo, Real_t *v, Real_t *delv, Real_t *arealg,
     Real_t *vdov, Real_t *e, Real_t *p, Real_t *q, Real_t *qq, Real_t *ql,
     Real_t *ss, Int_t *elemBC, Index_t *lxim, Index_t *lxip, Index_t *letam,
     Index_t *letap, Index_t *lzetam, Index_t *lzetap, Real_t *elemMass,
     Index_t numReg, Index_t *regElemSize, Index_t **regElemlist, Int_t cost,
     Index_t numElem, Index_t sizeX, Index_t sizeY, Index_t sizeZ,
     Real_t deltatime, Real_t v_cut, Real_t monoq_limiter_mult,
     Real_t monoq_max_slope, Real_t qlc_monoq, Real_t qqc_monoq, Real_t qstop,
     Real_t eosvmin, Real_t eosvmax, Real_t e_cut, Real_t p_cut, Real_t ss4o3,
     Real_t q_cut, Real_t pmin, Real_t emin, Real_t rho0) {
   Real_t *vnew =
       (Real_t *)malloc(numElem * sizeof(Real_t)); /* new relative vol -- temp */
 
   CalcLagrangeElements(x, y, z, xd, yd, zd, nodelist, volo, v, delv, arealg,
                        vdov, vnew, deltatime, numElem);
 
   /* Calculate Q.  (Monotonic q option requires communication) */
   CalcQForElems(x, y, z, xd, yd, zd, nodelist, volo, numReg, regElemSize,
                 regElemlist, elemBC, lxim, lxip, letam, letap, lzetam, lzetap,
                 vdov, elemMass, qq, ql, q, vnew, numElem, sizeX, sizeY, sizeZ,
                 monoq_limiter_mult, monoq_max_slope, qlc_monoq, qqc_monoq,
                 qstop);
 
   ApplyMaterialPropertiesForElems(v, e, p, q, qq, ql, delv, ss, vnew, numElem,
                                   numReg, regElemSize, regElemlist, cost,
                                   eosvmin, eosvmax, e_cut, p_cut, ss4o3, q_cut,
                                   pmin, emin, rho0);
 
   UpdateVolumesForElems(v, vnew, v_cut, numElem);
 
   free(vnew);
 }
 
 /******************************************/
 
 static inline void CalcCourantConstraintForElems(Real_t *ss, Real_t *vdov,
                                                  Real_t *arealg, Index_t length,
                                                  Index_t *regElemlist,
                                                  Real_t qqc,
                                                  Real_t *dtcourant) {
 
 #ifdef _OPENMP
   Index_t threads = omp_get_max_threads();
 #else
   Index_t threads = 1;
 #endif
   Index_t *courant_elem_per_thread =
       (Index_t *)malloc(threads * sizeof(Index_t));
   Real_t *dtcourant_per_thread = (Real_t *)malloc(threads * sizeof(Real_t));
 
 #pragma omp parallel firstprivate(length, qqc)
   {
     Real_t qqc2 = Real_t(64.0) * qqc * qqc;
     Real_t dtcourant_tmp = *dtcourant;
     Index_t courant_elem = -1;
 
 #if _OPENMP
     Index_t thread_num = omp_get_thread_num();
 #else
     Index_t thread_num = 0;
 #endif
 
 #pragma omp for
     for (Index_t i = 0; i < length; ++i) {
       Index_t indx = regElemlist[i];
       Real_t dtf = ss[indx] * ss[indx];
 
       if (vdov[indx] < Real_t(0.)) {
         dtf =
             dtf + qqc2 * arealg[indx] * arealg[indx] * vdov[indx] * vdov[indx];
       }
 
       dtf = SQRT(dtf);
       dtf = arealg[indx] / dtf;
 
       if (vdov[indx] != Real_t(0.)) {
         if (dtf < dtcourant_tmp) {
           dtcourant_tmp = dtf;
           courant_elem = indx;
         }
       }
     }
 
     dtcourant_per_thread[thread_num] = dtcourant_tmp;
     courant_elem_per_thread[thread_num] = courant_elem;
   }
 
   for (Index_t i = 1; i < threads; ++i) {
     if (dtcourant_per_thread[i] < dtcourant_per_thread[0]) {
       dtcourant_per_thread[0] = dtcourant_per_thread[i];
       courant_elem_per_thread[0] = courant_elem_per_thread[i];
     }
   }
 
   if (courant_elem_per_thread[0] != -1) {
     *dtcourant = dtcourant_per_thread[0];
   }
 
   free(courant_elem_per_thread);
   free(dtcourant_per_thread);
 
   return;
 }
 
 /******************************************/
 
 static inline void CalcHydroConstraintForElems(Real_t *vdov, Index_t length,
                                                Index_t *regElemlist,
                                                Real_t dvovmax,
                                                Real_t *dthydro) {
 
 #ifdef _OPENMP
   Index_t threads = omp_get_max_threads();
 #else
   Index_t threads = 1;
 #endif
   Index_t *hydro_elem_per_thread = (Index_t *)malloc(threads * sizeof(Index_t));
   Real_t *dthydro_per_thread = (Real_t *)malloc(threads * sizeof(Real_t));
 
 #pragma omp parallel firstprivate(length, dvovmax)
   {
     Real_t dthydro_tmp = *dthydro;
     Index_t hydro_elem = -1;
 
 #if _OPENMP
     Index_t thread_num = omp_get_thread_num();
 #else
     Index_t thread_num = 0;
 #endif
 
 #pragma omp for
     for (Index_t i = 0; i < length; ++i) {
       Index_t indx = regElemlist[i];
 
       if (vdov[indx] != Real_t(0.)) {
         Real_t dtdvov = dvovmax / (FABS(vdov[indx]) + Real_t(1.e-20));
 
         if (dthydro_tmp > dtdvov) {
           dthydro_tmp = dtdvov;
           hydro_elem = indx;
         }
       }
     }
 
     dthydro_per_thread[thread_num] = dthydro_tmp;
     hydro_elem_per_thread[thread_num] = hydro_elem;
   }
 
   for (Index_t i = 1; i < threads; ++i) {
     if (dthydro_per_thread[i] < dthydro_per_thread[0]) {
       dthydro_per_thread[0] = dthydro_per_thread[i];
       hydro_elem_per_thread[0] = hydro_elem_per_thread[i];
     }
   }
 
   if (hydro_elem_per_thread[0] != -1) {
     *dthydro = dthydro_per_thread[0];
   }
 
   free(hydro_elem_per_thread);
   free(dthydro_per_thread);
 
   return;
 }
 
 /******************************************/
 
 static inline void
 CalcTimeConstraintsForElems(Real_t *ss, Real_t *vdov, Real_t *arealg,
                             Index_t numReg, Index_t *regElemSize,
                             Index_t **regElemlist, Real_t qqc, Real_t dvovmax,
                             Real_t *dtcourant, Real_t *dthydro) {
 
   // Initialize conditions to a very large value
   *dtcourant = 1.0e+20;
   *dthydro = 1.0e+20;
 
   for (Index_t r = 0; r < numReg; ++r) {
     /* evaluate time constraint */
     CalcCourantConstraintForElems(ss, vdov, arealg, regElemSize[r],
                                   regElemlist[r], qqc, dtcourant);
 
     /* check hydro constraint */
     CalcHydroConstraintForElems(vdov, regElemSize[r], regElemlist[r], dvovmax,
                                 dthydro);
   }
 }
 
 /******************************************/
 
 static inline void LagrangeLeapFrog(
     Real_t *x, Real_t *y, Real_t *z, Real_t *xd, Real_t *yd, Real_t *zd,
     Real_t *xdd, Real_t *ydd, Real_t *zdd, Real_t *fx, Real_t *fy, Real_t *fz,
     Real_t *nodalMass, Index_t *nodelist, Real_t *volo, Real_t *v, Real_t *delv,
     Real_t *arealg, Real_t *vdov, Real_t *e, Real_t *p, Real_t *q, Real_t *qq,
     Real_t *ql, Real_t *ss, Real_t *elemMass, Int_t *elemBC, Index_t *lxim,
     Index_t *lxip, Index_t *letam, Index_t *letap, Index_t *lzetam,
     Index_t *lzetap, Index_t *nodeElemStart, Index_t *nodeElemCornerList,
     Index_t *symmX, Index_t *symmY, Index_t *symmZ, Index_t symmXempty,
     Index_t symmYempty, Index_t symmZempty, Index_t numReg,
     Index_t *regElemSize, Index_t **regElemlist, Int_t cost, Index_t numElem,
     Index_t numNode, Index_t sizeX, Index_t sizeY, Index_t sizeZ,
     Real_t deltatime, Real_t u_cut, Real_t v_cut, Real_t hgcoef,
     Real_t monoq_limiter_mult, Real_t monoq_max_slope, Real_t qlc_monoq,
     Real_t qqc_monoq, Real_t qstop, Real_t eosvmin, Real_t eosvmax,
     Real_t e_cut, Real_t p_cut, Real_t ss4o3, Real_t q_cut, Real_t pmin,
     Real_t emin, Real_t rho0, Real_t qqc, Real_t dvovmax, Real_t *dtcourant,
     Real_t *dthydro) {
   /* calculate nodal forces, accelerations, velocities, positions, with
    * applied boundary conditions and slide surface considerations */
   LagrangeNodal(x, y, z, xd, yd, zd, xdd, ydd, zdd, fx, fy, fz, nodalMass,
                 nodelist, volo, v, p, q, ss, elemMass, nodeElemStart,
                 nodeElemCornerList, symmX, symmY, symmZ, symmXempty, symmYempty,
                 symmZempty, deltatime, u_cut, hgcoef, numElem, numNode, sizeX);
 
   /* calculate element quantities (i.e. velocity gradient & q), and update
    * material states */
   LagrangeElements(x, y, z, xd, yd, zd, nodelist, volo, v, delv, arealg, vdov,
                    e, p, q, qq, ql, ss, elemBC, lxim, lxip, letam, letap,
                    lzetam, lzetap, elemMass, numReg, regElemSize, regElemlist,
                    cost, numElem, sizeX, sizeY, sizeZ, deltatime, v_cut,
                    monoq_limiter_mult, monoq_max_slope, qlc_monoq, qqc_monoq,
                    qstop, eosvmin, eosvmax, e_cut, p_cut, ss4o3, q_cut, pmin,
                    emin, rho0);
 
   CalcTimeConstraintsForElems(ss, vdov, arealg, numReg, regElemSize,
                               regElemlist, qqc, dvovmax, dtcourant, dthydro);
 }
 
 /******************************************/
 
 /* Helper function for converting strings to ints, with error checking */
 int StrToInt(const char *token, int *retVal) {
   const char *c;
   char *endptr;
   const int decimal_base = 10;
 
   if (token == nullptr)
     return 0;
 
   c = token;
   *retVal = (int)strtol(c, &endptr, decimal_base);
   if ((endptr != c) && ((*endptr == ' ') || (*endptr == '\0')))
     return 1;
   else
     return 0;
 }
 
 static void PrintCommandLineOptions(char *execname, int myRank) {
   if (myRank == 0) {
 
     printf("Usage: %s [opts]\n", execname);
     printf(" where [opts] is one or more of:\n");
     printf(" -q              : quiet mode - suppress all stdout\n");
     printf(" -i <iterations> : number of cycles to run\n");
     printf(" -s <size>       : length of cube mesh along side\n");
     printf(" -r <numregions> : Number of distinct regions (def: 11)\n");
     printf(" -b <balance>    : Load balance between regions of a domain (def: "
            "1)\n");
     printf(
         " -c <cost>       : Extra cost of more expensive regions (def: 1)\n");
     printf(" -p              : Print out progress\n");
     printf(" -h              : This message\n");
     printf("\n\n");
   }
 }
 
 static void ParseError(const char *message, int myRank) {
   if (myRank == 0) {
     printf("%s\n", message);
     exit(-1);
   }
 }
 
 void ParseCommandLineOptions(int argc, char **argv, int myRank, Int_t *its,
                              Int_t *nx, Int_t *numReg, Int_t *showProg,
                              Int_t *quiet, Int_t *cost, Int_t *balance) {
   if (argc > 1) {
     int i = 1;
 
     while (i < argc) {
       int ok;
       /* -i <iterations> */
       if (strcmp(argv[i], "-i") == 0) {
         if (i + 1 >= argc) {
           ParseError("Missing integer argument to -i", myRank);
         }
         ok = StrToInt(argv[i + 1], its);
         if (!ok) {
           ParseError("Parse Error on option -i integer value required after "
                      "argument\n",
                      myRank);
         }
         i += 2;
       }
       /* -s <size, sidelength> */
       else if (strcmp(argv[i], "-s") == 0) {
         if (i + 1 >= argc) {
           ParseError("Missing integer argument to -s\n", myRank);
         }
         ok = StrToInt(argv[i + 1], nx);
         if (!ok) {
           ParseError("Parse Error on option -s integer value required after "
                      "argument\n",
                      myRank);
         }
         i += 2;
       }
       /* -r <numregions> */
       else if (strcmp(argv[i], "-r") == 0) {
         if (i + 1 >= argc) {
           ParseError("Missing integer argument to -r\n", myRank);
         }
         ok = StrToInt(argv[i + 1], numReg);
         if (!ok) {
           ParseError("Parse Error on option -r integer value required after "
                      "argument\n",
                      myRank);
         }
         i += 2;
       }
       /* -p */
       else if (strcmp(argv[i], "-p") == 0) {
         *showProg = 1;
         i++;
       }
       /* -q */
       else if (strcmp(argv[i], "-q") == 0) {
         *quiet = 1;
         i++;
       } else if (strcmp(argv[i], "-b") == 0) {
         if (i + 1 >= argc) {
           ParseError("Missing integer argument to -b\n", myRank);
         }
         ok = StrToInt(argv[i + 1], balance);
         if (!ok) {
           ParseError("Parse Error on option -b integer value required after "
                      "argument\n",
                      myRank);
         }
         i += 2;
       } else if (strcmp(argv[i], "-c") == 0) {
         if (i + 1 >= argc) {
           ParseError("Missing integer argument to -c\n", myRank);
         }
         ok = StrToInt(argv[i + 1], cost);
         if (!ok) {
           ParseError("Parse Error on option -c integer value required after "
                      "argument\n",
                      myRank);
         }
         i += 2;
       }
       /* -h */
       else if (strcmp(argv[i], "-h") == 0) {
         PrintCommandLineOptions(argv[0], myRank);
         exit(0);
       } else {
         char msg[80];
         PrintCommandLineOptions(argv[0], myRank);
         sprintf(msg, "ERROR: Unknown command line argument: %s\n", argv[i]);
         ParseError(msg, myRank);
       }
     }
   }
 }
 
 /////////////////////////////////////////////////////////////////////
 
 void VerifyAndWriteFinalOutput(Real_t elapsed_time, Int_t cycle, Real_t *e,
                                Int_t nx, Int_t numRanks) {
   // GrindTime1 only takes a single domain into account, and is thus a good way
   // to measure processor speed indepdendent of MPI parallelism. GrindTime2
   // takes into account speedups from MPI parallelism
   Real_t grindTime1 = ((elapsed_time * 1e6) / cycle) / (nx * nx * nx);
   Real_t grindTime2 =
       ((elapsed_time * 1e6) / cycle) / (nx * nx * nx * numRanks);
 
   Index_t ElemId = 0;
   printf("Run completed:  \n");
   printf("   Problem size        =  %i \n", nx);
   printf("   MPI tasks           =  %i \n", numRanks);
   printf("   Iteration count     =  %i \n", cycle);
   printf("   Final Origin Energy = %12.6e \n", e[ElemId]);
 
   Real_t MaxAbsDiff = Real_t(0.0);
   Real_t TotalAbsDiff = Real_t(0.0);
   Real_t MaxRelDiff = Real_t(0.0);
 
   for (Index_t j = 0; j < nx; ++j) {
     for (Index_t k = j + 1; k < nx; ++k) {
       Real_t AbsDiff = FABS(e[j * nx + k] - e[k * nx + j]);
       TotalAbsDiff += AbsDiff;
 
       if (MaxAbsDiff < AbsDiff)
         MaxAbsDiff = AbsDiff;
 
       Real_t RelDiff = AbsDiff / e[k * nx + j];
 
       if (MaxRelDiff < RelDiff)
         MaxRelDiff = RelDiff;
     }
   }
 
   // Quick symmetry check
   printf("   Testing Plane 0 of Energy Array on rank 0:\n");
   printf("        MaxAbsDiff   = %12.6e\n", MaxAbsDiff);
   printf("        TotalAbsDiff = %12.6e\n", TotalAbsDiff);
   printf("        MaxRelDiff   = %12.6e\n\n", MaxRelDiff);
 
   // Timing information
   printf("\nElapsed time         = %10.2f (s)\n", elapsed_time);
   printf("Grind time (us/z/c)  = %10.8g (per dom)  (%10.8g overall)\n",
          grindTime1, grindTime2);
   printf("FOM                  = %10.8g (z/s)\n\n",
          1000.0 / grindTime2); // zones per second
 
   return;
 }
 
 /////////////////////////////////////////////////////////////////////
 // Domain constructor removed - initialization now happens in main()
 
 ////////////////////////////////////////////////////////////////////////////////
 void BuildMesh(Int_t nx, Int_t edgeNodes, Int_t edgeElems, Index_t *nodelist,
                Index_t *lxim, Index_t *lxip, Index_t *letam, Index_t *letap,
                Index_t *lzetam, Index_t *lzetap, Real_t *x, Real_t *y,
                Real_t *z, Index_t colLoc, Index_t rowLoc, Index_t planeLoc,
                Index_t sizeX, Index_t sizeY, Index_t sizeZ, Index_t *rowMin,
                Index_t *rowMax, Index_t *colMin, Index_t *colMax,
                Index_t *planeMin, Index_t *planeMax, Index_t tp) {
   Index_t meshEdgeElems = tp * nx;
 
   // initialize nodal coordinates
   Index_t nidx = 0;
   Real_t tz = Real_t(1.125) * Real_t(planeLoc * nx) / Real_t(meshEdgeElems);
   for (Index_t plane = 0; plane < edgeNodes; ++plane) {
     Real_t ty = Real_t(1.125) * Real_t(rowLoc * nx) / Real_t(meshEdgeElems);
     for (Index_t row = 0; row < edgeNodes; ++row) {
       Real_t tx = Real_t(1.125) * Real_t(colLoc * nx) / Real_t(meshEdgeElems);
       for (Index_t col = 0; col < edgeNodes; ++col) {
         x[nidx] = tx;
         y[nidx] = ty;
         z[nidx] = tz;
         ++nidx;
         // tx += ds ; // may accumulate roundoff...
         tx = Real_t(1.125) * Real_t(colLoc * nx + col + 1) /
              Real_t(meshEdgeElems);
       }
       // ty += ds ;  // may accumulate roundoff...
       ty =
           Real_t(1.125) * Real_t(rowLoc * nx + row + 1) / Real_t(meshEdgeElems);
     }
     // tz += ds ;  // may accumulate roundoff...
     tz = Real_t(1.125) * Real_t(planeLoc * nx + plane + 1) /
          Real_t(meshEdgeElems);
   }
 
   // embed hexehedral elements in nodal point lattice
   Index_t zidx = 0;
   Index_t element_id = 0;
   nidx = 0;
   for (Index_t plane = 0; plane < edgeElems; ++plane) {
     for (Index_t row = 0; row < edgeElems; ++row) {
       for (Index_t col = 0; col < edgeElems; ++col) {
         Index_t base = 8 * zidx;
         nodelist[base + 0] = nidx;
         nodelist[base + 1] = nidx + 1;
         nodelist[base + 2] = nidx + edgeNodes + 1;
         nodelist[base + 3] = nidx + edgeNodes;
         nodelist[base + 4] = nidx + edgeNodes * edgeNodes;
         nodelist[base + 5] = nidx + edgeNodes * edgeNodes + 1;
         nodelist[base + 6] = nidx + edgeNodes * edgeNodes + edgeNodes + 1;
         nodelist[base + 7] = nidx + edgeNodes * edgeNodes + edgeNodes;
         ++zidx;
         ++nidx;
         element_id++;
       }
       ++nidx;
     }
     nidx += edgeNodes;
   }
 }
 
 ////////////////////////////////////////////////////////////////////////////////
 void SetupCommBuffers(Int_t edgeNodes, Index_t *maxPlaneSize,
                       Index_t *maxEdgeSize, Index_t *rowMin, Index_t *rowMax,
                       Index_t *colMin, Index_t *colMax, Index_t *planeMin,
                       Index_t *planeMax, Index_t sizeX, Index_t sizeY,
                       Index_t sizeZ, Index_t colLoc, Index_t rowLoc,
                       Index_t planeLoc, Index_t tp) {
   // allocate a buffer large enough for nodal ghost data
   Index_t maxEdgeSizeCalc = MAX(sizeX, MAX(sizeY, sizeZ)) + 1;
   *maxPlaneSize = CACHE_ALIGN_REAL(maxEdgeSizeCalc * maxEdgeSizeCalc);
   *maxEdgeSize = CACHE_ALIGN_REAL(maxEdgeSizeCalc);
 
   // assume communication to 6 neighbors by default
   *rowMin = (rowLoc == 0) ? 0 : 1;
   *rowMax = (rowLoc == tp - 1) ? 0 : 1;
   *colMin = (colLoc == 0) ? 0 : 1;
   *colMax = (colLoc == tp - 1) ? 0 : 1;
   *planeMin = (planeLoc == 0) ? 0 : 1;
   *planeMax = (planeLoc == tp - 1) ? 0 : 1;
 }
 
 /////////////////////////////////////////////////////////////
 void SetupSymmetryPlanes(Int_t edgeNodes, Index_t **symmX, Index_t **symmY,
                          Index_t **symmZ, Int_t *symmX_size, Int_t *symmY_size,
                          Int_t *symmZ_size, Index_t colLoc, Index_t rowLoc,
                          Index_t planeLoc, Index_t sizeX, Index_t sizeY,
                          Index_t sizeZ) {
   Index_t symmSize = edgeNodes * edgeNodes;
 
   // Allocate and initialize symmetry plane arrays if this domain is on a
   // boundary
   *symmX = (Index_t *)malloc(symmSize * sizeof(Index_t));
   *symmX_size = symmSize;
   *symmY = (Index_t *)malloc(symmSize * sizeof(Index_t));
   *symmY_size = symmSize;
   *symmZ = (Index_t *)malloc(symmSize * sizeof(Index_t));
   *symmZ_size = symmSize;
   Index_t nidx = 0;
   for (Index_t i = 0; i < edgeNodes; ++i) {
     Index_t planeInc = i * edgeNodes * edgeNodes;
     Index_t rowInc = i * edgeNodes;
     for (Index_t j = 0; j < edgeNodes; ++j) {
       if (planeLoc == 0) {
         (*symmZ)[nidx] = rowInc + j;
       }
       if (rowLoc == 0) {
         (*symmY)[nidx] = planeInc + j;
       }
       if (colLoc == 0) {
         (*symmX)[nidx] = planeInc + j * edgeNodes;
       }
       ++nidx;
     }
   }
 }
 
 /////////////////////////////////////////////////////////////
 void SetupElementConnectivities(Int_t edgeElems, Index_t *lxim, Index_t *lxip,
                                 Index_t *letam, Index_t *letap, Index_t *lzetam,
                                 Index_t *lzetap, Index_t *nodelist,
                                 Index_t sizeX, Index_t sizeY, Index_t sizeZ) {
   Index_t numElem = sizeX * sizeY * sizeZ;
 
   lxim[0] = 0;
   for (Index_t i = 1; i < numElem; ++i) {
     lxim[i] = i - 1;
     lxip[i - 1] = i;
   }
   lxip[numElem - 1] = numElem - 1;
 
   for (Index_t i = 0; i < edgeElems; ++i) {
     letam[i] = i;
     letap[numElem - edgeElems + i] = numElem - edgeElems + i;
   }
   for (Index_t i = edgeElems; i < numElem; ++i) {
     letam[i] = i - edgeElems;
     letap[i - edgeElems] = i;
   }
 
   for (Index_t i = 0; i < edgeElems * edgeElems; ++i) {
     lzetam[i] = i;
     lzetap[numElem - edgeElems * edgeElems + i] =
         numElem - edgeElems * edgeElems + i;
   }
   for (Index_t i = edgeElems * edgeElems; i < numElem; ++i) {
     lzetam[i] = i - edgeElems * edgeElems;
     lzetap[i - edgeElems * edgeElems] = i;
   }
 }
 
 /////////////////////////////////////////////////////////////
 void SetupBoundaryConditions(Int_t edgeElems, Int_t *elemBC, Index_t *lxim,
                              Index_t *lxip, Index_t *letam, Index_t *letap,
                              Index_t *lzetam, Index_t *lzetap, Index_t colLoc,
                              Index_t rowLoc, Index_t planeLoc, Index_t sizeX,
                              Index_t sizeY, Index_t sizeZ, Int_t numRanks,
                              Index_t planeMin, Index_t planeMax, Index_t rowMin,
                              Index_t rowMax, Index_t colMin, Index_t colMax,
                              Index_t tp) {
   Index_t numElem = sizeX * sizeY * sizeZ;
   Index_t ghostIdx[6]; // offsets to ghost locations
 
   // set up boundary condition information
   for (Index_t i = 0; i < numElem; ++i) {
     elemBC[i] = Int_t(0);
   }
 
   for (Index_t i = 0; i < 6; ++i) {
     ghostIdx[i] = INT_MIN;
   }
 
   Int_t pidx = numElem;
   if (planeMin != 0) {
     ghostIdx[0] = pidx;
     pidx += sizeX * sizeY;
   }
 
   if (planeMax != 0) {
     ghostIdx[1] = pidx;
     pidx += sizeX * sizeY;
   }
 
   if (rowMin != 0) {
     ghostIdx[2] = pidx;
     pidx += sizeX * sizeZ;
   }
 
   if (rowMax != 0) {
     ghostIdx[3] = pidx;
     pidx += sizeX * sizeZ;
   }
 
   if (colMin != 0) {
     ghostIdx[4] = pidx;
     pidx += sizeY * sizeZ;
   }
 
   if (colMax != 0) {
     ghostIdx[5] = pidx;
   }
 
   // symmetry plane or free surface BCs
   for (Index_t i = 0; i < edgeElems; ++i) {
     Index_t planeInc = i * edgeElems * edgeElems;
     Index_t rowInc = i * edgeElems;
     for (Index_t j = 0; j < edgeElems; ++j) {
       if (planeLoc == 0) {
         elemBC[rowInc + j] |= ZETA_M_SYMM;
       } else {
         elemBC[rowInc + j] |= ZETA_M_COMM;
         lzetam[rowInc + j] = ghostIdx[0] + rowInc + j;
       }
 
       if (planeLoc == tp - 1) {
         elemBC[rowInc + j + numElem - edgeElems * edgeElems] |= ZETA_P_FREE;
       } else {
         elemBC[rowInc + j + numElem - edgeElems * edgeElems] |= ZETA_P_COMM;
         lzetap[rowInc + j + numElem - edgeElems * edgeElems] =
             ghostIdx[1] + rowInc + j;
       }
 
       if (rowLoc == 0) {
         elemBC[planeInc + j] |= ETA_M_SYMM;
       } else {
         elemBC[planeInc + j] |= ETA_M_COMM;
         letam[planeInc + j] = ghostIdx[2] + rowInc + j;
       }
 
       if (rowLoc == tp - 1) {
         elemBC[planeInc + j + edgeElems * edgeElems - edgeElems] |= ETA_P_FREE;
       } else {
         elemBC[planeInc + j + edgeElems * edgeElems - edgeElems] |= ETA_P_COMM;
         letap[planeInc + j + edgeElems * edgeElems - edgeElems] =
             ghostIdx[3] + rowInc + j;
       }
 
       if (colLoc == 0) {
         elemBC[planeInc + j * edgeElems] |= XI_M_SYMM;
       } else {
         elemBC[planeInc + j * edgeElems] |= XI_M_COMM;
         lxim[planeInc + j * edgeElems] = ghostIdx[4] + rowInc + j;
       }
 
       if (colLoc == tp - 1) {
         elemBC[planeInc + j * edgeElems + edgeElems - 1] |= XI_P_FREE;
       } else {
         elemBC[planeInc + j * edgeElems + edgeElems - 1] |= XI_P_COMM;
         lxip[planeInc + j * edgeElems + edgeElems - 1] =
             ghostIdx[5] + rowInc + j;
       }
     }
   }
 }
 
 ///////////////////////////////////////////////////////////////////////////
 void InitMeshDecomp(Int_t numRanks, Int_t myRank, Int_t *col, Int_t *row,
                     Int_t *plane, Int_t *side) {
   Int_t testProcs;
   Int_t dx, dy, dz;
   Int_t myDom;
 
   // Assume cube processor layout for now
   testProcs = Int_t(cbrt(Real_t(numRanks)) + 0.5);
   if (testProcs * testProcs * testProcs != numRanks) {
     printf("Num processors must be a cube of an integer (1, 8, 27, ...)\n");
     exit(-1);
   }
   if (sizeof(Real_t) != 4 && sizeof(Real_t) != 8) {
     printf("MPI operations only support float and double right now...\n");
     exit(-1);
   }
   if (MAX_FIELDS_PER_MPI_COMM > CACHE_COHERENCE_PAD_REAL) {
     printf("corner element comm buffers too small.  Fix code.\n");
     exit(-1);
   }
 
   dx = testProcs;
   dy = testProcs;
   dz = testProcs;
 
   // temporary test
   if (dx * dy * dz != numRanks) {
     printf("error -- must have as many domains as procs\n");
     exit(-1);
   }
   Int_t remainder = dx * dy * dz % numRanks;
   if (myRank < remainder) {
     myDom = myRank * (1 + (dx * dy * dz / numRanks));
   } else {
     myDom = remainder * (1 + (dx * dy * dz / numRanks)) +
             (myRank - remainder) * (dx * dy * dz / numRanks);
   }
 
   *col = myDom % dx;
   *row = (myDom / dx) % dy;
   *plane = myDom / (dx * dy);
   *side = testProcs;
 
   return;
 }
 
 /******************************************/
 
 int main(int argc, char **argv) {
   Int_t numRanks;
   Int_t myRank;
 
   numRanks = 1;
   myRank = 0;
 
   /* Set defaults that can be overridden by command line opts */
   Int_t its = 9999999;
   Int_t nx = 30;
   Int_t opt_numReg = 11;
   Int_t showProg = 0;
   Int_t quiet = 0;
   Int_t balance = 1;
   Int_t cost = 1;
 
   ParseCommandLineOptions(argc, argv, myRank, &its, &nx, &opt_numReg, &showProg,
                           &quiet, &cost, &balance);
 
   if ((myRank == 0) && (quiet == 0)) {
     printf("Running problem size %d^3 per domain until completion\n", nx);
     printf("Num processors: %d\n", numRanks);
 #if _OPENMP
     printf("Num threads: %d\n", omp_get_max_threads());
 #endif
     printf("Total number of elements: %lld\n\n",
            (long long int)(numRanks * nx * nx * nx));
     printf("To run other sizes, use -s <integer>.\n");
     printf("To run a fixed number of iterations, use -i <integer>.\n");
     printf("To run a more or less balanced region set, use -b <integer>.\n");
     printf("To change the relative costs of regions, use -c <integer>.\n");
     printf("To print out progress, use -p\n");
     printf("To write an output file for VisIt, use -v\n");
     printf("See help (-h) for more options\n\n");
   }
 
   // Set up the mesh and decompose. Assumes regular cubes for now
   Int_t col, row, plane, side;
   InitMeshDecomp(numRanks, myRank, &col, &row, &plane, &side);
 
   // Calculate mesh dimensions
   Index_t edgeElems = nx;
   Index_t edgeNodes = edgeElems + 1;
 
   // Domain dimensions
   Index_t numElem = edgeElems * edgeElems * edgeElems;
   Index_t numNode = edgeNodes * edgeNodes * edgeNodes;
   Index_t sizeX = edgeElems;
   Index_t sizeY = edgeElems;
   Index_t sizeZ = edgeElems;
   Index_t colLoc = col;
   Index_t rowLoc = row;
   Index_t planeLoc = plane;
   Index_t tp = side;
 
   // Physics constants
   Real_t e_cut = Real_t(1.0e-7);
   Real_t p_cut = Real_t(1.0e-7);
   Real_t q_cut = Real_t(1.0e-7);
   Real_t v_cut = Real_t(1.0e-10);
   Real_t u_cut = Real_t(1.0e-7);
   Real_t hgcoef = Real_t(3.0);
   Real_t ss4o3 = Real_t(4.0) / Real_t(3.0);
   Real_t qstop = Real_t(1.0e+12);
   Real_t monoq_max_slope = Real_t(1.0);
   Real_t monoq_limiter_mult = Real_t(2.0);
   Real_t qlc_monoq = Real_t(0.5);
   Real_t qqc_monoq = Real_t(2.0) / Real_t(3.0);
   Real_t qqc = Real_t(2.0);
   Real_t eosvmax = Real_t(1.0e+9);
   Real_t eosvmin = Real_t(1.0e-9);
   Real_t pmin = Real_t(0.0);
   Real_t emin = Real_t(-1.0e+15);
   Real_t dvovmax = Real_t(0.1);
   Real_t refdens = Real_t(1.0);
 
   // Timestep controls
   Real_t dtcourant = Real_t(1.0e+20);
   Real_t dthydro = Real_t(1.0e+20);
   Int_t cycle = Int_t(0);
   Real_t dtfixed = Real_t(-1.0e-6);
   Real_t time = Real_t(0.0);
   Real_t deltatime = Real_t(0.0);
   Real_t deltatimemultlb = Real_t(1.1);
   Real_t deltatimemultub = Real_t(1.2);
   Real_t dtmax = Real_t(1.0e-2);
   Real_t stoptime = Real_t(1.0e-2);
 
   // Region information
   Int_t numReg;
   Index_t *regElemSize = nullptr;
   Index_t *regNumList = (Index_t *)malloc(numElem * sizeof(Index_t));
   Index_t **regElemlist = nullptr;
 
   // Node-centered data - allocate with malloc
   Real_t *m_x = (Real_t *)malloc(numNode * sizeof(Real_t));
   Real_t *m_y = (Real_t *)malloc(numNode * sizeof(Real_t));
   Real_t *m_z = (Real_t *)malloc(numNode * sizeof(Real_t));
   Real_t *m_xd = (Real_t *)malloc(numNode * sizeof(Real_t));
   Real_t *m_yd = (Real_t *)malloc(numNode * sizeof(Real_t));
   Real_t *m_zd = (Real_t *)malloc(numNode * sizeof(Real_t));
   Real_t *m_xdd = (Real_t *)malloc(numNode * sizeof(Real_t));
   Real_t *m_ydd = (Real_t *)malloc(numNode * sizeof(Real_t));
   Real_t *m_zdd = (Real_t *)malloc(numNode * sizeof(Real_t));
   Real_t *m_fx = (Real_t *)malloc(numNode * sizeof(Real_t));
   Real_t *m_fy = (Real_t *)malloc(numNode * sizeof(Real_t));
   Real_t *m_fz = (Real_t *)malloc(numNode * sizeof(Real_t));
   Real_t *m_nodalMass = (Real_t *)malloc(numNode * sizeof(Real_t));
 
   // Symmetry plane nodesets (dynamic size)
   Index_t *m_symmX = nullptr;
   Index_t *m_symmY = nullptr;
   Index_t *m_symmZ = nullptr;
   Int_t m_symmX_size = 0;
   Int_t m_symmY_size = 0;
   Int_t m_symmZ_size = 0;
 
   // Element-centered data - allocate with malloc
   Index_t *m_nodelist = (Index_t *)malloc(8 * numElem * sizeof(Index_t));
   Index_t *m_lxim = (Index_t *)malloc(numElem * sizeof(Index_t));
   Index_t *m_lxip = (Index_t *)malloc(numElem * sizeof(Index_t));
   Index_t *m_letam = (Index_t *)malloc(numElem * sizeof(Index_t));
   Index_t *m_letap = (Index_t *)malloc(numElem * sizeof(Index_t));
   Index_t *m_lzetam = (Index_t *)malloc(numElem * sizeof(Index_t));
   Index_t *m_lzetap = (Index_t *)malloc(numElem * sizeof(Index_t));
   Int_t *m_elemBC = (Int_t *)malloc(numElem * sizeof(Int_t));
   Real_t *m_e = (Real_t *)malloc(numElem * sizeof(Real_t));
   Real_t *m_p = (Real_t *)malloc(numElem * sizeof(Real_t));
   Real_t *m_q = (Real_t *)malloc(numElem * sizeof(Real_t));
   Real_t *m_ql = (Real_t *)malloc(numElem * sizeof(Real_t));
   Real_t *m_qq = (Real_t *)malloc(numElem * sizeof(Real_t));
   Real_t *m_v = (Real_t *)malloc(numElem * sizeof(Real_t));
   Real_t *m_volo = (Real_t *)malloc(numElem * sizeof(Real_t));
   Real_t *m_vnew = (Real_t *)malloc(numElem * sizeof(Real_t));
   Real_t *m_delv = (Real_t *)malloc(numElem * sizeof(Real_t));
   Real_t *m_vdov = (Real_t *)malloc(numElem * sizeof(Real_t));
   Real_t *m_arealg = (Real_t *)malloc(numElem * sizeof(Real_t));
   Real_t *m_ss = (Real_t *)malloc(numElem * sizeof(Real_t));
   Real_t *m_elemMass = (Real_t *)malloc(numElem * sizeof(Real_t));
 
   // Temporary arrays (nullptr until needed)
   Real_t *m_dxx = nullptr;
   Real_t *m_dyy = nullptr;
   Real_t *m_dzz = nullptr;
   Real_t *m_delv_xi = nullptr;
   Real_t *m_delv_eta = nullptr;
   Real_t *m_delv_zeta = nullptr;
   Real_t *m_delx_xi = nullptr;
   Real_t *m_delx_eta = nullptr;
   Real_t *m_delx_zeta = nullptr;
 
   // OMP support structures
   Index_t *m_nodeElemStart = nullptr;
   Index_t *m_nodeElemCornerList = nullptr;
 
   // Setup tracking
   Index_t maxPlaneSize, maxEdgeSize;
   Index_t rowMin, rowMax, colMin, colMax, planeMin, planeMax;
 
   // Initialize element properties to zero
   for (Index_t i = 0; i < numElem; ++i) {
     m_e[i] = Real_t(0.0);
     m_p[i] = Real_t(0.0);
     m_q[i] = Real_t(0.0);
     m_ss[i] = Real_t(0.0);
   }
 
   // Initialize volumes to 1.0 (CRITICAL!)
   for (Index_t i = 0; i < numElem; ++i) {
     m_v[i] = Real_t(1.0);
   }
 
   // Initialize node velocities to zero
   for (Index_t i = 0; i < numNode; ++i) {
     m_xd[i] = Real_t(0.0);
     m_yd[i] = Real_t(0.0);
     m_zd[i] = Real_t(0.0);
   }
 
   // Initialize node accelerations to zero
   for (Index_t i = 0; i < numNode; ++i) {
     m_xdd[i] = Real_t(0.0);
     m_ydd[i] = Real_t(0.0);
     m_zdd[i] = Real_t(0.0);
   }
 
   // Initialize nodal mass to zero
   for (Index_t i = 0; i < numNode; ++i) {
     m_nodalMass[i] = Real_t(0.0);
   }
 
   // Setup communication buffers and boundary tracking
   SetupCommBuffers(edgeNodes, &maxPlaneSize, &maxEdgeSize, &rowMin, &rowMax,
                    &colMin, &colMax, &planeMin, &planeMax, sizeX, sizeY, sizeZ,
                    colLoc, rowLoc, planeLoc, tp);
 
   // Build mesh (populate m_x, m_y, m_z, m_nodelist)
   BuildMesh(nx, edgeNodes, edgeElems, m_nodelist, m_lxim, m_lxip, m_letam,
             m_letap, m_lzetam, m_lzetap, m_x, m_y, m_z, colLoc, rowLoc,
             planeLoc, sizeX, sizeY, sizeZ, &rowMin, &rowMax, &colMin, &colMax,
             &planeMin, &planeMax, tp);
 
   // Setup node-element indexing structures (inlined)
   {
     // set up node-centered indexing of elements
     Index_t *nodeElemCount = (Index_t *)malloc(numNode * sizeof(Index_t));
 
     for (Index_t i = 0; i < numNode; ++i) {
       nodeElemCount[i] = 0;
     }
 
     for (Index_t i = 0; i < numElem; ++i) {
       Index_t i8 = 8 * i;
       for (Index_t j = 0; j < 8; ++j) {
         ++(nodeElemCount[m_nodelist[i8 + j]]);
       }
     }
 
     m_nodeElemStart = (Index_t *)malloc((numNode + 1) * sizeof(Index_t));
 
     m_nodeElemStart[0] = 0;
 
     for (Index_t i = 1; i <= numNode; ++i) {
       m_nodeElemStart[i] = m_nodeElemStart[i - 1] + nodeElemCount[i - 1];
     }
 
     m_nodeElemCornerList =
         (Index_t *)malloc(m_nodeElemStart[numNode] * sizeof(Index_t));
 
     for (Index_t i = 0; i < numNode; ++i) {
       nodeElemCount[i] = 0;
     }
 
     for (Index_t i = 0; i < numElem; ++i) {
       Index_t i8 = 8 * i;
       for (Index_t j = 0; j < 8; ++j) {
         Index_t m = m_nodelist[i8 + j];
         Index_t k = i8 + j;
         Index_t offset = m_nodeElemStart[m] + nodeElemCount[m];
         m_nodeElemCornerList[offset] = k;
         ++(nodeElemCount[m]);
       }
     }
 
     Index_t clSize = m_nodeElemStart[numNode];
     for (Index_t i = 0; i < clSize; ++i) {
       Index_t clv = m_nodeElemCornerList[i];
       if ((clv < 0) || (clv > numElem * 8)) {
         fprintf(stderr, "AllocateNodeElemIndexes(): nodeElemCornerList entry "
                         "out of range!\n");
         exit(-1);
       }
     }
 
     free(nodeElemCount);
   }
 
   // Create region index sets (inlined)
   {
     srand(0);
     Index_t myRank = 0;
     numReg = opt_numReg;
     regElemSize = (Index_t *)malloc(numReg * sizeof(Index_t));
     Index_t nextIndex = 0;
     // if we only have one region just fill it
     //  Fill out the regNumList with material numbers, which are always
     //  the region index plus one
     if (numReg == 1) {
       while (nextIndex < numElem) {
         regNumList[nextIndex] = 1;
         nextIndex++;
       }
       regElemSize[0] = 0;
     }
     // If we have more than one region distribute the elements.
     else {
       Int_t regionNum;
       Int_t regionVar;
       Int_t lastReg = -1;
       Int_t binSize;
       Index_t elements;
       Index_t runto = 0;
       Int_t costDenominator = 0;
       Int_t *regBinEnd = (Int_t *)malloc(numReg * sizeof(Int_t));
       // Determine the relative weights of all the regions.  This is based off
       // the -b flag.  Balance is the value passed into b.
       for (Index_t i = 0; i < numReg; ++i) {
         regElemSize[i] = 0;
         costDenominator +=
             IPOW(i + 1, balance);       // Total sum of all regions weights
         regBinEnd[i] = costDenominator; // Chance of hitting a given region is
                                         // (regBinEnd[i]
                                         // - regBinEdn[i-1])/costDenominator
       }
       // Until all elements are assigned
       while (nextIndex < numElem) {
         // pick the region
         regionVar = rand() % costDenominator;
         Index_t i = 0;
         while (regionVar >= regBinEnd[i])
           i++;
         // rotate the regions based on MPI rank.  Rotation is Rank % NumRegions
         // this makes each domain have a different region with the highest
         // representation
         regionNum = ((i + myRank) % numReg) + 1;
         // make sure we don't pick the same region twice in a row
         while (regionNum == lastReg) {
           regionVar = rand() % costDenominator;
           i = 0;
           while (regionVar >= regBinEnd[i])
             i++;
           regionNum = ((i + myRank) % numReg) + 1;
         }
         // Pick the bin size of the region and determine the number of elements.
         binSize = rand() % 1000;
         if (binSize < 773) {
           elements = rand() % 15 + 1;
         } else if (binSize < 937) {
           elements = rand() % 16 + 16;
         } else if (binSize < 970) {
           elements = rand() % 32 + 32;
         } else if (binSize < 974) {
           elements = rand() % 64 + 64;
         } else if (binSize < 978) {
           elements = rand() % 128 + 128;
         } else if (binSize < 981) {
           elements = rand() % 256 + 256;
         } else
           elements = rand() % 1537 + 512;
         runto = elements + nextIndex;
         // Store the elements.  If we hit the end before we run out of elements
         // then just stop.
         while (nextIndex < runto && nextIndex < numElem) {
           regNumList[nextIndex] = regionNum;
           nextIndex++;
         }
         lastReg = regionNum;
       }
       free(regBinEnd);
     }
     // Convert regNumList to region index sets
     // First, count size of each region
     for (Index_t i = 0; i < numElem; ++i) {
       int r = regNumList[i] - 1; // region index == regnum-1
       regElemSize[r]++;
     }
     // Second, allocate each region index set
     Int_t maxRegElemSize = 0;
     for (Index_t i = 0; i < numReg; ++i) {
       if (regElemSize[i] > maxRegElemSize) {
         maxRegElemSize = regElemSize[i];
       }
     }
     regElemlist = (Index_t **)malloc(numReg * sizeof(Index_t *));
     for (Index_t i = 0; i < numReg; ++i) {
       regElemlist[i] = (Index_t *)malloc(maxRegElemSize * sizeof(Index_t));
       regElemSize[i] = 0;
     }
     // Third, fill index sets
     for (Index_t i = 0; i < numElem; ++i) {
       Index_t r = regNumList[i] - 1;     // region index == regnum-1
       Index_t regndx = regElemSize[r]++; // Note increment
       regElemlist[r][regndx] = i;
     }
   }
 
   // Setup symmetry planes
   SetupSymmetryPlanes(edgeNodes, &m_symmX, &m_symmY, &m_symmZ, &m_symmX_size,
                       &m_symmY_size, &m_symmZ_size, colLoc, rowLoc, planeLoc,
                       sizeX, sizeY, sizeZ);
 
   // Setup element connectivities
   SetupElementConnectivities(edgeElems, m_lxim, m_lxip, m_letam, m_letap,
                              m_lzetam, m_lzetap, m_nodelist, sizeX, sizeY,
                              sizeZ);
 
   // Setup boundary conditions
   SetupBoundaryConditions(edgeElems, m_elemBC, m_lxim, m_lxip, m_letam, m_letap,
                           m_lzetam, m_lzetap, colLoc, rowLoc, planeLoc, sizeX,
                           sizeY, sizeZ, numRanks, planeMin, planeMax, rowMin,
                           rowMax, colMin, colMax, tp);
 
   // Compute initial volumes and masses
   for (Index_t i = 0; i < numElem; ++i) {
     Real_t x_local[8], y_local[8], z_local[8];
     Index_t i8 = 8 * i;
     for (Index_t lnode = 0; lnode < 8; ++lnode) {
       Index_t gnode = m_nodelist[i8 + lnode];
       x_local[lnode] = m_x[gnode];
       y_local[lnode] = m_y[gnode];
       z_local[lnode] = m_z[gnode];
     }
     Real_t volume = CalcElemVolume(x_local, y_local, z_local);
     m_volo[i] = volume;
     m_elemMass[i] = volume;
     for (Index_t j = 0; j < 8; ++j) {
       Index_t idx = m_nodelist[i8 + j];
       m_nodalMass[idx] += volume / Real_t(8.0);
     }
   }
 
   // Deposit initial energy (Sedov blast)
   const Real_t ebase = Real_t(3.948746e+7);
   Real_t scale = (nx * tp) / Real_t(45.0);
   Real_t einit = ebase * scale * scale * scale;
   if (rowLoc + colLoc + planeLoc == 0) {
     m_e[0] = einit;
   }
 
   // Set initial deltatime based on analytic CFL calculation
   deltatime = (Real_t(0.5) * cbrt(m_volo[0])) / sqrt(Real_t(2.0) * einit);
 
   // BEGIN timestep to solution */
   timeval start;
   gettimeofday(&start, nullptr);
 
   // debug to see region sizes
   //    for(Int_t i = 0; i < locDom->numReg(); i++)
   //       std::cout << "region" << i + 1<< "size" << locDom->regElemSize(i)
   //       <<std::endl;
   while ((time < stoptime) && (cycle < its)) {
 
     if (cycle > 40) {
       /*
       for(int node_id = 0; node_id < locDom->numNode(); node_id++) {
               printf("%d) force prime: (%2.6e, %2.6e, %2.6e)\n", node_id,
                               locDom->fx(node_id), locDom->fy(node_id),
       locDom->fz(node_id)); printf("%d) velocity: (%2.6e, %2.6e, %2.6e)\n",
       node_id, locDom->xd(node_id), locDom->yd(node_id), locDom->zd(node_id));
               printf("%d) position: (%2.6e, %2.6e, %2.6e)\n", node_id,
                               locDom->x(node_id), locDom->y(node_id),
       locDom->z(node_id));
       }
 
       for(int element_id = 0; element_id < locDom->numElem(); ++element_id) {
 
               printf("%d) volume: ", element_id);
               printf("%2.6e\n", locDom->v(element_id));
 
               printf("%d) element_charastic_length: ", element_id);
               printf("%2.6e\n", locDom->arealg(element_id));
 
               printf("%d) volume derivative: ", element_id);
               printf("%2.6e\n", locDom->vdov(element_id));
 
               printf("%d) velocity gradient: (%2.6e, %2.6e, %2.6e)\n",
       element_id, locDom->delv_xi(element_id), locDom->delv_eta(element_id),
                               locDom->delv_zeta(element_id));
 
               printf("%d) position gradient: (%2.6e, %2.6e, %2.6e)\n",
       element_id, locDom->delx_xi(element_id), locDom->delx_eta(element_id),
                               locDom->delx_zeta(element_id));
 
               printf("%d) linear viscosity term: ", element_id);
               printf("%2.6e\n", locDom->ql(element_id));
 
               printf("%d) quadratic viscosity term: ", element_id);
               printf("%2.6e\n", locDom->qq(element_id));
 
               printf("%d) energy: ", element_id);
               printf("%2.6e\n", locDom->e(element_id));
 
               printf("%d) pressure: ", element_id);
               printf("%2.6e\n", locDom->p(element_id));
 
               printf("%d) viscosity: ", element_id);
               printf("%2.6e\n", locDom->q(element_id));
 
               printf("%d) sound speed: ", element_id);
               printf("%2.6e\n", locDom->ss(element_id));
       }
 */
     }
 
     TimeIncrement(&stoptime, &time, &dtfixed, &cycle, &deltatime, &dtcourant,
                   &dthydro, deltatimemultlb, deltatimemultub, dtmax);
 
     printf("iteration %d, delta time %f, energy %f\n", cycle, deltatime,
            m_e[0]);
 
     LagrangeLeapFrog(
         m_x, m_y, m_z, m_xd, m_yd, m_zd, m_xdd, m_ydd, m_zdd, m_fx, m_fy, m_fz,
         m_nodalMass, m_nodelist, m_volo, m_v, m_delv, m_arealg, m_vdov, m_e,
         m_p, m_q, m_qq, m_ql, m_ss, m_elemMass, m_elemBC, m_lxim, m_lxip,
         m_letam, m_letap, m_lzetam, m_lzetap, m_nodeElemStart,
         m_nodeElemCornerList, m_symmX, m_symmY, m_symmZ, (m_symmX_size == 0),
         (m_symmY_size == 0), (m_symmZ_size == 0), numReg, regElemSize,
         regElemlist, cost, numElem, numNode, sizeX, sizeY, sizeZ, deltatime,
         u_cut, v_cut, hgcoef, monoq_limiter_mult, monoq_max_slope, qlc_monoq,
         qqc_monoq, qstop, eosvmin, eosvmax, e_cut, p_cut, ss4o3, q_cut, pmin,
         emin, refdens, qqc, dvovmax, &dtcourant, &dthydro);
 
     for (int element_id = 0; element_id < numElem; ++element_id) {
 
       //		printf("%2.6e, %2.6e, %2.6e\n", m_dxx[element_id],
       // m_dyy[element_id], m_dzz[element_id]);
     }
 
     if ((showProg != 0) && (quiet == 0) && (myRank == 0)) {
       printf("cycle = %d, time = %e, dt=%e\n", cycle, double(time),
              double(deltatime));
     }
   }
 
   // Use reduced max elapsed time
   double elapsed_time;
   timeval end;
   gettimeofday(&end, nullptr);
   elapsed_time = (double)(end.tv_sec - start.tv_sec) +
                  ((double)(end.tv_usec - start.tv_usec)) / 1000000;
   double elapsed_timeG;
   elapsed_timeG = elapsed_time;
 
   if ((myRank == 0) && (quiet == 0)) {
     VerifyAndWriteFinalOutput(elapsed_timeG, cycle, m_e, nx, numRanks);
   }
 
   // Free all allocated memory
   free(m_x);
   free(m_y);
   free(m_z);
   free(m_xd);
   free(m_yd);
   free(m_zd);
   free(m_xdd);
   free(m_ydd);
   free(m_zdd);
   free(m_fx);
   free(m_fy);
   free(m_fz);
   free(m_nodalMass);
   free(m_symmX);
   free(m_symmY);
   free(m_symmZ);
   free(m_nodelist);
   free(m_lxim);
   free(m_lxip);
   free(m_letam);
   free(m_letap);
   free(m_lzetam);
   free(m_lzetap);
   free(m_elemBC);
   free(m_e);
   free(m_p);
   free(m_q);
   free(m_ql);
   free(m_qq);
   free(m_v);
   free(m_volo);
   free(m_vnew);
   free(m_delv);
   free(m_vdov);
   free(m_arealg);
   free(m_ss);
   free(m_elemMass);
   free(regNumList);
   free(m_nodeElemStart);
   free(m_nodeElemCornerList);
   for (Int_t r = 0; r < numReg; ++r) {
     free(regElemlist[r]);
   }
   free(regElemlist);
   free(regElemSize);
 
   return 0;
 }
 