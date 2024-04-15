#pragma once
#include <PreMixFEM_3D.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmdatypes.h>
#include <petscdmtypes.h>
#include <petscerror.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <petscviewer.h>

#define mmas 0.85
#define mmas0 0.15
#define artificial 1
#define m 1

typedef struct mma_text {
  Vec mmaL, mmaU, mmaLlast, mmaUlast, alpha, beta, xlast, xllast, xlllast, lbd,
      ubd, xsign, dgT, zzz1, zzz2, zzz;
  Vec p0, q0, p[m], q[m], b[m], g[m];
  PetscScalar bval[m];
  Vec rex, dx, delx, diagx;
  PetscScalar y[m], rey[m], dy[m], dely[m];
  PetscScalar z, rez, dz, delz;
  PetscScalar lam[m], relam[m], dlam[m];
  Vec xsi, eta, rexsi, reeta, gvec[m], dxsi, deta;
  PetscScalar mu[m], remu[m], dmu[m];
  PetscScalar zet, rezet, dzet;
  PetscScalar s[m], res[m], ds[m];
  PetscScalar c[m], d[m], a[m], a0;
  Vec temp, temp1, temp2, temp3;
} MMAx;

PetscErrorCode mmaInit(PCCtx *s_ctx, MMAx *mma_text);
PetscErrorCode mmaFinal(MMAx *mma_text);
PetscErrorCode adjointGradient(PCCtx *s_ctx, MMAx *mma_text, Mat A, Vec x,
                               Vec t, Vec dc, PetscInt penal);
PetscErrorCode adjointGradient1(PCCtx *s_ctx, Mat A, Vec x, Vec t, Vec dc,
                                PetscInt penal);

PetscErrorCode formLimit(PCCtx *s_ctx, MMAx *mma_text, PetscInt loop);
PetscErrorCode computeCostMMA(PCCtx *s_ctx, Vec t, PetscScalar *cost);
PetscErrorCode computeDerivative(PCCtx *s_ctx, PetscScalar y,
                                 PetscScalar *derivative, PetscScalar *dpartial,
                                 MMAx *mma_text, Vec dc, Vec x);
PetscErrorCode computeDerivative1(PCCtx *s_ctx, PetscScalar y,
                                  PetscScalar *derivative, MMAx *mma_text,
                                  Vec dc, Vec x);
PetscErrorCode mma(PCCtx *s_ctx, MMAx *mma_text, Vec dc, Vec x,
                   PetscScalar *initial);
PetscErrorCode mma1(PCCtx *s_ctx, MMAx *mma_text, Vec dc, Vec x,
                    PetscScalar *initial);
PetscErrorCode mma2(PCCtx *s_ctx, MMAx *mma_text, Vec dc, Vec x,
                    PetscScalar *initial);

PetscErrorCode findX(PCCtx *s_ctx, PetscScalar y, MMAx *mma_text, Vec dc,
                     Vec x);
PetscErrorCode computeChange(MMAx *mma_text, Vec x, PetscScalar *change);

PetscErrorCode mmatest(PCCtx *s_ctx, MMAx *mma_text, Vec dc, Vec x,
                       PetscScalar *initial);

PetscErrorCode computeWy(PCCtx *s_ctx, PetscScalar y, PetscScalar *wy,
                         MMAx *mma_text, Vec dc, Vec x);