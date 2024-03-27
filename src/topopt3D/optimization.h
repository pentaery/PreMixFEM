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

typedef struct mma_text {
  Vec mmaL, mmaU, mmaLlast, mmaUlast, alpha, beta, xlast, xllast, xlllast,lbd,ubd;
} MMAx;

PetscErrorCode mmaInit(PCCtx *s_ctx, MMAx *mma_text);
PetscErrorCode mmaFinal(MMAx *mma_text);
PetscErrorCode adjointGradient(PCCtx *s_ctx, Mat A, Vec x, Vec t, Vec dc,
                               PetscInt penal);
PetscErrorCode adjointGradient1(PCCtx *s_ctx, Mat A, Vec x, Vec t, Vec dc,
                                PetscInt penal);

PetscErrorCode formLimit(PCCtx *s_ctx, MMAx *mma_text, PetscInt loop);
PetscErrorCode computeCostMMA(PCCtx *s_ctx, Vec t, PetscScalar *cost);
PetscErrorCode computeDerivative(PCCtx *s_ctx, PetscScalar y,
                                 PetscScalar *derivative, MMAx *mma_text,
                                 Vec dc, Vec x);
PetscErrorCode mma(PCCtx *s_ctx, MMAx *mma_text, Vec dc, Vec x,
                   PetscScalar *initial);

PetscErrorCode findX(PCCtx *s_ctx, PetscScalar y, MMAx *mma_text, Vec dc,
                     Vec x);
PetscErrorCode computeChange(MMAx *mma_text, Vec x, PetscScalar *change);

PetscErrorCode mmatest(PCCtx *s_ctx, MMAx *mma_text, Vec dc, Vec x,
                       PetscScalar *initial);

PetscErrorCode computeWy(PCCtx *s_ctx, PetscScalar y, PetscScalar *wy,
                         MMAx *mma_text, Vec dc, Vec x);