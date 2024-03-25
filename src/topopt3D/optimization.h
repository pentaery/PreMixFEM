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

typedef struct mma_text {
  Vec mmaL, mmaU, mmaLlast, mmaUlast, alpha, beta, xlast, xllast, xlllast;
} MMAx;

PetscErrorCode mmaInit(PCCtx *s_ctx, MMAx *mma_text);

PetscErrorCode adjointGradient(PCCtx *s_ctx, Mat A, Vec x, Vec t, Vec dc,
                               PetscInt penal);
PetscErrorCode adjointGradient1(PCCtx *s_ctx, Mat A, Vec x, Vec t, Vec dc,
                                PetscInt penal);

PetscErrorCode formLimit(PCCtx *s_ctx, PetscInt loop, Vec xlast, Vec xllast,
                         Vec xlllast, Vec mmaL, Vec mmaU, Vec mmaLlast,
                         Vec mmaUlast, Vec alpha, Vec beta);
PetscErrorCode computeCostMMA(PCCtx *s_ctx, Vec t, PetscScalar *cost);
PetscErrorCode computeDerivative(PCCtx *s_ctx, PetscScalar y,
                                 PetscScalar *derivative, Vec xlast, Vec mmaU,
                                 Vec mmaL, Vec dc, Vec alpha, Vec beta, Vec x);
PetscErrorCode mma(PCCtx *s_ctx, Vec xlast, Vec mmaU, Vec mmaL, Vec dc,
                   Vec alpha, Vec beta, Vec x, PetscScalar *initial);

PetscErrorCode findX(PCCtx *s_ctx, PetscScalar y, Vec xlast, Vec mmaU, Vec mmaL,
                     Vec dc, Vec alpha, Vec beta, Vec x);

PetscErrorCode mmatest(PCCtx *s_ctx, Vec xlast, Vec mmaU, Vec mmaL, Vec dc,
                       Vec alpha, Vec beta, Vec x, PetscScalar *initial);

PetscErrorCode computeWy(PCCtx *s_ctx, PetscScalar y, PetscScalar *derivative,
                         Vec xlast, Vec mmaU, Vec mmaL, Vec dc, Vec alpha,
                         Vec beta, Vec x);