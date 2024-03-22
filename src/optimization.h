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

#define tD 1e2
#define rmin 1.1
#define volfrac 0.1
#define f0 1
#define penal 3
#define mmas 0.85
#define mmas0 0.15
#define xCont 1e-6
#define kH 1
#define kL 1e-6

PetscErrorCode formBoundary(PCCtx *s_ctx);
PetscErrorCode formkappa(PCCtx *s_ctx, Vec x);
PetscErrorCode formMatrix(PCCtx *s_ctx, Mat A);
PetscErrorCode formRHS(PCCtx *s_ctx, Vec rhs, Vec x);
PetscErrorCode computeGradient(PCCtx *s_ctx, Vec x, Vec t, Vec dc);
PetscErrorCode adjointGradient(PCCtx *s_ctx, Mat A, Vec x, Vec t, Vec dc);
PetscErrorCode adjointGradient1(PCCtx *s_ctx, Mat A, Vec x, Vec t, Vec dc);
PetscErrorCode filter(PCCtx *s_ctx, Vec dc, Vec x);
PetscErrorCode computeCost(PCCtx *s_ctx, Vec t, Vec rhs, PetscScalar *cost);
PetscErrorCode optimalCriteria(PCCtx *s_ctx, Vec x, Vec dc,
                               PetscScalar *change);

PetscErrorCode genOptimalCriteria(PCCtx *s_ctx, Vec x, Vec dc, PetscScalar *g,
                                  PetscScalar *glast, PetscScalar *lmid,
                                  PetscScalar *change, PetscScalar cost0);
PetscErrorCode computeCost1(PCCtx *s_ctx, Vec t, PetscScalar *cost);
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