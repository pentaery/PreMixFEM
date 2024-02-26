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
#define rmin 1.5
#define volfrac 0.2
#define cr 6
PetscErrorCode formBoundary(PCCtx *s_ctx);
PetscErrorCode formkappa(PCCtx *s_ctx, Vec x);
PetscErrorCode formMatrix(PCCtx *s_ctx, Mat A);
PetscErrorCode formRHS(PCCtx *s_ctx, Vec rhs, Vec x);
PetscErrorCode computeGradient(PCCtx *s_ctx, Vec x, Vec t, Vec dc);
PetscErrorCode filter(PCCtx *s_ctx, Vec dc, Vec x);
PetscErrorCode computeCost(PCCtx *s_ctx, Vec t, Vec rhs,
                           PetscScalar *cost);
PetscErrorCode optimalCriteria(PCCtx *s_ctx, Vec x, Vec dc,
                               PetscScalar *change);
PetscErrorCode mma(PCCtx *s_ctx, Vec x, Vec dc,
                               PetscScalar *change);
PetscErrorCode genoptimalCriteria(PCCtx *s_ctx, Vec x, Vec dc,
                               PetscScalar *change);
PetscErrorCode computeCost1(PCCtx *s_ctx, Vec t, PetscScalar *cost);

PetscErrorCode formBoundarytest(PCCtx *s_ctx);
PetscErrorCode formkappatest(PCCtx *s_ctx, Vec x);
PetscErrorCode formRHStest(PCCtx *s_ctx, Vec rhs, Vec x);
PetscErrorCode formMatrixtest(PCCtx *s_ctx, Mat A);
PetscErrorCode computeError(PCCtx *s_ctx, Vec t, PetscScalar *error);