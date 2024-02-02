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

#define tD 1e3
#define rmin 1.5

PetscErrorCode formx(PCCtx *s_ctx, Vec x);
PetscErrorCode formkappa(PCCtx *s_ctx);
PetscErrorCode formMatrix(PCCtx *s_ctx, Mat A);
PetscErrorCode formRHS(PCCtx *s_ctx, Vec rhs);
PetscErrorCode computeCost(PCCtx *s_ctx, Vec x, Vec t, Vec c, Vec dc);
PetscErrorCode filter(PCCtx *s_ctx, Vec dc);
// PetscErrorCode optimalCriteria(DM dm, Vec x, Vec dc, PetscScalar volfrac);
