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

PetscErrorCode formx(PCCtx *s_ctx);
PetscErrorCode formkappa(PCCtx *s_ctx);
PetscErrorCode formMatrix(PCCtx *s_ctx, Mat A);
PetscErrorCode formRHS(DM dm, Vec rhs, PetscInt N);
PetscErrorCode computeCost(DM dm, PetscScalar *cost, Vec u, Vec dc, Vec x);
PetscErrorCode filter(DM dm, Vec dc, Vec x);
PetscErrorCode optimalCriteria(DM dm, Vec x, Vec dc, PetscScalar volfrac);
