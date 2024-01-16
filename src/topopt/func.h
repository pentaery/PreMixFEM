#pragma once
#include <math.h>
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

PetscErrorCode formKE(PetscReal KE[8][8], PetscReal coef);
PetscErrorCode formMatrix(DM dm, Mat A, Vec x, PetscInt M, PetscInt N);
PetscErrorCode formRHS(DM dm, Vec rhs, PetscInt N);
PetscErrorCode computeCost(DM dm, PetscReal *cost, Vec u, Vec dc, Vec x);
PetscErrorCode filter(DM dm, Vec dc, Vec x, PetscInt M, PetscInt N);
PetscErrorCode optimalCriteria(DM dm, Vec x, Vec dc, PetscReal volfrac);