#pragma once
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

PetscErrorCode formx(DM dm, Vec x);
PetscErrorCode formkappa(DM dm, Vec x, Vec kappa);
PetscErrorCode formMatrix(DM dm, Mat A, Vec kappa);
PetscErrorCode formRHS(DM dm, Vec rhs, PetscInt N);
PetscErrorCode computeCost(DM dm, PetscScalar *cost, Vec u, Vec dc, Vec x);
PetscErrorCode filter(DM dm, Vec dc, Vec x);
PetscErrorCode optimalCriteria(DM dm, Vec x, Vec dc, PetscScalar volfrac);
