#include "PreMixFEM_3D.h"
#include "optimization.h"
#include "system.h"
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmdatypes.h>
#include <petscdmtypes.h>
#include <petscerror.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscmath.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <petscviewerhdf5.h>

PetscErrorCode xScaling(DM dm1, DM dm2, Vec x1, Vec x2);

int main(int argc, char **argv) {
  PetscCall(
      PetscInitialize(&argc, &argv, (char *)0, "Toplogical Optimiazation\n"));

  DM dm1, dm2;
  Vec x1, x2;
  PetscInt m = 10, n = 20;
  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                         DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, m, m, m,
                         PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL,
                         NULL, NULL, &dm1));
  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                         DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, n, n, n,
                         PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL,
                         NULL, NULL, &dm2));
  PetscCall(DMCreateGlobalVector(dm1, &x1));
  PetscCall(DMCreateGlobalVector(dm2, &x2));
  PetscCall(xScaling(dm1, dm2, x1, x2));
  PetscCall(PetscFinalize());
}
