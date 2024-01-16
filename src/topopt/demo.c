#include "func.h"
#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmdatypes.h>
#include <petscdmtypes.h>
#include <petscerror.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscvec.h>
#include <petscviewer.h>
static char help[] = "This is a demo.";
int main(int argc, char **args) {
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  DM dm;
  PetscInt M = 3, N = 3;
  Mat A;
  Vec x;
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                         DMDA_STENCIL_BOX, M, N, PETSC_DECIDE, PETSC_DECIDE, 2,
                         1, NULL, NULL, &dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMCreateMatrix(dm, &A));
  PetscCall(DMCreateGlobalVector(dm, &x));
  PetscCall(VecSet(x, 0.5));
  PetscCall(formMatrix(dm, A, x, M, N));
  PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(PetscFinalize());
}