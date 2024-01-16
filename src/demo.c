#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmdatypes.h>
#include <petscdmtypes.h>
#include <petscerror.h>
#include <petscsys.h>
#include <petscvec.h>
#include <petscviewer.h>
static char help[] = "hello";
int main(int argc, char **args) {
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  DM dm;
  Vec a, b;
  PetscInt M = 5, N = 4;
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                         DMDA_STENCIL_BOX, M, N, PETSC_DECIDE, PETSC_DECIDE, 1,
                         1, NULL, NULL, &dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMCreateGlobalVector(dm, &a));
  PetscCall(DMCreateGlobalVector(dm, &b));
  PetscCall(VecSet(a, 1));
  PetscCall(VecCopy(a, b));
  PetscCall(VecView(b, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscFinalize());
}