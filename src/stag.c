#include <petsc.h>
#include <petscsys.h>
#include <petscvec.h>
static char help[] = "hello";
int main(int argc, char **args) {
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  DM dm;
  Mat A;
  Vec x;
  // PetscReal change = 1, cost;
  PetscInt M = 5, N = 7, size;
  PetscInt startx, starty, nx, ny, ex, ey, sum = 0;
  PetscReal ***array;
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                         DMDA_STENCIL_BOX, M, N, PETSC_DECIDE, PETSC_DECIDE, 2,
                         1, NULL, NULL, &dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMCreateMatrix(dm, &A));
  PetscCall(DMDAGetCorners(dm, &startx, &starty, NULL, &nx, &ny, NULL));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "%d,%d\n", startx, starty));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "%d,%d\n", nx, ny));
  PetscCall(DMCreateGlobalVector(dm, &x));
  PetscCall(DMDAVecGetArrayDOF(dm, x, &array));
  for (ey = starty; ey < starty + ny; ey++) {
    for (ex = startx; ex < startx + nx; ex++) {
      array[ey][ex][0] = ex;
      array[ey][ex][1] = ey;
      sum += 1;
    }
  }

  PetscCall(PetscPrintf(PETSC_COMM_SELF, "sum: %d\n", sum));
  PetscCall(DMDAVecRestoreArrayDOF(dm, x, &array));
  PetscCall(VecGetSize(x, &size));

  PetscCall(VecView(x, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecDestroy(&x));
  PetscCall(MatDestroy(&A));
  PetscCall(DMDestroy(&dm));

  PetscCall(PetscFinalize());
  return 0;
}