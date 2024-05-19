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

int main(int argc, char **argv) {
  PetscCall(
      PetscInitialize(&argc, &argv, (char *)0, "Toplogical Optimiazation\n"));
  PCCtx test, test2;
  PetscInt grid = 2, m1, n1;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mesh", &grid, NULL));
  PetscInt mesh[3] = {grid, grid, grid};
  PetscInt mesh2[3] = {grid * 2, grid * 2, grid * 2};
  PetscInt ex, ey, ez, nx, ny, nz, startx, starty, startz;
  PetscScalar dom[3] = {1.0, 1.0, 1.0};
  PetscScalar ***arrayt1;
  Vec t1, t2;
  Mat A;
  PetscCall(PC_init(&test, dom, mesh));
  PetscCall(PC_init(&test2, dom, mesh2));
  PetscCall(DMCreateInterpolation(test.dm, test2.dm, &A, NULL));
  PetscCall(MatGetSize(A, &m1, &n1));
  PetscPrintf(PETSC_COMM_WORLD, "m1: %d, n1: %d\n", m1, n1);

  PetscCall(DMCreateGlobalVector(test.dm, &t1));
  PetscCall(DMDAGetCorners(test.dm, &startx, &starty, &startz, &nx, &ny, &nz));

  PetscCall(DMDAVecGetArray(test.dm, t1, &arrayt1));
  for (ez = startz; ez < startz + nz; ++ez) {
    for (ey = starty; ey < starty + ny; ++ey) {
      for (ex = startx; ex < startx + nx; ++ex) {
        if (ex == 0 && ey == 0 && ez == 0) {
          arrayt1[ez][ey][ex] = 1.0;
        }
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(test.dm, t1, &arrayt1));

  PetscCall(DMCreateGlobalVector(test2.dm, &t2));

  PetscCall(MatMult(A, t1, t2));
  PetscCall(VecView(t1, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecView(t2, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscFinalize());
}