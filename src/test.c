#include "PreMixFEM_3D.h"
#include "optimization.h"
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
  PCCtx test;
  PetscInt grid = 2;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mesh", &grid, NULL));
  PetscInt mesh[3] = {grid, grid, grid};
  PetscScalar dom[3] = {1.0, 1.0, 1.0};
  PetscScalar cost = 0, change = 0, error = 0;
  Mat A;
  Vec rhs, t, x, dc;
  KSP ksp;

  PetscCall(PC_init(&test, dom, mesh));
  PetscCall(PC_print_info(&test));

  PetscCall(DMCreateMatrix(test.dm, &A));
  PetscCall(DMCreateGlobalVector(test.dm, &rhs));
  PetscCall(DMCreateGlobalVector(test.dm, &x));
  PetscCall(VecSet(x, 0.5));
  PetscCall(DMCreateGlobalVector(test.dm, &t));
  PetscCall(DMCreateGlobalVector(test.dm, &dc));

  PetscCall(formBoundary(&test));
  PetscCall(formkappa(&test, x));
  PetscCall(formMatrix(&test, A));
  PetscCall(formRHS(&test, rhs, x));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, rhs, t));
  // PetscCall(VecView(rhs, PETSC_VIEWER_STDOUT_WORLD));
  // PetscCall(VecView(t, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(computeCost(&test, t, rhs, &cost));
  PetscPrintf(PETSC_COMM_WORLD, "cost: %f\n", cost);
  PetscCall(computeCost1(&test, t, &cost));
  PetscPrintf(PETSC_COMM_WORLD, "cost: %f\n", cost);
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&t));
  PetscCall(VecDestroy(&dc));
  PetscCall(VecDestroy(&x));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(PetscFinalize());
}