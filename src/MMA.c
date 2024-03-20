#include "PreMixFEM_3D.h"
#include "optimization.h"
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmdatypes.h>
#include <petscdmtypes.h>
#include <petscerror.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscoptions.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <petscviewerhdf5.h>

int main(int argc, char **argv) {
  PetscCall(
      PetscInitialize(&argc, &argv, (char *)0, "Toplogical Optimiazation\n"));
  PCCtx test;
  PetscInt grid = 20;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mesh", &grid, NULL));
  PetscInt mesh[3] = {grid, grid, grid};
  PetscScalar dom[3] = {1.0, 1.0, 1.0};
  PetscScalar cost = 0;
  Mat A;
  Vec rhs, t, x, dc, mmaL, mmaLlast, mmaU, mmaUlast, xlast, xllast, xlllast,
      alpha, beta;
  KSP ksp;
  PetscInt loop = 0, iter = 0;
  PetscScalar change = 1;

  char str[80];

  PetscCall(PC_init(&test, dom, mesh));
  PetscCall(PC_print_info(&test));

  PetscCall(DMCreateMatrix(test.dm, &A));
  PetscCall(DMCreateGlobalVector(test.dm, &rhs));
  PetscCall(DMCreateGlobalVector(test.dm, &x));
  PetscCall(DMCreateGlobalVector(test.dm, &t));
  PetscCall(DMCreateGlobalVector(test.dm, &dc));
  PetscCall(DMCreateGlobalVector(test.dm, &mmaL));
  PetscCall(DMCreateGlobalVector(test.dm, &mmaU));
  PetscCall(DMCreateGlobalVector(test.dm, &xlast));
  PetscCall(DMCreateGlobalVector(test.dm, &xllast));
  PetscCall(DMCreateGlobalVector(test.dm, &xlllast));
  PetscCall(DMCreateGlobalVector(test.dm, &mmaLlast));
  PetscCall(DMCreateGlobalVector(test.dm, &mmaUlast));
  PetscCall(DMCreateGlobalVector(test.dm, &alpha));
  PetscCall(DMCreateGlobalVector(test.dm, &beta));

  PetscCall(VecSet(xlast, volfrac));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(
      KSPSetTolerances(ksp, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
  PetscCall(KSPSetFromOptions(ksp));
  //   PetscCall(KSPSetUp(ksp));
  PetscCall(VecSet(x, 0.1));
  PetscCall(formBoundary(&test));

  while (change > 0.01) {
    loop += 1;
    PetscScalar initial = 0;
    // PetscViewer viewer;
    // sprintf(str, "../data/output/change%04d.vtr", loop);
    // PetscCall(
    //     PetscViewerVTKOpen(PETSC_COMM_WORLD, str, FILE_MODE_WRITE, &viewer));
    // PetscCall(VecView(x, viewer));
    // PetscCall(PetscViewerDestroy(&viewer));
    // PetscCall(VecView(x, PETSC_VIEWER_STDOUT_WORLD));
    if (loop == 2) {
      break;
    }
    PetscCall(formkappa(&test, x));
    PetscCall(formMatrix(&test, A));
    PetscCall(formRHS(&test, rhs, x));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSolve(ksp, rhs, t));
    PetscCall(KSPGetIterationNumber(ksp, &iter));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "iteration number: %d\n", iter));
    PetscCall(computeCostMMA(&test, t, &cost));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "cost: %f\n", cost));
    PetscCall(adjointGradient(&test, A, x, t, dc));
    PetscCall(formLimit(&test, loop, xlast, xllast, xlllast, mmaL, mmaU,
                        mmaLlast, mmaUlast, alpha, beta));
    // PetscCall(VecView(alpha, PETSC_VIEWER_STDOUT_WORLD));
    // PetscCall(VecView(beta, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(mmatest(&test, xlast, mmaU, mmaL, dc, alpha, beta, x, &initial));

    PetscCall(VecCopy(mmaL, mmaLlast));
    PetscCall(VecCopy(mmaU, mmaUlast));
    PetscCall(VecCopy(xllast, xlllast));
    PetscCall(VecCopy(xlast, xllast));

    PetscCall(VecAXPY(xlast, -1, x));
    PetscCall(VecNorm(xlast, NORM_INFINITY, &change));
    PetscCall(VecCopy(x, xlast));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "change: %f\n", change));
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "loop: %d\n", loop));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&t));
  PetscCall(VecDestroy(&dc));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&mmaL));
  PetscCall(VecDestroy(&mmaU));
  PetscCall(VecDestroy(&xlast));
  PetscCall(VecDestroy(&xllast));
  PetscCall(VecDestroy(&xlllast));
  PetscCall(VecDestroy(&mmaLlast));
  PetscCall(VecDestroy(&mmaUlast));
  PetscCall(VecDestroy(&alpha));
  PetscCall(VecDestroy(&beta));
  PetscCall(KSPDestroy(&ksp));

  PetscCall(PetscFinalize());
}