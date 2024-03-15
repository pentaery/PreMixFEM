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
  PetscScalar cost = 0, derivative = 0;
  Mat A;
  Vec rhs, t, x, dc, mmaL, mmaLlast, mmaU, mmaUlast, xlast, xllast, xlllast,
      lbd, ubd, alpha, beta;
  KSP ksp;
  PetscInt loop = 0, iter = 0;
  PetscScalar change = 1;
  PetscScalar cost0 = 0;

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
  PetscCall(DMCreateGlobalVector(test.dm, &ubd));
  PetscCall(DMCreateGlobalVector(test.dm, &lbd));
  PetscCall(DMCreateGlobalVector(test.dm, &mmaLlast));
  PetscCall(DMCreateGlobalVector(test.dm, &mmaUlast));
  PetscCall(DMCreateGlobalVector(test.dm, &alpha));
  PetscCall(DMCreateGlobalVector(test.dm, &beta));
  PetscCall(VecSet(ubd, 1));
  PetscCall(VecSet(lbd, 1e-6));
  PetscCall(VecSet(xlast, volfrac));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(
      KSPSetTolerances(ksp, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
  PetscCall(KSPSetFromOptions(ksp));
  //   PetscCall(KSPSetUp(ksp));
  PetscCall(VecSet(x, 0.5));
  PetscCall(formBoundary(&test));

  while (change > 0.01) {
    loop += 1;
    PetscScalar initial = 0;
    PetscViewer viewer;
    sprintf(str, "../data/output/change%04d.vtr", loop);
    PetscCall(
        PetscViewerVTKOpen(PETSC_COMM_WORLD, str, FILE_MODE_WRITE, &viewer));
    PetscCall(VecView(x, viewer));
    PetscCall(PetscViewerDestroy(&viewer));

    PetscCall(formLimit(&test, loop, xlast, xllast, xlllast, mmaL, mmaU,
                        mmaLlast, mmaUlast, alpha, beta, lbd, ubd));
    PetscCall(formkappa(&test, x));
    PetscCall(formMatrix(&test, A));
    PetscCall(formRHS(&test, rhs, x));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSolve(ksp, rhs, t));
    PetscCall(KSPGetIterationNumber(ksp, &iter));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "iteration number: %d\n", iter));
    
    PetscCall(computeCostMMA(&test, x, &cost));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "cost: %f\n", cost));

    PetscCall(adjointGradient(&test, A, x, t, dc));
    PetscCall(
        steepestDescent(&test, xlast, mmaU, mmaL, dc, alpha, beta, &initial));

    PetscCall(VecCopy(mmaL, mmaLlast));
    PetscCall(VecCopy(mmaU, mmaUlast));
    PetscCall(VecCopy(xllast, xlllast));
    PetscCall(VecCopy(xlast, xllast));
    PetscCall(VecCopy(x, xlast));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "change: %f\n", change));
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "loop: %d\n", loop));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&t));
  PetscCall(VecDestroy(&dc));
  PetscCall(VecDestroy(&x));
  PetscCall(KSPDestroy(&ksp));

  PetscCall(PetscFinalize());
}