#include "PreMixFEM_3D.h"
#include "oCriteria.h"
#include "optimization.h"
#include "system.h"
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
  MMAx mmax;
  PetscInt grid = 20;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mesh", &grid, NULL));
  PetscInt mesh[3] = {grid, grid, grid};
  PetscScalar dom[3] = {1.0, 1.0, 1.0};
  PetscScalar cost = 0;
  Mat A;
  Vec rhs, t, x, dc;
  KSP ksp;
  PetscInt loop = 0, iter = 0, penal = 3;
  PetscScalar change = 1, tau = 0, initial = 0, maxdc = 0, mindc = 0;

  char str[80];

  PetscCall(PC_init(&test, dom, mesh));
  PetscCall(mmaInit(&test, &mmax));
  PetscCall(PC_print_info(&test));

  PetscCall(DMCreateMatrix(test.dm, &A));
  PetscCall(DMCreateGlobalVector(test.dm, &rhs));
  PetscCall(DMCreateGlobalVector(test.dm, &x));
  PetscCall(DMCreateGlobalVector(test.dm, &t));
  PetscCall(DMCreateGlobalVector(test.dm, &dc));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(
      KSPSetTolerances(ksp, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
  PetscCall(KSPSetFromOptions(ksp));
  //   PetscCall(KSPSetUp(ksp));
  PetscCall(VecSet(x, volfrac));
  PetscCall(formBoundary(&test));
  while (change > 1e-3) {
    if (loop <= 40) {
      penal = 1;
    } else if (loop <= 60) {
      penal = 2;
    } else {
      penal = 3;
    }
    // if (loop == 60) {
    //   break;
    // }
    loop += 1;
    PetscViewer viewer;
    sprintf(str, "../data/output/change%04d.vtr", loop);
    PetscCall(
        PetscViewerVTKOpen(PETSC_COMM_WORLD, str, FILE_MODE_WRITE, &viewer));
    PetscCall(VecView(x, viewer));
    PetscCall(PetscViewerDestroy(&viewer));

    PetscCall(formkappa(&test, x, penal));
    PetscCall(formMatrix(&test, A));
    PetscCall(formRHS(&test, rhs, x, penal));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSolve(ksp, rhs, t));
    PetscCall(KSPGetIterationNumber(ksp, &iter));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "iteration number: %d\n", iter));

    PetscCall(VecMax(t, NULL, &tau));
    tau -= tD;
    tau *= kL / f0;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "tau: %f\n", tau));

    PetscCall(computeCostMMA(&test, t, &cost));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "cost: %f\n", cost));
    PetscCall(adjointGradient(&test, A, x, t, dc, penal));
    // PetscCall(computeGradient(&test, x, t, dc, penal));
    //
    // PetscCall(VecMax(dc, NULL, &maxdc));
    // PetscCall(VecMin(dc, NULL, &mindc));
    // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "maxdc: %f\n", maxdc));
    // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "mindc: %f\n", mindc));

    PetscCall(formLimit(&test, &mmax, loop));

    PetscCall(mma(&test, &mmax, dc, x, &initial));
    PetscCall(mmatest(&test, &mmax, dc, x, &initial));

    PetscCall(computeChange(&mmax, x, &change));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "loop: %d\n", loop));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "change: %f\n", change));
  }

  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&t));
  PetscCall(VecDestroy(&dc));
  PetscCall(VecDestroy(&x));

  PetscCall(KSPDestroy(&ksp));
  PetscCall(mmaFinal(&mmax));
  // PetscCall(PC_final(&test));

  PetscCall(PetscFinalize());
}