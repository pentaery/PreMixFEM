#include "PreMixFEM_3D.h"
#include "optimization.h"
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
#include <petscviewerhdf5.h>

int main(int argc, char **argv) {
  PetscCall(
      PetscInitialize(&argc, &argv, (char *)0, "Toplogical Optimiazation\n"));
  PCCtx test;
  PetscInt mesh[3] = {12, 12, 12};
  PetscScalar dom[3] = {1.0, 1.0, 1.0};
  PetscScalar cost = 0;
  Mat A;
  Vec rhs, t, x, dc;
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

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(
      KSPSetTolerances(ksp, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
  PetscCall(KSPSetFromOptions(ksp));
  //   PetscCall(KSPSetUp(ksp));
  PetscCall(VecSet(x, 0.1));
  PetscCall(formBoundary(&test));

  while (change > 0.01) {
    loop += 1;

    PetscViewer viewer;
    sprintf(str, "../data/output/change%03d.vtr", loop);
    PetscCall(
        PetscViewerVTKOpen(PETSC_COMM_WORLD, str, FILE_MODE_WRITE, &viewer));
    PetscCall(VecView(x, viewer));
    PetscCall(PetscViewerDestroy(&viewer));

    PetscCall(VecSet(dc, 0));
    PetscCall(formkappa(&test, x));
    PetscCall(formMatrix(&test, A));
    PetscCall(formRHS(&test, rhs, x));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSolve(ksp, rhs, t));
    PetscCall(KSPGetIterationNumber(ksp, &iter));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "iteration number: %d\n", iter));
    PetscCall(computeCost1(&test, t, &cost));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "cost: %f\n", cost));

    PetscCall(computeGradient(&test, x, t, dc));
    // PetscCall(filter(&test, dc, x));
    PetscCall(optimalCriteria(&test, x, dc, &change));
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