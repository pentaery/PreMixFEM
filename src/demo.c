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
  PetscInt mesh[3] = {16, 16, 16};
  PetscScalar dom[3] = {1.0, 1.0, 1.0};
  PetscScalar cost = 0, cost1 = 0;
  Mat A;
  Vec rhs, t, x, dc, c;
  KSP ksp;
  PetscInt loop = 0;
  PetscScalar change = 1;

  PetscCall(PC_init(&test, dom, mesh));
  PetscCall(PC_print_info(&test));

  PetscCall(DMCreateMatrix(test.dm, &A));
  PetscCall(DMCreateGlobalVector(test.dm, &rhs));
  PetscCall(DMCreateGlobalVector(test.dm, &x));
  PetscCall(DMCreateGlobalVector(test.dm, &t));
  PetscCall(DMCreateGlobalVector(test.dm, &dc));

  PetscCall(DMCreateGlobalVector(test.dm, &c));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetFromOptions(ksp));
  //   PetscCall(KSPSetUp(ksp));
  PetscCall(formx(&test, x));

  while (change > 0.01) {
    loop += 1;
    PetscCall(VecSet(dc, 0));
    PetscCall(VecSet(c, 0));
    PetscCall(formkappa(&test, x));
    PetscCall(formMatrix(&test, A));
    PetscCall(formRHS(&test, rhs, x));
    PetscCall(KSPSetOperators(ksp, A, A));
    PetscCall(KSPSolve(ksp, rhs, t));
    PetscCall(computeCost(&test, A, t, &cost));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "cost: %f\n", cost));
    PetscCall(computeCost1(&test, t, c));
    PetscCall(VecSum(c, &cost1));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "cost1: %f\n", cost1));

    PetscCall(computeGradient(&test, x, t, dc));
    // PetscCall(filter(&test, dc, x));
    PetscCall(optimalCriteria(&test, x, dc, &change));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "change: %f\n", change));
  }

  PetscViewer viewer;
  PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, "../data/output.h5",
  FILE_MODE_WRITE,
                                &viewer));
  PetscCall(VecView(x, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "loop: %d\n", loop));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&t));
  PetscCall(VecDestroy(&dc));
  PetscCall(VecDestroy(&x));
  PetscCall(KSPDestroy(&ksp));

  PetscCall(PetscFinalize());
}