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

int main(int argc, char **argv) {
  PetscCall(
      PetscInitialize(&argc, &argv, (char *)0, "Toplogical Optimiazation\n"));
  PCCtx test;
  PetscInt mesh[3] = {64, 64, 64};
  PetscScalar dom[3] = {1.0, 1.0, 1.0};
  PetscScalar cost = 0;
  Mat A;
  Vec rhs, t, x, dc, c;
  KSP ksp;

  PetscCall(PC_init(&test, dom, mesh));
  PetscCall(PC_print_info(&test));

  PetscCall(DMCreateMatrix(test.dm, &A));
  PetscCall(DMCreateGlobalVector(test.dm, &rhs));
  PetscCall(DMCreateGlobalVector(test.dm, &x));
  PetscCall(DMCreateGlobalVector(test.dm, &t));
  PetscCall(DMCreateGlobalVector(test.dm, &c));
  PetscCall(DMCreateGlobalVector(test.dm, &dc));

  PetscCall(VecSet(c, 0));
  PetscCall(VecSet(dc, 0));

  PetscCall(formx(&test, x));
  PetscCall(formkappa(&test));
  PetscCall(formMatrix(&test, A));
  PetscCall(formRHS(&test, rhs, x));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, rhs, t));

  PetscCall(ccost(&test, A, t, &cost));
  PetscPrintf(PETSC_COMM_WORLD, "Cost: %f\n", cost);
  PetscCall(computeCost(&test, x, t, c, dc));
  PetscCall(VecSum(c, &cost));
  PetscPrintf(PETSC_COMM_WORLD, "Cost: %f\n", cost);

  PetscCall(filter(&test, dc));

  PetscCall(optimalCriteria(&test, x, dc));

  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&t));
  PetscCall(VecDestroy(&dc));
  PetscCall(VecDestroy(&x));
  PetscCall(KSPDestroy(&ksp));

  PetscCall(PetscFinalize());
}