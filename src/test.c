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
  PetscInt mesh[3] = {3, 3, 3};
  PetscScalar dom[3] = {1.0, 1.0, 1.0};
  PetscScalar cost = 0, change = 0;
  Mat A;
  Vec rhs, t, x, dc;
  KSP ksp;

  PetscCall(PC_init(&test, dom, mesh));
  PetscCall(PC_print_info(&test));

  PetscCall(DMCreateMatrix(test.dm, &A));
  PetscCall(DMCreateGlobalVector(test.dm, &rhs));
  PetscCall(DMCreateGlobalVector(test.dm, &x));
  PetscCall(DMCreateGlobalVector(test.dm, &t));
  PetscCall(DMCreateGlobalVector(test.dm, &dc));

  PetscCall(VecSet(dc, 0));

  PetscCall(formx(&test, x));
  PetscCall(formkappa(&test, x));
  PetscCall(formMatrix(&test, A));
  PetscCall(formRHS(&test, rhs, x));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, rhs, t));

  PetscCall(computeCost(&test, A, t, &cost));
  PetscCall(computeGradient(&test, x, t, dc));
  PetscCall(filter(&test, dc, x));

  PetscCall(optimalCriteria(&test, x, dc, &change));
  PetscCall(VecView(x, PETSC_VIEWER_STDOUT_WORLD));





  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&t));
  PetscCall(VecDestroy(&dc));
  PetscCall(VecDestroy(&x));
  PetscCall(KSPDestroy(&ksp));

  PetscCall(PetscFinalize());
}