#include "PreMixFEM_3D.h"
#include "optimization.h"
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmdatypes.h>
#include <petscdmtypes.h>
#include <petscerror.h>
#include <petscsys.h>

int main(int argc, char **argv) {
  PetscCall(
      PetscInitialize(&argc, &argv, (char *)0, "Toplogical Optimiazation\n"));

  DM dm;
  PetscInt M = 32, N = 32, P = 32;
  Mat A;
  Vec x, b, u, dc, kappa;
  KSP ksp;
  PetscScalar cost;
  PetscScalar volfrac = M * N * 0.3;
  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                         DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, M, N, P,
                         PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, 1, NULL,
                         NULL, NULL, &dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMCreateMatrix(dm, &A));
  PetscCall(DMCreateGlobalVector(dm, &x));
  PetscCall(DMCreateGlobalVector(dm, &b));
  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(DMCreateGlobalVector(dm, &dc));
  PetscCall(DMCreateGlobalVector(dm, &kappa));
  PetscCall(formx(dm, x));
  PetscCall(formkappa(dm, x, kappa));
  PetscCall(formMatrix(dm, A, x));
  PetscCall(formRHS(dm, b, N));
  PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSolve(ksp, b, u));

  PetscCall(computeCost(dm, &cost, u, dc, x));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "cost: %f\n", cost));

  PetscCall(optimalCriteria(dm, x, dc, volfrac));

  PetscCall(PetscFinalize());
}