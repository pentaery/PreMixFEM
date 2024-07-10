#include "MMA.h"
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
#include <petsclog.h>
#include <petscmat.h>
#include <petscoptions.h>
#include <petscpc.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petsctime.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <petscviewerhdf5.h>
#include <slepceps.h>

int main(int argc, char **argv) {
  PetscCall(
      SlepcInitialize(&argc, &argv, (char *)0, "Toplogical Optimiazation\n"));
  PCCtx test;
  MMAx mmax;
  PetscInt grid = 60;
  PetscInt iter_number = 80;
  PetscLogEvent linearsolve, optimize;
  PetscBool is_petsc_default = PETSC_FALSE;
  PetscCall(PetscLogEventRegister("LinearSolve", 0, &linearsolve));
  PetscCall(PetscLogEventRegister("Optimization", 1, &optimize));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mesh", &grid, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-iter", &iter_number, NULL));
  PetscCall(
      PetscOptionsHasName(NULL, NULL, "-petsc_default", &is_petsc_default));
  PetscInt mesh[3] = {grid, grid, grid};
  PetscScalar dom[3] = {1.0, 1.0, 1.0};
  PetscScalar cost = 0;
  Mat A;
  Vec rhs, t, x, dc;
  KSP ksp;
  PetscInt loop = 0, iter = 0, penal = 3;
  PetscScalar change = 1, tau = 0, xvolfrac = 0;

  char str[80];

  PetscCall(mmaInit(&mmax, dom, mesh));


  PetscCall(DMCreateMatrix(mmax.dm, &A));
  PetscCall(DMCreateGlobalVector(mmax.dm, &rhs));
  PetscCall(DMCreateGlobalVector(mmax.dm, &x));
  PetscCall(DMCreateGlobalVector(mmax.dm, &t));
  PetscCall(DMCreateGlobalVector(mmax.dm, &dc));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(
      KSPSetTolerances(ksp, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
  PetscCall(KSPSetFromOptions(ksp));
  //   PetscCall(KSPSetUp(ksp));
  PetscCall(VecSet(mmax.xlast, volfrac));
  PetscCall(VecSet(x, volfrac));
  PetscCall(formBoundary(&test, &mmax));
  while (PETSC_TRUE) {
    if (loop == iter_number) {
      break;
    }
    loop += 1;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "loop: %d\n", loop));

    PetscCall(PC_init(&test, dom, mesh, mmax.dm));
    // PetscCall(PC_print_info(&test));

    PetscCall(VecSum(x, &xvolfrac));
    xvolfrac /= test.M * test.N * test.P;

    // PetscViewer viewer;
    // sprintf(str, "../data/output/change%04d.vtr", loop);
    // PetscCall(
    //     PetscViewerVTKOpen(PETSC_COMM_WORLD, str, FILE_MODE_WRITE, &viewer));
    // PetscCall(VecView(x, viewer));
    // PetscCall(PetscViewerDestroy(&viewer));

    PetscCall(formkappa(&test, x, penal));
    PetscCall(formMatrix(&test, A));
    PetscCall(formRHS(&test, rhs, x, penal));
    PetscCall(KSPSetOperators(ksp, A, A));
    PC pc;
    if (!is_petsc_default) {
      PetscCall(KSPGetPC(ksp, &pc));
      PetscCall(PCSetType(pc, PCSHELL));
      PetscCall(PCShellSetContext(pc, &test));
      PetscCall(PCShellSetSetUp(pc, PC_setup));
      PetscCall(PCShellSetApply(pc, PC_apply_vec));
      PetscCall(PCShellSetName(
          pc, "3levels-MG-via-GMsFEM-with-velocity-elimination"));
    }
    PetscCall(PetscLogEventBegin(linearsolve, 0, 0, 0, 0));
    PetscCall(KSPSolve(ksp, rhs, t));
    PetscCall(PetscLogEventEnd(linearsolve, 0, 0, 0, 0));

    PetscCall(KSPGetIterationNumber(ksp, &iter));

    PetscCall(VecMax(t, NULL, &tau));
    tau -= tD;
    tau *= kL / f0;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "tau: %f\n", tau));

    PetscCall(computeCostMMA(&test, t, &cost));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "cost: %f\n", cost));
    PetscCall(VecSet(dc, 0));

    PetscCall(PetscLogEventBegin(optimize, 0, 0, 0, 0));
    PetscCall(adjointGradient(&test, &mmax, ksp, A, mmax.xlast, t, dc, penal));
    PetscCall(mmaLimit(&test, &mmax, loop));
    PetscCall(mmaSub(&test, &mmax, dc));
    PetscCall(subSolv(&test, &mmax, x));
    PetscCall(PetscLogEventEnd(optimize, 0, 0, 0, 0));

    PetscCall(computeChange(&mmax, x, &change));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "change: %f\n", change));
    PetscCall(PC_final(&test));
    PetscCall(PCDestroy(&pc));
  }
  PetscCall(mmaFinal(&mmax));


  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&t));
  PetscCall(VecDestroy(&dc));
  PetscCall(VecDestroy(&x));
  PetscCall(KSPDestroy(&ksp));

  PetscCall(SlepcFinalize());
}