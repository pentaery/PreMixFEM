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
  PetscInt grid = 20;
  PetscInt iter_number = 60;
  PetscInt output_frequency = 20;
  PetscLogEvent linearsolve, optimize;
  PetscBool petsc_default = PETSC_FALSE;
  PetscCall(PetscLogEventRegister("LinearSolve", 0, &linearsolve));
  PetscCall(PetscLogEventRegister("Optimization", 1, &optimize));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mesh", &grid, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-iter", &iter_number, NULL));
  PetscCall(PetscOptionsHasName(NULL, NULL, "-petsc_default", &petsc_default));
  PetscCall(
      PetscOptionsGetInt(NULL, NULL, "-frequency", &output_frequency, NULL));
  PetscInt mesh[3] = {grid, grid, grid};
  PetscScalar dom[3] = {1.0, 1.0, 1.0};
  PetscScalar cost = 0;
  Mat A;
  Vec rhs, t, x, dc;
  KSP ksp;
  PetscInt loop = 0, iter = 0, penal = 3;
  PetscScalar change = 1, tau = 0, xvolfrac = 0;

  char str[80];

  PetscCall(PC_init(&test, dom, mesh));
  PetscCall(mmaInit(&test, &mmax));
  PetscCall(PC_print_info(&test));

  PetscCall(DMCreateMatrix(test.dm, &A));
  PetscCall(DMCreateGlobalVector(test.dm, &rhs));
  PetscCall(DMCreateGlobalVector(test.dm, &x));
  PetscCall(PetscObjectSetName((PetscObject)x, "xvalue"));
  PetscCall(PetscObjectSetName((PetscObject)mmax.xlast, "xvalue"));
  PetscCall(DMCreateGlobalVector(test.dm, &t));
  PetscCall(DMCreateGlobalVector(test.dm, &dc));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(
      KSPSetTolerances(ksp, 1e-6, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT));
  PetscCall(KSPSetFromOptions(ksp));
  //   PetscCall(KSPSetUp(ksp));
  
    PetscViewer h5_viewer;
  PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD,
                                "../data/output/change0011.h5", FILE_MODE_READ,
                                &h5_viewer));
  PetscCall(VecLoad(x, h5_viewer));
  PetscCall(PetscViewerDestroy(&h5_viewer));
  PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD,
                                "../data/output/change0010.h5", FILE_MODE_READ,
                                &h5_viewer));
  PetscCall(VecLoad(mmax.xlast, h5_viewer));
  PetscCall(PetscViewerDestroy(&h5_viewer));

  
  // PetscCall(VecSet(mmax.xlast, volfrac));
  // PetscCall(VecSet(x, volfrac));
  PetscCall(formBoundary(&test));
  while (PETSC_TRUE) {
    if (loop >= iter_number) {
      break;
    }
    loop += 1;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "loop: %d\n", loop));

    PetscCall(VecSum(x, &xvolfrac));
    xvolfrac /= test.M * test.N * test.P;

    if (loop % output_frequency == 0) {
      PetscViewer viewer;
      sprintf(str, "../data/output/change%04d.vtr", loop);
      PetscCall(
          PetscViewerVTKOpen(PETSC_COMM_WORLD, str, FILE_MODE_WRITE, &viewer));
      PetscCall(VecView(x, viewer));
      PetscCall(PetscViewerDestroy(&viewer));
    }
    
    PetscCall(formkappa(&test, x, penal));
    PetscCall(formMatrix(&test, A));
    PetscCall(formRHS(&test, rhs, x, penal));
    PetscCall(KSPSetOperators(ksp, A, A));

    PC pc;
    PetscCall(KSPGetPC(ksp, &pc));
    if (!petsc_default) {
      PetscCall(PCSetType(pc, PCSHELL));
      PetscCall(PCShellSetContext(pc, &test));
      PetscCall(PCShellSetSetUp(pc, PC_setup));
      PetscCall(PCShellSetApply(pc, PC_apply_vec));
      PetscCall(PCShellSetName(
          pc, "3levels-MG-via-GMsFEM-with-velocity-elimination"));
    } else {
      PetscCall(PCSetType(pc, PCGAMG));
    }
    PetscCall(PetscLogEventBegin(linearsolve, 0, 0, 0, 0));
    PetscCall(KSPSolve(ksp, rhs, t));
    PetscCall(PetscLogEventEnd(linearsolve, 0, 0, 0, 0));

    PetscCall(KSPGetIterationNumber(ksp, &iter));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "iter: %d\n, iter));

    PetscCall(VecMax(t, NULL, &tau));
    tau -= tD;
    tau *= kL;
    tau /= f0;
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
    // PetscCall(PCDestroy(&pc));
  }
  if (!petsc_default) {
    PetscCall(PC_final(&test));
  }

  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&t));
  PetscCall(VecDestroy(&dc));
  PetscCall(VecDestroy(&x));

  PetscCall(KSPDestroy(&ksp));
  PetscCall(mmaFinal(&mmax));

  PetscCall(SlepcFinalize());
}
