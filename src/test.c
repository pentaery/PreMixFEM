#include "PreMixFEM_3D.h"
#include "optimization.h"
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmdatypes.h>
#include <petscdmtypes.h>
#include <petscerror.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <petscviewer.h>

int main(int argc, char **argv) {
  PetscCall(
      PetscInitialize(&argc, &argv, (char *)0, "Toplogical Optimiazation\n"));

  PetscInt mesh[3] = {3, 3, 3};
  PCCtx test;
  PetscScalar dom[3] = {1.0, 1.0, 1.0};
  Mat A;
  PetscCall(DMCreateMatrix(test.dm, &A));
  PetscCall(PC_init(&test, dom, mesh));
  PetscCall(PC_print_info(&test));
  PetscCall(formx(&test));
  PetscCall(formkappa(&test));
  PetscCall(formMatrix(&test, A));
  // PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(PetscFinalize());
}