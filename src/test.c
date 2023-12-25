#include "PreMixFEM_3D.h"
#include "mpi.h"
#include <petscdmda.h>
#include <petscerror.h>
#include <petscoptions.h>
#include <petscsys.h>
#include <petscsystypes.h>

#define MAX_ARGS 24
static char help[] = "A test";

int main(int argc, char **argv) {
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  Vec x, b, u;
  Mat A;
  KSP ksp;
  PC pc;
  PetscReal norm;
  PetscInt i, n = 10, col[3], its;
  PetscMPIInt size;
  PetscScalar value[3];

  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "hhh");

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));

  PetscCall(VecCreate(PETSC_COMM_SELF, &x));
  PetscCall(PetscObjectSetName((PetscObject)x, "Solution"));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, n));
  

  return 0;
}