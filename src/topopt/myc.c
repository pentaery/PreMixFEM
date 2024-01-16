
#include "mpi.h"
#include <petsc.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmdatypes.h>
#include <petscdmtypes.h>
#include <petscerror.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscsys.h>
#include <petscvec.h>
#include <petscviewer.h>
static char help[] = "This is a demo.";
int main(int argc, char **args) {
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscInt rank;
  PetscInt x = 1;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "rank: %d\n", rank));
  if (rank == 0) {
    x = 10;
  }
  PetscCallMPI(MPI_Bcast(&x, 1, MPI_INT, 0, PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "x: %d\n", x));
  PetscCall(PetscFinalize());
}