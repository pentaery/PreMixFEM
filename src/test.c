#include "PreMixFEM_3D.h"
#include "optimization.h"
#include "system.h"
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

PetscErrorCode xScaling(DM dm1, DM dm2, Vec x1, Vec x2);

int main(int argc, char **argv) {
  PetscCall(
      PetscInitialize(&argc, &argv, (char *)0, "Toplogical Optimiazation\n"));

  DM dm;
  Mat A_cc;
  PetscInt rank;
  const PetscMPIInt *ng_ranks;
  PetscInt A_cc_range[NEIGH + 1][2];
  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                         DM_BOUNDARY_NONE, DMDA_STENCIL_STAR, 2, 2, 2, 2, 2, 2,
                         1, 1, NULL, NULL, NULL, &dm));
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, 2, 2, PETSC_DETERMINE,
                         PETSC_DETERMINE, 1, NULL, 6, NULL, &A_cc));
  PetscCall(DMSetUp(dm));
  PetscCall(MatGetOwnershipRange(A_cc, &A_cc_range[0][0], &A_cc_range[0][1]));
  PetscCall(DMDAGetNeighbors(dm, &ng_ranks));
  PetscCall(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "1: %d, 2: %d\n", A_cc_range[0][0],
                        A_cc_range[0][1]));
  PetscCall(PetscFinalize());
}
