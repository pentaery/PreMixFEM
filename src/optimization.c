#include "optimization.h"
#include "mpi.h"
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

PetscErrorCode formx(DM dm, Vec x) {
  PetscFunctionBeginUser;
  PetscInt startx, starty, startz, nx, ny, nz, ex, ey, ez;
  PetscScalar ***array;
  PetscCall(DMDAGetCorners(dm, &startx, &starty, &startz, &nx, &ny, &nz));
  PetscCall(DMDAVecGetArray(dm, x, &array));
  for (ez = startz; ez < startz + nz; ++ez) {
    for (ey = starty; ey < starty + ny; ++ey) {
      for (ex = startx; ex < startx + nx; ++ex) {
        array[ez][ey][ex] = 0.5;
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(dm, x, &array));
  PetscFunctionReturn(0);
}

PetscErrorCode(formkappa(DM dm, Vec x, Vec kappa)) {
  PetscFunctionBeginUser;
  PetscInt startx, starty, startz, nx, ny, nz, ex, ey, ez;
  PetscScalar ***arrayx, ***arraykappa;
  PetscCall(DMDAGetCorners(dm, &startx, &starty, &startz, &nx, &ny, &nz));
  PetscCall(DMDAVecGetArray(dm, x, &arrayx));
  PetscCall(DMDAVecGetArray(dm, kappa, &arraykappa));
  for (ez = startz; ez < startz + nz; ++ez) {
    for (ey = starty; ey < starty + ny; ++ey) {
      for (ex = startx; ex < startx + nx; ++ex) {
        arraykappa[ez][ey][ex] =
            (1 - 1e-6) * PetscPowScalar(arrayx[ez][ey][ex], 3) + 1e-6;
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(dm, x, &arrayx));
  PetscCall(DMDAVecRestoreArray(dm, kappa, &arraykappa));
  PetscFunctionReturn(0);
}

PetscErrorCode formMatrix(DM dm, Mat A, Vec kappa) {
  PetscFunctionBeginUser;
  PetscScalar value[2][2];
  PetscInt startx, starty, nx, ny, ex, ey;
  PetscScalar ***array;
  MatStencil col[2];
  PetscCall(DMDAGetCorners(dm, &startx, &starty, NULL, &nx, &ny, NULL));
  for (ey = starty; ey < starty + ny; ey++) {
    for (ex = startx; ex < startx + nx; ex++) {
      col[0] = (MatStencil){.i = ex, .j = ey, .c = 0};
      col[1] = (MatStencil){.i = ex, .j = ey, .c = 1};
      PetscCall(DMDAVecGetArrayDOF(dm, x, &array));
      coef = array[ey][ex][0] * array[ey][ex][0] * array[ey][ex][0];
      PetscCall(DMDAVecRestoreArrayDOF(dm, x, &array));
      PetscCall(
          MatSetValuesStencil(A, 8, col, 8, col, &value[0][0], ADD_VALUES));
    }
  }

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscFunctionReturn(0);
}

PetscErrorCode formRHS(DM dm, Vec rhs, PetscInt N) {
  PetscFunctionBeginUser;
  PetscScalar ***array;
  PetscInt startx, starty, nx, ny, ex, ey;
  PetscCall(DMDAGetCorners(dm, &startx, &starty, NULL, &nx, &ny, NULL));

  // Set RHS
  PetscCall(DMDAVecGetArrayDOF(dm, rhs, &array));
  for (ey = starty; ey < starty + ny; ey++) {
    for (ex = startx; ex < startx + nx; ex++) {
      if (ex == 0) {
        array[ey][ex][0] = 0;
        if (ey == N - 1) {
          array[ey][ex][1] = -1;
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayDOF(dm, rhs, &array));
  // PetscCall(VecView(rhs, PETSC_VIEWER_STDOUT_WORLD));
  PetscFunctionReturn(0);
}

PetscErrorCode computeCost(DM dm, PetscScalar *cost, Vec u, Vec dc, Vec x) {
  PetscFunctionBeginUser;
  PetscInt startx, starty, nx, ny, ex, ey, i, j;
  PetscScalar value[8][8];
  PetscScalar ***array, ***arraydc, ***arrayx;
  PetscScalar Ue[8];
  PetscScalar v;
  Vec localu;
  PetscCall(DMDAGetCorners(dm, &startx, &starty, NULL, &nx, &ny, NULL));

  PetscCall(DMGetLocalVector(dm, &localu));
  PetscCall(DMGlobalToLocal(dm, u, INSERT_VALUES, localu));

  PetscCall(DMDAVecGetArrayDOF(dm, localu, &array));
  PetscCall(DMDAVecGetArrayDOF(dm, x, &arrayx));
  PetscCall(DMDAVecGetArrayDOF(dm, dc, &arraydc));
  PetscScalar localcost = 0;
  // PetscCall(
  //     PetscPrintf(PETSC_COMM_SELF, "cost before calculating: %f\n", *cost));
  for (ey = starty; ey < starty + ny; ++ey) {
    for (ex = startx; ex < startx + nx; ++ex) {
      Ue[0] = array[ey + 1][ex][0];
      Ue[1] = array[ey + 1][ex][1];
      Ue[2] = array[ey + 1][ex + 1][0];
      Ue[3] = array[ey + 1][ex + 1][1];
      Ue[4] = array[ey][ex + 1][0];
      Ue[5] = array[ey][ex + 1][1];
      Ue[6] = array[ey][ex][0];
      Ue[7] = array[ey][ex][1];
      v = 0;
      for (i = 0; i < 8; i++) {
        for (j = 0; j < 8; j++) {
          v += Ue[j] * value[i][j] * Ue[i];
        }
      }
      localcost += v;
      arraydc[ey][ex][0] = -3 * v * arrayx[ey][ex][0] * arrayx[ey][ex][0];
    }
  }
  // PetscCall(
  //     PetscPrintf(PETSC_COMM_SELF, "cost after calculating: %f \n", *cost));

  PetscCall(DMDAVecRestoreArrayDOF(dm, localu, &array));
  PetscCall(DMDAVecRestoreArrayDOF(dm, x, &arrayx));
  PetscCall(DMDAVecRestoreArrayDOF(dm, dc, &arraydc));
  PetscCallMPI(MPI_Allreduce(&localcost, cost, 1, MPI_DOUBLE, MPI_SUM,
                             PETSC_COMM_WORLD));
  PetscFunctionReturn(0);
}

PetscErrorCode filter(DM dm, Vec dc, Vec x) {
  PetscFunctionBeginUser;
  PetscInt startx, starty, nx, ny, ex, ey;
  PetscScalar ***arraydc, ***arrayx, ***arraydcn;
  Vec dcn, localx, localdcn, localdc;
  PetscCall(DMCreateGlobalVector(dm, &dcn));
  PetscCall(DMDAGetCorners(dm, &startx, &starty, NULL, &nx, &ny, NULL));

  PetscCall(DMCreateLocalVector(dm, &localx));
  PetscCall(DMCreateLocalVector(dm, &localdcn));
  PetscCall(DMCreateLocalVector(dm, &localdc));

  PetscCall(DMGlobalToLocal(dm, x, INSERT_VALUES, localx));
  PetscCall(DMGlobalToLocal(dm, dcn, INSERT_VALUES, localdcn));
  PetscCall(DMGlobalToLocal(dm, dc, INSERT_VALUES, localdc));

  PetscCall(DMDAVecGetArrayDOF(dm, localdc, &arraydc));
  PetscCall(DMDAVecGetArrayDOF(dm, localdcn, &arraydcn));
  PetscCall(DMDAVecGetArrayDOF(dm, localx, &arrayx));

  PetscCall(DMCreateGlobalVector(dm, &dcn));

  for (ey = starty; ey < starty + ny; ++ey) {
    for (ex = startx; ex < startx + nx; ex++) {

      arraydcn[ey][ex][0] =
          (0.6 * arraydc[ey][ex][0] * arrayx[ey][ex][0] +
           0.1 * arraydc[ey - 1][ex][0] * arrayx[ey - 1][ex][0] +
           0.1 * arraydc[ey + 1][ex][0] * arrayx[ey + 1][ex][0] +
           0.1 * arraydc[ey][ex - 1][0] * arrayx[ey][ex - 1][0] +
           0.1 * arraydc[ey][ex + 1][0] * arrayx[ey][ex + 1][0]) /
          arrayx[ey][ex][0];
    }
  }

  PetscCall(DMDAVecRestoreArrayDOF(dm, localdc, &arraydc));
  PetscCall(DMDAVecRestoreArrayDOF(dm, localx, &arrayx));
  PetscCall(DMDAVecRestoreArrayDOF(dm, localdcn, &arraydcn));
  PetscCall(DMLocalToGlobal(dm, localdcn, INSERT_VALUES, dcn));
  PetscCall(VecDuplicate(dcn, &dc));
  PetscCall(VecDestroy(&dcn));
  PetscCall(VecDestroy(&localx));
  PetscCall(VecDestroy(&localdc));
  PetscCall(VecDestroy(&localdcn));
  PetscFunctionReturn(0);
}

PetscErrorCode optimalCriteria(DM dm, Vec x, Vec dc, PetscScalar volfrac) {
  PetscFunctionBeginUser;
  PetscScalar l1 = 0, l2 = 100000, move = 0.2, lmid;
  PetscInt startx, starty, nx, ny, ex, ey;
  PetscScalar ***arraydc, ***arrayx;
  PetscScalar sum;

  PetscCall(DMDAGetCorners(dm, &startx, &starty, NULL, &nx, &ny, NULL));

  PetscCall(DMDAVecGetArrayDOF(dm, dc, &arraydc));
  PetscCall(DMDAVecGetArrayDOF(dm, x, &arrayx));

  while (l2 - l1 > 1e-4) {
    lmid = (l1 + l2) / 2;

    for (ey = starty; ey < starty + ny; ++ey) {
      for (ex = startx; ex < startx + nx; ++ex) {
        if (-arrayx[ey][ex][0] * arraydc[ey][ex][0] / lmid <
            PetscMax(0.001, arrayx[ey][ex][0] - move)) {
          arrayx[ey][ex][0] = PetscMax(0.001, arrayx[ey][ex][0] - move);
        } else if (-arrayx[ey][ex][0] * arraydc[ey][ex][0] / lmid >
                   PetscMin(1, arrayx[ey][ex][0] + move)) {
          arrayx[ey][ex][0] = PetscMin(1, arrayx[ey][ex][0] + move);
        } else {
          arrayx[ey][ex][0] = -arrayx[ey][ex][0] * arraydc[ey][ex][0] / lmid;
        }
      }
    }

    PetscCall(DMDAVecRestoreArrayDOF(dm, dc, &arraydc));
    PetscCall(DMDAVecRestoreArrayDOF(dm, x, &arrayx));

    PetscCall(VecSum(x, &sum));

    if (sum > volfrac) {
      l1 = lmid;
    } else {
      l2 = lmid;
    }
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "lambda: %f\n", lmid));
  }

  PetscFunctionReturn(0);
}