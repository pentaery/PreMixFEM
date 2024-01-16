#include "func.h"
#include "mpi.h"
#include <math.h>
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

PetscErrorCode formMatrix(DM dm, Mat A, Vec x, PetscInt M, PetscInt N) {
  PetscFunctionBeginUser;
  // PetscInt sizea, sizeb;
  PetscReal value[8][8];
  PetscReal coef;
  PetscInt startx, starty, nx, ny, ex, ey;
  PetscReal ***array;
  MatStencil col[8];
  PetscCall(DMDAGetCorners(dm, &startx, &starty, NULL, &nx, &ny, NULL));
  for (ey = starty; ey < starty + ny; ey++) {
    for (ex = startx; ex < startx + nx; ex++) {
      if (ex < M - 1 && ey < N - 1) {
        col[0] = (MatStencil){.i = ex, .j = ey, .c = 0};
        col[1] = (MatStencil){.i = ex, .j = ey, .c = 1};
        col[2] = (MatStencil){.i = ex + 1, .j = ey, .c = 0};
        col[3] = (MatStencil){.i = ex + 1, .j = ey, .c = 1};
        col[4] = (MatStencil){.i = ex, .j = ey + 1, .c = 0};
        col[5] = (MatStencil){.i = ex, .j = ey + 1, .c = 1};
        col[6] = (MatStencil){.i = ex + 1, .j = ey + 1, .c = 0};
        col[7] = (MatStencil){.i = ex + 1, .j = ey + 1, .c = 1};
        // row[0] = (MatStencil){.i = ex, .j = ey, .c = 0};
        // row[1] = (MatStencil){.i = ex, .j = ey, .c = 1};
        // row[2] = (MatStencil){.i = ex + 1, .j = ey, .c = 0};
        // row[3] = (MatStencil){.i = ex + 1, .j = ey, .c = 1};
        // row[4] = (MatStencil){.i = ex, .j = ey + 1, .c = 0};
        // row[5] = (MatStencil){.i = ex, .j = ey + 1, .c = 1};
        // row[6] = (MatStencil){.i = ex + 1, .j = ey + 1, .c = 0};
        // row[7] = (MatStencil){.i = ex + 1, .j = ey + 1, .c = 1};
        PetscCall(DMDAVecGetArrayDOF(dm, x, &array));
        coef = array[ey][ex][0] * array[ey][ex][0] * array[ey][ex][0];
        PetscCall(DMDAVecRestoreArrayDOF(dm, x, &array));
        formKE(value, coef);
        PetscCall(
            MatSetValuesStencil(A, 8, col, 8, col, &value[0][0], ADD_VALUES));
      }
    }
  }

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  // PetscCall(MatGetOwnershipRange(A, &M, &N));
  // PetscCall(PetscPrintf(PETSC_COMM_SELF, "m: %d, n: %d\n", M, N));
  // PetscCall(PetscPrintf(PETSC_COMM_SELF,
  //                       "startx: %d, starty: %d, nx:%d, ny: %d\n", startx,
  //                       starty, nx, ny));
  // PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));
  // MatStencil row0;
  // for (ey = starty; ey < starty + ny; ++ey) {
  //   for (ex = startx; ex < startx + nx; ++ex) {
  //     if (ex == 0) {
  //       row0 = (MatStencil){.i = ex, .j = ey, .c = 1};
  //       PetscCall(MatZeroRowsStencil(A, 1, &row0, 1, NULL, NULL));
  //     }
  //   }
  // }
  // if (startx == 0 || starty == 0) {
  //   // PetscCall(MatZeroRows(A, 1, &row1, 1, NULL, NULL));
  //   PetscCall(MatZeroRowsColumns(A, 1, &row1, 1, NULL, NULL));
  // };
  // row0 = (MatStencil){.i = 0, .j = 0, .c = 1};
  // PetscCall(MatZeroRowsStencil(A, 1, &row0, 1, NULL, NULL));

  // PetscCall(DMCreateGlobalVector(dm, &a));
  // PetscCall(DMCreateLocalVector(dm, &b));
  // PetscCall(VecGetSize(a, &sizea));
  // PetscCall(VecGetSize(b, &sizeb));
  // PetscCall(
  //     PetscPrintf(PETSC_COMM_SELF, "sizea: %d, sizeb:%d\n", sizea,
  // sizeb));

  PetscFunctionReturn(0);
}

PetscErrorCode formKE(PetscReal KE[8][8], PetscReal coef) {
  PetscFunctionBeginUser;
  PetscInt i, j;
  PetscReal nu = 0.3, factor = coef / (1 - nu * nu);
  PetscReal k[8] = {1.0 / 2.0 - nu / 6.0,
                    1.0 / 8.0 + nu / 8.0,
                    -1.0 / 4.0 - nu / 12.0,
                    -1.0 / 8.0 + 3.0 * nu / 8.0,
                    -1.0 / 4.0 + nu / 12.0,
                    -1.0 / 8.0 - nu / 8.0,
                    nu / 6.0,
                    1.0 / 8.0 - 3.0 * nu / 8.0};
  for (i = 0; i < 8; i++) {
    KE[i][i] = factor * k[0];
  }
  for (i = 2; i < 8; i++) {
    KE[1][i] = factor * k[9 - i];
  }
  KE[2][3] = factor * k[5];
  KE[2][4] = factor * k[6];
  KE[2][5] = factor * k[3];
  KE[2][6] = factor * k[4];
  KE[2][7] = factor * k[1];
  KE[3][4] = factor * k[7];
  KE[3][5] = factor * k[2];
  KE[3][6] = factor * k[1];
  KE[3][7] = factor * k[4];
  KE[4][5] = factor * k[1];
  KE[4][6] = factor * k[2];
  KE[4][7] = factor * k[3];
  KE[5][6] = factor * k[7];
  KE[5][7] = factor * k[6];
  KE[6][7] = factor * k[5];
  for (i = 1; i < 8; i++) {
    for (j = 0; j < i - 1; j++) {
      KE[i][j] = KE[j][i];
    }
  }

  PetscFunctionReturn(0);
}

PetscErrorCode formRHS(DM dm, Vec rhs, PetscInt N) {
  PetscFunctionBeginUser;
  PetscReal ***array;
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

PetscErrorCode computeCost(DM dm, PetscReal *cost, Vec u, Vec dc, Vec x,
                           PetscInt M, PetscInt N) {
  PetscFunctionBeginUser;
  PetscInt startx, starty, nx, ny, ex, ey, i, j;
  PetscReal value[8][8];
  PetscReal ***array, ***arraydc, ***arrayx;
  PetscReal Ue[8];
  PetscReal v;
  Vec localu;
  formKE(value, 1);
  PetscCall(DMDAGetCorners(dm, &startx, &starty, NULL, &nx, &ny, NULL));
  PetscCall(DMGetLocalVector(dm, &localu));
  PetscCall(DMGlobalToLocal(dm, u, INSERT_VALUES, localu));
  PetscCall(DMDAVecGetArrayDOF(dm, localu, &array));
  PetscCall(DMDAVecGetArrayDOF(dm, x, &arrayx));
  PetscCall(DMDAVecGetArrayDOF(dm, dc, &arraydc));
  *cost = 0;
  // PetscCall(
  //     PetscPrintf(PETSC_COMM_SELF, "cost before calculating: %f\n", *cost));
  for (ey = starty; ey < starty + ny; ++ey) {
    for (ex = startx; ex < startx + nx; ++ex) {
      if (ex < M - 1 && ey < N - 1) {
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
        *cost += v;
        arraydc[ey][ex][0] = -3 * v * arrayx[ey][ex][0] * arrayx[ey][ex][0];
      }
    }
  }
  // PetscCall(
  //     PetscPrintf(PETSC_COMM_SELF, "cost after calculating: %f \n", *cost));
  PetscCall(DMDAVecRestoreArrayDOF(dm, localu, &array));
  PetscCall(DMDAVecRestoreArrayDOF(dm, x, &arrayx));
  PetscCall(DMDAVecRestoreArrayDOF(dm, dc, &arraydc));
  PetscFunctionReturn(0);
}

PetscErrorCode filter(DM dm, Vec dc, Vec x, PetscInt M, PetscInt N) {
  PetscFunctionBeginUser;
  PetscInt startx, starty, nx, ny, ex, ey;
  PetscReal ***arraydc, ***arrayx, ***arraydcn;
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
      if (ex < M - 2 && ey < N - 2 && ex > 0 && ey > 0) {
        arraydcn[ey][ex][0] =
            (0.6 * arraydc[ey][ex][0] * arrayx[ey][ex][0] +
             0.1 * arraydc[ey - 1][ex][0] * arrayx[ey - 1][ex][0] +
             0.1 * arraydc[ey + 1][ex][0] * arrayx[ey + 1][ex][0] +
             0.1 * arraydc[ey][ex - 1][0] * arrayx[ey][ex - 1][0] +
             0.1 * arraydc[ey][ex + 1][0] * arrayx[ey][ex + 1][0]) /
            arrayx[ey][ex][0];
      }
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

PetscErrorCode optimalCriteria(DM dm, Vec x, Vec dc, PetscReal volfrac,
                               PetscInt M, PetscInt N) {
  PetscFunctionBeginUser;
  PetscReal l1 = 0, l2 = 100000, move = 0.2, lmid;
  PetscInt startx, starty, nx, ny, ex, ey, rank;
  PetscReal ***arraydc, ***arrayx;
  PetscReal sum, allsum;
  PetscCall(DMDAVecGetArrayDOF(dm, dc, &arraydc));
  PetscCall(DMDAVecGetArrayDOF(dm, x, &arrayx));
  PetscCall(DMDAGetCorners(dm, &startx, &starty, NULL, &nx, &ny, NULL));
  while (l2 - l1 > 1e-4) {
    lmid = (l1 + l2) / 2;
    sum = 0;
    for (ey = starty; ey < starty + ny; ey++) {
      for (ex = startx; ex < startx + nx; ex++) {
        if (ex < M - 1 && ey < N - 1) {
          if (-arrayx[ey][ex][0] * arraydc[ey][ex][0] / lmid <
              fmax(0.001, arrayx[ey][ex][0] - move)) {
            arrayx[ey][ex][0] = fmax(0.001, arrayx[ey][ex][0] - move);
          } else if (-arrayx[ey][ex][0] * arraydc[ey][ex][0] / lmid >
                     fmin(1, arrayx[ey][ex][0] + move)) {
            arrayx[ey][ex][0] = fmin(1, arrayx[ey][ex][0] + move);
          } else
            arrayx[ey][ex][0] = -arrayx[ey][ex][0] * arraydc[ey][ex][0] / lmid;
          sum += arrayx[ey][ex][0];
        }
      }
    }
    PetscCall(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "rank: %d\n", rank));
    PetscCallMPI(MPI_Allreduce(&sum, &allsum, 1, MPIU_SCALAR, MPI_SUM,
                               PETSC_COMM_WORLD));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "allsum: %f\n", lmid));
    if (sum > volfrac) {
      l1 = lmid;
    } else {
      l2 = lmid;
    }
  }

  PetscCall(DMDAVecRestoreArrayDOF(dm, dc, &arraydc));
  PetscCall(DMDAVecRestoreArrayDOF(dm, x, &arrayx));
  PetscFunctionReturn(0);
}