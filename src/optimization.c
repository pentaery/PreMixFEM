#include "optimization.h"
#include "PreMixFEM_3D.h"
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

PetscErrorCode formx(PCCtx *s_ctx) {
  PetscFunctionBeginUser;
  PetscInt startx, starty, startz, nx, ny, nz, ex, ey, ez;
  PetscScalar ***array;
  PetscCall(
      DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));
  PetscCall(DMDAVecGetArray(s_ctx->dm, s_ctx->x, &array));
  for (ez = startz; ez < startz + nz; ++ez) {
    for (ey = starty; ey < starty + ny; ++ey) {
      for (ex = startx; ex < startx + nx; ++ex) {
        array[ez][ey][ex] = 0.5;
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, s_ctx->x, &array));
  PetscFunctionReturn(0);
}

PetscErrorCode(formkappa(PCCtx *s_ctx)) {
  PetscFunctionBeginUser;
  PetscInt startx, starty, startz, nx, ny, nz, ex, ey, ez, i;
  PetscScalar ***arrayx, ***arraykappa[DIM];
  PetscCall(
      DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));
  PetscCall(DMDAVecGetArray(s_ctx->dm, s_ctx->x, &arrayx));
  for (i = 0; i < DIM; ++i) {
    PetscCall(DMDAVecGetArray(s_ctx->dm, s_ctx->kappa[i], &arraykappa[i]));
  }
  for (ez = startz; ez < startz + nz; ++ez) {
    for (ey = starty; ey < starty + ny; ++ey) {
      for (ex = startx; ex < startx + nx; ++ex) {
        for (i = 0; i < DIM; ++i) {
          arraykappa[i][ez][ey][ex] =
              (1 - PetscPowScalar(10, -s_ctx->cr)) *
                  PetscPowScalar(arrayx[ez][ey][ex], 3) +
              PetscPowScalar(10, -s_ctx->cr);
        }
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, s_ctx->x, &arrayx));
  for (i = 0; i < DIM; ++i) {
    PetscCall(DMDAVecRestoreArray(s_ctx->dm, s_ctx->kappa[i], &arraykappa[i]));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode formMatrix(PCCtx *s_ctx, Mat A) {
  PetscFunctionBeginUser;

  Vec kappa_loc[DIM]; // Destroy later.
  PetscInt proc_startx, proc_starty, proc_startz, proc_nx, proc_ny, proc_nz, ex,
      ey, ez, i;
  // 赋值迭代变量
  PetscScalar ***arr_kappa_3d[DIM], val_A[2][2], meas_elem, meas_face_yz,
      meas_face_zx, meas_face_xy, avg_kappa_e;
  MatStencil row[2], col[2];

  for (i = 0; i < DIM; ++i) {
    PetscCall(DMGetLocalVector(s_ctx->dm, &kappa_loc[i]));
    PetscCall(DMGlobalToLocal(s_ctx->dm, s_ctx->kappa[i], INSERT_VALUES,
                              kappa_loc[i]));
    PetscCall(DMDAVecGetArrayRead(s_ctx->dm, kappa_loc[i], &arr_kappa_3d[i]));
  }
  // 将每个进程中的kappa取出来

  meas_elem = s_ctx->H_x * s_ctx->H_y * s_ctx->H_z;
  meas_face_yz = s_ctx->H_y * s_ctx->H_z;
  meas_face_zx = s_ctx->H_z * s_ctx->H_x;
  meas_face_xy = s_ctx->H_x * s_ctx->H_y;
  // 计算网格体积和面积

  PetscCall(DMDAGetCorners(s_ctx->dm, &proc_startx, &proc_starty, &proc_startz,
                           &proc_nx, &proc_ny, &proc_nz));
  for (ez = proc_startz; ez < proc_startz + proc_nz; ++ez)
    for (ey = proc_starty; ey < proc_starty + proc_ny; ++ey)
      for (ex = proc_startx; ex < proc_startx + proc_nx; ++ex) {
        if (ex >= 1) {
          row[0] = (MatStencil){.i = ex - 1, .j = ey, .k = ez};
          row[1] = (MatStencil){.i = ex, .j = ey, .k = ez};
          col[0] = (MatStencil){.i = ex - 1, .j = ey, .k = ez};
          col[1] = (MatStencil){.i = ex, .j = ey, .k = ez};
          avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[0][ez][ey][ex - 1] +
                               1.0 / arr_kappa_3d[0][ez][ey][ex]);
          val_A[0][0] = meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e;
          val_A[0][1] = -meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e;
          val_A[1][0] = -meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e;
          val_A[1][1] = meas_face_yz * meas_face_yz / meas_elem * avg_kappa_e;
          PetscCall(MatSetValuesStencil(A, 2, &row[0], 2, &col[0], &val_A[0][0],
                                        ADD_VALUES));    
        }
        if (ey >= 1) {
          row[0] = (MatStencil){.i = ex, .j = ey - 1, .k = ez};
          row[1] = (MatStencil){.i = ex, .j = ey, .k = ez};
          col[0] = (MatStencil){.i = ex, .j = ey - 1, .k = ez};
          col[1] = (MatStencil){.i = ex, .j = ey, .k = ez};
          avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[1][ez][ey - 1][ex] +
                               1.0 / arr_kappa_3d[1][ez][ey][ex]);
          val_A[0][0] = meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e;
          val_A[0][1] = -meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e;
          val_A[1][0] = -meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e;
          val_A[1][1] = meas_face_zx * meas_face_zx / meas_elem * avg_kappa_e;
          PetscCall(MatSetValuesStencil(A, 2, &row[0], 2, &col[0], &val_A[0][0],
                                        ADD_VALUES));
        }
        if (ez >= 1) {
          row[0] = (MatStencil){.i = ex, .j = ey, .k = ez - 1};
          row[1] = (MatStencil){.i = ex, .j = ey, .k = ez};
          col[0] = (MatStencil){.i = ex, .j = ey, .k = ez - 1};
          col[1] = (MatStencil){.i = ex, .j = ey, .k = ez};
          avg_kappa_e = 2.0 / (1.0 / arr_kappa_3d[2][ez - 1][ey][ex] +
                               1.0 / arr_kappa_3d[2][ez][ey][ex]);
          val_A[0][0] = meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e;
          val_A[0][1] = -meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e;
          val_A[1][0] = -meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e;
          val_A[1][1] = meas_face_xy * meas_face_xy / meas_elem * avg_kappa_e;
          PetscCall(MatSetValuesStencil(A, 2, &row[0], 2, &col[0], &val_A[0][0],
                                        ADD_VALUES));
        }
        if (ex >= PetscFloorReal((0.5 - s_ctx->lengthportion / 2) * s_ctx->M) &&
            ex <= PetscCeilReal((0.5 + s_ctx->lengthportion / 2) * s_ctx->M) &&
            ey >= PetscFloorReal((0.5 - s_ctx->widthportion / 2) * s_ctx->M) &&
            ey <= PetscCeilReal((0.5 + s_ctx->widthportion / 2) * s_ctx->M) &&
            ez == 0) {
          col[0] = (MatStencil){.i = ex, .j = ey, .k = ez};
          row[0] = (MatStencil){.i = ex, .j = ey, .k = ez};
          val_A[0][0] = 2 * meas_face_xy * meas_face_xy / meas_elem *
                        arr_kappa_3d[2][ez][ey][ex];
          PetscCall(MatSetValuesStencil(A, 1, &col[0], 1, &row[0], &val_A[0][0],
                                        ADD_VALUES));
          // PetscCall(PetscPrintf(PETSC_COMM_SELF, "ex: %d, ey: %d, ez: %d\n", ex,
          //                       ey, ez));
        }
      }
  // A的赋值
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  for (i = 0; i < DIM; ++i) {
    PetscCall(
        DMDAVecRestoreArrayRead(s_ctx->dm, kappa_loc[i], &arr_kappa_3d[i]));
    PetscCall(DMRestoreLocalVector(s_ctx->dm, &kappa_loc[i]));
    PetscCall(VecDestroy(&kappa_loc[i]));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode formRHS(PCCtx *s_ctx, Vec rhs) {
  PetscFunctionBeginUser;
  PetscScalar ***array, ***arraykappa;
  PetscInt startx, starty, startz, nx, ny, nz, ex, ey, ez;
  PetscCall(
      DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));

  // Set RHS
  PetscCall(DMDAVecGetArray(s_ctx->dm, s_ctx->kappa[2], &arraykappa));
  PetscCall(DMDAVecGetArray(s_ctx->dm, rhs, &array));
  for (ez = startz; ez < startz + nz; ++ez)
    for (ey = starty; ey < starty + ny; ey++) {
      for (ex = startx; ex < startx + nx; ex++) {
        if (ex >= PetscFloorReal((0.5 - s_ctx->lengthportion / 2) * s_ctx->M) &&
            ex <= PetscCeilReal((0.5 + s_ctx->lengthportion / 2) * s_ctx->M) &&
            ey >= PetscFloorReal((0.5 - s_ctx->widthportion / 2) * s_ctx->M) &&
            ey <= PetscCeilReal((0.5 + s_ctx->widthportion / 2) * s_ctx->M) &&
            ez == 0) {
          array[ez][ey][ex] = 2 * arraykappa[ez][ey][ex] * 1e3 * s_ctx->H_x *
                              s_ctx->H_y / s_ctx->H_z;
        }
      }
    }

  PetscCall(DMDAVecRestoreArray(s_ctx->dm, rhs, &array));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, s_ctx->kappa[2], &arraykappa));

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