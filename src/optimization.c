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

PetscErrorCode formx(PCCtx *s_ctx, Vec x) {
  PetscFunctionBeginUser;
  PetscInt startx, starty, startz, nx, ny, nz, ex, ey, ez;
  PetscScalar ***array;
  PetscCall(
      DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));
  PetscCall(DMDAVecGetArray(s_ctx->dm, x, &array));
  for (ez = startz; ez < startz + nz; ++ez) {
    for (ey = starty; ey < starty + ny; ++ey) {
      for (ex = startx; ex < startx + nx; ++ex) {
        array[ez][ey][ex] = 0.5;
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, x, &array));
  PetscFunctionReturn(0);
}

PetscErrorCode formkappa(PCCtx *s_ctx, Vec x) {
  PetscFunctionBeginUser;
  PetscInt startx, starty, startz, nx, ny, nz, ex, ey, ez, i;
  PetscScalar ***arrayx, ***arraykappa[DIM];
  PetscCall(
      DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, x, &arrayx));
  for (i = 0; i < DIM; ++i) {
    PetscCall(DMDAVecGetArray(s_ctx->dm, s_ctx->kappa[i], &arraykappa[i]));
  }
  for (ez = startz; ez < startz + nz; ++ez) {
    for (ey = starty; ey < starty + ny; ++ey) {
      for (ex = startx; ex < startx + nx; ++ex) {
        for (i = 0; i < DIM; ++i) {
          arraykappa[i][ez][ey][ex] =
              (1 - PetscPowScalar(10, -cr)) *
                  PetscPowScalar(arrayx[ez][ey][ex], 3) +
              PetscPowScalar(10, -cr);
        }
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, x, &arrayx));
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
          // PetscCall(PetscPrintf(PETSC_COMM_SELF, "ex: %d, ey: %d, ez: %d\n",
          // ex,
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

PetscErrorCode formRHS(PCCtx *s_ctx, Vec rhs, Vec x) {
  PetscFunctionBeginUser;
  PetscScalar ***array, ***arraykappa, ***arrayx;
  PetscInt startx, starty, startz, nx, ny, nz, ex, ey, ez;
  PetscCall(
      DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));

  // Set RHS
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, x, &arrayx));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, s_ctx->kappa[2], &arraykappa));
  PetscCall(DMDAVecGetArray(s_ctx->dm, rhs, &array));
  for (ez = startz; ez < startz + nz; ++ez)
    for (ey = starty; ey < starty + ny; ey++) {
      for (ex = startx; ex < startx + nx; ex++) {
        array[ez][ey][ex] +=
            1 - arrayx[ez][ey][ex] * arrayx[ez][ey][ex] * arrayx[ez][ey][ex];
        if (ex >= PetscFloorReal((0.5 - s_ctx->lengthportion / 2) * s_ctx->M) &&
            ex <= PetscCeilReal((0.5 + s_ctx->lengthportion / 2) * s_ctx->M) &&
            ey >= PetscFloorReal((0.5 - s_ctx->widthportion / 2) * s_ctx->M) &&
            ey <= PetscCeilReal((0.5 + s_ctx->widthportion / 2) * s_ctx->M) &&
            ez == 0) {
          array[ez][ey][ex] = 2 * arraykappa[ez][ey][ex] * tD * s_ctx->H_x *
                              s_ctx->H_y / s_ctx->H_z;
        }
      }
    }
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, x, &arrayx));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, s_ctx->kappa[2], &arraykappa));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, rhs, &array));

  PetscFunctionReturn(0);
}

PetscErrorCode computeCost(PCCtx *s_ctx, Mat A, Vec t, PetscScalar *cost) {
  PetscFunctionBeginUser;
  PetscInt startx, starty, startz, nx, ny, nz, ex, ey, ez;
  Vec t_mod;
  Vec temp;
  PetscScalar ***arrayt, ***arraycost;
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &t_mod));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &temp));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, t, &arrayt));
  PetscCall(DMDAVecGetArray(s_ctx->dm, t_mod, &arraycost));

  PetscCall(
      DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));

  for (ez = startz; ez < startz + nz; ++ez) {
    for (ey = starty; ey < starty + ny; ey++) {
      for (ex = startx; ex < startx + nx; ex++) {
        arraycost[ez][ey][ex] = arrayt[ez][ey][ex];
        if (ex >= PetscFloorReal((0.5 - s_ctx->lengthportion / 2) * s_ctx->M) &&
            ex <= PetscCeilReal((0.5 + s_ctx->lengthportion / 2) * s_ctx->M) &&
            ey >= PetscFloorReal((0.5 - s_ctx->widthportion / 2) * s_ctx->M) &&
            ey <= PetscCeilReal((0.5 + s_ctx->widthportion / 2) * s_ctx->M) &&
            ez == 0) {
          arraycost[ez][ey][ex] -= tD;
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, t, &arrayt));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, t_mod, &arraycost));
  PetscCall(MatMult(A, t_mod, temp));
  PetscCall(VecDot(temp, t_mod, cost));

  PetscCall(VecDestroy(&t_mod));
  PetscCall(VecDestroy(&temp));
  PetscFunctionReturn(0);
}

PetscErrorCode computeGradient(PCCtx *s_ctx, Vec x, Vec t, Vec dc) {
  PetscFunctionBeginUser;
  PetscInt startx, starty, startz, nx, ny, nz, ex, ey, ez, i;
  PetscScalar ***arrayt, ***arraydc, ***arrayx, ***arr_kappa_3d[DIM];
  Vec t_loc;
  Vec kappa_loc[DIM];

  PetscCall(
      DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));

  PetscCall(DMGetLocalVector(s_ctx->dm, &t_loc));
  PetscCall(DMGlobalToLocal(s_ctx->dm, t, INSERT_VALUES, t_loc));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, t_loc, &arrayt));
  for (i = 0; i < DIM; ++i) {
    PetscCall(DMGetLocalVector(s_ctx->dm, &kappa_loc[i]));
    PetscCall(DMGlobalToLocal(s_ctx->dm, s_ctx->kappa[i], INSERT_VALUES,
                              kappa_loc[i]));
    PetscCall(DMDAVecGetArrayRead(s_ctx->dm, kappa_loc[i], &arr_kappa_3d[i]));
  }

  PetscCall(DMDAVecGetArray(s_ctx->dm, dc, &arraydc));
  PetscCall(DMDAVecGetArray(s_ctx->dm, x, &arrayx));

  for (ez = startz; ez < startz + nz; ++ez) {
    for (ey = starty; ey < starty + ny; ++ey) {
      for (ex = startx; ex < startx + nx; ++ex) {
        if (ex >= 1) {
          arraydc[ez][ey][ex] += 6 * (1 - 10e-6) / s_ctx->H_x * s_ctx->H_y *
                                 s_ctx->H_z * arrayx[ez][ey][ex] *
                                 arrayx[ez][ey][ex] *
                                 (arrayt[ez][ey][ex] - arrayt[ez][ey][ex - 1]) *
                                 (arrayt[ez][ey][ex] - arrayt[ez][ey][ex - 1]) /
                                 ((1 + arr_kappa_3d[0][ez][ey][ex] /
                                           arr_kappa_3d[0][ez][ey][ex - 1]) *
                                  (1 + arr_kappa_3d[0][ez][ey][ex] /
                                           arr_kappa_3d[0][ez][ey][ex - 1]));
        }
        if (ex <= s_ctx->M - 1) {
          arraydc[ez][ey][ex] += 6 * (1 - 10e-6) / s_ctx->H_x * s_ctx->H_y *
                                 s_ctx->H_z * arrayx[ez][ey][ex] *
                                 arrayx[ez][ey][ex] *
                                 (arrayt[ez][ey][ex + 1] - arrayt[ez][ey][ex]) *
                                 (arrayt[ez][ey][ex + 1] - arrayt[ez][ey][ex]) /
                                 ((1 + arr_kappa_3d[0][ez][ey][ex] /
                                           arr_kappa_3d[0][ez][ey][ex + 1]) *
                                  (1 + arr_kappa_3d[0][ez][ey][ex] /
                                           arr_kappa_3d[0][ez][ey][ex + 1]));
        }
        if (ey >= 1) {
          arraydc[ez][ey][ex] += 6 * (1 - 10e-6) * s_ctx->H_x / s_ctx->H_y *
                                 s_ctx->H_z * arrayx[ez][ey][ex] *
                                 arrayx[ez][ey][ex] *
                                 (arrayt[ez][ey][ex] - arrayt[ez][ey - 1][ex]) *
                                 (arrayt[ez][ey][ex] - arrayt[ez][ey - 1][ex]) /
                                 ((1 + arr_kappa_3d[1][ez][ey][ex] /
                                           arr_kappa_3d[1][ez][ey - 1][ex]) *
                                  (1 + arr_kappa_3d[1][ez][ey][ex] /
                                           arr_kappa_3d[1][ez][ey - 1][ex]));
        }
        if (ey < s_ctx->N - 1) {
          arraydc[ez][ey][ex] += 6 * (1 - 10e-6) * s_ctx->H_x / s_ctx->H_y *
                                 s_ctx->H_z * arrayx[ez][ey][ex] *
                                 arrayx[ez][ey][ex] *
                                 (arrayt[ez][ey + 1][ex] - arrayt[ez][ey][ex]) *
                                 (arrayt[ez][ey + 1][ex] - arrayt[ez][ey][ex]) /
                                 ((1 + arr_kappa_3d[1][ez][ey][ex] /
                                           arr_kappa_3d[1][ez][ey + 1][ex]) *
                                  (1 + arr_kappa_3d[1][ez][ey][ex] /
                                           arr_kappa_3d[1][ez][ey + 1][ex]));
        }
        if (ez >= 1) {
          arraydc[ez][ey][ex] += 6 * (1 - 10e-6) * s_ctx->H_x * s_ctx->H_y /
                                 s_ctx->H_z * arrayx[ez][ey][ex] *
                                 arrayx[ez][ey][ex] *
                                 (arrayt[ez][ey][ex] - arrayt[ez - 1][ey][ex]) *
                                 (arrayt[ez][ey][ex] - arrayt[ez - 1][ey][ex]) /
                                 ((1 + arr_kappa_3d[2][ez][ey][ex] /
                                           arr_kappa_3d[2][ez - 1][ey][ex]) *
                                  (1 + arr_kappa_3d[2][ez][ey][ex] /
                                           arr_kappa_3d[2][ez - 1][ey][ex]));
        }
        if (ez < s_ctx->P - 1) {
          arraydc[ez][ey][ex] += 6 * (1 - 10e-6) * s_ctx->H_x * s_ctx->H_y /
                                 s_ctx->H_z * arrayx[ez][ey][ex] *
                                 arrayx[ez][ey][ex] *
                                 (arrayt[ez + 1][ey][ex] - arrayt[ez][ey][ex]) *
                                 (arrayt[ez + 1][ey][ex] - arrayt[ez][ey][ex]) /
                                 ((1 + arr_kappa_3d[2][ez][ey][ex] /
                                           arr_kappa_3d[2][ez + 1][ey][ex]) *
                                  (1 + arr_kappa_3d[2][ez][ey][ex] /
                                           arr_kappa_3d[2][ez + 1][ey][ex]));
        }
        if (ex >= PetscFloorReal((0.5 - s_ctx->lengthportion / 2) * s_ctx->M) &&
            ex <= PetscCeilReal((0.5 + s_ctx->lengthportion / 2) * s_ctx->M) &&
            ey >= PetscFloorReal((0.5 - s_ctx->widthportion / 2) * s_ctx->M) &&
            ey <= PetscCeilReal((0.5 + s_ctx->widthportion / 2) * s_ctx->M) &&
            ez == 0) {
          arraydc[ez][ey][ex] +=
              6 * (1 - 10e-6) * s_ctx->H_x * s_ctx->H_y / s_ctx->H_z *
              arrayx[ez][ey][ex] * arrayx[ez][ey][ex] *
              (arrayt[ez][ey][ex] - tD) * (arrayt[ez][ey][ex] - tD);
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, t_loc, &arrayt));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, x, &arrayx));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, dc, &arraydc));
  for (i = 0; i < DIM; ++i) {
    PetscCall(
        DMDAVecRestoreArrayRead(s_ctx->dm, kappa_loc[i], &arr_kappa_3d[i]));
  }

  PetscCall(VecDestroy(&t_loc));
  for (i = 0; i < DIM; ++i) {
    PetscCall(VecDestroy(&kappa_loc[i]));
  }

  PetscFunctionReturn(0);
}

PetscErrorCode filter(PCCtx *s_ctx, Vec dc, Vec x) {
  PetscFunctionBeginUser;
  PetscInt startx, starty, startz, nx, ny, nz, ex, ey, ez;
  PetscScalar ***arraydc, ***arrayx, ***arraydcn, ***arraycoef;
  Vec localx, localdc, coef, dcn;
  PetscCall(
      DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &dcn));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &coef));
  PetscCall(VecSet(dcn, 0));
  PetscCall(VecSet(coef, 0));
  PetscCall(DMDAVecGetArray(s_ctx->dm, coef, &arraycoef));

  PetscCall(DMCreateLocalVector(s_ctx->dm, &localx));
  PetscCall(DMCreateLocalVector(s_ctx->dm, &localdc));

  PetscCall(DMGlobalToLocal(s_ctx->dm, x, INSERT_VALUES, localx));
  PetscCall(DMGlobalToLocal(s_ctx->dm, dc, INSERT_VALUES, localdc));

  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, localdc, &arraydc));
  PetscCall(DMDAVecGetArray(s_ctx->dm, dcn, &arraydcn));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, localx, &arrayx));

  for (ez = startz; ez < startz + nz; ++ez) {
    for (ey = starty; ey < starty + ny; ++ey) {
      for (ex = startx; ex < startx + nx; ++ex) {
        arraydcn[ez][ey][ex] += rmin * arraydc[ez][ey][ex] * arrayx[ez][ey][ex];
        if (ex >= 1) {
          arraydcn[ez][ey][ex] +=
              (rmin - 1) * arraydc[ez][ey][ex - 1] * arrayx[ez][ey][ex - 1];
          arraycoef[ez][ey][ex] += (rmin - 1) * arrayx[ez][ey][ex];
        }
        if (ex < s_ctx->M - 1) {
          arraydcn[ez][ey][ex] +=
              (rmin - 1) * arraydc[ez][ey][ex + 1] * arrayx[ez][ey][ex + 1];
          arraycoef[ez][ey][ex] += (rmin - 1) * arrayx[ez][ey][ex];
        }
        if (ey >= 1) {
          arraydcn[ez][ey][ex] +=
              (rmin - 1) * arraydc[ez][ey - 1][ex] * arrayx[ez][ey - 1][ex];
          arraycoef[ez][ey][ex] += (rmin - 1) * arrayx[ez][ey][ex];
          // PetscCall(PetscPrintf(PETSC_COMM_SELF, "%f\n",
          // arraycoef[ez][ey][ex]));
        }
        if (ey < s_ctx->N - 1) {
          arraydcn[ez][ey][ex] +=
              (rmin - 1) * arraydc[ez][ey + 1][ex] * arrayx[ez][ey + 1][ex];
          arraycoef[ez][ey][ex] += (rmin - 1) * arrayx[ez][ey][ex];
        }
        if (ez >= 1) {
          arraydcn[ez][ey][ex] +=
              (rmin - 1) * arraydc[ez - 1][ey][ex] * arrayx[ez - 1][ey][ex];
          arraycoef[ez][ey][ex] += (rmin - 1) * arrayx[ez][ey][ex];
        }
        if (ez < s_ctx->P - 1) {
          arraydcn[ez][ey][ex] +=
              (rmin - 1) * arraydc[ez + 1][ey][ex] * arrayx[ez + 1][ey][ex];
          arraycoef[ez][ey][ex] += (rmin - 1) * arrayx[ez][ey][ex];
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, localdc, &arraydc));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, localx, &arrayx));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, dcn, &arraydcn));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, coef, &arraycoef));

  // PetscCall(VecView(coef, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecPointwiseDivide(dc, dcn, coef));

  PetscCall(VecDestroy(&localx));
  PetscCall(VecDestroy(&localdc));
  PetscCall(VecDestroy(&dcn));
  PetscCall(VecDestroy(&coef));

  PetscFunctionReturn(0);
}

PetscErrorCode optimalCriteria(PCCtx *s_ctx, Vec x, Vec dc,
                               PetscScalar *change) {
  PetscFunctionBeginUser;
  PetscScalar l1 = 0, l2 = 100000, lmid;
  PetscInt startx, starty, startz, nx, ny, nz, ex, ey, ez;
  PetscScalar move = 0.2;
  PetscScalar ***arraydc, ***arrayx;
  PetscScalar sum;
  PetscScalar volume = volfrac * s_ctx->M * s_ctx->N * s_ctx->P;
  Vec xold;
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &xold));
  PetscCall(VecCopy(x, xold));
  PetscCall(
      DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, dc, &arraydc));

  while (l2 - l1 > 1e-4) {

    PetscCall(DMDAVecGetArray(s_ctx->dm, x, &arrayx));
    lmid = (l1 + l2) / 2;
    for (ez = startz; ez < startz + nz; ++ez) {
      for (ey = starty; ey < starty + ny; ++ey) {
        for (ex = startx; ex < startx + nx; ++ex) {
          if (arrayx[ez][ey][ex] * arraydc[ez][ey][ex] / lmid <
              PetscMax(0.001, arrayx[ez][ey][ex] - move)) {
            arrayx[ez][ey][ex] = PetscMax(0.001, arrayx[ez][ey][ex] - move);
          } else if (arrayx[ez][ey][ex] * arraydc[ez][ey][ex] / lmid >
                     PetscMin(1, arrayx[ez][ey][ex] + move)) {
            arrayx[ez][ey][ex] = PetscMin(1, arrayx[ez][ey][ex] + move);
          } else {
            arrayx[ez][ey][ex] =
                arrayx[ez][ey][ex] * arraydc[ez][ey][ex] / lmid;
          }
        }
      }
    }

    PetscCall(DMDAVecRestoreArray(s_ctx->dm, x, &arrayx));

    PetscCall(VecSum(x, &sum));

    if (sum > volume) {
      l1 = lmid;
    } else {
      l2 = lmid;
    }
    // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "lambda: %f\n", lmid));
  }
  PetscCall(VecAXPY(xold, -1, x));
  PetscCall(VecMax(xold, NULL, change));

  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, dc, &arraydc));
  PetscCall(VecDestroy(&xold));
  PetscFunctionReturn(0);
}

PetscErrorCode computeCost1(PCCtx *s_ctx, Vec x, Vec t, Vec c, Vec dc) {
  PetscFunctionBeginUser;
  PetscInt startx, starty, startz, nx, ny, nz, ex, ey, ez, i;
  PetscScalar ***arrayt, ***arrayc, ***arraydc, ***arrayx, ***arr_kappa_3d[DIM];
  Vec t_loc;
  Vec kappa_loc[DIM];

  PetscCall(
      DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));

  PetscCall(DMGetLocalVector(s_ctx->dm, &t_loc));
  PetscCall(DMGlobalToLocal(s_ctx->dm, t, INSERT_VALUES, t_loc));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, t_loc, &arrayt));
  for (i = 0; i < DIM; ++i) {
    PetscCall(DMGetLocalVector(s_ctx->dm, &kappa_loc[i]));
    PetscCall(DMGlobalToLocal(s_ctx->dm, s_ctx->kappa[i], INSERT_VALUES,
                              kappa_loc[i]));
    PetscCall(DMDAVecGetArrayRead(s_ctx->dm, kappa_loc[i], &arr_kappa_3d[i]));
  }

  PetscCall(DMDAVecGetArray(s_ctx->dm, c, &arrayc));
  PetscCall(DMDAVecGetArray(s_ctx->dm, dc, &arraydc));
  PetscCall(DMDAVecGetArray(s_ctx->dm, x, &arrayx));

  for (ez = startz; ez < startz + nz; ++ez) {
    for (ey = starty; ey < starty + ny; ++ey) {
      for (ex = startx; ex < startx + nx; ++ex) {
        if (ex >= 1) {
          arrayc[ez][ey][ex] += 2 * s_ctx->H_y * s_ctx->H_z / s_ctx->H_x *
                                (arrayt[ez][ey][ex] - arrayt[ez][ey][ex - 1]) *
                                (arrayt[ez][ey][ex] - arrayt[ez][ey][ex - 1]) /
                                (1 / arr_kappa_3d[0][ez][ey][ex] +
                                 1 / arr_kappa_3d[0][ez][ey][ex - 1]);
          arraydc[ez][ey][ex] += 6 * (1 - 10e-6) / s_ctx->H_x * s_ctx->H_y *
                                 s_ctx->H_z * arrayx[ez][ey][ex] *
                                 arrayx[ez][ey][ex] *
                                 (arrayt[ez][ey][ex] - arrayt[ez][ey][ex - 1]) *
                                 (arrayt[ez][ey][ex] - arrayt[ez][ey][ex - 1]) /
                                 ((1 + arr_kappa_3d[0][ez][ey][ex] /
                                           arr_kappa_3d[0][ez][ey][ex - 1]) *
                                  (1 + arr_kappa_3d[0][ez][ey][ex] /
                                           arr_kappa_3d[0][ez][ey][ex - 1]));
        }
        if (ex <= s_ctx->M - 1) {
          arraydc[ez][ey][ex] += 6 * (1 - 10e-6) / s_ctx->H_x * s_ctx->H_y *
                                 s_ctx->H_z * arrayx[ez][ey][ex] *
                                 arrayx[ez][ey][ex] *
                                 (arrayt[ez][ey][ex + 1] - arrayt[ez][ey][ex]) *
                                 (arrayt[ez][ey][ex + 1] - arrayt[ez][ey][ex]) /
                                 ((1 + arr_kappa_3d[0][ez][ey][ex] /
                                           arr_kappa_3d[0][ez][ey][ex + 1]) *
                                  (1 + arr_kappa_3d[0][ez][ey][ex] /
                                           arr_kappa_3d[0][ez][ey][ex + 1]));
        }
        if (ey >= 1) {
          arrayc[ez][ey][ex] += 2 * s_ctx->H_z * s_ctx->H_x / s_ctx->H_y *
                                (arrayt[ez][ey][ex] - arrayt[ez][ey - 1][ex]) *
                                (arrayt[ez][ey][ex] - arrayt[ez][ey - 1][ex]) /
                                (1 / arr_kappa_3d[1][ez][ey][ex] +
                                 1 / arr_kappa_3d[1][ez][ey - 1][ex]);
          arraydc[ez][ey][ex] += 6 * (1 - 10e-6) * s_ctx->H_x / s_ctx->H_y *
                                 s_ctx->H_z * arrayx[ez][ey][ex] *
                                 arrayx[ez][ey][ex] *
                                 (arrayt[ez][ey][ex] - arrayt[ez][ey - 1][ex]) *
                                 (arrayt[ez][ey][ex] - arrayt[ez][ey - 1][ex]) /
                                 ((1 + arr_kappa_3d[1][ez][ey][ex] /
                                           arr_kappa_3d[1][ez][ey - 1][ex]) *
                                  (1 + arr_kappa_3d[1][ez][ey][ex] /
                                           arr_kappa_3d[1][ez][ey - 1][ex]));
        }
        if (ey < s_ctx->N - 1) {
          arraydc[ez][ey][ex] += 6 * (1 - 10e-6) * s_ctx->H_x / s_ctx->H_y *
                                 s_ctx->H_z * arrayx[ez][ey][ex] *
                                 arrayx[ez][ey][ex] *
                                 (arrayt[ez][ey + 1][ex] - arrayt[ez][ey][ex]) *
                                 (arrayt[ez][ey + 1][ex] - arrayt[ez][ey][ex]) /
                                 ((1 + arr_kappa_3d[1][ez][ey][ex] /
                                           arr_kappa_3d[1][ez][ey + 1][ex]) *
                                  (1 + arr_kappa_3d[1][ez][ey][ex] /
                                           arr_kappa_3d[1][ez][ey + 1][ex]));
        }
        if (ez >= 1) {
          arrayc[ez][ey][ex] += 2 * s_ctx->H_x * s_ctx->H_y / s_ctx->H_z *
                                (arrayt[ez][ey][ex] - arrayt[ez - 1][ey][ex]) *
                                (arrayt[ez][ey][ex] - arrayt[ez - 1][ey][ex]) /
                                (1 / arr_kappa_3d[2][ez][ey][ex] +
                                 1 / arr_kappa_3d[2][ez - 1][ey][ex]);
          arraydc[ez][ey][ex] += 6 * (1 - 10e-6) * s_ctx->H_x * s_ctx->H_y /
                                 s_ctx->H_z * arrayx[ez][ey][ex] *
                                 arrayx[ez][ey][ex] *
                                 (arrayt[ez][ey][ex] - arrayt[ez - 1][ey][ex]) *
                                 (arrayt[ez][ey][ex] - arrayt[ez - 1][ey][ex]) /
                                 ((1 + arr_kappa_3d[2][ez][ey][ex] /
                                           arr_kappa_3d[2][ez - 1][ey][ex]) *
                                  (1 + arr_kappa_3d[2][ez][ey][ex] /
                                           arr_kappa_3d[2][ez - 1][ey][ex]));
        }
        if (ez < s_ctx->P - 1) {
          arraydc[ez][ey][ex] += 6 * (1 - 10e-6) * s_ctx->H_x * s_ctx->H_y /
                                 s_ctx->H_z * arrayx[ez][ey][ex] *
                                 arrayx[ez][ey][ex] *
                                 (arrayt[ez + 1][ey][ex] - arrayt[ez][ey][ex]) *
                                 (arrayt[ez + 1][ey][ex] - arrayt[ez][ey][ex]) /
                                 ((1 + arr_kappa_3d[2][ez][ey][ex] /
                                           arr_kappa_3d[2][ez + 1][ey][ex]) *
                                  (1 + arr_kappa_3d[2][ez][ey][ex] /
                                           arr_kappa_3d[2][ez + 1][ey][ex]));
        }
        if (ex >= PetscFloorReal((0.5 - s_ctx->lengthportion / 2) * s_ctx->M) &&
            ex <= PetscCeilReal((0.5 + s_ctx->lengthportion / 2) * s_ctx->M) &&
            ey >= PetscFloorReal((0.5 - s_ctx->widthportion / 2) * s_ctx->M) &&
            ey <= PetscCeilReal((0.5 + s_ctx->widthportion / 2) * s_ctx->M) &&
            ez == 0) {
          arrayc[ez][ey][ex] += 2 * s_ctx->H_x * s_ctx->H_y / s_ctx->H_z *
                                (arrayt[ez][ey][ex] - tD) *
                                (arrayt[ez][ey][ex] - tD) *
                                arr_kappa_3d[2][ez][ey][ex];
          arraydc[ez][ey][ex] +=
              6 * (1 - 10e-6) * s_ctx->H_x * s_ctx->H_y / s_ctx->H_z *
              arrayx[ez][ey][ex] * arrayx[ez][ey][ex] *
              (arrayt[ez][ey][ex] - tD) * (arrayt[ez][ey][ex] - tD);
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, t_loc, &arrayt));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, x, &arrayx));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, dc, &arraydc));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, c, &arrayc));
  for (i = 0; i < DIM; ++i) {
    PetscCall(
        DMDAVecRestoreArrayRead(s_ctx->dm, kappa_loc[i], &arr_kappa_3d[i]));
  }

  PetscCall(VecDestroy(&t_loc));
  for (i = 0; i < DIM; ++i) {
    PetscCall(VecDestroy(&kappa_loc[i]));
  }

  PetscFunctionReturn(0);
}