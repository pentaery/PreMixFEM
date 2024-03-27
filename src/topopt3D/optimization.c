#include "optimization.h"
#include "PreMixFEM_3D.h"
#include "mpi.h"
#include "system.h"
#include <H5Spublic.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmdatypes.h>
#include <petscdmtypes.h>
#include <petscerror.h>
#include <petscksp.h>
#include <petsclog.h>
#include <petscmat.h>
#include <petscmath.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <petscviewer.h>

PetscErrorCode mmaInit(PCCtx *s_ctx, MMAx *mma_text) {
  PetscFunctionBeginUser;
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->mmaL));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->mmaU));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->xlast));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->xllast));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->xlllast));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->mmaLlast));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->mmaUlast));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->alpha));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->beta));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->lbd));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->ubd));
  PetscCall(VecSet(mma_text->xlast, volfrac));
  PetscCall(VecSet(mma_text->lbd, 0));
  PetscCall(VecSet(mma_text->ubd, 1));
  PetscFunctionReturn(0);
}

PetscErrorCode mmaFinal(MMAx *mma_text) {
  PetscFunctionBeginUser;
  PetscCall(VecDestroy(&mma_text->mmaL));
  PetscCall(VecDestroy(&mma_text->mmaU));
  PetscCall(VecDestroy(&mma_text->xlast));
  PetscCall(VecDestroy(&mma_text->xllast));
  PetscCall(VecDestroy(&mma_text->xlllast));
  PetscCall(VecDestroy(&mma_text->mmaLlast));
  PetscCall(VecDestroy(&mma_text->mmaUlast));
  PetscCall(VecDestroy(&mma_text->alpha));
  PetscCall(VecDestroy(&mma_text->beta));
  PetscCall(VecDestroy(&mma_text->lbd));
  PetscCall(VecDestroy(&mma_text->ubd));
  PetscFunctionReturn(0);
}
PetscErrorCode computeChange(MMAx *mma_text, Vec x, PetscScalar *change) {
  PetscFunctionBeginUser;
  PetscCall(VecCopy(mma_text->mmaL, mma_text->mmaLlast));
  PetscCall(VecCopy(mma_text->mmaU, mma_text->mmaUlast));
  PetscCall(VecCopy(mma_text->xllast, mma_text->xlllast));
  PetscCall(VecCopy(mma_text->xlast, mma_text->xllast));

  PetscCall(VecAXPY(mma_text->xlast, -1, x));
  PetscCall(VecNorm(mma_text->xlast, NORM_INFINITY, change));
  PetscCall(VecCopy(x, mma_text->xlast));
  PetscFunctionReturn(0);
}

PetscErrorCode formLimit(PCCtx *s_ctx, MMAx *mma_text, PetscInt loop) {
  PetscFunctionBeginUser;
  if (loop <= 2) {
    PetscCall(VecSet(mma_text->mmaL, -mmas0));
    PetscCall(VecSet(mma_text->mmaU, mmas0));
    PetscCall(VecAXPBYPCZ(mma_text->alpha, 0.1, 0.9, 0, mma_text->xlast,
                          mma_text->mmaL));
    PetscCall(VecAXPBYPCZ(mma_text->beta, 0.1, 0.9, 0, mma_text->xlast,
                          mma_text->mmaU));
    PetscCall(VecPointwiseMax(mma_text->alpha, mma_text->alpha, mma_text->lbd));
    PetscCall(VecPointwiseMin(mma_text->beta, mma_text->beta, mma_text->ubd));
  } else {
    PetscScalar ***arrayxlast, ***arrayxllast, ***arrayxlllast, ***arrayL,
        ***arrayU, ***arrayLlast, ***arrayUlast;
    PetscInt startx, starty, startz, nx, ny, nz, ex, ey, ez;
    PetscCall(
        DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));
    PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mma_text->xlast, &arrayxlast));
    PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mma_text->xllast, &arrayxllast));
    PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mma_text->xlllast, &arrayxlllast));
    PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mma_text->mmaLlast, &arrayLlast));
    PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mma_text->mmaUlast, &arrayUlast));
    PetscCall(DMDAVecGetArray(s_ctx->dm, mma_text->mmaL, &arrayL));
    PetscCall(DMDAVecGetArray(s_ctx->dm, mma_text->mmaU, &arrayU));
    for (ez = startz; ez < startz + nz; ++ez) {
      for (ey = starty; ey < starty + ny; ++ey) {
        for (ex = startx; ex < startx + nx; ++ex) {
          if ((arrayxlast[ez][ey][ex] - arrayxllast[ez][ey][ex]) *
                  (arrayxllast[ez][ey][ex] - arrayxlllast[ez][ey][ex]) <
              0) {
            arrayL[ez][ey][ex] =
                arrayxlast[ez][ey][ex] -
                (arrayxllast[ez][ey][ex] - arrayLlast[ez][ey][ex]) * mmas;
            arrayU[ez][ey][ex] =
                arrayxlast[ez][ey][ex] +
                (arrayUlast[ez][ey][ex] - arrayxllast[ez][ey][ex]) * mmas;
          } else {
            arrayL[ez][ey][ex] =
                arrayxlast[ez][ey][ex] -
                (arrayxllast[ez][ey][ex] - arrayLlast[ez][ey][ex]) / mmas;
            arrayU[ez][ey][ex] =
                arrayxlast[ez][ey][ex] +
                (arrayUlast[ez][ey][ex] - arrayxllast[ez][ey][ex]) / mmas;
          }
        }
      }
    }
    PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mma_text->xlast, &arrayxlast));
    PetscCall(
        DMDAVecRestoreArrayRead(s_ctx->dm, mma_text->xllast, &arrayxllast));
    PetscCall(
        DMDAVecRestoreArrayRead(s_ctx->dm, mma_text->xlllast, &arrayxlllast));
    PetscCall(
        DMDAVecRestoreArrayRead(s_ctx->dm, mma_text->mmaLlast, &arrayLlast));
    PetscCall(
        DMDAVecRestoreArrayRead(s_ctx->dm, mma_text->mmaUlast, &arrayUlast));
    PetscCall(DMDAVecRestoreArray(s_ctx->dm, mma_text->mmaL, &arrayL));
    PetscCall(DMDAVecRestoreArray(s_ctx->dm, mma_text->mmaU, &arrayU));

    PetscCall(VecAXPBYPCZ(mma_text->alpha, 0.1, 0.9, 0, mma_text->xlast,
                          mma_text->mmaL));
    PetscCall(VecAXPBYPCZ(mma_text->beta, 0.1, 0.9, 0, mma_text->xlast,
                          mma_text->mmaU));
    PetscCall(VecPointwiseMax(mma_text->alpha, mma_text->alpha, mma_text->lbd));
    PetscCall(VecPointwiseMin(mma_text->beta, mma_text->beta, mma_text->ubd));
  }

  PetscFunctionReturn(0);
}
PetscErrorCode computeCostMMA(PCCtx *s_ctx, Vec t, PetscScalar *cost) {
  PetscFunctionBeginUser;
  PetscCall(VecSum(t, cost));
  *cost = *cost / s_ctx->M / s_ctx->N / s_ctx->P;
  PetscFunctionReturn(0);
}
PetscErrorCode adjointGradient(PCCtx *s_ctx, Mat A, Vec x, Vec t, Vec dc,
                               PetscInt penal) {
  PetscFunctionBeginUser;
  PetscInt ex, ey, ez, nx, ny, nz, startx, starty, startz, i;
  PetscScalar ***arrayBoundary, ***arrayx, ***arraydc, ***arrayt,
      ***arraylambda, ***arraykappa[DIM];
  Vec lambda, dgT, kappa_loc[DIM], t_loc, lambda_loc;
  KSP ksp1;
  // calculate lambda
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &dgT));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &lambda));
  PetscCall(VecSet(dgT, 1.0 / s_ctx->M / s_ctx->N / s_ctx->P));
  PetscCall(VecSet(dc, 0));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp1));
  PetscCall(KSPSetOperators(ksp1, A, A));
  PetscCall(KSPSetFromOptions(ksp1));
  PetscCall(KSPSolve(ksp1, dgT, lambda));
  PetscCall(KSPDestroy(&ksp1));

  PetscCall(
      DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, s_ctx->boundary, &arrayBoundary));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, x, &arrayx));

  PetscCall(DMGetLocalVector(s_ctx->dm, &t_loc));
  PetscCall(DMGlobalToLocal(s_ctx->dm, t, INSERT_VALUES, t_loc));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, t_loc, &arrayt));
  for (i = 0; i < DIM; ++i) {
    PetscCall(DMGetLocalVector(s_ctx->dm, &kappa_loc[i]));
    PetscCall(DMGlobalToLocal(s_ctx->dm, s_ctx->kappa[i], INSERT_VALUES,
                              kappa_loc[i]));
    PetscCall(DMDAVecGetArrayRead(s_ctx->dm, kappa_loc[i], &arraykappa[i]));
  }

  PetscCall(DMGetLocalVector(s_ctx->dm, &lambda_loc));
  PetscCall(DMGlobalToLocal(s_ctx->dm, lambda, INSERT_VALUES, lambda_loc));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, lambda_loc, &arraylambda));

  // start computing dA
  PetscCall(DMDAVecGetArray(s_ctx->dm, dc, &arraydc));
  for (ez = startz; ez < startz + nz; ++ez) {
    for (ey = starty; ey < starty + ny; ++ey) {
      for (ex = startx; ex < startx + nx; ++ex) {
        arraydc[ez][ey][ex] -= penal * f0 * arraylambda[ez][ey][ex] *
                               (s_ctx->H_x * s_ctx->H_y * s_ctx->H_z) *
                               PetscPowScalar(arrayx[ez][ey][ex], penal - 1);
        if (ex > 0) {
          arraydc[ez][ey][ex] -=
              2 * penal * (kH - kL) *
              PetscPowScalar(arrayx[ez][ey][ex], penal - 1) *
              (s_ctx->H_y * s_ctx->H_z / s_ctx->H_x) *
              (arrayt[ez][ey][ex] - arrayt[ez][ey][ex - 1]) *
              (arraylambda[ez][ey][ex] - arraylambda[ez][ey][ex - 1]) /
              (PetscPowScalar(
                  (arraykappa[0][ez][ey][ex] / arraykappa[0][ez][ey][ex - 1]) +
                      1,
                  2));
        }
        if (ex < s_ctx->M - 1) {
          arraydc[ez][ey][ex] -=
              2 * penal * (kH - kL) *
              PetscPowScalar(arrayx[ez][ey][ex], penal - 1) *
              (s_ctx->H_y * s_ctx->H_z / s_ctx->H_x) *
              (arrayt[ez][ey][ex + 1] - arrayt[ez][ey][ex]) *
              (arraylambda[ez][ey][ex + 1] - arraylambda[ez][ey][ex]) /
              (PetscPowScalar(
                  (arraykappa[0][ez][ey][ex] / arraykappa[0][ez][ey][ex + 1]) +
                      1,
                  2));
        }
        if (ey > 0) {
          arraydc[ez][ey][ex] -=
              2 * penal * (kH - kL) *
              PetscPowScalar(arrayx[ez][ey][ex], penal - 1) *
              (s_ctx->H_x * s_ctx->H_z / s_ctx->H_y) *
              (arrayt[ez][ey][ex] - arrayt[ez][ey - 1][ex]) *
              (arraylambda[ez][ey][ex] - arraylambda[ez][ey - 1][ex]) /
              (PetscPowScalar(
                  (arraykappa[1][ez][ey][ex] / arraykappa[1][ez][ey - 1][ex]) +
                      1,
                  2));
        }
        if (ey < s_ctx->N - 1) {
          arraydc[ez][ey][ex] -=
              2 * penal * (kH - kL) *
              PetscPowScalar(arrayx[ez][ey][ex], penal - 1) *
              (s_ctx->H_x * s_ctx->H_z / s_ctx->H_y) *
              (arrayt[ez][ey + 1][ex] - arrayt[ez][ey][ex]) *
              (arraylambda[ez][ey + 1][ex] - arraylambda[ez][ey][ex]) /
              (PetscPowScalar(
                  (arraykappa[1][ez][ey][ex] / arraykappa[1][ez][ey + 1][ex]) +
                      1,
                  2));
        }
        if (ez > 0) {
          arraydc[ez][ey][ex] -=
              2 * penal * (kH - kL) *
              PetscPowScalar(arrayx[ez][ey][ex], penal - 1) *
              (s_ctx->H_x * s_ctx->H_y / s_ctx->H_z) *
              (arrayt[ez][ey][ex] - arrayt[ez - 1][ey][ex]) *
              (arraylambda[ez][ey][ex] - arraylambda[ez - 1][ey][ex]) /
              (PetscPowScalar(
                  (arraykappa[2][ez][ey][ex] / arraykappa[2][ez - 1][ey][ex]) +
                      1,
                  2));
        }
        if (ez < s_ctx->P - 1) {
          arraydc[ez][ey][ex] -=
              2 * penal * (kH - kL) *
              PetscPowScalar(arrayx[ez][ey][ex], penal - 1) *
              (s_ctx->H_x * s_ctx->H_y / s_ctx->H_z) *
              (arrayt[ez + 1][ey][ex] - arrayt[ez][ey][ex]) *
              (arraylambda[ez + 1][ey][ex] - arraylambda[ez][ey][ex]) /
              (PetscPowScalar(
                  (arraykappa[2][ez][ey][ex] / arraykappa[2][ez + 1][ey][ex]) +
                      1,
                  2));
        }
        if (arrayBoundary[ez][ey][ex] > 0.5) {
          arraydc[ez][ey][ex] -= 2 * penal * (kH - kL) *
                                 PetscPowScalar(arrayx[ez][ey][ex], penal - 1) *
                                 (s_ctx->H_x * s_ctx->H_y / s_ctx->H_z) *
                                 arrayt[ez][ey][ex] * arraylambda[ez][ey][ex];
          arraydc[ez][ey][ex] += 2 * penal * arraylambda[ez][ey][ex] *
                                 (kH - kL) *
                                 (s_ctx->H_x * s_ctx->H_y / s_ctx->H_z) * tD *
                                 PetscPowScalar(arrayx[ez][ey][ex], penal - 1);
        }
      }
    }
  }
  PetscCall(
      DMDAVecRestoreArrayRead(s_ctx->dm, s_ctx->boundary, &arrayBoundary));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, dc, &arraydc));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, x, &arrayx));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, t_loc, &arrayt));
  for (i = 0; i < DIM; ++i) {
    PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, kappa_loc[i], &arraykappa[i]));
    PetscCall(VecDestroy(&kappa_loc[i]));
  }
  PetscCall(VecDestroy(&dgT));
  PetscCall(VecDestroy(&t_loc));
  PetscCall(VecDestroy(&lambda_loc));

  PetscFunctionReturn(0);
}

PetscErrorCode mma(PCCtx *s_ctx, MMAx *mma_text, Vec dc, Vec x,
                   PetscScalar *initial) {
  PetscFunctionBeginUser;
  PetscScalar yL = 0.0, yU = 1000000.0;
  PetscScalar derivative = 0, y = *initial;
  while (yU - yL > 1e-6) {
    y = (yL + yU) / 2;
    PetscCall(computeDerivative(s_ctx, y, &derivative, mma_text, dc, x));
    if (derivative > 0) {
      yL = y;
    } else {
      yU = y;
    }
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "y: %f\n", y));
  PetscFunctionReturn(0);
}

PetscErrorCode computeDerivative(PCCtx *s_ctx, PetscScalar y,
                                 PetscScalar *derivative, MMAx *mma_text,
                                 Vec dc, Vec x) {
  PetscFunctionBeginUser;
  PetscCall(findX(s_ctx, y, mma_text, dc, x));
  // PetscCall(VecView(x, PETSC_VIEWER_STDOUT_WORLD));
  PetscInt startx, starty, startz, nx, ny, nz, ex, ey, ez;
  PetscScalar ***arrayxlast, ***arrayU, ***arrayderivative, ***arrayx;
  Vec Derivative;
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &Derivative));
  PetscCall(VecSet(Derivative, 0));
  PetscCall(
      DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mma_text->xlast, &arrayxlast));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mma_text->mmaU, &arrayU));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, x, &arrayx));
  PetscCall(DMDAVecGetArray(s_ctx->dm, Derivative, &arrayderivative));

  for (ez = startz; ez < startz + nz; ++ez) {
    for (ey = starty; ey < starty + ny; ++ey) {
      for (ex = startx; ex < startx + nx; ++ex) {
        arrayderivative[ez][ey][ex] +=
            ((arrayU[ez][ey][ex] - arrayxlast[ez][ey][ex]) *
             (arrayU[ez][ey][ex] - arrayxlast[ez][ey][ex])) /
            (arrayU[ez][ey][ex] - arrayx[ez][ey][ex]);
        arrayderivative[ez][ey][ex] +=
            (2 * arrayxlast[ez][ey][ex] - arrayU[ez][ey][ex]);
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mma_text->xlast, &arrayxlast));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mma_text->mmaU, &arrayU));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, x, &arrayx));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, Derivative, &arrayderivative));
  PetscCall(VecSum(Derivative, derivative));
  *derivative -= s_ctx->M * s_ctx->N * s_ctx->P * volfrac;
  PetscCall(VecDestroy(&Derivative));
  // PetscCall(PetscPrintf(PETSC_COMM_WORLD, "derivative: %f\n",
  // *derivative));
  PetscFunctionReturn(0);
}

PetscErrorCode findX(PCCtx *s_ctx, PetscScalar y, MMAx *mma_text, Vec dc,
                     Vec x) {
  PetscFunctionBeginUser;
  PetscInt startx, starty, startz, nx, ny, nz, ex, ey, ez;
  PetscScalar ***arrayxlast, ***arrayU, ***arrayL, ***arraydc, ***arrayalpha,
      ***arraybeta, ***arrayx, ***arrayl1, ***arrayl2;
  Vec l1, l2;
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &l1));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &l2));
  PetscCall(
      DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));
  PetscCall(DMDAVecGetArray(s_ctx->dm, l1, &arrayl1));
  PetscCall(DMDAVecGetArray(s_ctx->dm, l2, &arrayl2));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mma_text->xlast, &arrayxlast));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mma_text->mmaU, &arrayU));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mma_text->mmaL, &arrayL));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, dc, &arraydc));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mma_text->alpha, &arrayalpha));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mma_text->beta, &arraybeta));
  PetscCall(DMDAVecGetArray(s_ctx->dm, x, &arrayx));

  for (ez = startz; ez < startz + nz; ++ez) {
    for (ey = starty; ey < starty + ny; ++ey) {
      for (ex = startx; ex < startx + nx; ++ex) {
        if (arraydc[ez][ey][ex] >= 0) {
          arrayx[ez][ey][ex] = arrayalpha[ez][ey][ex];
        } else {
          arrayl1[ez][ey][ex] =
              (((arrayU[ez][ey][ex] - arrayxlast[ez][ey][ex]) *
                (arrayU[ez][ey][ex] - arrayxlast[ez][ey][ex])) *
               y /
               ((arrayU[ez][ey][ex] - arrayalpha[ez][ey][ex]) *
                (arrayU[ez][ey][ex] - arrayalpha[ez][ey][ex]))) +

              ((arraydc[ez][ey][ex] *
                (arrayxlast[ez][ey][ex] - arrayL[ez][ey][ex]) *
                (arrayxlast[ez][ey][ex] - arrayL[ez][ey][ex])) /
               ((arrayalpha[ez][ey][ex] - arrayL[ez][ey][ex]) *
                (arrayalpha[ez][ey][ex] - arrayL[ez][ey][ex])));
          arrayl2[ez][ey][ex] =
              (((arrayU[ez][ey][ex] - arrayxlast[ez][ey][ex]) *
                (arrayU[ez][ey][ex] - arrayxlast[ez][ey][ex])) *
               y /
               ((arrayU[ez][ey][ex] - arraybeta[ez][ey][ex]) *
                (arrayU[ez][ey][ex] - arraybeta[ez][ey][ex]))) +

              ((arraydc[ez][ey][ex] *
                (arrayxlast[ez][ey][ex] - arrayL[ez][ey][ex]) *
                (arrayxlast[ez][ey][ex] - arrayL[ez][ey][ex])) /
               ((arraybeta[ez][ey][ex] - arrayL[ez][ey][ex]) *
                (arraybeta[ez][ey][ex] - arrayL[ez][ey][ex])));

          if (arrayl1[ez][ey][ex] >= 0) {
            arrayx[ez][ey][ex] = arrayalpha[ez][ey][ex];
          } else if (arrayl2[ez][ey][ex] <= 0) {
            arrayx[ez][ey][ex] = arraybeta[ez][ey][ex];
          } else {
            arrayx[ez][ey][ex] =
                ((PetscSqrtScalar(
                      y * (arrayU[ez][ey][ex] - arrayxlast[ez][ey][ex]) *
                      (arrayU[ez][ey][ex] - arrayxlast[ez][ey][ex])) *
                  arrayL[ez][ey][ex]) +
                 (PetscSqrtScalar(
                      -arraydc[ez][ey][ex] *
                      (arrayxlast[ez][ey][ex] - arrayL[ez][ey][ex]) *
                      (arrayxlast[ez][ey][ex] - arrayL[ez][ey][ex])) *
                  arrayU[ez][ey][ex])) /
                (PetscSqrtScalar(
                     y * (arrayU[ez][ey][ex] - arrayxlast[ez][ey][ex]) *
                     (arrayU[ez][ey][ex] - arrayxlast[ez][ey][ex])) +
                 PetscSqrtScalar(
                     (-arraydc[ez][ey][ex]) *
                     (arrayxlast[ez][ey][ex] - arrayL[ez][ey][ex]) *
                     (arrayxlast[ez][ey][ex] - arrayL[ez][ey][ex])));
          }
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArray(s_ctx->dm, l1, &arrayl1));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, l2, &arrayl2));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, x, &arrayx));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mma_text->alpha, &arrayalpha));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mma_text->beta, &arraybeta));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, dc, &arraydc));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mma_text->xlast, &arrayxlast));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mma_text->mmaU, &arrayU));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mma_text->mmaL, &arrayL));
  // PetscCall(VecView(x, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecDestroy(&l1));
  PetscCall(VecDestroy(&l2));
  PetscFunctionReturn(0);
}

PetscErrorCode computeWy(PCCtx *s_ctx, PetscScalar y, PetscScalar *wy,
                         MMAx *mma_text, Vec dc, Vec x) {
  PetscFunctionBeginUser;
  PetscCall(findX(s_ctx, y, mma_text, dc, x));
  PetscInt startx, starty, startz, nx, ny, nz, ex, ey, ez;
  PetscScalar ***arrayxlast, ***arrayU, ***arrayL, ***arraywy, ***arrayx,
      ***arraydc;
  Vec value;
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &value));
  PetscCall(
      DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mma_text->xlast, &arrayxlast));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mma_text->mmaU, &arrayU));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mma_text->mmaU, &arrayL));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, x, &arrayx));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, dc, &arraydc));
  PetscCall(DMDAVecGetArray(s_ctx->dm, value, &arraywy));

  for (ez = startz; ez < startz + nz; ++ez) {
    for (ey = starty; ey < starty + ny; ++ey) {
      for (ex = startx; ex < startx + nx; ++ex) {
        if (arraydc[ez][ey][ex] > 0) {
          arraywy[ez][ey][ex] +=
              ((y + arraydc[ez][ey][ex]) *
               (arrayU[ez][ey][ex] - arrayxlast[ez][ey][ex]) *
               (arrayU[ez][ey][ex] - arrayxlast[ez][ey][ex])) /
              (arrayU[ez][ey][ex] - arrayx[ez][ey][ex]);
        } else {
          arraywy[ez][ey][ex] +=
              ((arrayU[ez][ey][ex] - arrayxlast[ez][ey][ex]) *
               (arrayU[ez][ey][ex] - arrayxlast[ez][ey][ex]) /
               (arrayU[ez][ey][ex] - arrayx[ez][ey][ex])) -
              ((arrayxlast[ez][ey][ex] - arrayL[ez][ey][ex]) *
                   (arrayxlast[ez][ey][ex] - arrayL[ez][ey][ex]) /
                   arrayx[ez][ey][ex] -
               arrayL[ez][ey][ex]);
        }
        arraywy[ez][ey][ex] +=
            2 * y * arrayxlast[ez][ey][ex] - arrayU[ez][ey][ex];
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mma_text->xlast, &arrayxlast));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mma_text->mmaU, &arrayU));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mma_text->mmaU, &arrayL));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, x, &arrayx));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, dc, &arraydc));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, value, &arraywy));
  PetscCall(VecSum(value, wy));
  *wy -= y * s_ctx->M * s_ctx->N * s_ctx->P * volfrac;
  PetscCall(VecDestroy(&value));
  PetscFunctionReturn(0);
}
PetscErrorCode mmatest(PCCtx *s_ctx, MMAx *mma_text, Vec dc, Vec x,
                       PetscScalar *initial) {
  PetscFunctionBeginUser;
  PetscScalar y = 0, derivative = 0, value = 0;
  for (y = 0; y < 10; y += 0.1) {

    PetscCall(computeDerivative(s_ctx, y, &derivative, mma_text, dc, x));
    PetscCall(computeWy(s_ctx, y, &value, mma_text, dc, x));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "y: %f\n", y));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "derivative: %f\n", derivative));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "value: %f\n", value));
  }

  PetscFunctionReturn(0);
}

// PetscErrorCode adjointGradient1(PCCtx *s_ctx, Mat A, Vec x, Vec t, Vec dc,
//                                 PetscInt penal) {
//   PetscFunctionBeginUser;
//   PetscInt ex, ey, ez, nx, ny, nz, startx, starty, startz, i;
//   PetscScalar ***arrayBoundary, ***arrayx, ***arraydf, ***arraydc,
//   ***arrayt,
//       ***arraylambda, ***arraykappa[DIM];
//   Vec lambda, dgT, dF, df, kappa_loc[DIM], t_loc, lambda_loc;
//   KSP ksp1;
//   // calculate lambda
//   PetscCall(DMCreateGlobalVector(s_ctx->dm, &dgT));
//   PetscCall(DMCreateGlobalVector(s_ctx->dm, &lambda));
//   PetscCall(DMCreateGlobalVector(s_ctx->dm, &df));
//   PetscCall(DMCreateGlobalVector(s_ctx->dm, &dF));
//   PetscCall(VecSet(dgT, 1.0 / s_ctx->M / s_ctx->N / s_ctx->P));
//   PetscCall(VecSet(dc, 0));
//   PetscCall(VecSet(df, 0));
//   PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp1));
//   PetscCall(KSPSetOperators(ksp1, A, A));
//   PetscCall(KSPSetFromOptions(ksp1));
//   PetscCall(KSPSolve(ksp1, dgT, lambda));
//   PetscCall(KSPDestroy(&ksp1));

//   PetscCall(
//       DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));
//   PetscCall(DMDAVecGetArray(s_ctx->dm, df, &arraydf));
//   PetscCall(DMDAVecGetArrayRead(s_ctx->dm, s_ctx->boundary,
//   &arrayBoundary)); PetscCall(DMDAVecGetArrayRead(s_ctx->dm, x, &arrayx));

//   PetscCall(DMGetLocalVector(s_ctx->dm, &t_loc));
//   PetscCall(DMGlobalToLocal(s_ctx->dm, t, INSERT_VALUES, t_loc));
//   PetscCall(DMDAVecGetArrayRead(s_ctx->dm, t_loc, &arrayt));
//   for (i = 0; i < DIM; ++i) {
//     PetscCall(DMGetLocalVector(s_ctx->dm, &kappa_loc[i]));
//     PetscCall(DMGlobalToLocal(s_ctx->dm, s_ctx->kappa[i], INSERT_VALUES,
//                               kappa_loc[i]));
//     PetscCall(DMDAVecGetArrayRead(s_ctx->dm, kappa_loc[i],
//     &arraykappa[i]));
//   }

//   for (ez = startz; ez < startz + nz; ++ez) {
//     for (ey = starty; ey < starty + ny; ++ey) {
//       for (ex = startx; ex < startx + nx; ++ex) {
//         arraydf[ez][ey][ex] -= 3 * f0 * s_ctx->H_x * s_ctx->H_y *
//         s_ctx->H_z
//         *
//                                arrayx[ez][ey][ex] * arrayx[ez][ey][ex];
//         if (arrayBoundary[ez][ey][ex] > 0.5) {
//           arraydf[ez][ey][ex] += 6 * (kH - kL) * s_ctx->H_x * s_ctx->H_y /
//                                  s_ctx->H_z * tD * arrayx[ez][ey][ex] *
//                                  arrayx[ez][ey][ex];
//         }
//       }
//     }
//   }

//   PetscCall(DMDAVecRestoreArray(s_ctx->dm, df, &arraydf));
//   PetscCall(VecPointwiseMult(dF, df, lambda));
//   // finish computing dF

//   PetscCall(DMGetLocalVector(s_ctx->dm, &lambda_loc));
//   PetscCall(DMGlobalToLocal(s_ctx->dm, lambda, INSERT_VALUES, lambda_loc));
//   PetscCall(DMDAVecGetArrayRead(s_ctx->dm, lambda_loc, &arraylambda));

//   // start computing dA
//   PetscCall(DMDAVecGetArray(s_ctx->dm, dc, &arraydc));
//   for (ez = startz; ez < startz + nz; ++ez) {
//     for (ey = starty; ey < starty + ny; ++ey) {
//       for (ex = startx; ex < startx + nx; ++ex) {
//         if (ex > 0) {
//           arraydc[ez][ey][ex] +=
//               6 * (1 - xCont) * PetscPowScalar(arrayx[ez][ey][ex], penal -
//               1)
//               * s_ctx->H_y * s_ctx->H_z / s_ctx->H_x * (arrayt[ez][ey][ex]
//               - arrayt[ez][ey][ex - 1]) * (arraylambda[ez][ey][ex] -
//               arraylambda[ez][ey][ex - 1]) / (PetscPowScalar(
//                   (arraykappa[0][ez][ey][ex] / arraykappa[0][ez][ey][ex -
//                   1])
//                   +
//                       1,
//                   2));
//         }
//         if (ex < s_ctx->M - 1) {
//           arraydc[ez][ey][ex] +=
//               6 * (1 - xCont) * PetscPowScalar(arrayx[ez][ey][ex], penal -
//               1)
//               * s_ctx->H_y * s_ctx->H_z / s_ctx->H_x * (arrayt[ez][ey][ex +
//               1] - arrayt[ez][ey][ex]) * (arraylambda[ez][ey][ex + 1] -
//               arraylambda[ez][ey][ex]) / (PetscPowScalar(
//                   (arraykappa[0][ez][ey][ex] / arraykappa[0][ez][ey][ex +
//                   1])
//                   +
//                       1,
//                   2));
//         }
//         if (ey > 0) {
//           arraydc[ez][ey][ex] +=
//               6 * (1 - xCont) * PetscPowScalar(arrayx[ez][ey][ex], penal -
//               1)
//               * s_ctx->H_x * s_ctx->H_z / s_ctx->H_y * (arrayt[ez][ey][ex]
//               - arrayt[ez][ey - 1][ex]) * (arraylambda[ez][ey][ex] -
//               arraylambda[ez][ey - 1][ex]) / (PetscPowScalar(
//                   (arraykappa[1][ez][ey][ex] / arraykappa[1][ez][ey -
//                   1][ex])
//                   +
//                       1,
//                   2));
//         }
//         if (ey < s_ctx->N - 1) {
//           arraydc[ez][ey][ex] +=
//               6 * (1 - xCont) * PetscPowScalar(arrayx[ez][ey][ex], penal -
//               1)
//               * s_ctx->H_x * s_ctx->H_z / s_ctx->H_y * (arrayt[ez][ey +
//               1][ex] - arrayt[ez][ey][ex]) * (arraylambda[ez][ey + 1][ex] -
//               arraylambda[ez][ey][ex]) / (PetscPowScalar(
//                   (arraykappa[1][ez][ey][ex] / arraykappa[1][ez][ey +
//                   1][ex])
//                   +
//                       1,
//                   2));
//         }
//         if (ez > 0) {
//           arraydc[ez][ey][ex] +=
//               6 * (1 - xCont) * PetscPowScalar(arrayx[ez][ey][ex], penal -
//               1)
//               * s_ctx->H_x * s_ctx->H_y / s_ctx->H_z * (arrayt[ez][ey][ex]
//               - arrayt[ez - 1][ey][ex]) * (arraylambda[ez][ey][ex] -
//               arraylambda[ez - 1][ey][ex]) / (PetscPowScalar(
//                   (arraykappa[2][ez][ey][ex] / arraykappa[2][ez -
//                   1][ey][ex])
//                   +
//                       1,
//                   2));
//         }
//         if (ez < s_ctx->P - 1) {
//           arraydc[ez][ey][ex] +=
//               6 * (1 - xCont) * PetscPowScalar(arrayx[ez][ey][ex], penal -
//               1)
//               * s_ctx->H_x * s_ctx->H_y / s_ctx->H_z * (arrayt[ez +
//               1][ey][ex] - arrayt[ez][ey][ex]) * (arraylambda[ez +
//               1][ey][ex]
//               - arraylambda[ez][ey][ex]) / (PetscPowScalar(
//                   (arraykappa[2][ez][ey][ex] / arraykappa[2][ez +
//                   1][ey][ex])
//                   +
//                       1,
//                   2));
//         }
//         if (arrayBoundary[ez][ey][ex] > 0.5) {
//           arraydc[ez][ey][ex] += 6 * (1 - xCont) * arrayx[ez][ey][ex] *
//                                  arrayx[ez][ey][ex] * s_ctx->H_x *
//                                  s_ctx->H_y / s_ctx->H_z *
//                                  arrayt[ez][ey][ex] *
//                                  arraylambda[ez][ey][ex];
//         }
//       }
//     }
//   }
//   PetscCall(
//       DMDAVecRestoreArrayRead(s_ctx->dm, s_ctx->boundary, &arrayBoundary));
//   PetscCall(DMDAVecRestoreArray(s_ctx->dm, dc, &arraydc));
//   PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, x, &arrayx));
//   PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, t_loc, &arrayt));
//   for (i = 0; i < DIM; ++i) {
//     PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, kappa_loc[i],
//     &arraykappa[i])); PetscCall(VecDestroy(&kappa_loc[i]));
//   }
//   PetscCall(VecAYPX(dc, -1, dF));

//   PetscCall(VecDestroy(&dgT));
//   PetscCall(VecDestroy(&t_loc));
//   PetscCall(VecDestroy(&lambda_loc));
//   PetscCall(VecDestroy(&df));
//   PetscCall(VecDestroy(&dF));
//   PetscFunctionReturn(0);
// }
