#include "MMA.h"
#include "system.h"
#include <PreMixFEM_3D.h>
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
PetscErrorCode mmaInit(PCCtx *s_ctx, MMAx *mma_text) {
  PetscFunctionBeginUser;
  PetscInt i = 0;
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
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->xsign));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->dgT));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->p0));
  PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->q0));
  for (i = 0; i < m; i++) {
    PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->p[i]));
    PetscCall(DMCreateGlobalVector(s_ctx->dm, &mma_text->q[i]));
    mma_text->a[i] = 0;
    mma_text->d[i] = 1;
    mma_text->c[i] = 100;
  }
  mma_text->a0 = 1;
  PetscCall(VecSet(mma_text->dgT, 1.0 / s_ctx->M / s_ctx->N / s_ctx->P));
  PetscCall(VecSet(mma_text->xlast, volfrac));
  PetscCall(VecSet(mma_text->lbd, xmin));
  PetscCall(VecSet(mma_text->ubd, xmax));
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
  PetscCall(VecDestroy(&mma_text->xsign));
  PetscCall(VecDestroy(&mma_text->dgT));
  PetscFunctionReturn(0);
}
PetscErrorCode mmaLimit(PCCtx *s_ctx, MMAx *mmax, Vec x, Vec t, PetscInt loop) {
  PetscFunctionBeginUser;
  PetscScalar asyinit = 0.5;
  PetscScalar asyincr = 1 / 0.85;
  PetscScalar asydecr = 0.85;
  PetscScalar sign = 0;
  PetscScalar albefa = 0.1;
  PetscScalar move = 0.5;
  PetscInt startx, starty, startz, nx, ny, nz, ex, ey, ez;
  PetscScalar ***xval, ***xold1, ***xold2, ***low, ***upp;
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->xlast, &xval));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->xllast, &xold1));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->xlllast, &xold2));
  PetscCall(
      DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));
  if (loop <= 2) {
    PetscCall(VecWAXPY(mmax->mmaL, -asyinit, mmax->ubd, mmax->xlast));
    PetscCall(VecAXPY(mmax->mmaL, asyinit, mmax->lbd));
    PetscCall(VecWAXPY(mmax->mmaL, asyinit, mmax->ubd, mmax->xlast));
    PetscCall(VecAXPY(mmax->mmaL, -asyinit, mmax->lbd));
  } else {
    PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->mmaL, &low));
    PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->mmaU, &upp));
    for (ez = startz; ez < startz + nz; ++ez) {
      for (ey = starty; ey < starty + ny; ++ey) {
        for (ex = startx; ex < startx + nx; ++ex) {
          sign = (xval[ez][ey][ex] - xold1[ez][ey][ex]) *
                 (xold1[ez][ey][ex] - xold2[ez][ey][ex]);
          if (sign > 0) {
            low[ez][ey][ex] = xval[ez][ey][ex] -
                              asyincr * (xold1[ez][ey][ex] - low[ez][ey][ex]);

            upp[ez][ey][ex] = xval[ez][ey][ex] +
                              asyincr * (upp[ez][ey][ex] - xold1[ez][ey][ex]);
          } else if (sign < 0) {
            low[ez][ey][ex] = xval[ez][ey][ex] -
                              asydecr * (xold1[ez][ey][ex] - low[ez][ey][ex]);
            upp[ez][ey][ex] = xval[ez][ey][ex] +
                              asydecr * (upp[ez][ey][ex] - xold1[ez][ey][ex]);
          }
          low[ez][ey][ex] = PetscMax(low[ez][ey][ex],
                                     xval[ez][ey][ex] - 10.0 * (xmax - xmin));
          low[ez][ey][ex] = PetscMin(low[ez][ey][ex],
                                     xval[ez][ey][ex] - 0.01 * (xmax - xmin));
          upp[ez][ey][ex] = PetscMax(upp[ez][ey][ex],
                                     xval[ez][ey][ex] + 0.01 * (xmax - xmin));
          upp[ez][ey][ex] = PetscMin(upp[ez][ey][ex],
                                     xval[ez][ey][ex] + 10.0 * (xmax - xmin));
        }
      }
    }
    PetscCall(DMDAVecRestoreArray(s_ctx->dm, mmax->mmaL, &low));
    PetscCall(DMDAVecRestoreArray(s_ctx->dm, mmax->mmaU, &upp));
  }

  PetscCall(VecCopy(mmax->mmaL, mmax->zzz1));
  PetscCall(
      VecAXPBYPCZ(mmax->zzz1, albefa, -albefa, 1, mmax->xlast, mmax->mmaL));
  PetscCall(VecCopy(mmax->xlast, mmax->zzz2));
  PetscCall(VecShift(mmax->zzz2, move * (xmin - xmax)));
  PetscCall(VecPointwiseMax(mmax->zzz, mmax->zzz1, mmax->zzz2));
  PetscCall(VecPointwiseMax(mmax->alpha, mmax->zzz, mmax->lbd));

  PetscCall(VecCopy(mmax->mmaU, mmax->zzz1));
  PetscCall(
      VecAXPBYPCZ(mmax->zzz1, albefa, -albefa, 1, mmax->xlast, mmax->mmaU));
  PetscCall(VecCopy(mmax->xlast, mmax->zzz2));
  PetscCall(VecShift(mmax->zzz2, move * (xmax - xmin)));
  PetscCall(VecPointwiseMin(mmax->zzz, mmax->zzz1, mmax->zzz2));
  PetscCall(VecPointwiseMin(mmax->beta, mmax->zzz, mmax->ubd));

  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->xlast, &xval));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->xllast, &xold1));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->xlllast, &xold2));
  PetscFunctionReturn(0);
}
PetscErrorCode mmaSub(PCCtx *s_ctx, MMAx *mmax, Vec x, Vec t, Vec dc) {
  PetscFunctionBeginUser;
  PetscInt ex, ey, ez, nx, ny, nz, startx, starty, startz, i;
  PetscScalar ***arraydc, ***arrayp0, ***arrayq0, ***arrayp[m], ***arrayq[m],
      ***arrayb[m], ***arrayL, ***arrayU, ***arrayx;
  PetscCall(
      DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, dc, &arraydc));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->mmaL, &arrayL));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->mmaU, &arrayU));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->xlast, &arrayx));
  PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->p0, &arrayp0));
  PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->q0, &arrayq0));
  for (i = 0; i < m; i++) {
    PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->p[i], &arrayp[i]));
    PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->q[i], &arrayq[i]));
    PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->b[i], &arrayb[i]));
  }
  for (ez = startz; ez < startz + nz; ++ez) {
    for (ey = starty; ey < starty + ny; ++ey) {
      for (ex = startx; ex < startx + nx; ++ex) {
        arrayp0[ez][ey][ex] =
            PetscPowScalar(arrayU[ez][ey][ex] - arrayx[ez][ey][ex], 2) *
            (1.001 * PetscMax(0.0, arraydc[ez][ey][ex]) +
             0.001 * PetscMax(0.0, -arraydc[ez][ey][ex]) +
             1e-5 / (xmax - xmin));
        arrayq0[ez][ey][ex] =
            PetscPowScalar(arrayx[ez][ey][ex] - arrayL[ez][ey][ex], 2) *
            (1.001 * PetscMax(0.0, -arraydc[ez][ey][ex]) +
             0.001 * PetscMax(0.0, arraydc[ez][ey][ex]) + 1e-5 / (xmax - xmin));
        for (i = 0; i < m; ++i) {
          arrayp[i][ez][ey][ex] =
              PetscPowScalar(arrayU[ez][ey][ex] - arrayx[ez][ey][ex], 2) *
              (1.001 + 1e-5 / (xmax - xmin));
          arrayq[i][ez][ey][ex] =
              PetscPowScalar(arrayx[ez][ey][ex] - arrayL[ez][ey][ex], 2) *
              (0.001 + 1e-5 / (xmax - xmin));
          arrayb[i][ez][ey][ez] = (arrayU[ez][ey][ex] - arrayx[ez][ey][ex]) *
                                      (1.001 + 1e-5 / (xmax - xmin)) +
                                  (arrayx[ez][ey][ex] - arrayL[ez][ey][ex]) *
                                      (0.001 + 1e-5 / (xmax - xmin));
        }
      }
    }
  }
  PetscScalar fval = 0;
  PetscCall(VecSum(mmax->xlast, &fval));
  for (i = 0; i < m; ++i) {
    PetscCall(DMDAVecRestoreArray(s_ctx->dm, mmax->b[i], &arrayb[i]));
    PetscCall(VecSum(mmax->b[i], &mmax->bval[i]));
    mmax->bval[i] -= fval;
    mmax->bval[i] += s_ctx->M * s_ctx->N * s_ctx->P * volfrac;
  }
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, dc, &arraydc));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->mmaL, &arrayL));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->mmaU, &arrayU));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->xlast, &arrayx));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, mmax->p0, &arrayp0));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, mmax->q0, &arrayq0));
  for (i = 0; i < m; i++) {
    PetscCall(DMDAVecRestoreArray(s_ctx->dm, mmax->p[i], &arrayp[i]));
    PetscCall(DMDAVecRestoreArray(s_ctx->dm, mmax->q[i], &arrayq[i]));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode subSolv(PCCtx *s_ctx, MMAx *mmax, Vec x, Vec t) {
  PetscFunctionBeginUser;
  PetscCall(omegaInitial(s_ctx, mmax, x));
  PetscInt itera = 0;
  PetscScalar epsi = 1, residumax, residunorm;
  while (epsi > epsimin) {
    PetscCall(computeResidual(s_ctx, mmax, x, epsi, &residumax, &residunorm));
    PetscInt ittt = 0;
    while (residumax > 0.9 * epsi && ittt < 200) {
      ittt += 1;
      itera += 1;
    }
  }

  PetscFunctionReturn(0);
}
PetscErrorCode omegaInitial(PCCtx *s_ctx, MMAx *mmax, Vec x) {
  PetscFunctionBeginUser;
  PetscInt startx, starty, startz, nx, ny, nz, ex, ey, ez, i;
  PetscScalar ***arrayeta, ***arrayxsi, ***arrayx, ***arrayalpha, ***arraybeta;
  PetscCall(
      DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));
  PetscCall(VecAXPBYPCZ(x, 0.5, 0.5, 0, mmax->alpha, mmax->beta));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->beta, &arraybeta));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->alpha, &arrayalpha));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, x, &arrayx));
  PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->eta, &arrayeta));
  PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->xsi, &arrayxsi));
  for (ez = startz; ez < startz + nz; ++ez) {
    for (ey = starty; ey < starty + ny; ++ey) {
      for (ex = startx; ex < startx + nx; ++ex) {
        arrayeta[ez][ey][ex] =
            PetscMax(1, 1 / (arraybeta[ez][ey][ex] - arrayx[ez][ey][ex]));
        arrayxsi[ez][ey][ex] =
            PetscMax(1, 1 / (arrayx[ez][ey][ex] - arrayalpha[ez][ey][ex]));
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->beta, &arraybeta));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->alpha, &arrayalpha));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, x, &arrayx));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, mmax->eta, &arrayeta));
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, mmax->xsi, &arrayxsi));
  mmax->z = 1;
  mmax->zet = 1;
  for (i = 0; i < m; ++i) {
    mmax->y[i] = 1;
    mmax->lam[i] = 1;
    mmax->s[i] = 1;
    mmax->mu[i] = PetscMax(1, mmax->c[i] / 2);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode computeResidual(PCCtx *s_ctx, MMAx *mmax, Vec x,
                               PetscScalar epsi, PetscScalar *residumax,
                               PetscScalar *residunorm) {
  PetscFunctionBeginUser;
  PetscInt startx, starty, startz, nx, ny, nz, ex, ey, ez, i;
  PetscScalar ***arrayx, ***arrayrex, ***arrayxsi, ***arrayeta, ***arrayU,
      ***arrayL, ***arrayp0, ***arrayq0, ***arrayrexsi, ***arrayreeta,
      ***arrayalpha, ***arraybeta, ***arrayp[m], ***arrayq[m], ***arraygvec[m];
  PetscScalar residuanorm, resudumax;
  PetscCall(
      DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->alpha, &arrayalpha));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->beta, &arraybeta));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->xsi, &arrayxsi));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->eta, &arrayeta));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, x, &arrayx));
  PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->rex, &arrayrex));
  PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->rexsi, &arrayrexsi));
  PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->reeta, &arrayreeta));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->mmaL, &arrayL));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->mmaU, &arrayU));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->p0, &arrayp0));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->q0, &arrayq0));
  for (i = 0; i < m; i++) {
    PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->p[i], &arrayp[i]));
    PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->q[i], &arrayq[i]));
    PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->gvec[i], &arraygvec[i]));
  }

  for (ez = startz; ez < startz + nz; ++ez) {
    for (ey = starty; ey < starty + ny; ++ey) {
      for (ex = startx; ex < startx + nx; ++ex) {
        arrayrex[ez][ey][ex] =
            arrayp0[ez][ey][ex] / ((arrayU[ez][ey][ex] - arrayx[ez][ey][ex]) *
                                   (arrayU[ez][ey][ex] - arrayx[ez][ey][ex])) +
            arrayq0[ez][ey][ex] / ((arrayx[ez][ey][ex] - arrayL[ez][ey][ex]) *
                                   (arrayx[ez][ey][ex] - arrayL[ez][ey][ex]));
        arrayrexsi[ez][ey][ex] =
            arrayxsi[ez][ey][ex] *
                (arrayx[ez][ey][ex] - arrayalpha[ez][ey][ex]) -
            1;
        arrayreeta[ez][ey][ex] = arrayeta[ez][ey][ex] * (arraybeta[ez][ey][ex] -
                                                         arrayx[ez][ey][ex]) -
                                 1;

        for (int i = 0; i < m; i++) {
          arrayrex[ez][ey][ex] +=
              arrayp[i][ez][ey][ex] /
                  ((arrayU[ez][ey][ex] - arrayx[ez][ey][ex]) *
                   (arrayU[ez][ey][ex] - arrayx[ez][ey][ex])) +
              arrayq[i][ez][ey][ex] /
                  ((arrayx[ez][ey][ex] - arrayL[ez][ey][ex]) *
                   (arrayx[ez][ey][ex] - arrayL[ez][ey][ex]));
          arrayrex[ez][ey][ex] += arrayeta[ez][ey][ex] - arrayxsi[ez][ey][ex];
          arraygvec[i][ez][ey][ex] =
              arrayp[i][ez][ey][ex] /
                  (arrayU[ez][ey][ex] - arrayx[ez][ey][ex]) +
              arrayq[i][ez][ey][ex] / (arrayx[ez][ey][ex] - arrayL[ez][ey][ex]);
        }
      }
    }
  }
  for (i = 0; i < m; i++) {
    mmax->rey[i] =
        mmax->c[i] + mmax->d[i] * mmax->y[i] - mmax->mu[i] - mmax->lam[i];
    mmax->remu[i] = mmax->mu[i] * mmax->y[i] - epsi;
    mmax->res[i] = mmax->lam[i] * mmax->s[i] - epsi;
    PetscCall(DMDAVecRestoreArray(s_ctx->dm, mmax->gvec[i], &arraygvec[i]));
    PetscCall(VecSum(mmax->gvec[i], &mmax->relam[i]));
    mmax->relam[i] +=
        -mmax->a[i] * mmax->z - mmax->y[i] + mmax->s[i] - mmax->bval[i];
  }
  mmax->rez = mmax->a0 - mmax->zet;
  mmax->rezet = mmax->zet * mmax->z - epsi;
  for (i = 0; i < m; i++) {
    mmax->rez -= mmax->a[i] * mmax->lam[i];
  }

  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->alpha, &arrayalpha));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->beta, &arraybeta));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->xsi, &arrayxsi));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->eta, &arrayeta));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, x, &arrayx));
  PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->rex, &arrayrex));
  PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->rexsi, &arrayrexsi));
  PetscCall(DMDAVecGetArray(s_ctx->dm, mmax->reeta, &arrayreeta));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->mmaL, &arrayL));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->mmaU, &arrayU));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->p0, &arrayp0));
  PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->q0, &arrayq0));
  for (i = 0; i < m; ++i) {
    PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->p[i], &arrayp[i]));
    PetscCall(DMDAVecGetArrayRead(s_ctx->dm, mmax->q[i], &arrayq[i]));
  }

  PetscScalar maxrex, normrex, maxrexsi, normrexsi, maxreeta, normreeta;
  PetscCall(VecNorm(mmax->rex, NORM_INFINITY, &maxrex));
  PetscCall(VecNorm(mmax->rex, NORM_2, &normrex));
  PetscCall(VecNorm(mmax->rexsi, NORM_INFINITY, &maxrexsi));
  PetscCall(VecNorm(mmax->rexsi, NORM_2, &normrexsi));
  PetscCall(VecNorm(mmax->reeta, NORM_INFINITY, &maxreeta));
  PetscCall(VecNorm(mmax->reeta, NORM_2, &normreeta));
  *residunorm = normrex * normrex;
  *residunorm += normrexsi * normrexsi;
  *residunorm += normreeta * normreeta;
  *residumax = PetscMax(maxrex, maxrexsi);
  *residumax = PetscMax(*residumax, maxreeta);
  for (i = 0; i < m; ++i) {
    *residumax = PetscMax(*residumax, PetscAbsScalar(mmax->rey[i]));
    *residumax = PetscMax(*residumax, PetscAbsScalar(mmax->remu[i]));
    *residumax = PetscMax(*residumax, PetscAbsScalar(mmax->res[i]));
    *residumax = PetscMax(*residumax, PetscAbsScalar(mmax->relam[i]));
    *residunorm += mmax->rey[i] * mmax->rey[i];
    *residunorm += mmax->remu[i] * mmax->remu[i];
    *residunorm += mmax->res[i] * mmax->res[i];
    *residunorm += mmax->relam[i] * mmax->relam[i];
  }
  *residumax = PetscMax(*residumax, PetscAbsScalar(mmax->rez));
  *residumax = PetscMax(*residumax, PetscAbsScalar(mmax->rezet));
  *residunorm += mmax->rez * mmax->rez;
  *residunorm += mmax->rezet * mmax->rezet;
  *residunorm = PetscSqrtScalar(*residunorm);

  PetscFunctionReturn(0);
}

PetscErrorCode computeDelta(PCCtx *s_ctx, MMAx *mmax, Vec x, PetscScalar epsi) {
  PetscFunctionBeginUser;
  PetscScalar mat11, mat12, mat21, mat22, rhs1, rhs2;
  PetscInt i;

  PetscFunctionReturn(0);
}
PetscErrorCode findStep(PCCtx *s_ctx, MMAx *mmax, Vec x) {}