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
PetscErrorCode mmaSub(PCCtx *s_ctx, MMAx *mmax, Vec x, Vec t, Vec dc,
                      PetscInt loop) {
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
  PetscCall(VecPointwiseMin(mmax->alpha, mmax->zzz, mmax->ubd));

  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->xlast, &xval));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->xllast, &xold1));
  PetscCall(DMDAVecRestoreArrayRead(s_ctx->dm, mmax->xlllast, &xold2));
  PetscFunctionReturn(0);
}