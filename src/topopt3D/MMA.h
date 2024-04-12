#include "optimization.h"
#include <PreMixFEM_3D.h>
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
#define rmin 1.1
#define xmin 0.0
#define xmax 1.0
#define epsimin 1e-7
PetscErrorCode mmaInit(PCCtx *s_ctx, MMAx *mma_text);
PetscErrorCode mmaFinal(MMAx *mma_text);
PetscErrorCode mmaLimit(PCCtx *s_ctx, MMAx *mmax, Vec x, Vec t,
                        PetscInt penal);
PetscErrorCode mmaSub(PCCtx *s_ctx, MMAx *mmax, Vec x, Vec t, Vec dc);
PetscErrorCode subSolv(PCCtx *s_ctx, MMAx *mmax, Vec x, Vec t);
PetscErrorCode computeResidual(PCCtx *s_ctx, MMAx *mmax, Vec x);