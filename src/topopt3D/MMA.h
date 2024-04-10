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
PetscErrorCode mmaInit(PCCtx *s_ctx, MMAx *mma_text);
PetscErrorCode mmaFinal(MMAx *mma_text);
PetscErrorCode mmaSub(PCCtx *s_ctx, MMAx *mmax, Vec x, Vec t, Vec dc,
                      PetscInt penal);
