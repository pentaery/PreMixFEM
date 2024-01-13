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
static char help[] = "A test for TOOP.";

PetscErrorCode formKE(PetscReal KE[8][8], PetscReal coef);
PetscErrorCode formMatrix(DM dm, Mat A, Vec x);
PetscErrorCode formRHS(DM dm, Vec rhs, PetscInt N);
PetscErrorCode computeCost(DM dm, PetscReal *cost, Vec dc);

int main(int argc, char **args) {
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  DM dm;
  Mat A;
  Vec x;
  Vec dc;
  Vec b, u;
  KSP ksp;
  PetscInt M = 5, N = 4;
  PetscReal cost = 0;
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                         DMDA_STENCIL_BOX, M, N, PETSC_DECIDE, PETSC_DECIDE, 2,
                         1, NULL, NULL, &dm));
  PetscCall(DMSetUp(dm));

  PetscCall(DMCreateMatrix(dm, &A));

  PetscCall(DMCreateGlobalVector(dm, &x));
  PetscCall(DMCreateGlobalVector(dm, &b));
  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(DMCreateGlobalVector(dm, &dc));
  PetscCall(VecSet(x, 0.5));

  formMatrix(dm, A, x);
  formRHS(dm, b, N);
  computeCost(dm, &cost, dc);

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, b, u));

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&u));
  PetscCall(MatDestroy(&A));
  PetscCall(DMDestroy(&dm));
  PetscCall(KSPDestroy(&ksp));

  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode formMatrix(DM dm, Mat A, Vec x) {
  PetscFunctionBeginUser;
  // PetscInt sizea, sizeb;
  PetscReal value[8][8];
  PetscReal coef;
  PetscInt startx, starty, nx, ny, ex, ey;
  PetscReal ***array;
  MatStencil col[8], row[8];
  PetscCall(DMDAGetCorners(dm, &startx, &starty, NULL, &nx, &ny, NULL));
  for (ey = starty; ey < starty + ny - 1; ey++) {
    for (ex = startx; ex < startx + nx - 1; ex++) {
      col[0] = (MatStencil){.i = ex, .j = ey, .c = 0};
      col[1] = (MatStencil){.i = ex, .j = ey, .c = 1};
      col[2] = (MatStencil){.i = ex + 1, .j = ey, .c = 0};
      col[3] = (MatStencil){.i = ex + 1, .j = ey, .c = 1};
      col[4] = (MatStencil){.i = ex, .j = ey + 1, .c = 0};
      col[5] = (MatStencil){.i = ex, .j = ey + 1, .c = 1};
      col[6] = (MatStencil){.i = ex + 1, .j = ey + 1, .c = 0};
      col[7] = (MatStencil){.i = ex + 1, .j = ey + 1, .c = 1};
      row[0] = (MatStencil){.i = ex, .j = ey, .c = 0};
      row[1] = (MatStencil){.i = ex, .j = ey, .c = 1};
      row[2] = (MatStencil){.i = ex + 1, .j = ey, .c = 0};
      row[3] = (MatStencil){.i = ex + 1, .j = ey, .c = 1};
      row[4] = (MatStencil){.i = ex, .j = ey + 1, .c = 0};
      row[5] = (MatStencil){.i = ex, .j = ey + 1, .c = 1};
      row[6] = (MatStencil){.i = ex + 1, .j = ey + 1, .c = 0};
      row[7] = (MatStencil){.i = ex + 1, .j = ey + 1, .c = 1};
      PetscCall(DMDAVecGetArrayDOF(dm, x, &array));
      coef = array[ey][ex][0] * array[ey][ex][0] * array[ey][ex][0];
      PetscCall(DMDAVecRestoreArrayDOF(dm, x, &array));
      formKE(value, coef);
      PetscCall(
          MatSetValuesStencil(A, 8, col, 8, row, &value[0][0], ADD_VALUES));
    }
  }

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscInt M, N;
  PetscCall(MatGetOwnershipRange(A, &M, &N));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "m: %d, n: %d\n", M, N));
  PetscCall(PetscPrintf(PETSC_COMM_SELF,
                        "startx: %d, starty: %d, nx:%d, ny: %d\n", startx,
                        starty, nx, ny));
  PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));
  MatStencil row0;
  for (ey = starty; ey < starty + ny; ++ey) {
    for (ex = startx; ex < startx + nx; ++ex) {
      if (ex == 0) {
        row0 = (MatStencil){.i = ex, .j = ey, .c = 1};
        PetscCall(MatZeroRowsStencil(A, 1, &row0, 1, NULL, NULL));
      }
    }
  }
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

PetscErrorCode computeCost(DM dm, PetscReal *cost, Vec dc) {
  PetscFunctionBeginUser;
  PetscInt startx, starty, nx, ny, ex, ey;
  PetscReal value[8][8];
  formKE(value, 1);
  PetscCall(DMDAGetCorners(dm, &startx, &starty, NULL, &nx, &ny, NULL));
  for (ey = starty; ey < starty + ny - 1; ey++) {
    for (ex = startx; ex < startx + nx - 1; ex++) {
    }
  }
  PetscFunctionReturn(0);
}