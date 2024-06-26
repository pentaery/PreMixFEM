#include "PreMixFEM_3D.h"
#include <petscdmda.h>
#include <petscmath.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <slepceps.h>

#define MAX_ARGS 24
#define CELL_LEN 8

PetscErrorCode create_cross_kappa(PCCtx *s_ctx, PetscInt cr) {
  PetscFunctionBeginUser;
  PetscInt startx, nx, ex, ex_r, starty, ny, ey, ey_r, startz, nz, ez, ez_r, i;
  PetscCall(
      DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));
  PetscScalar ***arr_kappa_array[DIM];
  for (i = 0; i < DIM; ++i) {
    PetscCall(DMCreateGlobalVector(s_ctx->dm, &s_ctx->kappa[i]));
    PetscCall(DMDAVecGetArray(s_ctx->dm, s_ctx->kappa[i], &arr_kappa_array[i]));
  }
  for (ez = startz; ez < startz + nz; ++ez) {
    ez_r = ez % CELL_LEN;
    for (ey = starty; ey < starty + ny; ++ey) {
      ey_r = ey % CELL_LEN;
      for (ex = startx; ex < startx + nx; ++ex) {
        ex_r = ex % CELL_LEN;
        if ((ex_r >= 3 && ex_r < 5 && ey_r >= 3 && ey_r < 5) ||
            (ex_r >= 3 && ex_r < 5 && ez_r >= 3 && ez_r < 5) ||
            (ey_r >= 3 && ey_r < 5 && ez_r >= 3 && ez_r < 5)) {
          arr_kappa_array[0][ez][ey][ex] = PetscPowInt(10.0, cr);
          arr_kappa_array[1][ez][ey][ex] = PetscPowInt(10.0, cr);
          arr_kappa_array[2][ez][ey][ex] = PetscPowInt(10.0, cr);
        } else {
          arr_kappa_array[0][ez][ey][ex] = 1.0;
          arr_kappa_array[1][ez][ey][ex] = 1.0;
          arr_kappa_array[2][ez][ey][ex] = 1.0;
        }
      }
    }
  }
  for (i = 0; i < DIM; ++i)
    PetscCall(
        DMDAVecRestoreArray(s_ctx->dm, s_ctx->kappa[i], &arr_kappa_array[i]));

  PetscFunctionReturn(0);
}
//创建一个高对比度的kappa项

PetscErrorCode create_well_source_XxY_rhs(PCCtx *s_ctx, Vec *rhs) {
  PetscFunctionBeginUser;
  PetscCall(DMCreateGlobalVector(s_ctx->dm, rhs));
  PetscInt ex, ey, ez, startx, starty, startz, nx, ny, nz;
  PetscScalar ***arr_source_3d,
      meas_elem = s_ctx->H_x * s_ctx->H_y * s_ctx->H_z;
  PetscCall(
      DMDAGetCorners(s_ctx->dm, &startx, &starty, &startz, &nx, &ny, &nz));
  PetscCall(DMDAVecGetArray(s_ctx->dm, *rhs, &arr_source_3d));
  for (ez = startz; ez < startz + nz; ++ez)
    for (ey = starty; ey < starty + ny; ++ey)
      for (ex = startx; ex < startx + nx; ++ex) {
        /* Change the source term values here. */
        if ((ex == 0 || ex == s_ctx->M - 1) && (ey == 0 || ey == s_ctx->N - 1))
          arr_source_3d[ez][ey][ex] = 1.0e+3 * meas_elem;
        else if ((ex == s_ctx->M / 2 - 1 || ex == s_ctx->M / 2) &&
                 (ey == s_ctx->N / 2 - 1 || ey == s_ctx->N / 2))
          arr_source_3d[ez][ey][ex] = -1.0e+3 * meas_elem;
        else
          arr_source_3d[ez][ey][ex] = 0.0;
      }
  PetscCall(DMDAVecRestoreArray(s_ctx->dm, *rhs, &arr_source_3d));
  PetscFunctionReturn(0);
}
//创建右端项

int main(int argc, char **argv) {
  PetscCall(SlepcInitialize(&argc, &argv, (char *)0,
                            "This is a code for strong/weak scalability tests "
                            "with a homogeneous Neumann BC!\n"));
  PetscInt mesh[3] = {8, 8, 8}, cr = 0, i;
  //网格数，对比度初始化
  PetscBool is_petsc_default = PETSC_FALSE;
  PetscScalar dom[3] = {1.0, 1.0, 1.0}, norm_rhs;
  //总长度初始化

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-size", &mesh[0], NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-size", &mesh[1], NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-size", &mesh[2], NULL));
  PetscCall(
      PetscOptionsHasName(NULL, NULL, "-petsc_default", &is_petsc_default));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-cr", &cr, NULL));

  PCCtx s_ctx;
  PetscCall(PC_init(&s_ctx, &dom[0], &mesh[0]));
  // DMDA初始化
  PetscCall(PC_print_info(&s_ctx));
  // 打印一些关键信息
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Use contrast config=%d.\n", cr));
  // 打印对比度
  PetscLogDouble main_stage[2] = {0.0, 0.0}, time_tmp;
  PetscCall(PetscTime(&time_tmp));

  // Build the system.
  Vec rhs, u, r;
  Mat A;
  PetscCall(create_cross_kappa(&s_ctx, cr));
  //创建kappa，保存在s_ctx中的kappa里
  PetscCall(PC_create_A(&s_ctx, &A));
  //**创建A**
  PetscCall(create_well_source_XxY_rhs(&s_ctx, &rhs));
  //创建右端项
  PetscCall(VecNormalize(rhs, &norm_rhs));

  PetscCall(PetscTimeSubtract(&time_tmp));
  main_stage[0] -= time_tmp;

  // Solve the system.
  PetscCall(PetscTime(&time_tmp));
  // PetscCall(_PC_setup(&s_ctx));
  KSP ksp;
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));

  if (!is_petsc_default) {
    PC pc;
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCSHELL));
    //自定义的preconditiner
    PetscCall(PCShellSetContext(pc, &s_ctx));
    PetscCall(PCShellSetSetUp(pc, PC_setup));
    PetscCall(PCShellSetApply(pc, PC_apply_vec));
    PetscCall(
        PCShellSetName(pc, "3levels-MG-via-GMsFEM-with-velocity-elimination"));
  }
  //设定preconditioner

  PetscCall(KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
  PetscCall(KSPSetFromOptions(ksp));
  // Default is KSP_NORM_PRECONDITIONED, which is not the real residual norm.
  // Why?
  PetscCall(KSPSetUp(ksp));
  PetscCall(VecDuplicate(rhs, &u));
  PetscCall(KSPSolve(ksp, rhs, u));

  PetscCall(PetscTimeSubtract(&time_tmp));
  main_stage[1] -= time_tmp;

  PetscInt iter_count;
  PetscScalar residual;
  PetscCall(VecDuplicate(u, &r));
  PetscCall(MatMult(A, u, r));
  PetscCall(VecAXPY(r, -1.0, rhs));
  //求误差
  PetscCall(VecNorm(r, NORM_2, &residual));
  PetscCall(VecNorm(rhs, NORM_2, &norm_rhs));
  PetscCall(KSPGetIterationNumber(ksp, &iter_count));
  PetscCall(
      PetscPrintf(PETSC_COMM_WORLD, "The iteration number=%d.\n", iter_count));
  PetscCall(PetscPrintf(
      PETSC_COMM_WORLD,
      "The absolute residual in L2-norm=%.5f, the rhs in L2-norm=%.5f.\n",
      residual, norm_rhs));
  PetscCall(KSPConvergedReasonView(ksp, 0));

  if (!is_petsc_default)
    PetscCall(PC_print_stat(&s_ctx));

  PetscLogDouble main_stage_range[2][2];
  PetscCall(PC_get_range(&main_stage[0], &main_stage_range[0][0], MPI_DOUBLE));
  PetscCall(PC_get_range(&main_stage[1], &main_stage_range[1][0], MPI_DOUBLE));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "Pre=[%.5f, %.5f], Sol=[%.5f, %.5f].\n",
                        main_stage_range[0][0], main_stage_range[0][1],
                        main_stage_range[1][0], main_stage_range[1][1]));

  // ----------------
  // Cleaning.
  // ----------------
  PetscCall(VecDestroy(&r));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&rhs));
  for (i = 0; i < DIM; ++i)
    PetscCall(VecDestroy(&s_ctx.kappa[i]));
  PetscCall(MatDestroy(&A));
  if (!is_petsc_default) {
    PetscCall(PC_final(&s_ctx));
  } else {
    PetscCall(PC_final_default(&s_ctx));
  }
  PetscCall(SlepcFinalize());
  return 0;
}
