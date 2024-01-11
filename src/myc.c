#include <stdio.h>

void lk(double KE[8][8]) {
  double E = 1.0;
  double nu = 0.3;
  double k[8] = {1.0 / 2.0 - nu / 6.0,
                 1.0 / 8.0 + nu / 8.0,
                 -1.0 / 4.0 - nu / 12.0,
                 -1.0 / 8.0 + 3.0 * nu / 8.0,
                 -1.0 / 4.0 + nu / 12.0,
                 -1.0 / 8.0 - nu / 8.0,
                 nu / 6.0,
                 1.0 / 8.0 - 3.0 * nu / 8.0};

  double factor = E / (1.0 - nu * nu);

  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      KE[i][j] = factor * k[i] * k[j];
    }
  }
}

int main() {
  double KE[8][8];
  lk(KE);

  printf("Stiffness matrix KE:\n");
  for (int i = 0; i < 8; i++) {
    for (int j = 0; j < 8; j++) {
      printf("%f ", KE[i][j]);
    }
    printf("\n");
  }

  return 0;
}