/***********************

Conway Game of Life

serial version

************************/

#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

#define NI 200        /* array sizes */

#define NJ 200

#define NSTEPS 500    /* number of time steps */

int main(int argc, char *argv[])
{
  int i, j, n, im, ip, jm, jp, ni, nj, nsum, isum;
  int **old, **new;  
  float x;

  double start_time = omp_get_wtime();  // Início do tempo
  int threads = atoi(argv[1]);

  /* allocate arrays */

  ni = NI + 2;  /* add 2 for left and right ghost cells */
  nj = NJ + 2;
  old = malloc(ni*sizeof(int*));
  new = malloc(ni*sizeof(int*));

  for(i=0; i<ni; i++){
    old[i] = malloc(nj*sizeof(int));
    new[i] = malloc(nj*sizeof(int));
  }

  /* initialize elements of old to 0 or 1 */
  for(i=1; i<=NI; i++){
    for(j=1; j<=NJ; j++){
      x = rand()/((float)RAND_MAX + 1);
      if(x<0.5){
        old[i][j] = 0;
      } else {
        old[i][j] = 1;
      }
    }
  }

  /*  time steps */
  for(n=0; n<NSTEPS; n++){

    /* corner boundary conditions */
    old[0][0] = old[NI][NJ];
    old[0][NJ+1] = old[NI][1];
    old[NI+1][NJ+1] = old[1][1];
    old[NI+1][0] = old[1][NJ];

    /* left-right boundary conditions */

    for(i=1; i<=NI; i++){
      old[i][0] = old[i][NJ];
      old[i][NJ+1] = old[i][1];
    }

    /* top-bottom boundary conditions */
    for(j=1; j<=NJ; j++){
      old[0][j] = old[NI][j];
      old[NI+1][j] = old[1][j];
    }

    #pragma omp parallel num_threads(threads) private(i, j, im, ip, jm, jp, nsum) shared(old, new)
    {
      #pragma omp for
      for(i=1; i<=NI; i++){
        for(j=1; j<=NJ; j++){
          im = i-1;
          ip = i+1;
          jm = j-1;
          jp = j+1;

          nsum =  old[im][jp] + old[i][jp] + old[ip][jp]
                + old[im][j ]              + old[ip][j ] 
                + old[im][jm] + old[i][jm] + old[ip][jm];

          switch(nsum){

          case 3:
            new[i][j] = 1;
            break;

          case 2:
            new[i][j] = old[i][j];
            break;

          default:
            new[i][j] = 0;
          }
        }
    }

    /* copy new state into old state */
    #pragma omp parallel for private(i, j)
    for(i=1; i<=NI; i++){
      for(j=1; j<=NJ; j++){
        old[i][j] = new[i][j];
      }
    }
    }
  }

  /*  Iterations are done; sum the number of live cells */
  isum = 0;

  #pragma omp parallel for private(i, j) shared(new) reduction(+:isum)
  for(i=1; i<=NI; i++){
    //printf("\nlinha: %d\n", i);
    for(j=1; j<=NJ; j++){
      isum = isum + new[i][j];
      //printf("%d ", new[i][j]);
    }
  }

  printf("\nNumber of live cells = %d\n", isum);
  
  double end_time = omp_get_wtime();  // Fim do tempo
  printf("Tempo de execução: %f segundos\n", end_time - start_time);
  return 0;
}
