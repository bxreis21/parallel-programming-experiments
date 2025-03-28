/***********************

Conway Game of Life

serial version

************************/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define NI 200      /* array sizes */
#define NJ 200
#define NSTEPS 500    /* number of time steps */

int main(int argc, char *argv[])
{
  int i, j, n, im, ip, jm, jp, ni, nj, nsum, isum;
  int **old, **new, **parcial;
  float x;

  double start_time = MPI_Wtime();  // Início do tempo

  int size, my_rank;
  int size_ni_local, ni_start_local;
  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  ni = NI + 2;  /* add 2 for left and right ghost cells */
  nj = NJ + 2;

  int parts_NI = NI/size;
  int rest_NI = NI%size;

  ni_start_local = 1;
  size_ni_local = my_rank < rest_NI ? parts_NI + 1 : parts_NI;

  /* allocate arrays */
  old = malloc(ni*sizeof(int*));
  new = malloc(ni*sizeof(int*));
  parcial = malloc(size_ni_local*sizeof(int*));

  for(i=0; i<ni; i++){
    old[i] = malloc(nj*sizeof(int));
    new[i] = malloc(nj*sizeof(int));
  }
  for(i=0; i<size_ni_local; i++){
      parcial[i] = malloc(nj*sizeof(int));
  }
  
  if (my_rank == 0) {
    int index_NI = size_ni_local + 1;

    for(int p = 1; p < size; p++){
      //enviando quantidade de passos no indice ni (linhas da matriz) para cada processo
      int parts_NI_temp = p < rest_NI ? parts_NI + 1 : parts_NI;
      MPI_Send(&parts_NI_temp, 1, MPI_INT, p, 0, MPI_COMM_WORLD);

      //enviando indice inicial ni (linha da matriz) de cada processo
      MPI_Send(&index_NI, 1, MPI_INT, p, 1, MPI_COMM_WORLD);

      //printf("enviado para o processo: %d, start: %d, ni_size: %d\n", p, index_NI, parts_NI_temp);
      //Atualizando proximo indice inicial ni
      index_NI += parts_NI_temp;
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
  }

  else{
      //recebendo quantidade de passos no indice ni (linhas da matriz) para cada processo
      MPI_Recv(&size_ni_local, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
      
      //recebendo indice inicial ni (linha da matriz) para cada processo
      MPI_Recv(&ni_start_local, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);

      //printf("recebido, processo: %d, start: %d, ni_size: %d\n", my_rank, ni_start_local, size_ni_local);
  }
    
  /*  time steps */
  for(n=0; n<NSTEPS; n++){

    if(my_rank == 0) {
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
    }

    //enviando old atualizado para todos os processos
    for(int i = 0; i < ni; i++){
      MPI_Bcast(old[i], nj, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // printf("\nrecebendo primeira linha OLD, processo %d\n", my_rank);
    // for(j = 0; j < 10; j++){
    //   printf("%d ", old[0][j]);
    // }
    // printf("\n");

    //printf("processo %d: i:%d até: %d\n", my_rank, ni_start_local, ni_start_local + size_ni_local);

    // cada processo ira pegar linhas diferentes da matriz para trabalhar
    for(i=ni_start_local; i < ni_start_local + size_ni_local; i++){
      for(j=1; j<=NJ; j++){
        im = i-1;
        ip = i+1;
        jm = j-1;
        jp = j+1;

        nsum =  old[im][jp] + old[i][jp] + old[ip][jp]
              + old[im][j ]              + old[ip][j ] 
              + old[im][jm] + old[i][jm] + old[ip][jm];
        

        //printf("processo:%d, i: %d, j:%d, nsum:%d \n", my_rank, i, j, nsum);
        
        //preechendo submatriz parcial com os resultados parciais de cada processo
        if(my_rank != 0){
          switch(nsum){
            case 3:
              parcial[i - ni_start_local][j] = 1;
              break;

            case 2:
              parcial[i - ni_start_local][j] = old[i][j];
              break;

            default:
              parcial[i - ni_start_local][j] = 0;
          }
          //printf("processo:%d, i: %d, j:%d, parcial:%d \n", my_rank, i, j, parcial[i - ni_start_local][j]);
        }

        //processo 0 já atualiza diretamente na matriz new
        else {
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
          //printf("processo:%d, i: %d, j:%d, new:%d \n", my_rank, i, j, new[i - ni_start_local][j]);
        }
        
      }
    }

    //printf("-----------------------------------------------------------------------");
    // cada processo irá enviar a submatriz parcial que trabalhou para o processo master
    if(my_rank != 0){
      MPI_Send(&size_ni_local, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
      //printf("\nenviando size: %d, processo %d\n", size_ni_local, my_rank);
      for(int i = 0; i < size_ni_local; i++){
        MPI_Send(parcial[i], nj, MPI_INT, 0, 3, MPI_COMM_WORLD);

          // printf("\nenviando primeira linha, processo %d\n", my_rank);
          // for(j = 0; j < nj; j++){
          //   printf("%d ", parcial[i][j]);
          // }
          // printf("\n");
      }
    }


    else {
      //printf("-----------------------------------------------------------------------");
      // processo master recebe os resultados parciais de cada processo que é responsável
      // por trabalhar em uma região da old, e alocar os resultados na posição certa na
      // matriz new

      int start = size_ni_local + 1;

      for(int p = 1; p < size; p++){
        int size_ni_parcial = 0;
        MPI_Recv(&size_ni_parcial, 1, MPI_INT, p, 2, MPI_COMM_WORLD, &status);
        //printf("\nrecebendo size: %d, processo %d\n", size_ni_parcial, p);

        // printf("limite new processo %d: i:%d até: %d\n", p, start, start + size_ni_parcial);
        for(int i = start; i < start + size_ni_parcial; i++){
          MPI_Recv(new[i], nj, MPI_INT, p, 3, MPI_COMM_WORLD, &status);
          
          // printf("\nrecebendo linha %d, processo %d\n", i, p);
          // for(j = 0; j < nj; j++){
          //   printf("%d ", new[i][j]);
          // }
          // printf("\n");
        }
        
        start += size_ni_parcial;
      }
      
      /* copy new state into old state */
      for(i=1; i<=NI; i++){
        //printf("\nlinha: %d\n", i);
        for(j=1; j<=NJ; j++){
          old[i][j] = new[i][j];
          //printf("%d ", new[i][j]);
        }
      }
    }
  }

  /*  Iterations are done; sum the number of live cells */
  isum = 0;

  for(i=1; i<=NI; i++){
    for(j=1; j<=NJ; j++){
      isum = isum + new[i][j];
    }
  }

  if(my_rank==0){
    printf("\nNumber of live cells = %d\n", isum);
    double end_time = MPI_Wtime();  // Fim do tempo
    printf("Tempo de execução: %f segundos\n", end_time - start_time);
    }

  free(old);
  free(new);
  free(parcial);

  

  MPI_Finalize();

  return 0;
}
