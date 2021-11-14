#include <stdio.h>

/*
Programa en Cuda que realiza la Multiplicación de Matrices MxN (4x4),
 por bloques y memoria global
*/

const int TILE_WIDTH = 2;
const int dim = 4;

__global__ void multiplicarMatriz(int (*M_d)[dim], int (*N_d)[dim],
                                  int (*R_d)[dim])
{
    // Calcula el índice de la fila del elemento valor de M
    int fila = blockIdx.y * TILE_WIDTH + threadIdx.y;
    // Calcula el índice de la fila del elemento valor de N
    int columna = blockIdx.x * TILE_WIDTH + threadIdx.x;
    
    int valor = 0;
    
    // Cada hilo computa un elemento del bloque de la submatriz
    for (int k = 0; k < dim; k++)
    {
        valor += M_d[fila][k] * N_d[k][columna];
    }
    
    R_d[fila][columna] = valor;
}

int main(int argc, char **argv)
{
    // Declarar todas las variables
    const int longitud = dim * dim * sizeof(int);
    int M_h[dim][dim] = 
    {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16}
    };
    int N_h[dim][dim] = 
    {
        {4, 3, 2, 1},
        {8, 7, 6, 5},
        {12, 11, 10, 9},
        {16, 15, 14, 13}
    };
    int R_h[dim][dim] = {};
    int (*M_d)[dim];
    int (*N_d)[dim];
    int (*R_d)[dim];
    
    // Asignar memoria en el dispositivo
    cudaMalloc((void **) &M_d, longitud);
    cudaMalloc((void **) &N_d, longitud);
    cudaMalloc((void **) &R_d, longitud);
    
    // Transferir datos al dispositivo
    cudaMemcpy(M_d, M_h, longitud, cudaMemcpyHostToDevice);
    cudaMemcpy(N_d, N_h, longitud, cudaMemcpyHostToDevice);
    
    // Ejecutar kernel en el dispositivo, dos bloques con ochos hilos cada uno
    dim3 bloques(dim / TILE_WIDTH, dim / TILE_WIDTH);
    dim3 hilos(TILE_WIDTH, TILE_WIDTH);
    multiplicarMatriz<<<bloques, hilos>>>(M_d, N_d, R_d);
    
    // Transferir resultados al anfitrión
    cudaMemcpy(R_h, R_d, longitud, cudaMemcpyDeviceToHost);
    
    // Mostrar resultados
    printf("{\n"
           "  {1, 2, 3, 4},\n"
           "  {5, 6, 7, 8},\n"
           "  {9, 10, 11, 12},\n"
           "  {13, 14, 15, 16},\n"
           "}\n"
           "*\n"
           "{\n"
           "  {4, 3, 2, 1},\n"
           "  {8, 7, 6, 5},\n"
           "  {12, 11, 10, 9},\n"
           "  {16, 15, 14, 13},\n"
           "}\n"
           "=\n"
           "{\n"
           "  {%d, %d, %d, %d},\n"
           "  {%d, %d, %d, %d},\n"
           "  {%d, %d, %d, %d},\n"
           "  {%d, %d, %d, %d},\n"
           "}\n",
           R_h[0][0], R_h[0][1], R_h[0][2], R_h[0][3],
           R_h[1][0], R_h[1][1], R_h[1][2], R_h[1][3],
           R_h[2][0], R_h[2][1], R_h[2][2], R_h[2][3],
           R_h[3][0], R_h[3][1], R_h[3][2], R_h[3][3]);
    
    // Liberar memoria del dispositivo
    cudaFree(M_d);
    cudaFree(N_d);
    cudaFree(R_d);
    
    return 0;
}

