#include <stdio.h>

// Programa en Cuda que permita calcular la transpuesta de una matriz

const int dim = 4;
__constant__ int matrizInicial_d[dim][dim] = 
{
    {1, 2, 3, 4},
    {5, 6, 7, 8},
    {9, 10, 11, 12},
    {13, 14, 15, 16}
};

__global__ void transpuestaMatriz(int (*matrizT_d)[dim])
{
    matrizT_d[threadIdx.x][threadIdx.y] = matrizInicial_d[threadIdx.y][threadIdx.x];
}

int main(int argc, char **argv)
{
    // Declarar todas las variables
    const int longitud = dim * dim * sizeof(int);
    int matrizT_h[dim][dim] = {};
    int (*matrizT_d)[dim];
    
    // Asignar memoria en el dispositivo
    cudaMalloc((void **) &matrizT_d, longitud);
    
    // Ejecutar kernel en el dispositivo, dos bloques con ochos hilos cada uno
    dim3 bloques(1, 1);
    dim3 hilos(dim, dim);
    transpuestaMatriz<<<bloques, hilos>>>(matrizT_d);
    
    // Transferir resultados al anfitrión
    cudaMemcpy(matrizT_h, matrizT_d, longitud, cudaMemcpyDeviceToHost);
    
    // Mostrar resultados
    printf("Matriz inicial:\n"
           "{\n"
           "  {1, 2, 3, 4},\n"
           "  {5, 6, 7, 8},\n"
           "  {9, 10, 11, 12},\n"
           "  {13, 14, 15, 16},\n"
           "}\n"
           "Matriz transpuesta:\n"
           "{\n"
           "  {%d, %d, %d, %d},\n"
           "  {%d, %d, %d, %d},\n"
           "  {%d, %d, %d, %d},\n"
           "  {%d, %d, %d, %d},\n"
           "}\n",
           matrizT_h[0][0], matrizT_h[0][1], matrizT_h[0][2], matrizT_h[0][3],
           matrizT_h[1][0], matrizT_h[1][1], matrizT_h[1][2], matrizT_h[1][3],
           matrizT_h[2][0], matrizT_h[2][1], matrizT_h[2][2], matrizT_h[2][3],
           matrizT_h[3][0], matrizT_h[3][1], matrizT_h[3][2], matrizT_h[3][3]);
    
    // Liberar memoria del dispositivo
    cudaFree(matrizT_d);
    
    return 0;
}

