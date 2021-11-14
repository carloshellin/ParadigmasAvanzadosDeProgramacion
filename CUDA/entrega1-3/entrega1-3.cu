#include <stdio.h>

// Programa en Cuda con memoria dinámica que realiza la suma de dos matrices de 4x4

__global__ void sumaMatrices(int *matrixA_d, int *matrixB_d, int *matrixR_d)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    matrixR_d[i] = matrixA_d[i] + matrixB_d[i];
}

int main(int argc, char **argv)
{
    // Declarar todas las variables
    const int longitud = 4 * 4 * sizeof(int);
    int *matrixA_h, *matrixB_h, *matrixR_h;
    int *matrixA_d, *matrixB_d, *matrixR_d;
    
    // Asignar memoria en el anfitrión
    matrixA_h = (int *) malloc(longitud);
    matrixB_h = (int *) malloc(longitud);
    matrixR_h = (int *) malloc(longitud);
    
    // Asignar memoria en el dispositivo
    cudaMalloc((void **) &matrixA_d, longitud);
    cudaMalloc((void **) &matrixB_d, longitud);
    cudaMalloc((void **) &matrixR_d, longitud);
    
    // Escribir en la memoria del anfitrión
    for (int i = 0; i < longitud; i++)
    {
        matrixA_h[i] = matrixB_h[i] = i + 1;
    }
    
    // Transferir datos al dispositivo
    cudaMemcpy(matrixA_d, matrixA_h, longitud, cudaMemcpyHostToDevice);
    cudaMemcpy(matrixB_d, matrixB_h, longitud, cudaMemcpyHostToDevice);
    
    // Ejecutar kernel en el dispositivo, dos bloques con ochos hilos cada uno
    sumaMatrices<<<2, 8>>>(matrixA_d, matrixB_d, matrixR_d);
    
    // Transferir resultados al anfitrión
    cudaMemcpy(matrixR_h, matrixR_d, longitud, cudaMemcpyDeviceToHost);
    
    // Mostrar resultados
    printf("{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16} + "
           "{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16} = "
           "{%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d}\n",
           matrixR_h[0], matrixR_h[1], matrixR_h[2], matrixR_h[3], matrixR_h[4],
           matrixR_h[5], matrixR_h[6], matrixR_h[7], matrixR_h[8], matrixR_h[9],
           matrixR_h[10], matrixR_h[11], matrixR_h[12], matrixR_h[13], matrixR_h[14],
           matrixR_h[15]);
    
    // Liberar memoria del anfitrión
    free(matrixA_h);
    free(matrixB_h);
    free(matrixR_h);
    
    // Liberar memoria del dispositivo
    cudaFree(matrixA_d);
    cudaFree(matrixB_d);
    cudaFree(matrixR_d);
    
    return 0;
}

