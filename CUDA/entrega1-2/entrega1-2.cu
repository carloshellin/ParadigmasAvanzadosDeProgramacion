#include <stdio.h>

/*
Programa en Cuda que permite la suma de un entero a cada uno de los
 8 elementos de un vector
*/

__global__ void sumaEntero(int entero_h, int *resultado_d)
{
    int i = threadIdx.x;
    resultado_d[i] += entero_h;
}

int main(int argc, char **argv)
{
    // Declarar todas las variables
    int entero_h = 1;
    const int longitud = 8;
    int vector_h[longitud] = {1, 2, 3, 4, 5, 6, 7, 8};
    int *resultado_d;
    
    // Asignar memoria en el dispositivo
    cudaMalloc((void **) &resultado_d, sizeof(vector_h));
    
    // Transferir datos al dispositivo
    cudaMemcpy(resultado_d, vector_h, sizeof(vector_h), cudaMemcpyHostToDevice);
    
    // Ejecutar kernel en el dispositivo
    sumaEntero<<<1, longitud>>>(entero_h, resultado_d);
    
    // Transferir resultados al anfitrión
    cudaMemcpy(vector_h, resultado_d, sizeof(vector_h), cudaMemcpyDeviceToHost);
    
    // Mostrar resultados
    printf("{1, 2, 3, 4, 5, 6, 7, 8} + 1 = {%d, %d, %d, %d, %d, %d, %d, %d}\n", 
           vector_h[0], vector_h[1], vector_h[2], vector_h[3], vector_h[4], 
           vector_h[5], vector_h[6], vector_h[7]);
    
    // Liberar memoria del dispositivo
    cudaFree(resultado_d);
    
    return 0;
}

