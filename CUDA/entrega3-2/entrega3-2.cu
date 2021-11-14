#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* 
Programa en Cuda, donde se aplican  los Patrones de computación paralela en matrices. 

(Gather (many to one), Scatter (one to many), Stencil (several to one))

*/

const int dim = 16;

__global__ void gather(int (*matrizA_d)[dim], int (*matrizR_d)[dim])
{
    // Un hilo calcula la media de 3 elementos vecinos en memoria
    // y el resultado lo escribe en un solo sitio
    
    int valor = 0;
    
    int izquierda = (threadIdx.x - 1) == UINT_MAX ? dim - 1 : threadIdx.x - 1;
    int derecha = (threadIdx.x + 1) == dim ? 0 : threadIdx.x + 1;
    
    valor += matrizA_d[threadIdx.y][izquierda];
    valor += matrizA_d[threadIdx.y][threadIdx.x];
    valor += matrizA_d[threadIdx.y][derecha];
    
    
    matrizR_d[threadIdx.y][threadIdx.x] = valor / 3;
}

__global__ void scatter(int (*matrizA_d)[dim], int (*matrizR_d)[dim])
{
    // Un hilo lee una localización en memoria, suma 1/3 al valor del elemento
    // y el resultado lo replica en 3 sitios vecinos
    
    int valor = matrizA_d[threadIdx.y][threadIdx.x];
    
    int izquierda = (threadIdx.x - 1) == UINT_MAX ? dim - 1 : threadIdx.x - 1;
    int derecha = (threadIdx.x + 1) == dim ? 0 : threadIdx.x + 1;
    
    matrizR_d[threadIdx.y][izquierda] = valor + valor/3;
    matrizR_d[threadIdx.y][threadIdx.x] = valor + valor/3;
    matrizR_d[threadIdx.y][derecha] = valor + valor/3;
}

__global__ void stencil(int (*matrizA_d)[dim], int (*matrizR_d)[dim])
{
    // Un hilo suma los valores que tiene a izquierda, derecha, arriba y abajo
    // y el resultado lo escribe en un solo sitio
    
    int izquierda = (threadIdx.x - 1) == UINT_MAX ? dim - 1 : threadIdx.x - 1;
    int derecha = (threadIdx.x + 1) == dim ? 0 : threadIdx.x + 1;
    int arriba = (threadIdx.y - 1) == UINT_MAX ? dim - 1 : threadIdx.y - 1;
    int abajo = (threadIdx.y + 1) == dim ? 0 : threadIdx.y + 1;
    
    int valor = 0;
    valor += matrizA_d[threadIdx.y][izquierda];
    valor += matrizA_d[threadIdx.y][derecha];
    valor += matrizA_d[arriba][threadIdx.x];
    valor += matrizA_d[abajo][threadIdx.x];
    
    matrizR_d[threadIdx.y][threadIdx.x] = valor;
}

void pintarMatriz(int (*matriz_h)[dim])
{
    for (int fila = 0; fila < dim; fila++)
    {
        printf("  {%d", matriz_h[fila][0]);
        for (int columna = 1; columna < dim; columna++)
        {
            printf(", %d", matriz_h[fila][columna]);
        }
        
        if (fila == (dim - 1))
        {
            printf("}\n");
        }
        else
        {
            printf("},\n");
        }
    }
    printf("}\n");
}

int main(int argc, char **argv)
{
    // Declarar todas las variables
    const int longitud = dim * dim * sizeof(int);
    int matrizA_h[dim][dim] = {};
    int matrizGatherR_h[dim][dim] = {};
    int matrizScatterR_h[dim][dim] = {};
    int matrizStencilR_h[dim][dim] = {};
    int (*matrizA_d)[dim];
    int (*matrizGatherR_d)[dim];
    int (*matrizScatterR_d)[dim];
    int (*matrizStencilR_d)[dim];
    
    // Semilla para mejorar los números pseudoaleatorios
    srand((unsigned int) time(NULL));
    
    // Asignar memoria en el dispositivo
    cudaMalloc((void **) &matrizA_d, longitud);
    cudaMalloc((void **) &matrizGatherR_d, longitud);
    cudaMalloc((void **) &matrizScatterR_d, longitud);
    cudaMalloc((void **) &matrizStencilR_d, longitud);
    
    // Escribir en la memoria del anfitrión
    for (int fila = 0; fila < dim; fila++)
    {
        for (int columna = 0; columna < dim; columna++)
        {
            matrizA_h[fila][columna] = rand() % 99 + 1;
        }
    }
    
    // Transferir datos al dispositivo
    cudaMemcpy(matrizA_d, matrizA_h, longitud, cudaMemcpyHostToDevice);
    
    // Ejecutar tres kernels en el dispositivo, un bloque con 256 hilos
    dim3 bloques(1, 1);
    dim3 hilos(dim, dim);
    gather<<<bloques, hilos>>>(matrizA_d, matrizGatherR_d);
    scatter<<<bloques, hilos>>>(matrizA_d, matrizScatterR_d);
    stencil<<<bloques, hilos>>>(matrizA_d, matrizStencilR_d);
    
    // Transferir resultados al anfitrión
    cudaMemcpy(matrizGatherR_h, matrizGatherR_d, longitud, cudaMemcpyDeviceToHost);
    cudaMemcpy(matrizScatterR_h, matrizScatterR_d, longitud, cudaMemcpyDeviceToHost);
    cudaMemcpy(matrizStencilR_h, matrizStencilR_d, longitud, cudaMemcpyDeviceToHost);
    
    // Mostrar resultados
    printf("Matriz origen A\n"
           "{\n");
    pintarMatriz(matrizA_h);
    
    printf("Gather: la media de 3 elementos vecinos y resultado a un solo sitio\n");
    pintarMatriz(matrizGatherR_h);
    
    printf("Scatter: suma 1/3 al valor de un elemento y se replica en 3 elementos " 
           "vecinos\n");
    pintarMatriz(matrizScatterR_h);
    
    printf("Stencil: suma los valores que tiene a izquierda, derecha, arriba "
           "y abajo, y resultado a un solo sitio\n");
    pintarMatriz(matrizStencilR_h);
    
    // Liberar memoria del dispositivo
    cudaFree(matrizA_d);
    cudaFree(matrizGatherR_d);
    cudaFree(matrizScatterR_d);
    cudaFree(matrizStencilR_d);
    
    return 0;
}

