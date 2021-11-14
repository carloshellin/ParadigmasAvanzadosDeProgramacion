#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/*
Programa en Cuda que realiza la convolución de matrices en GPU mediante
múltiples bloques y memoria global
*/

// Tamaño de la matriz A
const int dimA = 32;

// Tamaño de la matriz B
const int dimB = 3;

const int TILE_WIDTH = 16;

__global__ void girar(int (*matrizB_d)[dimB], int (*matrizBR_d)[dimB])
{
    // Kernel para girar la matriz B de 3x3 en 180º
    
    unsigned int posicionX = threadIdx.x;
    unsigned int posicionY = threadIdx.y;
    
    switch (posicionX)
    {
        case 0:
        {
            // En el caso de estar en el hilo 0, se busca la posición 2 de x
            posicionX += 2;
            break;
        }
        
        case 2:
        {
            // En el caso de estar en el hilo 2, se busca la posición 0 de x
            posicionX -= 2;
            break;
        }
    }
    
    switch (posicionY)
    {
        case 0:
        {
            // En el caso de estar en el hilo 0, se busca la posición 2 de y
            posicionY += 2;
            break;
        }
        
        case 2:
        {
            // En el caso de estar en el hilo 2, se busca la posición 0 de y
            posicionY -= 2;
            break;
        }
    }
    
    // Se realiza el giro con posiciones e hilos
    matrizBR_d[posicionY][posicionX] = matrizB_d[threadIdx.y][threadIdx.x];
}

__global__ void convolucion(int (*matrizA_d)[dimA], int (*matrizB_d)[dimB],
                            int (*matrizR_d)[dimA])
{
    int fila = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int columna = blockIdx.x * TILE_WIDTH + threadIdx.x;
    
    int izquierda = columna - 1;
    int derecha = columna + 1;
    int arriba = fila - 1;
    int abajo = fila + 1;
    
    int valor1 = 0;
    int valor2 = 0;
    int resultado = 0;
    
    if (arriba >= 0)
    {
        // Se multiplica los valores de arriba
        valor1 = matrizA_d[arriba][columna];
        valor2 = matrizB_d[0][1];
        
        resultado += valor1 * valor2;
        
        if (izquierda >= 0)
        {
            // Se multiplica los valores de arriba e izquierda
            valor1 = matrizA_d[arriba][izquierda];
            valor2 = matrizB_d[0][0];
            
            resultado += valor1 * valor2;
        }
        
        if (derecha < dimA)
        {
            // Se multiplica los valores de arriba y izquierda
            valor1 = matrizA_d[arriba][derecha];
            valor2 = matrizB_d[0][2];
            
            resultado += valor1 * valor2;
        }
    }
    
    if (izquierda >= 0)
    {
        // Se multiplica los valores de izquierda
        valor1 = matrizA_d[fila][izquierda];
        valor2 = matrizB_d[1][0];
        
        resultado += valor1 * valor2;
    }
    
    
    // Se multiplica los valores del centro
    valor1 = matrizA_d[fila][columna];
    valor2 = matrizB_d[1][1];
    
    resultado += valor1 * valor2;
    
    if (derecha < dimA)
    {
        // Se multiplica los valores de derecha
        valor1 = matrizA_d[fila][derecha];
        valor2 = matrizB_d[1][2];
        
        resultado += valor1 * valor2;
    }
    
    if (abajo < dimA)
    {
        // Se multiplica los valores de abajo
        valor1 = matrizA_d[abajo][columna];
        valor2 = matrizB_d[2][1];
        
        resultado += valor1 * valor2;
        
        if (izquierda >= 0)
        {
            // Se multiplica los valores de abajo e izquierda
            valor1 = matrizA_d[abajo][izquierda];
            valor2 = matrizB_d[2][0];
            
            resultado += valor1 * valor2;
        }
        
        if (derecha < dimA)
        {
            // Se multiplica los valores de abajo y derecha
            valor1 = matrizA_d[abajo][derecha];
            valor2 = matrizB_d[2][2];
            
            resultado += valor1 * valor2;
        }
    }
    
    // El resultado se almacena en la matriz resultante usando fila y columna
    matrizR_d[fila][columna] = resultado;
}

void pintarMatriz(int (*matriz_h)[dimA])
{
    for (int fila = 0; fila < dimA; fila++)
    {
        printf("  {%d", matriz_h[fila][0]);
        for (int columna = 1; columna < dimA; columna++)
        {
            printf(", %d", matriz_h[fila][columna]);
        }
        
        if (fila == (dimA - 1))
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
    const int longitudA = dimA * dimA * sizeof(int);
    const int longitudB = dimB * dimB * sizeof(int);
    int matrizA_h[dimA][dimA] = {};
    int matrizB_h[dimB][dimB] = 
    {
        {0, 1, 0},
        {1, 0, 1},
        {0, 1, 0}
    };
    
    // Matriz B de prueba para demostrar el giro de 180º
    /*int matrizB_h[dimB][dimB] = 
    {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };*/
    int matrizR_h[dimA][dimA] = {};
    int matrizBR_h[dimB][dimB] = {};
    
    int (*matrizA_d)[dimA];
    int (*matrizB_d)[dimB];
    int (*matrizR_d)[dimA];
    int (*matrizBR_d)[dimB];
    
    // Semilla para mejorar los números pseudoaleatorios
    srand((unsigned int) time(NULL));
    
    // Escribir en la memoria del anfitrión
    for (int fila = 0; fila < dimA; fila++)
    {
        for (int columna = 0; columna < dimA; columna++)
        {
            int R = rand() % 255;
            int G = rand() % 255;
            int B = rand() % 255;
            matrizA_h[fila][columna] = R + G + B;
        }
    }
    
    // Asignar memoria en el dispositivo
    cudaMalloc((void **) &matrizA_d, longitudA);
    cudaMalloc((void **) &matrizB_d, longitudB);
    cudaMalloc((void **) &matrizR_d, longitudA);
    cudaMalloc((void **) &matrizBR_d, longitudB);
    
    // Transferir datos al dispositivo
    cudaMemcpy(matrizA_d, matrizA_h, longitudA, cudaMemcpyHostToDevice);
    cudaMemcpy(matrizB_d, matrizB_h, longitudB, cudaMemcpyHostToDevice);
    
    // Ejecutar kernel de girar en el dispositivo, un bloque con nueve hilos
    dim3 bloquesB(1, 1);
    dim3 hilosB(dimB, dimB);
    girar<<<bloquesB, hilosB>>>(matrizB_d, matrizBR_d);
    
    // Transferir el resultado de girar la matriz B al anfitrión
    cudaMemcpy(matrizBR_h, matrizBR_d, longitudB, cudaMemcpyDeviceToHost);
    
    // Ejecutar kernel de convolución en el dispositivo, multiples bloques con 
    // TILE_WIDTH * TILEWIDTH hilos
    dim3 bloques(dimA / TILE_WIDTH, dimA / TILE_WIDTH);
    dim3 hilos(TILE_WIDTH , TILE_WIDTH);
    convolucion<<<bloques, hilos>>>(matrizA_d, matrizBR_d, matrizR_d);
    
    // Transferir resultados al anfitrión
    cudaMemcpy(matrizR_h, matrizR_d, longitudA, cudaMemcpyDeviceToHost);
    
    // Mostrar resultados
    printf("Matriz A\n"
           "{\n");
    pintarMatriz(matrizA_h);
    
    printf("\nMatriz B\n");
    printf("{\n"
           "  {%d, %d, %d},\n"
           "  {%d, %d, %d},\n"
           "  {%d, %d, %d},\n"
           "}\n",
           matrizBR_h[0][0], matrizBR_h[0][1], matrizBR_h[0][2],
           matrizBR_h[1][0], matrizBR_h[1][1], matrizBR_h[1][2],
           matrizBR_h[2][0], matrizBR_h[2][1], matrizBR_h[2][2]);
    
    printf("\nMatriz resultante\n"
           "{\n");
    pintarMatriz(matrizR_h);
    
    // Liberar memoria del dispositivo
    cudaFree(matrizA_d);
    cudaFree(matrizB_d);
    cudaFree(matrizR_d);
    cudaFree(matrizBR_d);
    
    return 0;
}

