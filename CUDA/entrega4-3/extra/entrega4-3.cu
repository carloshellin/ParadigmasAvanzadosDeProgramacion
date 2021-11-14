#include <stdio.h>
#include <stdlib.h>
#include <time.h>


/*
Programa en Cuda que realiza la convolución completa de matrices en GPU 
mediante múltiples bloques y memoria compartida
*/

// Tamaño de la matriz A
const int dimA = 32;

// Tamaño de la matriz B
const int dimB = 3;

// Tamaño de la matriz resultante
const int dimR = dimA + 2;

const int TILE_WIDTH = dimR / 2;

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
                            int (*matrizR_d)[dimR])
{
    __shared__ int matrizA_ds[TILE_WIDTH - 1][TILE_WIDTH - 1];
    __shared__ int matrizB_ds[dimB][dimB];
    
    int fila = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int columna = blockIdx.x * TILE_WIDTH + threadIdx.x;
    
    int posicionX = columna - 1;
    int posicionY = fila - 1;
    
    int posicionYs = posicionY - ((TILE_WIDTH - 1) * blockIdx.y);
    int posicionXs = posicionX - ((TILE_WIDTH - 1) * blockIdx.x);
    
    int izquierdaS = posicionXs - 1;
    int derechaS = posicionXs + 1;
    int arribaS = posicionYs - 1;
    int abajoS = posicionYs + 1;
    
    int izquierda = posicionX - 1;
    int derecha = posicionX + 1;
    int arriba = posicionY - 1;
    int abajo = posicionY + 1;
    
    int valor1 = 0;
    int valor2 = 0;
    int resultado = 0;
    
    int posicionXA = threadIdx.x - 1;
    int posicionYA = threadIdx.y - 1;
    if (posicionXA >= 0 && posicionYA >= 0 && posicionXA < dimA && posicionYA < dimA)
    {
        int columnaA = blockIdx.x * (TILE_WIDTH - 1) + posicionXA;
        int filaA = blockIdx.y * (TILE_WIDTH - 1) + posicionYA;
        
        matrizA_ds[posicionYA][posicionXA] = matrizA_d[filaA][columnaA];
        
        for (int i = 0; i < dimB; i++) 
        {
            for (int j = 0; j < dimB; j++)
            {
                matrizB_ds[i][j] = matrizB_d[i][j];
            }
        }
        
        __syncthreads();
    }
    
    if (posicionXs > 0 && posicionXs < TILE_WIDTH - 2 && posicionYs > 0 && posicionYs < TILE_WIDTH - 2)
    {
        resultado += matrizA_ds[arribaS][izquierdaS] * matrizB_ds[0][0];
        resultado += matrizA_ds[arribaS][posicionXs] * matrizB_ds[0][1];
        resultado += matrizA_ds[arribaS][derechaS] * matrizB_ds[0][2];
        resultado += matrizA_ds[posicionYs][izquierdaS] * matrizB_ds[1][0];
        resultado += matrizA_ds[posicionYs][posicionXs] * matrizB_ds[1][1];
        resultado += matrizA_ds[posicionYs][derechaS] * matrizB_ds[1][2];
        resultado += matrizA_ds[abajoS][izquierdaS] * matrizB_ds[2][0];
        resultado += matrizA_ds[abajoS][posicionXs] * matrizB_ds[2][1];
        resultado += matrizA_ds[abajoS][derechaS] * matrizB_ds[2][2];
        
        __syncthreads();
    }
    else
    {
        
        if (arriba >= 0 && izquierda >= 0)
        {
            valor1 = matrizA_d[arriba][izquierda];
            valor2 = matrizB_d[0][0];
            
            resultado += valor1 * valor2;
        }
        
        if (arriba >= 0 && posicionX >= 0 && posicionX < dimA)
        {
            valor1 = matrizA_d[arriba][posicionX];
            valor2 = matrizB_d[0][1];
            
            resultado += valor1 * valor2;
        }
        
        if (arriba >= 0 && derecha < dimA)
        {
            valor1 = matrizA_d[arriba][derecha];
            valor2 = matrizB_d[0][2];
            
            resultado += valor1 * valor2;
        }
        
        if (izquierda >= 0 && posicionY >= 0 && posicionY < dimA)
        {
            valor1 = matrizA_d[posicionY][izquierda];
            valor2 = matrizB_d[1][0];
            
            resultado += valor1 * valor2;
        }
        
        if (posicionX >= 0 && posicionY >= 0 && posicionX < dimA && posicionY < dimA)
        {
            valor1 = matrizA_d[posicionY][posicionX];
            valor2 = matrizB_d[1][1];
            
            resultado += valor1 * valor2;
        }
        
        if (derecha < dimA && posicionY >= 0 && posicionY < dimA)
        {
            valor1 = matrizA_d[posicionY][derecha];
            valor2 = matrizB_d[1][2];
            
            resultado += valor1 * valor2;
        }
        
        if (abajo < dimA && izquierda >= 0)
        {
            valor1 = matrizA_d[abajo][izquierda];
            valor2 = matrizB_d[2][0];
            
            resultado += valor1 * valor2;
        }
        
        if (abajo < dimA && posicionX >= 0 && posicionX < dimA)
        {
            valor1 = matrizA_d[abajo][posicionX];
            valor2 = matrizB_d[2][1];
            
            resultado += valor1 * valor2;
        }
        
        if (abajo < dimA && derecha < dimA)
        {
            valor1 = matrizA_d[abajo][derecha];
            valor2 = matrizB_d[2][2];
            
            resultado += valor1 * valor2;
        }
        
    }
    
    
    matrizR_d[fila][columna] = resultado;
}

void pintarMatrizA(int (*matriz_h)[dimA])
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

void pintarMatrizR(int (*matriz_h)[dimR])
{
    for (int fila = 0; fila < dimR; fila++)
    {
        printf("  {%d", matriz_h[fila][0]);
        for (int columna = 1; columna < dimR; columna++)
        {
            printf(", %d", matriz_h[fila][columna]);
        }
        
        if (fila == (dimR - 1))
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
    const int longitudR = dimR * dimR * sizeof(int);
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
    int matrizR_h[dimR][dimR] = {};
    int matrizBR_h[dimB][dimB] = {};
    
    int (*matrizA_d)[dimA];
    int (*matrizB_d)[dimB];
    int (*matrizR_d)[dimR];
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
    cudaMalloc((void **) &matrizR_d, longitudR);
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
    dim3 bloques(dimR / TILE_WIDTH, dimR / TILE_WIDTH);
    dim3 hilos(TILE_WIDTH, TILE_WIDTH);
    convolucion<<<bloques, hilos>>>(matrizA_d, matrizBR_d, matrizR_d);
    
    // Transferir resultados al anfitrión
    cudaMemcpy(matrizR_h, matrizR_d, longitudR, cudaMemcpyDeviceToHost);
    
    // Mostrar resultados
    printf("Matriz A\n"
           "{\n");
    pintarMatrizA(matrizA_h);
    
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
    pintarMatrizR(matrizR_h);
    
    // Liberar memoria del dispositivo
    cudaFree(matrizA_d);
    cudaFree(matrizB_d);
    cudaFree(matrizR_d);
    cudaFree(matrizBR_d);
    
    return 0;
}

