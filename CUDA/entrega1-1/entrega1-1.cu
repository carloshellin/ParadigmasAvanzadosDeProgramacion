#include <stdio.h>

int main(int argc, char **argv)
{
    cudaDeviceProp prop = {};
    int count = 0;
    
    cudaGetDeviceCount(&count);
    for (int device = 0; device < count; device++)
    {
        cudaGetDeviceProperties(&prop, device);
        
        printf("Nombre del dispositivo: %s\n", prop.name);
        printf("Cantidad de memoria global (bytes): %zd\n", 
               prop.totalGlobalMem);
        printf("Cantidad maxima de memoria compartida por bloque (bytes): "
               "%zd\n", prop.sharedMemPerBlock);
        printf("Numero de registros de 32 bits disponible por bloque: %d\n"
               , prop.regsPerBlock);
        printf("Numeros de hilos en un warp: %d\n\n", prop.warpSize);
        
        printf("Numero maximo de hilos que un bloque puede contener: %d\n" 
               , prop.maxThreadsPerBlock);
        printf("Numero maximos de hilos permitidos a lo largo de cada "
               "dimension de un bloque: [x -> %d] [y -> %d] [z -> %d]\n", 
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], 
               prop.maxThreadsDim[2]);
        printf("Numero de bloques permitidos a lo largo de cada "
               "dimension de una cuadricula: "
               "[x -> %d] [y - > %d] [z -> %d]\n",
               prop.maxGridSize[0], prop.maxGridSize[1], 
               prop.maxGridSize[2]);
        printf("Cantidad de memoria constante disponible (bytes): %zd\n", 
               prop.totalConstMem);
        printf("Revision mayor y menor del dispositivo: %i.%i\n", 
               prop.major, prop.minor);
        printf("Requisito del dispositivo para la alineacion de "
               "la textura: %zd\n", prop.textureAlignment);
        printf("El despositivo puede realizar simultaneamente "
               "una ejecucion " "de cudaMemcpy y de kernel: %s\n", 
               prop.deviceOverlap ? "Verdadero" : "Falso");
        printf("Numero de multiprocesadores en el dispositivo: %i\n", 
               prop.multiProcessorCount);
        printf("El dispositivo tiene un limite de tiempo de "
               "ejecucion para los kernels ejecutados: %s\n", 
               prop.kernelExecTimeoutEnabled ? "Verdadero" : "Falso");
        printf("El dispositivo es una GPU integrada: %s\n", 
               prop.integrated ? "Verdadero" : "Falso");
        printf("El dispositivo puede asignar memoria del host "
               "al espacio de direcciones del dispositivo CUDA: %s\n",
               prop.canMapHostMemory ? "Verdadero" : "Falso");
    }
    
    return 0;
}