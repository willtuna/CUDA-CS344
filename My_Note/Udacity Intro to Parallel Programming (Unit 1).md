# Udacity Intro to Parallel Programming (Unit 1)

## L1.16
### Main Concept
![](https://d2mxuefqeaa7sj.cloudfront.net/s_2B1568633CB6911594204E84ACB7F23851E082D723291FC588643E29AF20F892_1527295589365_image.png)


CUDA offering C programming extension allow us to compile C programming code into 2 parts, host - for CPU, device - for GPU.: cudaMemcpy
Moving Data From Host’s memory to Device’s memory: cudaMalloc
allocate memory in device’s memory: kernel

## L1.18
GPU could 
- respond to CPU request to send data from GPU to CPU
- respond to CPU request to receive data from CPU to GPU
- compute a kernel launched by CPU

CUDA Program flow

- CPU allocate storage on GPU: cudaMalloc
- CPU copies input data from CPU to GPU: cudaMemcpy
- CPU lanuch kernels on GPU to process the data: kernel launch
- CPU copies result from GPU back to CPU: cudaMemcpy


Core Concept of CUDA
- kernel look like serial programs
- write your program as if it will run on one thread
- the GPU will run that program on many threads



## Example Code
``` cuda
// kernel program
    __global__ void square(float * d_out, float * d_in){// __global__: declaration specifier , tell the cuda this is kernel program
        int idx = threadIdx.x; // each thread knows its index, so threadIdx is a C struct: dim3, which has members: x,y,z
        float f = d_in[idx];
        d_out[idx] = f*f;
    }
    // naming habits:  h_ means host
    //                 d_ means device
    int main(int argc, char ** argv) {
        const int ARRAY_SIZE = 64;
        const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);
        /*----------- host init -----------*/
        // generate the input array on the host
        float h_in[ARRAY_SIZE];
        // initial value
        for (int i = 0; i < ARRAY_SIZE; i++) {
            h_in[i] = float(i);
        }
        float h_out[ARRAY_SIZE];
```
#### Allocate memory in cuda by double pointer
``` cuda
        /*----------- device init -----------*/
        // declare GPU memory pointers
        float * d_in;
        float * d_out;
        // allocate GPU memory, using double pointer (void **) is 
        // because we gonna allocate memory in "device", so
        // so from host we veiw this as pointer (d_in) in host, point to another pointer in deive (ptr_in_device),
        // it would malloc memory "in device not host" to (ptr_in_device)
        cudaMalloc((void**) &d_in, ARRAY_BYTES);
        cudaMalloc((void**) &d_out, ARRAY_BYTES);
        // transfer the array to the GPU
        cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);
```
#### Launch Kernel
```Cuda
        // launch the kernel, configure the kernel
        // 1 block, ARRAY_SIZE threads
        square<<<1, ARRAY_SIZE>>>(d_out, d_in);
        // copy back the result array to the CPU
        cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);
        // print out the resulting array
        for (int i =0; i < ARRAY_SIZE; i++) {
            printf("%f", h_out[i]);
            printf(((i % 4) != 3) ? "\t" : "\n");
        }
        cudaFree(d_in);
        cudaFree(d_out);
        return 0;
    }
    
```
![](https://i.imgur.com/LltT5E5.png)
* number of threads in each block is limited by GPU Hardware
* So for different GPU , it needs different configuration.

Given current GPU has maximum 256 threads for each block.
![](https://i.imgur.com/8Xop2HP.png)

#### Multidimensional Block & Thread configuration.
* Green: launch 128 blocks in 1-D, each block has 128 threads in 1-D
* Red  : launch 64 blocks in 2-D ($8\times8$), each block has 256 threads in 2-D($16\times16$) 
![](https://i.imgur.com/v4xGNIo.png)

By default it would be 1-D block
![](https://i.imgur.com/pSSam0I.png)

![](https://i.imgur.com/Dcuuqe3.png)
##### threadIdx blockDim blockIdx gridDim
  * threadIdx: thread within block
               threadIdx.x threadIdx.y
  * blockDim : size of block (number of thread in each block dimension)
  * blockIdx : block within grid, choose block from grid
  * gridDim  : size of grid (number of block in each grid dimension)
![](https://i.imgur.com/FmeDDdt.png)

### MAP Operation
![](https://i.imgur.com/ECeqXEZ.png)

## HW1 Assignment
``` c
// Homework 1
// Color to Greyscale Conversion

//A common way to represent color images is known as RGBA - the color
//is specified by how much Red, Green, and Blue is in it.
//The 'A' stands for Alpha and is used for transparency; it will be
//ignored in this homework.

//Each channel Red, Blue, Green, and Alpha is represented by one byte.
//Since we are using one byte for each color there are 256 different
//possible values for each color.  This means we use 4 bytes per pixel.

//Greyscale images are represented by a single intensity value per pixel
//which is one byte in size.

//To convert an image from color to grayscale one simple method is to
//set the intensity to the average of the RGB channels.  But we will
//use a more sophisticated method that takes into account how the eye 
//perceives color and weights the channels unequally.

//The eye responds most strongly to green followed by red and then blue.
//The NTSC (National Television System Committee) recommends the following
//formula for color to greyscale conversion:

//I = .299f * R + .587f * G + .114f * B

//Notice the trailing f's on the numbers which indicate that they are 
//single precision floating point constants and not double precision
//constants.

//You should fill in the kernel as well as set the block and grid sizes
//so that the entire image is processed.

#include "reference_calc.cpp"
#include "utils.h"
#include <stdio.h>

__global__
void rgba_to_greyscale(const uchar4* const rgbaImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols)
{
  //TODO
  //Fill in the kernel to convert from color to greyscale
  //the mapping from components of a uchar4 to RGBA is:
  // .x -> R ; .y -> G ; .z -> B ; .w -> A
  //
  //The output (greyImage) at each pixel should be the result of
  //applying the formula: output = .299f * R + .587f * G + .114f * B;
  //Note: We will be ignoring the alpha channel for this conversion

  //First create a mapping from the 2D block and grid locations
  //to an absolute 2D location in the image, then use that to
  //calculate a 1D offset
  // x select col, y select row
  int y_idx = threadIdx.y + (blockIdx.y * blockDim.y );
  int x_idx = threadIdx.x + (blockIdx.x * blockDim.x );

  if(x_idx < numCols && y_idx < numRows)
      int index = x_idx + numRows*y_idx;
  uchar4 color = rgbaImage[index];
  unsigned char grey = (unsigned char) (.299f*color.x + .587f*color.y +.114f*color.z)
  greyImage[index] = grey;
}

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                            unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{
  //numRows represent the size of image in each row
  //numCols represent the size of image in each col

  //You must fill in the correct sizes for the blockSize and gridSize
  //currently only one block with one thread is being launched
  int blockwidth = 8;
  const dim3 blockSize(blockwidth, blockwidth, 1);  //TODO

  int numBlockX = numRows/blockwidth+1;
  int numBlockY = numCols/blockwidth+1;
  const dim3 gridSize( numBlockX, numBlockY, 1);  //TODO

  rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

```