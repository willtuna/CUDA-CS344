# Udacity Intro to Parallel Programming (Unit 2)

### Map & Gather & Scatter
* Map: Task read from and write to "specific data element".
        1 to 1 (both of them are determined)
        <img src=https://i.imgur.com/4klubUC.png width=400></img>
* Gather: Each task gather input data element together from different place.
        Multiple to 1 
        <img src=https://i.imgur.com/MlEwXeR.png width=400></img>

* Scatter: Each task compute input data element to muliple output .
    1 to Multiple (output is undetermined with multiple choice)(issue: $write collision$)
        <img src=https://i.imgur.com/lsme7mH.png width=400></img>

    Sorting Problem is a scatter problem, since each thread operation is computing where to write its result(output is undetermined).
* stencil: task read input from a fixed neighborhood in an array (Feature:  $Data Reuse !$)
<img src=https://i.imgur.com/tsszbqQ.png width=400></img>
## Transpose
**scattering operation form**
<img src=https://i.imgur.com/isiHB22.png width=300></img>
**gathering opeartion form**
<img src=https://i.imgur.com/ja59aa0.png width=300></img>
### Structure of Array (SOA) vs Array of Structure (AOS)
<img src=https://i.imgur.com/EsxTfN0.png width=300></img>

## Quiz:
![](https://i.imgur.com/frjXMpz.png)
Last 2 not stencil, because not all output would be computed to.
# Parallel Computing Patterns
![](https://i.imgur.com/aYbLoAE.png)
## GPU Hardware Programmer's View
Streaming Multiprocessor(SM): run multiple threads.
Thread block is just like allocate the work in memory, then <p><font size=5><B>GPU is responsible for allocating blocks to SM.</B></font></p>

* A thread block contains many threads. (Memory)
* An SM may run more than one block. (Many Small size thread block allocate to the same SM)
* All the threads in a thread block may cooperate to solve a subproblem
* Not all thread run on a given SM should cooperate to sovle a subproblem(since, there might independent thread block in the same SM)

<h1> Important Concept</h1>

* The Programmer is responsible for defining thread block in software.
    - the prgrammer could not specify Block's execution order.
* The GPU is responsible for allocating thread blocks to hardware streaming multiprocessors(SMs).

# What CUDA guarantee ?
- All thread in a block run on the same SM at the same time.
- All blocks in a kernel finish before any blocks from the next kernel run

# CUDA Memory Model
<img src=https://i.imgur.com/cRtGzAc.png width=400>
</img>
- All threads from a block can access the same variable in that block's shared memory
- Thread in different block can access the same variable in global memory
- Threads from different blocks have their own copy of local varaible in local memory.
- Threads from the same blocks have their own copy of lcoal variables in local memory.
(看圖最準!)


# Synchronization 

tool to solve synchronization problem.
<h4>Barrier</h4>

point in the program whare threads stop and wait.
When all threads have reached the barrier, they can proceed.
<img src=https://i.imgur.com/i0RQkD2.png ></img>

<h5> Example </h5>

Check the number of barrier inserted to make sure synchronization. Ans:3

``` c
// collision example
int idx=threadIdx.x;
__share__ int array [128];
array[idx] = threadIdx.x;
if(idx < 127)
    array[idx] = array[idx+1];
```

``` c
// free of collision
int idx=threadIdx.x;
__share__ int array [128];
__syncthreads();
array[idx] = threadIdx.x;// assign to shared memory
if(idx < 127)
    int temp = array[idx+1];// assign to local memory
    __syncthreads();
    array[idx] = temp;// assign back to shared memory
    __syncthreads();
```
**__syncthreads()** create barrier for block operation.
![](https://i.imgur.com/i20OALK.png)


# Writing Efficient Programs
### Maximize arithmetic intensity.
    - maximize compute operation per thread
    - minimize time spent on memory per thread
Minimize time spent on memory (Memory Hierarchy Issue)

## Great Example:
``` C
// Using different memory spaces in CUDA
#include <stdio.h>

/**********************
 * using local memory *
 **********************/

// a __device__ or __global__ function runs on the GPU
__global__ void use_local_memory_GPU(float in)
{
    float f;    // variable "f" is in local memory and private to each thread
    f = in;     // parameter "in" is in local memory and private to each thread
    // ... real code would presumably do other stuff here ... 
}

/**********************
 * using global memory *
 **********************/

// a __global__ function runs on the GPU & can be called from host
__global__ void use_global_memory_GPU(float *array)
{
    // "array" is a pointer into global memory on the device
    array[threadIdx.x] = 2.0f * (float) threadIdx.x;
}

/**********************
 * using shared memory *
 **********************/

// (for clarity, hardcoding 128 threads/elements and omitting out-of-bounds checks)
__global__ void use_shared_memory_GPU(float *array)
{
    // local variables, private to each thread
    int i, index = threadIdx.x;
    float average, sum = 0.0f;

    // __shared__ variables are visible to all threads in the thread block
    // and have the same lifetime as the thread block
    __shared__ float sh_arr[128];

    // copy data from "array" in global memory to sh_arr in shared memory.
    // here, each thread is responsible for copying a single element.
    sh_arr[index] = array[index];

    __syncthreads();    // ensure all the writes to shared memory have completed

    // now, sh_arr is fully populated. Let's find the average of all previous elements
    for (i=0; i<index; i++) { sum += sh_arr[i]; }
    average = sum / (index + 1.0f);

    // if array[index] is greater than the average of array[0..index-1], replace with average.
    // since array[] is in global memory, this change will be seen by the host (and potentially 
    // other thread blocks, if any)
    if (array[index] > average) { array[index] = average; }

    // the following code has NO EFFECT: it modifies shared memory, but 
    // the resulting modified data is never copied back to global memory
    // and vanishes when the thread block completes
    sh_arr[index] = 3.14;
}

int main(int argc, char **argv)
{
    /*
     * First, call a kernel that shows using local memory 
     */
    use_local_memory_GPU<<<1, 128>>>(2.0f);

    /*
     * Next, call a kernel that shows using global memory
     */
    float h_arr[128];   // convention: h_ variables live on host
    float *d_arr;       // convention: d_ variables live on device (GPU global mem)

    // allocate global memory on the device, place result in "d_arr"
    cudaMalloc((void **) &d_arr, sizeof(float) * 128);
    // now copy data from host memory "h_arr" to device memory "d_arr"
    cudaMemcpy((void *)d_arr, (void *)h_arr, sizeof(float) * 128, cudaMemcpyHostToDevice);
    // launch the kernel (1 block of 128 threads)
    use_global_memory_GPU<<<1, 128>>>(d_arr);  // modifies the contents of array at d_arr
    // copy the modified array back to the host, overwriting contents of h_arr
    cudaMemcpy((void *)h_arr, (void *)d_arr, sizeof(float) * 128, cudaMemcpyDeviceToHost);
    // ... do other stuff ...

    /*
     * Next, call a kernel that shows using shared memory
     */

    // as before, pass in a pointer to data in global memory
    use_shared_memory_GPU<<<1, 128>>>(d_arr); 
    // copy the modified array back to the host
    cudaMemcpy((void *)h_arr, (void *)d_arr, sizeof(float) * 128, cudaMemcpyHostToDevice);
    // ... do other stuff ...
    return 0;
}
```

## Coalesce Memory Access
![](https://i.imgur.com/KfyGuRb.png)

## Atomic Memory Opeartion
Problems: Lots of threads reading and writing the same memory locations.
![](https://i.imgur.com/mlnltGh.png)

### Limitation of atomic
* Only certain opeartion ,data type are supported.
    - so using atomic CAS() ... atomic compare and swap , detail would be mutex and critical section.
* Still no ordering constraints
    - floating-point arithmetic non-associative (huge issue !!)
* Slow


## High Arithmetic Intensity
![](https://i.imgur.com/tWkQh81.png)
#### Branch would cause divergence (Unbalanced).
![](https://i.imgur.com/8niETuX.png)
#### Loop would cause divergence (Unbalanced).
![](https://i.imgur.com/Qwa9Amv.png)


# Summary
1. Communication Patterns
    - gather ,scatter ,stencil ,transpose
2. GPU hardware & Programming Model
    - Stream Multiprocessors, Blocks, Ordering
    - Synchronization
    - Memory Model - local,global,shared,atomics
3. Efficient GPU Programming
    - Access Memory Faster
        - coalescing global memory
        - use faster memory
    - Avoid Thread Divergence








