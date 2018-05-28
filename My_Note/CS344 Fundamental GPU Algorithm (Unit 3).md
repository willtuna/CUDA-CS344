# CS344 Fundamental GPU Algorithm (Unit 3)

## Main Algorithm for this lecture:
* Reduce
* Scan
* Histogram

## Step & Work
* Step Complexity
* Work Complexity
<img src=https://i.imgur.com/TXomWri.png width=300 ></img>

### Algorithm Reduce
1. Set of Elements
2. Reduction Operator
    - Binary
    - Associative
<img src=https://i.imgur.com/LPY6dQm.png width=300></img>
* Serial Implementation of Reduce (Complexity $O(N)$)
* Parallel Implementation of Reduce (Complexity $O(log_2N)$)
<img src=https://i.imgur.com/PI90hjX.png width=300></img>
### Algorithm Scan
- Address Set of Problems Otherwise difficult to Parallelize
- Not Useful in Serial World But Useful in Parallel

Input to Scan:
- input array
- binary associative operator
- identity operator


|    OP    |  Idenity |  Cause   |
| -------- | -------- | -------- |
|    $+$   |  $\emptyset$ | $\emptyset+a=a$ |
| min(u_char)|  0xFF | $min(0xFF,a)=a$ |

#### Share Memory Reduce to accelerate
Kernel Code
```c
__global__ void shmem_reduce_kernel(float * d_out, const float * d_in)
{
    // sdata is allocated in the kernel call: 3rd arg to <<<b, t, shmem>>>
    extern __shared__ float sdata[];

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid  = threadIdx.x;

    // load shared mem from global mem
    sdata[tid] = d_in[myId];
    __syncthreads();            // make sure entire block is loaded!

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0)
    {
        d_out[blockIdx.x] = sdata[0];
    }
}

Usage of Kernel Code with SharedMem
```C
shmem_reduce_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_intermediate, d_in);
```

****


#### Exclusive SCAN
<img src=https://i.imgur.com/FIStokm.png width=300></img>

<img src=https://i.imgur.com/YL2HAUh.png width=300></img>


<h3> We can parallelize SCAN in GPU </h3>


#### Inclusive vs Exclusive
<img src=https://i.imgur.com/7ejEnWs.png width=300></img>

##### Serial Implementation of Inclusive SCAN
```  
// pseudo
int acc = idenity
for (i=0;i<elements.len();i++){
    acc = acc op element[i];
    acc[i] = acc;
}
```

##### Serial Implementation of Exclusive SCAN
```  
// pseudo
int acc = idenity
for (i=0;i<elements.len();i++){
    acc[i] = acc;
    acc = acc op element[i];
}
```

### Hillis & Steele 's Algorithm (Step Efficient)
Greedy Like Concept
<img src=https://i.imgur.com/g4KGEVt.png width=400></img>
Work Complexity: O($Nlog_2 N$)
Step Complexity: O($log_2 N$)
### Blelloch 's Algorithm (Work Efficient)
Depth First Concept

<img src=https://i.imgur.com/5m7NdHR.png ></img>
Downsweep: 
1. Put Identity to Right Most Element
2. Copy R to L and Operate on L and R to R
Work Complexity: O($N$)
Step Complexity: O($2*log_2 N$)
###### Algorithm Deployment
<img src=https://i.imgur.com/pI68t56.png width=300></img>
Data Element More Than Processor: Work Efficient
Data Element Less Than Processor: Time Efficient


### Tips of Histogram Problem

128 items 8 thread 3 BINS
#### Simple Atomic Operation
#### Private Histogram per thread & reduce (Saving Atomic Opeartion)
Privated (Local) Histogram per thread, then reduce.
- 16 items operated in each thread
<img src=https://i.imgur.com/LT60Ssw.png width=300></img>
- reduce
<img src=https://i.imgur.com/WzypIS2.png width=300><\img>
#### Sort & Reduced by Key
<img src=https://i.imgur.com/aGn4cIk.png width=300><\img>










