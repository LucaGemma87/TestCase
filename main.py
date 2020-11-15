#!/usr/bin/python
from __future__ import division
from numba import cuda, float32
import numpy
import math 
import sys

sys.path.append(".")

from prova_gemma_logic import glogic

TPB = 5

@cuda.jit
def cuda_vector_gnot(vector,result):
    temp = vector
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    x,y = cuda.grid(2)
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.x
    if result.shape[0] > 1 and result.shape[1]>1:
       # Quit if (x, y) is outside of valid vector boundary
       return
       #print(temp) 
    iter = len(temp)-1
    #print(iter)
    #range_data = int(vector.size/TPB)
    range_data = int(vector.shape[1] / TPB)
    print(range_data)
    tmp = 0.
    for i in range(range_data):
        # Preload data into shared memory
        print(i)
        temp_1 = vector[x, ty + i * TPB]
        sA[tx] = temp_1
        print(temp_1)
        temp_2 = vector[tx + i * TPB, y]
        sB[ty] = temp_2
        print(temp_2)

        tmp += math.floor(abs(temp_1 - temp_2 - tmp)/3)


        # Wait until all threads finish preloading
        cuda.syncthreads()

        #print(sA)
        #print(sB)
        # Computes partial product on the shared memory
        #for j in range(TPB):
        #    print(j)
        #    tmp = 0
        #    tmp += abs(sA[tx, j] - tmp)
        #    print(tmp)
        # Wait until all threads finish computing
        #cuda.syncthreads()

    result[x] = tmp
    
    #print(sA)
    #print(sB)
    #while iter>=1:
    #      for x in range(0,range_data,1):
    #          temp[x] = abs(temp[x]-temp[x+1])
    #          print(temp[x])   
    #      range_data = range_data -1
    #      iter = iter - 1
          #print(iter)
              
    #result = temp[0]
    return  


def main():
    print("START MAIN")
    array = [0,1,2,3,4,5,6,7,8,9]
    num_of_rows = max(array)+1
    num_of_cols = max(array)+1
    null_matrix = numpy.full((num_of_rows, num_of_cols), numpy.zeros)
    
    # The data array
    matrix = [0,3,4,6,9]
    A = numpy.full((TPB,TPB), 1, numpy.float) # [32 x 48] matrix containing all 3's
    print(A)
    print(A*matrix)
    A_global_mem = cuda.to_device(A*matrix)
    C_global_mem = cuda.device_array(((1, 1))) # [32 x 16] matrix result
    p1 = glogic(array,null_matrix)
    #p1.print_gnot_matrix()
    #p1.print_gexor_matrix()
    #p1.print_gexand_matrix()
    #p1.print_gnot_matrix()
    #p1.print_gand_matrix()
    #p1.print_gor_matrix()
    #p1.print_gnand_matrix()
    #p1.print_gnor_matrix()
    #p1.print_gexor_matrix()
    #matrix = [1,9]
    # Configure the blocks
    threadsperblock = (TPB, TPB)
    blockspergrid_x = int(math.ceil(A.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x)
    cuda_vector_gnot[blockspergrid, threadsperblock](A_global_mem,C_global_mem)
    res = C_global_mem.copy_to_host()
    print("REsult:")
    print(res)
    #matrix =  [1,5,9]
    #out2 = p1.vector_gand(matrix)
    #print(out2)
    #matrix =  [1,5,9]
    #out3 = p1.vector_gor(matrix)
    #print(out3)
    #matrix =  [1,5,9]
    #out4 = p1.vector_gnand(matrix)
    #print(out4)
    #matrix =  [1,5,9]
    #out5 = p1.vector_gnor(matrix)
    #print(out5)
    #matrix =  [1,5,9]
    #out6 = p1.vector_gexor(matrix)
    #print(out6)
    #matrix =  [1,5,9]
    #out7 = p1.vector_gexand(matrix)
    #print(out7)
    print("END MAIN")

if __name__ == "__main__":
    main()