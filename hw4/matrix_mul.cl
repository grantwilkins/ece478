__kernel void matrix_mul(__global float * inputMatrix1, __global float * inputMatrix2, __global float *result)
{
	int size = get_global_size(0);
	int i = get_global_id(0);
	int j = get_global_id(0);

	int prod = 0;

	if(i < size && j < size)
	{
		for (int k=0; k<size; k++)
	    	prod += inputMatrix1[j*size + k] * inputMatrix2[k*size + i];

	    result[j*size + i] = prod;
	}
}