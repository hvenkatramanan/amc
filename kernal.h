const char* programSource =
"__kernel void vecadd(__global int *A[], __global int *B, __global int *C) \n"
"{\n"
" int idx = get_global_id(0);\n"
" C[idx] = A[idx] + B[idx]; \n"
" } \n"
;
