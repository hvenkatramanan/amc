//#ifdef NGRAM_2
const char* programSource =
"__kernel void vecadd(__global int *items[0], __global int *items[1], __global int *out) \n"
"{\n"
" int idx = get_global_id(0);\n"
" out[idx] += items[0][idx] * items[1][idx]; \n"
" } \n"
;
//#endif

#ifdef NGRAM_3
const char* programSource =
"__kernel void vecadd(__global int *A[], __global int *B, __global int *C) \n"
"{\n"
" int idx = get_global_id(0);\n"
" C[idx] = A[idx] + B[idx]; \n"
" } \n"
;

#endif

//#if NGRAM == 2

//#endif
