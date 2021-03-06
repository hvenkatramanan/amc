// random_shuffle example
#include <iostream>     // std::cout
#include <algorithm>    // std::random_shuffle
#include <vector>       // std::vector
#include <ctime>        // std::time  // Needed for the true randomization
#include <cstdlib>      // std::rand, std::srand
#include <unordered_map>
#include <fstream>
#include <string>


using namespace std;
// random generator function:
int myrandom(int i) { return  rand() % i; }
void createvector(int d, unordered_map<char, vector<int>> &m);

int main() {
	unordered_map<char, vector<int>> m; //map with alphabets as key and vector of size d as value
	int d = 20;

	createvector(d, m);



	string line;
	string dir = "C:\\Users\\poorn\\Documents\\Visual Studio 2015\\Projects\\OpenCLProject3\\Files\\test\\";
	vector<string> trainingfile;
	ifstream listfile(dir + "list1.txt");
	int index = 0;
	if (listfile.is_open())
	{
		while (getline(listfile, line))
		{
			//cout << line << '\n';
			//trainingfile.push_back(line);

			char c;
			ifstream tfile(dir + line);
			cout << "opening the file : " << line << '\n';
			if (tfile.is_open())
			{

				while (tfile.good())//(getline(tfile, line))
				{
					tfile.get(c);
					cout << c << '\n';

				}
				tfile.close();
			}

			//cout << trainingfile.at(index) << '\n';
			//index++;
		}
		listfile.close();
	}

	else cout << "Unable to open file";


	getchar();
	return 0;
}






/*

// reading training data from a list text file
#include <iostream>
#include <fstream>
#include <string>
#include <vector>       // std::vector
using namespace std;

int main() {


	string line;
	string dir = "C:\\Users\\poorn\\Documents\\Visual Studio 2015\\Projects\\OpenCLProject3\\Files\\test\\";
	vector<string> trainingfile;
	ifstream myfile(dir+"list1.txt");
	int index = 0;
	if (myfile.is_open())
	{
		while (getline(myfile, line))
		{
			//cout << line << '\n';
			//trainingfile.push_back(line);
			
			char c;
			ifstream tfile(dir + line);
			cout << "opening the file : " << line << '\n';
			if (tfile.is_open())
			{

				while (tfile.good())//(getline(tfile, line))
				{
					tfile.get(c);
					cout << c << '\n';

				}
				tfile.close();
			}

			//cout << trainingfile.at(index) << '\n';
			//index++;
		}
		myfile.close();
	}

	else cout << "Unable to open file";

	getchar();
	return 0;
}






*/






/*

// This program implements a vector addition using OpenCL
// System includes
#include < stdio.h>
#include < stdlib.h>
// OpenCL includes
#include < CL/cl.h>
// OpenCL kernel to perform an element-wise
// add of two arrays
const char* programSource =
"__kernel void vecadd(__global int *A, __global int *B, __global int *C) \n"
"{\n"
" int idx = get_global_id(0);\n"
" C[idx] = A[idx] + B[idx]; \n"
" } \n"
;
int main() {
	// This code executes on the OpenCL host
	// Host data


	int *A = NULL; // Input array
	int *B = NULL; // Input array
	int *C = NULL; // Output array
				   // Elements in each array
	const int elements = 10000;
	// Compute the size of the data
	size_t datasize = sizeof(int)*elements;
	// Allocate space for input/output data
	A = (int*)malloc(datasize);
	B = (int*)malloc(datasize);
	C = (int*)malloc(datasize);


	// Initialize the random input data
	for (int i = 0; i < elements; i++) {
		A[i] = i;
		B[i] = i;
	}
	// Use this to check the output of each API call
	cl_int status;
	//覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧�
	// STEP 1: Discover and initialize the platforms
	//覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧�
	cl_uint numPlatforms = 0;
	cl_platform_id *platforms = NULL;
	// Use clGetPlatformIDs() to retrieve the number of
	// platforms
	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	// Allocate enough space for each platform
	platforms =
		(cl_platform_id*)malloc(
			numPlatforms * sizeof(cl_platform_id));
	// Fill in platforms with clGetPlatformIDs()
	status = clGetPlatformIDs(numPlatforms, platforms,
		NULL);
	//覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧�
	// STEP 2: Discover and initialize the devices
	//覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧�
	cl_uint numDevices = 0;
	cl_device_id *devices = NULL;
	// Use clGetDeviceIDs() to retrieve the number of
	// devices present
	status = clGetDeviceIDs(


		platforms[0],
		CL_DEVICE_TYPE_ALL,
		0,
		NULL,
		&numDevices);
	// Allocate enough space for each device
	devices =
		(cl_device_id*)malloc(
			numDevices * sizeof(cl_device_id));
	// Fill in devices with clGetDeviceIDs()
	status = clGetDeviceIDs(
		platforms[0],
		CL_DEVICE_TYPE_ALL,
		numDevices,
		devices,
		NULL);
	//覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧�
	// STEP 3: Create a context
	//覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧�
	cl_context context = NULL;
	// Create a context using clCreateContext() and
	// associate it with the devices
	context = clCreateContext(
		NULL,
		numDevices,
		devices,
		NULL,
		NULL,
		&status);
	//覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧�
	// STEP 4: Create a command queue
	//覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧�
	cl_command_queue cmdQueue;
	// Create a command queue using clCreateCommandQueue(),
	// and associate it with the device you want to execute
	// on
	cmdQueue = clCreateCommandQueue(
		context,
		devices[0],
		0,
		&status);
	//覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧�
	// STEP 5: Create device buffers
	//覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧�


	cl_mem bufferA; // Input array on the device
	cl_mem bufferB; // Input array on the device
	cl_mem bufferC; // Output array on the device
					// Use clCreateBuffer() to create a buffer object (d_A)
					// that will contain the data from the host array A
	bufferA = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		datasize,
		NULL,
		&status);
	// Use clCreateBuffer() to create a buffer object (d_B)
	// that will contain the data from the host array B
	bufferB = clCreateBuffer(
		context,
		CL_MEM_READ_ONLY,
		datasize,
		NULL,
		&status);
	// Use clCreateBuffer() to create a buffer object (d_C)
	// with enough space to hold the output data
	bufferC = clCreateBuffer(
		context,
		CL_MEM_WRITE_ONLY,
		datasize,
		NULL,
		&status);
	//覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧�
	// STEP 6: Write host data to device buffers
	//覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧�
	// Use clEnqueueWriteBuffer() to write input array A to
	// the device buffer bufferA
	status = clEnqueueWriteBuffer(
		cmdQueue,
		bufferA,
		CL_FALSE,
		0,
		datasize,
		A,
		0,
		NULL,
		NULL);
	// Use clEnqueueWriteBuffer() to write input array B to
	// the device buffer bufferB
	status = clEnqueueWriteBuffer(
		cmdQueue,


		bufferB,
		CL_FALSE,
		0,
		datasize,
		B,
		0,
		NULL,
		NULL);
	//覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧�
	// STEP 7: Create and compile the program
	//覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧�
	// Create a program using clCreateProgramWithSource()
	cl_program program = clCreateProgramWithSource(
		context,
		1,
		(const char**)&programSource,
		NULL,
		&status);
	// Build (compile) the program for the devices with
	// clBuildProgram()
	status = clBuildProgram(
		program,
		numDevices,
		devices,
		NULL,
		NULL,
		NULL);
	//覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧�
	// STEP 8: Create the kernel
	//覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧�
	cl_kernel kernel = NULL;
	// Use clCreateKernel() to create a kernel from the
	// vector addition function (named "vecadd")
	kernel = clCreateKernel(program, "vecadd", &status);
	//覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧�
	// STEP 9: Set the kernel arguments
	//覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧�
	// Associate the input and output buffers with the
	// kernel
	// using clSetKernelArg()
	status = clSetKernelArg(
		kernel,
		0,


		sizeof(cl_mem),
		&bufferA);
	status = clSetKernelArg(
		kernel,
		1,
		sizeof(cl_mem),
		&bufferB);
	status = clSetKernelArg(
		kernel,
		2,
		sizeof(cl_mem),
		&bufferC);
	//覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧�
	// STEP 10: Configure the work-item structure
	//覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧�
	// Define an index space (global work size) of work
	// items for
	// execution. A workgroup size (local work size) is not
	// required,
	// but can be used.
	size_t globalWorkSize[1];
	// There are 弾lements� work-items
	globalWorkSize[0] = elements;
	//覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧�
	// STEP 11: Enqueue the kernel for execution
	//覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧�
	// Execute the kernel by using
	// clEnqueueNDRangeKernel().
	// 暖lobalWorkSize� is the 1D dimension of the
	// work-items
	status = clEnqueueNDRangeKernel(
		cmdQueue,
		kernel,
		1,
		NULL,
		globalWorkSize,
		NULL,
		0,
		NULL,
		NULL);
	//覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧�
	// STEP 12: Read the output buffer back to the host
	//覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧�
	// Use clEnqueueReadBuffer() to read the OpenCL output

	// buffer (bufferC)
	// to the host output array (C)
	clEnqueueReadBuffer(
		cmdQueue,
		bufferC,
		CL_TRUE,
		0,
		datasize,
		C,
		0,
		NULL,
		NULL);
	// Verify the output
	bool result = true;
	for (int i = 0; i < elements; i++) {
		//printf("result: %d", C[i]);
		if (C[i] != i + i) {
			result = false;
			break;
		}
	}
	if (result) {
		printf("Output for addition is correct\n");
		printf(" ");
		printf(" ");
		printf(" ");
		printf(" ");
		getchar();
	}
	else {
		printf("Output for addition is incorrect\n");
	}
	//覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧�
	// STEP 13: Release OpenCL resources
	//覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧覧�
	// Free OpenCL resources
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(cmdQueue);
	clReleaseMemObject(bufferA);
	clReleaseMemObject(bufferB);
	clReleaseMemObject(bufferC);
	clReleaseContext(context);
	// Free host resources
	free(A);
	free(B);
	free(C);
	free(platforms);
	free(devices);
}*/

void createvector(int d, unordered_map<char, vector<int>>& m)
{
	srand(unsigned(time(0))); // This will ensure randomized number by help of time.

	vector<int> dvector;
	char alph[27] = { 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ' };

	for (int i = 0; i < d / 2; i++) {
		dvector.push_back(1);
	}

	for (int i = d / 2; i < d; i++) {
		dvector.push_back(-1);
	}

	for (int j = 0; j < 27; j++) {

		random_shuffle(dvector.begin(), dvector.end()); //Shuffling

		random_shuffle(dvector.begin(), dvector.end(), myrandom); //test if necessary

		m[alph[j]] = dvector;

		cout << "m[" << alph[j] << "] = ";
		for (std::vector<int>::const_iterator i = dvector.begin(); i != dvector.end(); ++i)
			std::cout << *i << ' ';

		cout << '\n';

	}
}
