// random_shuffle example
#include <iostream>     // std::cout
#include <algorithm>    // std::random_shuffle
#include <vector>       // std::vector
#include <ctime>        // std::time  // Needed for the true randomization
#include <cstdlib>      // std::rand, std::srand
#include <unordered_map>
#include <fstream>
#include <string>
#include "kernal.h"

#ifdef TARGET_OS_WIN32
#include <CL/cl.h>
#else
#include <OpenCL/OpenCL.h>
#endif

using namespace std;

//#define PRINT_ITEM_MEMORY;                            //Prints item memory if 1
//#define GENERATE_TRAINING_DATA;                       //Creates Associative memory if 1
//#define SPLIT_BY_SPACE;                               //Split input by space instead of using NGRAMS

const int       DIMEN       = 1000;                     //Vector dimension
const int       NGRAM       = 2;                        //Ngram-characters read from the training and test files
const size_t    DATASIZE    = sizeof(int) * DIMEN;
const string    BASE_DIR    = "C:\\Users\\poorn\\Documents\\Visual Studio 2015\\Projects\\OpenCLProject3\\Files\\test\\";
const string    TRAIN_FILE  = "list1.txt";
const string    TEST_FILE   = "";
const char *    KERNAL_NAME = "abcde";

int     *out = NULL;
int     *items;
string  news_type;
size_t globalWorkSize[1];

unordered_map<char,     vector<int>>    ITEM_MEMORY;        //map with alphabets as key and vector of size d as value
unordered_map<string,   vector<int>>    ASSOCIATIVE_MEMORY;

/** OpenCL parameter initializations -- BEGIN **/
/**/ cl_int status;
/**/ cl_uint numPlatforms = 0;
/**/ cl_platform_id *platforms = NULL;
/**/ cl_uint numDevices = 0;
/**/ cl_device_id *devices = NULL;
/**/ cl_context context = NULL;
/**/ cl_command_queue cmdQueue;
/**/ cl_mem buffer;
/**/ cl_mem bufferOut;
/**/ cl_program program;
/**/ cl_kernel kernel = NULL;
/** OpenCL parameter initializations -- END **/


/** Function Declarations -- BEGIN **/
void generateItemMemory();
int myrandom(int i);
void initializeAssociativeMemory();
void kernalInitialize();
void processFile(string filename);
vector<string> ReadFile(string filename);
void loadKernalBuffer_Ngram(string data);
void loadKernalBuffer_space(string str);
void writeDataIntoBuffer();
void runKernal();
void addToAssociativeMemory();
void releaseMemory();
/** Function Declarations -- END **/


int main()
{
    generateItemMemory();
    initializeAssociativeMemory();
    kernalInitialize();
    //processFile(TRAIN_FILE);
    
    
#ifdef GENERATE_TRAINING_DATA
    string line;
    vector<string> trainingfile;
    ifstream listfile(BASE_DIR + TRAIN_FILE);
    //int index = 0;
    if (listfile.is_open())
    {
        while (getline(listfile, line))
        {
            //cout << line << '\n';
            //trainingfile.push_back(line);
            char c;
            ifstream tfile(BASE_DIR + line);
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
#endif
    
    getchar();
    return 0;
}

/********************************************************************************************************************/
void generateItemMemory()
{
    srand(unsigned(time(0)));                           //This will ensure randomized number by help of time.
    vector<int> dvector;
    char alph[27] = { 'a', 'b', 'c', 'd', 'e',
        'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
        'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
        'v', 'w', 'x', 'y', 'z', ' ' };
    
    for (int i = 0; i < DIMEN / 2; i++)
    {
        dvector.push_back(1);                           //Fills half of the vector with 1s
    }
    
    for (int i = DIMEN / 2; i < DIMEN; i++)
    {
        dvector.push_back(-1);                          //Fills other hald of the vector with -1s
    }
    
    for (int j = 0; j < 27; j++)
    {
        random_shuffle(dvector.begin(), dvector.end()); //Shuffles the vector to rearrange the 1s and -1s
        random_shuffle(dvector.begin(), dvector.end(),  //Reshuffling the vector by feeding random index postions
                       myrandom);                       //to make sure ITEM_MEMORY is unique
        
        ITEM_MEMORY[alph[j]] = dvector;                 //Fill the ITEM_MEMORY
        
#ifdef PRINT_ITEM_MEMORY
        cout << "ITEM_MEMORY[" << alph[j] << "] = ";
        for (std::vector<int>::const_iterator i = dvector.begin(); i != dvector.end(); ++i)
            std::cout << *i << ' ';
        cout << '\n';
#endif
    }
}

/********************************************************************************************************************/
int myrandom(int i)
{
    return  rand() % i;
}

/********************************************************************************************************************/
void initializeAssociativeMemory()
{
    for(int i = 0; i < DIMEN; i++)
        ASSOCIATIVE_MEMORY[news_type].push_back(0);
}

/********************************************************************************************************************/
void kernalInitialize()
{
    out = (int*)malloc(DATASIZE);
    
    items = NULL;
    items = (int*)malloc(DATASIZE*NGRAM);
    
    //*STEP 1: Initialize the platforms
    // Use clGetPlatformIDs() to retrieve the number of platforms
    status = clGetPlatformIDs(0, NULL, &numPlatforms);
    // Allocate enough space for each platform
    platforms = (cl_platform_id*)malloc(numPlatforms * sizeof(cl_platform_id));
    // Fill in platforms with clGetPlatformIDs()
    status = clGetPlatformIDs(numPlatforms, platforms, NULL);
    
    //*STEP 2: Discover and initialize the devices
    // Use clGetDeviceIDs() to retrieve the number of devices present
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
    // Allocate enough space for each device
    devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));
    // Fill in devices with clGetDeviceIDs()
    status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_ALL, numDevices, devices, NULL);
    
    //*STEP 3: Create a context
    // Create a context using clCreateContext() and associate it with the devices
    context = clCreateContext(NULL, numDevices, devices, NULL, NULL, &status);
    
    //*STEP 4: Create a command queue
    // Create a command queue and associate it with the device you want to execute on
    cmdQueue = clCreateCommandQueue(context, devices[0], 0, &status);
    
    //*STEP 5: Create device buffers
    // Create a buffer object (d_A) that will contain the data from the host array
    for(int i=0;i<NGRAM;i++)
    {
        buffer = clCreateBuffer(
                                context,
                                CL_MEM_READ_ONLY,
                                DATASIZE,
                                NULL,
                                &status);
    }
    // Create a buffer object (d_C) with enough space to hold the output data
    bufferOut = clCreateBuffer(context, CL_MEM_READ_WRITE, DATASIZE, NULL, &status);
    
    
    //*STEP 6: Write host data to input and output buffers
    writeDataIntoBuffer();
    status = clEnqueueWriteBuffer(cmdQueue, bufferOut, CL_FALSE, 0, DATASIZE, out, 0, NULL, NULL);
    
    
    //*STEP 7: Create and compile the program
    // Create a program using clCreateProgramWithSource()
    program = clCreateProgramWithSource(context, 1, (const char**)&programSource, NULL, &status);
    // Build (compile) the program for the devices with clBuildProgram()
    status = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);
    
    //*STEP 8: Create the kernel
    // Use clCreateKernel() to create a kernel from the
    // vector addition function (named "vecadd")
    kernel = clCreateKernel(program, KERNAL_NAME, &status);
    
    // STEP 9: Set the kernel arguments
    // Associate the input and output buffers with the kernel
    status = clSetKernelArg(kernel,
                            0,                          //Argument position
                            sizeof(cl_mem),
                            &items);                    //TODO: change input based on kernal
    
    status = clSetKernelArg(kernel,
                            1,                          //Argument position
                            sizeof(cl_mem),
                            &bufferOut);
    
    // STEP 10: Configure the work-item structure
    // Define an index space (global work size) of work items for execution
    
    // There are ’elements’ work-items
    globalWorkSize[0] = DIMEN;
    
}

/********************************************************************************************************************/
void processFile(string filename)
{
    vector<string> path = ReadFile(filename);
    std::vector<string>::const_iterator i;
    for (i = path.begin(); i != path.end(); ++i)
    {
        string str = (string)*i;
        string strWords[2];
        int counter=0;
        
        for(int i=0; i<str.length(); i++)
        {
            if (str[i] == '\t')
                counter++;
            else
                strWords[counter] += str[i];
        }
        
        cout<<"Key is--"<<strWords[0]<<"\n";
        news_type = strWords[0];
        
        initializeAssociativeMemory();
        
        cout<<"Value --"<<strWords[1]<<"\n\n";
        
#ifdef SPLIT_BY_SPACE
        loadKernalBuffer_space(strWords[1]);
#else
        loadKernalBuffer_Ngram(strWords[1]);
#endif
    }
}

/********************************************************************************************************************/
vector<string> ReadFile(string filename)
{
    ifstream inFile(filename);
    vector<string> vecstr;
    if (!inFile) {
        cerr << "File -- "<<filename<<" --not found." << endl;
        return vecstr;
    }
    
    
    // Using getline() to read one line at a time.
    string line;
    while (getline(inFile, line)) {
        if (line.empty()) continue;
        
        // Using istringstream to read the line into integers.
        /*	istringstream iss(line);
         int sum = 0, next = 0;
         while (iss >> next) sum += next;
         outFile << sum << endl;
         */
        
        vecstr.push_back(line);
    }
    
    inFile.close();
    //	outFile.close();
    return vecstr;
}

/********************************************************************************************************************/
void loadKernalBuffer_Ngram(string data)
{
    for(int i = 0; i < data.length()-NGRAM+1; ++i)
    {
        for(int j=0; j < NGRAM; j++)
        {
            for(int k=0; k < DIMEN; k++)
            {
                items[k]=ITEM_MEMORY[data.at(i+j)][k];
            }
        }
        //Load kernal arguments
        //Run Kernal
        cout<<data.substr(i,NGRAM);
        cout<<"----";
    }
    //get output from kernal and put it to associative mem
}

/********************************************************************************************************************/
void loadKernalBuffer_space(string str)
{
    //const char *str = data.c_str();
    char * writable = new char[str.size()+1];
    std::copy(str.begin(), str.end(), writable);
    writable[str.size()] = '\0';
    char * pch;
    pch = strtok (writable," ");
    while (pch != NULL) {
        cout<<pch;
        cout<<"----";
        pch = strtok (NULL, " ");
    }
    
    /*	for(unsigned int i = 0; i < data.length()-n+1; ++i) {
     cout<<data.substr(i,n);
     cout<<"----";
     }*/
}

/********************************************************************************************************************/
void writeDataIntoBuffer()
{
    //———————————————————————————————————————————————————
    // STEP 6: Write host data to device buffers
    //———————————————————————————————————————————————————
    // Use clEnqueueWriteBuffer() to write input array A to
    // the device buffer bufferA
    //for(int i=0; i<NGRAM; i++)
    {
        status = clEnqueueWriteBuffer(cmdQueue,
                                      buffer,
                                      CL_FALSE,
                                      0,
                                      DATASIZE*NGRAM,
                                      items,
                                      0,
                                      NULL,
                                      NULL);
    }
}

/********************************************************************************************************************/
void runKernal()
    {
    // Enqueue the kernel for execution
    status = clEnqueueNDRangeKernel(cmdQueue,
                                    kernel,
                                    1,
                                    NULL,
                                    globalWorkSize,
                                    NULL,
                                    0,
                                    NULL,
                                    NULL);
}

/********************************************************************************************************************/
void addToAssociativeMemory(){
    //Read the output buffer back to the host
    clEnqueueReadBuffer(cmdQueue,
                        bufferOut,
                        CL_TRUE,
                        0,
                        DATASIZE,
                        out,
                        0,
                        NULL,
                        NULL);
    for(int i=0; i < sizeof(out); i++)
    {
        ASSOCIATIVE_MEMORY[news_type][i] += out[i];
    }
}

/********************************************************************************************************************/
void releaseMemory()
{
    // Free OpenCL resources
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdQueue);
    clReleaseMemObject(buffer);
    clReleaseMemObject(bufferOut);
    clReleaseContext(context);
    
    // Free host resources
    free(out);
    free(items);
    free(platforms);
    free(devices);
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
	//———————————————————————————————————————————————————
	// STEP 1: Discover and initialize the platforms
	//———————————————————————————————————————————————————
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
	//———————————————————————————————————————————————————
	// STEP 2: Discover and initialize the devices
	//———————————————————————————————————————————————————
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
	//———————————————————————————————————————————————————
	// STEP 3: Create a context
	//———————————————————————————————————————————————————
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
	//———————————————————————————————————————————————————
	// STEP 4: Create a command queue
	//———————————————————————————————————————————————————
	cl_command_queue cmdQueue;
	// Create a command queue using clCreateCommandQueue(),
	// and associate it with the device you want to execute
	// on
	cmdQueue = clCreateCommandQueue(
		context,
		devices[0],
		0,
		&status);
	//———————————————————————————————————————————————————
	// STEP 5: Create device buffers
	//———————————————————————————————————————————————————


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
	//———————————————————————————————————————————————————
	// STEP 6: Write host data to device buffers
	//———————————————————————————————————————————————————
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
	//———————————————————————————————————————————————————
	// STEP 7: Create and compile the program
	//———————————————————————————————————————————————————
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
	//———————————————————————————————————————————————————
	// STEP 8: Create the kernel
	//———————————————————————————————————————————————————
	cl_kernel kernel = NULL;
	// Use clCreateKernel() to create a kernel from the
	// vector addition function (named "vecadd")
	kernel = clCreateKernel(program, "vecadd", &status);
	//———————————————————————————————————————————————————
	// STEP 9: Set the kernel arguments
	//———————————————————————————————————————————————————
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
	//———————————————————————————————————————————————————
	// STEP 10: Configure the work-item structure
	//———————————————————————————————————————————————————
	// Define an index space (global work size) of work
	// items for
	// execution. A workgroup size (local work size) is not
	// required,
	// but can be used.
	size_t globalWorkSize[1];
	// There are ’elements’ work-items
	globalWorkSize[0] = elements;
	//———————————————————————————————————————————————————
	// STEP 11: Enqueue the kernel for execution
	//———————————————————————————————————————————————————
	// Execute the kernel by using
	// clEnqueueNDRangeKernel().
	// ’globalWorkSize’ is the 1D dimension of the
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
	//———————————————————————————————————————————————————
	// STEP 12: Read the output buffer back to the host
	//———————————————————————————————————————————————————
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
	//———————————————————————————————————————————————————
	// STEP 13: Release OpenCL resources
	//———————————————————————————————————————————————————
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

