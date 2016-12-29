// random_shuffle example
#include <iostream>     // std::cout
#include <algorithm>    // std::random_shuffle
#include <vector>       // std::vector
#include <ctime>        // std::time  // Needed for the true randomization
#include <cstdlib>      // std::rand, std::srand
#include <unordered_map>
#include <fstream>
#include <string>
#include "kernel.h"

#ifdef _WIN32
#include <CL/cl.h>
#else
#include <OpenCL/OpenCL.h>
#endif
using namespace std;

#define NGRAM_3

//#define PRINT_ITEM_MEMORY 0;                            //Prints item memory if defined
//#define SPLIT_BY_SPACE;                               //Split input by space instead of using NGRAMS

const int       DIMEN       = 10;                     //Vector dimension
const int       NGRAM       = 2;                        //Ngram-characters read from the training and test files
const size_t    DATASIZE    = sizeof(int) * DIMEN;
const string    BASE_DIR    = "/Users/hariharanvenkatramanan/Desktop/OpenCL/Final/amc/amc/";
const string    TRAIN_FILE  = "Train.txt";
const string    TEST_FILE   = "";
const char *    kernel_NAME = "vecadd";

int     *out = NULL;
int     *items[NGRAM];
string  news_type;
size_t  globalWorkSize[1];
bool    trainingDone        = false;

unordered_map<char,     vector<int>>    ITEM_MEMORY;        //map with alphabets as key and vector of size d as value
unordered_map<string,   vector<int>>    ASSOCIATIVE_MEMORY;
unordered_map<string,   vector<int>>    TESTING_DATA_MAP;

/** OpenCL parameter initializations -- BEGIN **/
/**/ cl_int status;
/**/ cl_uint numPlatforms = 0;
/**/ cl_platform_id *platforms = NULL;
/**/ cl_uint numDevices = 0;
/**/ cl_device_id *devices = NULL;
/**/ cl_context context = NULL;
/**/ cl_command_queue cmdQueue;
/**/ cl_mem buffer[NGRAM];
/**/ cl_mem bufferOut;
/**/ cl_program program;
/**/ cl_kernel kernel = NULL;
/** OpenCL parameter initializations -- END **/


/** Function Declarations -- BEGIN **/
void generateItemMemory();
int myrandom(int i);
void initializeAssociativeMemory();
void initializeTestMap();
void kernelInitialize();
void processFile(string filename);
vector<string> ReadFile(string filename);
void loadkernelBuffer_Ngram(string data);
void loadkernelBuffer_space(string str);
void writeDataIntoBuffer();
void readWriteOut();
void resetOut();
void runkernel();
void addToAssociativeMemory();
void addToTestMap();
void releaseMemory();
/** Function Declarations -- END **/


int main()
{
    kernelInitialize();
    generateItemMemory();
    
    processFile(BASE_DIR + TRAIN_FILE);
    cout << "ASSOCIATIVE_MEMORY[earn] Size: "<<ASSOCIATIVE_MEMORY["earn"].size()<< "\n";
    for (std::vector<int>::iterator it = ASSOCIATIVE_MEMORY["earn"].begin(); it != ASSOCIATIVE_MEMORY["earn"].end(); ++it)
        cout << '\t' << *it;
    cout << '\n';
    
  //  trainingDone = true;
  //  processFile(TEST_FILE);
    
    
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
    if(ASSOCIATIVE_MEMORY["earn"].size() == 0)
        for(int i = 0; i < DIMEN; i++)
            ASSOCIATIVE_MEMORY[news_type].push_back(0);
        
}

/********************************************************************************************************************/
void initializeTestMap()
{
    for(int i = 0; i < DIMEN; i++)
    {
        TESTING_DATA_MAP[news_type].push_back(0);
    }
}

/********************************************************************************************************************/
void kernelInitialize()
{
    out = (int*)malloc(DATASIZE);
    for(int i = 0; i<NGRAM; i++)
    {
    items[i] = NULL;
    items[i] = (int*)malloc(DATASIZE);
        for(int j=0; j<DIMEN; j++){
            items[i][j] = 0;
            out[j] = 0;
        }
    }
    
    cout<<"Out Size = "<<sizeof(out)<<"\n";
    cout<<"items size "<<sizeof(items)<<"\n";
    
    
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
        buffer[i] = clCreateBuffer(
                                context,
                                CL_MEM_READ_ONLY,
                                DATASIZE,
                                NULL,
                                &status);
    }
    // Create a buffer object (d_C) with enough space to hold the output data
    bufferOut = clCreateBuffer(context, CL_MEM_READ_WRITE, DATASIZE, NULL, &status);
    
    
    //*STEP 6: Write host data to input and output buffers
    for(int i=0; i<NGRAM; i++)
    {
        status = clEnqueueWriteBuffer(cmdQueue,
                                      buffer[i],
                                      CL_FALSE,
                                      0,
                                      DATASIZE,
                                      items[i],
                                      0,
                                      NULL,
                                      NULL);
    }
    status = clEnqueueWriteBuffer(cmdQueue, bufferOut, CL_FALSE, 0, DATASIZE, out, 0, NULL, NULL);
    
    
    //*STEP 7: Create and compile the program
    // Create a program using clCreateProgramWithSource()
    program = clCreateProgramWithSource(context, 1, (const char**)&programSource, NULL, &status);
    // Build (compile) the program for the devices with clBuildProgram()
    status = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);
    
    //*STEP 8: Create the kernel
    // Use clCreateKernel() to create a kernel from the
    // vector addition function (named "vecadd")
    kernel = clCreateKernel(program, "vecadd", &status);
    
    // STEP 9: Set the kernel arguments
    // Associate the input and output buffers with the kernel
    for(int i=0; i<NGRAM; i++){
    status = clSetKernelArg(kernel,
                            i,                          //Argument position
                            sizeof(cl_mem),
                            &buffer[i]);                    //TODO: change input based on kernel
    }
    status = clSetKernelArg(kernel,
                            NGRAM,                          //Argument position
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
        
        if(trainingDone)
            initializeTestMap();
        else
            initializeAssociativeMemory();
        
        cout<<"Value --"<<strWords[1]<<"\n\n";
        
#ifdef SPLIT_BY_SPACE
        loadkernelBuffer_space(strWords[1]);
#else
        loadkernelBuffer_Ngram(strWords[1]);
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
void loadkernelBuffer_Ngram(string data)
{
    cout<<"Method: LoadKernalBuffer\n";
    cout <<"Data length : "<<data.length()<<"\n";
    for(int i = 0; i < data.length()-NGRAM; i++)
    {
        for(int j=0; j < NGRAM; j++)
        {
            cout <<"Char:"<<data.at(i+j)<<"\t";
            for(int k=0; k < DIMEN; k++)
            {
                items[j][k]=ITEM_MEMORY[data.at(i+j)][k];
                cout<<items[j][k]<<"\t";
            }
            cout << "\n";
        }
        cout << "-----------------------------\n";
        //Load kernel arguments
        writeDataIntoBuffer();
        
        //Run kernel
        runkernel();
        readWriteOut();
      //  addToAssociativeMemory();
    }
    cout<<"\nDone\n\n";
    //get output from kernel and put it to associative mem
    if(trainingDone)
        addToTestMap();
    else
        addToAssociativeMemory();
    resetOut();
}

/********************************************************************************************************************/
void loadkernelBuffer_space(string str)         //TODO: Need to change this
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
    for(int i=0; i<NGRAM; i++)
    {
        status = clEnqueueWriteBuffer(cmdQueue,
                                      buffer[i],
                                      CL_FALSE,
                                      0,
                                      DATASIZE,
                                      items[i],
                                      0,
                                      NULL,
                                      NULL);
    }
}

/********************************************************************************************************************/
void runkernel()
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
void readWriteOut(){
    clEnqueueReadBuffer(cmdQueue,
                        bufferOut,
                        CL_TRUE,
                        0,
                        DATASIZE,
                        out,
                        0,
                        NULL,
                        NULL);
    status = clEnqueueWriteBuffer(cmdQueue, bufferOut, CL_FALSE, 0, DATASIZE, out, 0, NULL, NULL);
}

/********************************************************************************************************************/
void resetOut(){
    for(int i=0; i<DIMEN; i++){
        out[i] = 0;
    }
    status = clEnqueueWriteBuffer(cmdQueue, bufferOut, CL_FALSE, 0, DATASIZE, out, 0, NULL, NULL);
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
    for(int i=0; i<DIMEN; i++){
        cout<<"Pre_AM "<<ASSOCIATIVE_MEMORY[news_type][i]<<"\t";
    }
    cout<<"\n";
    for(int i=0; i < DIMEN; i++)
    {
        ASSOCIATIVE_MEMORY[news_type][i] += out[i];
        cout<<"out    "<<out[i]<<"\t";
    }
    cout<<"\n";
    for(int i=0; i<DIMEN; i++){
        cout<<"AM     "<<ASSOCIATIVE_MEMORY[news_type][i]<<"\t";
    }
    cout<<"\n\n";
    
}

/********************************************************************************************************************/
void addToTestMap(){
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
        TESTING_DATA_MAP[news_type][i] += out[i];
    }
}


/********************************************************************************************************************/
void releaseMemory()
{
    // Free OpenCL resources
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(cmdQueue);
    for(int i=0; i<NGRAM; i++)
        clReleaseMemObject(buffer[i]);
    clReleaseMemObject(bufferOut);
    clReleaseContext(context);
    
    // Free host resources
    free(out);
    free(items);
    free(platforms);
    free(devices);
}





