/*
 * image convolution code based on AMD examples, NVIDIA SimpleGL example and Andy Johnson's colliding galaxies example
 *
 */

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// OpenGL Graphics Includes
#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
    #include <OpenGL/OpenGL.h>
    #include <GLUT/glut.h>
#else
    #include <GL/freeglut.h>
    #ifdef UNIX
       #include <GL/glx.h>
    #endif
#endif

// Includes
#include <memory>
#include <iostream>
#include <cassert>

// Utilities, OpenCL and system includes
#include <oclUtils.h>
#include <shrQATest.h>

#if defined (__APPLE__) || defined(MACOSX)
   #define GL_SHARING_EXTENSION "cl_APPLE_gl_sharing"
#else
   #define GL_SHARING_EXTENSION "cl_khr_gl_sharing"
#endif


//My additions
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>

#include <stdlib.h>
#include <assert.h>
#include <png.h> 
#include <cstdio>
#include <string>

#define TEXTURE_LOAD_ERROR 0

using namespace std;

// Constants, defines, typedefs and global declarations
//*****************************************************************************
#define REFRESH_DELAY	  10 //ms

// Rendering window vars
const unsigned int window_width = 1024;
const unsigned int window_height = 768;

int whichKernel;
png_byte *image_data;

#define BLOCK_DIM 16

// OpenCL vars
cl_platform_id cpPlatform;
cl_context cxGPUContext;
cl_device_id* cdDevices;
cl_uint uiDevCount;
cl_command_queue cqCommandQueue;
cl_kernel ckKernel;
cl_program cpProgram;
cl_int ciErrNum;
char* cPathAndName = NULL;          // var for full paths to data, src, etc.
char* cSourceCL = NULL;             // Buffer to hold source for compilation 
const char* cExecutableName = NULL;

//cl_mem data variables

//image data as texture
cl_mem cl_inputImage;
cl_sampler cl_inputSampler;
cl_mem cl_outputImage;
cl_sampler cl_outputSampler;

//image data as buffer of floats
cl_mem cl_inputDataBuffer;

cl_mem cl_filter;

cl_mem cl_imageWidth;
cl_mem cl_imageHeight;
cl_mem cl_filterWidth;

int iGLUTWindowHandle = 0;          // handle to the GLUT window

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;

// Sim and Auto-Verification parameters 
shrBOOL bQATest = shrFALSE;
shrBOOL bNoPrompt = shrFALSE;  

int *pArgc = NULL;
char **pArgv = NULL;

int computationHeight = 1;  //1d data list- so can change kernel dimensionality

string* imageFile; 

//Data
unsigned int anim = 0;//counter 
int imageWidth;
int* constImageWidth;
int imageHeight;
int* constImageHeight;
int filterWidth;
int* constFilterWidth;
GLuint inputImage;//input image texture
GLuint outputImage;//output image texture

//GLuint inputImageBuffer;
//GLuint outputImageBuffer;
float* inputImageDataBuffer;//input image as a buffer of floats
float* outputImageDataBuffer;//output image as a buffer of floats
float* filter;

//camera control
float minEyeX, minEyeY, minEyeZ, maxEyeX, maxEyeY, maxEyeZ, eyeX, eyeY, eyeZ;
int minX = 16;
int maxX = 4096+16;
int minY = 16;
int maxY = 2258+16;
float deltaX = 0.0;
float deltaY = 0.0;
float deltaZ = 0.0;

// Forward Function declarations
//*****************************************************************************
// OpenCL functionality
void runKernel();

// GL functionality
void InitGL(int* argc, char** argv);
void DisplayGL();
void KeyboardGL(unsigned char key, int x, int y);
void processSpecialKeys( int key, int x, int y );
void mouse(int button, int state, int x, int y);
void mouseWheel(int button, int dir, int x, int y);
void motion(int x, int y);
void timerEvent(int value);

//data
void initFilter( int filterWidth );
void createImageDataBuffers(); //called if running in mode = 1, otherwise use textures 

//loadTexture
GLuint loadTexture(const string filename, int &width, int &height);

// Helpers
void Cleanup(int iExitCode);
void (*pCleanup)(int) = &Cleanup;
void framesPerSecond();

////////////////////////////////////////////////////////////////


// Main program
//*****************************************************************************
int main( int argc, char** argv)
{
	pArgc = &argc;
	pArgv = argv;
    
    if( argc < 1+5 )//if no dimensions specified or image input file
    {
        printf("too few arguments.  exiting");
        Cleanup(EXIT_FAILURE); 
    }
    else if( argc == 6 )//right number of arguments 
    {
        printf("processing command line input" );
        imageFile = new string(argv[1]);//"test.png"); //later arg[1]
        imageWidth = atoi( argv[2] );//512; //
        imageHeight = atoi( argv[3] );//512; //
        filterWidth = atoi( argv[4] );//5;//3;//
        whichKernel = atoi( argv[5] );//2; //
        
        if( whichKernel == 3 )
            whichKernel = 4;
        if( whichKernel == 5 )
            whichKernel = 4;
    }
    else
    {
        printf("too few arguments.  exiting");
        Cleanup(EXIT_FAILURE); 
    }
        
    // start logs 
    shrQAStart(argc, argv);
	cExecutableName = argv[0];
    shrSetLogFileName ("convolve.txt");
    shrLog("%s Starting...\n\n", argv[0]); 
    
    // Initialize OpenGL items (if not No-GL QA test)
	shrLog("%sInitGL...\n\n", bQATest ? "Skipping " : "Calling "); 
    if(!bQATest)
    {
        InitGL(&argc, argv);
    }
    
    //Load and create textures
    inputImage = loadTexture(imageFile->c_str(), imageWidth, imageHeight);
    outputImage = loadTexture(imageFile->c_str(), imageWidth, imageHeight); //don't really need to do this, but want to properly init output image
    
    //----------- OPENCL STUFF------------//
    std::cout << "opencl stuff " << std::endl;

    //Get the NVIDIA platform-- can be non-nvidia
    ciErrNum = oclGetPlatformID(&cpPlatform);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    
    // Get the number of GPU devices available to the platform
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 0, NULL, &uiDevCount);//will return back one or more gpus
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    //typically want gpu not integrated graphics.  
    
    // Create the device list
    cdDevices = new cl_device_id [uiDevCount];
    ciErrNum = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, uiDevCount, cdDevices, NULL);
    oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    
    // Get device requested on command line, if any
    unsigned int uiDeviceUsed = 0;
    unsigned int uiEndDev = uiDevCount - 1;
    if(shrGetCmdLineArgumentu(argc, (const char**)argv, "device", &uiDeviceUsed ))
    {
        uiDeviceUsed = CLAMP(uiDeviceUsed, 0, uiEndDev);
        uiEndDev = uiDeviceUsed; 
    } 
    
    // Check if the requested device (or any of the devices if none requested) supports context sharing with OpenGL
    if(!bQATest)
    {
        bool bSharingSupported = false;
        for(unsigned int i = uiDeviceUsed; (!bSharingSupported && (i <= uiEndDev)); ++i) 
        {
            size_t extensionSize;
            ciErrNum = clGetDeviceInfo(cdDevices[i], CL_DEVICE_EXTENSIONS, 0, NULL, &extensionSize );
            oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
            if(extensionSize > 0) 
            {
                char* extensions = (char*)malloc(extensionSize);
                ciErrNum = clGetDeviceInfo(cdDevices[i], CL_DEVICE_EXTENSIONS, extensionSize, extensions, &extensionSize);
                oclCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
                std::string stdDevString(extensions);
                free(extensions);
                
                size_t szOldPos = 0;
                size_t szSpacePos = stdDevString.find(' ', szOldPos); // extensions string is space delimited
                while (szSpacePos != stdDevString.npos)
                {
                    if( strcmp(GL_SHARING_EXTENSION, stdDevString.substr(szOldPos, szSpacePos - szOldPos).c_str()) == 0 ) 
                    {
                        // Device supports context sharing with OpenGL
                        uiDeviceUsed = i;
                        bSharingSupported = true;
                        break;
                    }
                    do 
                    {
                        szOldPos = szSpacePos + 1;
                        szSpacePos = stdDevString.find(' ', szOldPos);
                    } 
                    while (szSpacePos == szOldPos);
                }
            }
        }
        
        shrLog("%s...\n\n", bSharingSupported ? "Using CL-GL Interop" : "No device found that supports CL/GL context sharing");  
        oclCheckErrorEX(bSharingSupported, true, pCleanup);
        
        // Define OS-specific context properties and create the OpenCL context
#if defined (__APPLE__)
        CGLContextObj kCGLContext = CGLGetCurrentContext();
        CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);
        cl_context_properties props[] = 
        {
            CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE, (cl_context_properties)kCGLShareGroup, 
            0 
        };
        cxGPUContext = clCreateContext(props, 0,0, NULL, NULL, &ciErrNum);
#else
#ifdef UNIX
        cl_context_properties props[] = 
        {
            CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(), 
            CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(), 
            CL_CONTEXT_PLATFORM, (cl_context_properties)cpPlatform, 
            0
        };
        cxGPUContext = clCreateContext(props, 1, &cdDevices[uiDeviceUsed], NULL, NULL, &ciErrNum);
#else // Win32
        cl_context_properties props[] = 
        {
            CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(), 
            CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(), 
            CL_CONTEXT_PLATFORM, (cl_context_properties)cpPlatform, 
            0
        };
        cxGPUContext = clCreateContext(props, 1, &cdDevices[uiDeviceUsed], NULL, NULL, &ciErrNum);
#endif
#endif
    }
    else 
    {
        cl_context_properties props[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)cpPlatform, 0};
        cxGPUContext = clCreateContext(props, 1, &cdDevices[uiDeviceUsed], NULL, NULL, &ciErrNum);
    }
    
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);//if problem , bug out
    
    // Log device used (reconciled for requested requested and/or CL-GL interop capable devices, as applies)
    shrLog("Device # %u, ", uiDeviceUsed);
    oclPrintDevName(LOGBOTH, cdDevices[uiDeviceUsed]); // expect to get GPU
    shrLog("\n");
    
    // create a command-queue use 0 for no profiling or CL_QUEUE_PROFILING_ENABLE to enable profiling
    cqCommandQueue = clCreateCommandQueue(cxGPUContext, cdDevices[uiDeviceUsed], CL_QUEUE_PROFILING_ENABLE, &ciErrNum);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    std::cout << "done with opencl stuff" << std::endl;
    //----------- END OPENCL STUFF------------//
    
    //------------KERNEL and PROG SETUP-------//
    std::cout << "kernel and prog setup" << std::endl;
    //initialize args for kernel
    cl_inputImage = clCreateFromGLTexture2D( cxGPUContext,  CL_MEM_READ_ONLY, GL_TEXTURE_2D, 0, inputImage, &ciErrNum );  
        //clCreateImage2D( context, 0, &format, texWidth, texHeight, 0, NULL, NULL );
    oclCheckError(ciErrNum, CL_SUCCESS);
    
    cl_outputImage = clCreateFromGLTexture2D( cxGPUContext,  CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, outputImage, &ciErrNum );  
        //clCreateImage2D( context, 0, &format, texWidth, texHeight, 0, NULL, NULL );
    oclCheckError(ciErrNum, CL_SUCCESS);
    
    cl_inputSampler = clCreateSampler( cxGPUContext, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);

    cl_outputSampler = clCreateSampler( cxGPUContext, CL_FALSE, CL_ADDRESS_CLAMP_TO_EDGE, CL_FILTER_NEAREST, &ciErrNum);
    oclCheckError(ciErrNum, CL_SUCCESS);
    
    //setup filter 
    filter = (float *) malloc (filterWidth*filterWidth * sizeof(float));
    initFilter( filterWidth );
    
    cl_filter = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                 (filterWidth*filterWidth) * sizeof(float), filter, &ciErrNum);//gives back ptr to buffer
    oclCheckError(ciErrNum, CL_SUCCESS);
    
    
    //setup input image buffer- for non-texture kernels 
    inputImageDataBuffer = (float *) malloc (imageWidth*imageHeight*4*sizeof(float));
    for(int i = 0; i < imageWidth*imageHeight*4; i++)
    {
        inputImageDataBuffer[i] = 0.0; //init to 0
    }
    
    cl_inputDataBuffer = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                               (imageWidth*imageHeight*4) * sizeof(float), inputImageDataBuffer, &ciErrNum);//gives back ptr to buffer
    oclCheckError(ciErrNum, CL_SUCCESS);

    
    constImageWidth = (int*)malloc( 1*sizeof(int) );
    constImageWidth[0] = imageWidth;
    cl_imageWidth = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                               1 * sizeof(int), constImageWidth, &ciErrNum);//gives back ptr to buffer
    oclCheckError(ciErrNum, CL_SUCCESS);
    
    constImageHeight = (int*)malloc( 1*sizeof(int) );
    constImageHeight[0] = imageHeight;
    cl_imageHeight = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   1 * sizeof(int), constImageHeight, &ciErrNum);//gives back ptr to buffer
    oclCheckError(ciErrNum, CL_SUCCESS);
    
    constFilterWidth = (int*)malloc( 1*sizeof(int) );
    constFilterWidth[0] = filterWidth;
    cl_filterWidth = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    1 * sizeof(int), constFilterWidth, &ciErrNum);//gives back ptr to buffer
    oclCheckError(ciErrNum, CL_SUCCESS);
//    
//    if(!bQATest)
//    {    
//        //load and create input data buffer
//        glGenBuffers(1, &inputImageBuffer);
//        glBindBuffer(GL_ARRAY_BUFFER, inputImageBuffer);
//        glBufferData(GL_ARRAY_BUFFER, imageWidth*imageHeight*4,  0, GL_DYNAMIC_DRAW);
//                //(GLvoid*) image_data, GL_DYNAMIC_DRAW);
//    
//        //setup image as buffer
//#ifdef GL_INTEROP
//        // create OpenCL buffer from GL VBO
//        cl_inputDataBuffer = clCreateFromGLBuffer(cxGPUContext, CL_MEM_READ_WRITE, *inputImageBuffer, NULL);
//#else
//        // create standard OpenCL mem buffer
//        cl_inputDataBuffer = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, imageWidth*imageHeight*4, NULL, &ciErrNum);
//#endif
//        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
//    }
//    else    
//    {
//        // create standard OpenCL mem buffer
//        cl_inputDataBuffer = clCreateBuffer(cxGPUContext, CL_MEM_READ_WRITE, imageWidth*imageHeight*4,  NULL, &ciErrNum);
//        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
//    }


    // Program Setup
    size_t program_length;
    cPathAndName = shrFindFilePath("p4.cl", argv[0]);// get kernel file
    shrCheckErrorEX(cPathAndName != NULL, shrTRUE, pCleanup);
    cSourceCL = oclLoadProgSource(cPathAndName, "", &program_length);
    shrCheckErrorEX(cSourceCL != NULL, shrTRUE, pCleanup);
    
    // create the program
    cpProgram = clCreateProgramWithSource(cxGPUContext, 1,
                                          (const char **) &cSourceCL, &program_length, &ciErrNum);
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    
    // build the program
    ciErrNum = clBuildProgram(cpProgram, 0, NULL, NULL, NULL, NULL);
    if (ciErrNum != CL_SUCCESS)
    {
        // write out standard error, Build Log and PTX, then cleanup and exit
        shrLogEx(LOGBOTH | ERRORMSG, ciErrNum, STDERROR);
        oclLogBuildInfo(cpProgram, oclGetFirstDev(cxGPUContext));
        oclLogPtx(cpProgram, oclGetFirstDev(cxGPUContext), "convolve.ptx");
        Cleanup(EXIT_FAILURE); 
    }
    
    // create the kernel
    if( whichKernel == 0 )
        ckKernel = clCreateKernel(cpProgram, "convolve", &ciErrNum);
    else if( whichKernel == 1 )
        ckKernel = clCreateKernel(cpProgram, "convolveGloballMem", &ciErrNum);
    else if( whichKernel == 2 )
        ckKernel = clCreateKernel(cpProgram, "convolveConstant", &ciErrNum);
    else if( whichKernel == 3 )
        ckKernel = clCreateKernel(cpProgram, "convolveGloballMemConstant", &ciErrNum);
    else if( whichKernel == 4 )
        ckKernel = clCreateKernel(cpProgram, "convolveConstantLocal", &ciErrNum);
    else if( whichKernel == 5 )
        ckKernel = clCreateKernel(cpProgram, "convolveGloballMemLocal", &ciErrNum);
    
        
        
    if(ciErrNum == CL_INVALID_PROGRAM_EXECUTABLE )
        printf("INVALID" );
    //shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    
    // if there is a problem creating the kernel print out some detailed information about it
    if (ciErrNum) {
   		char log[10240] = "";
   		ciErrNum = clGetProgramBuildInfo(cpProgram, cdDevices[uiDeviceUsed], CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
   		fprintf(stderr, "Error(s) creating the kernel:\n%s\n", log);
    }
	shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    
	int computationWidth = (imageWidth*imageHeight)/computationHeight;
    
    computationWidth = (unsigned int) imageWidth;
    computationHeight = (unsigned int) imageHeight; 
    
    // set the args values  for kernel-- will get any error- note, order and type must matter- still one param left
    if( whichKernel == 0 )//convolve
    {
        ciErrNum |= clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void *) &cl_inputImage); //ptr to mem buf on card    
        ciErrNum |= clSetKernelArg(ckKernel, 1, sizeof(cl_mem), (void *) &cl_outputImage); //ptr to mem buf on card
        ciErrNum |= clSetKernelArg(ckKernel, 2, sizeof(cl_sampler), (void *) &cl_inputSampler); //ptr to mem buf on card   
        ciErrNum |= clSetKernelArg(ckKernel, 3, sizeof(cl_sampler), (void *) &cl_outputSampler); //ptr to mem buf on card    
        ciErrNum |= clSetKernelArg(ckKernel, 4, sizeof(int), (void *) &imageWidth);
        ciErrNum |= clSetKernelArg(ckKernel, 5, sizeof(int), (void *) &imageHeight);
        ciErrNum |= clSetKernelArg(ckKernel, 6, sizeof(cl_mem), (void *) &cl_filter); //ptr to mem buf on card
        ciErrNum |= clSetKernelArg(ckKernel, 7, sizeof(int), (void *) &filterWidth);    
        ciErrNum |= clSetKernelArg(ckKernel, 8, sizeof(unsigned int), (void *) &computationWidth);
        ciErrNum |= clSetKernelArg(ckKernel, 9, sizeof(unsigned int), (void *) &computationHeight);
    } 
    else if( whichKernel == 1 )//convolveGlobalMem
    {
        ciErrNum |= clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void *) &cl_inputDataBuffer); //ptr to mem buf on card    
        ciErrNum |= clSetKernelArg(ckKernel, 1, sizeof(cl_mem), (void *) &cl_inputImage); //ptr to mem buf on card    
        ciErrNum |= clSetKernelArg(ckKernel, 2, sizeof(cl_mem), (void *) &cl_outputImage); //ptr to mem buf on card
        ciErrNum |= clSetKernelArg(ckKernel, 3, sizeof(cl_sampler), (void *) &cl_inputSampler); //ptr to mem buf on card   
        ciErrNum |= clSetKernelArg(ckKernel, 4, sizeof(cl_sampler), (void *) &cl_outputSampler); //ptr to mem buf on card    
        ciErrNum |= clSetKernelArg(ckKernel, 5, sizeof(int), (void *) &imageWidth);
        ciErrNum |= clSetKernelArg(ckKernel, 6, sizeof(int), (void *) &imageHeight);
        ciErrNum |= clSetKernelArg(ckKernel, 7, sizeof(cl_mem), (void *) &cl_filter); //ptr to mem buf on card
        ciErrNum |= clSetKernelArg(ckKernel, 8, sizeof(int), (void *) &filterWidth);    
        ciErrNum |= clSetKernelArg(ckKernel, 9, sizeof(unsigned int), (void *) &computationWidth);
        ciErrNum |= clSetKernelArg(ckKernel, 10, sizeof(unsigned int), (void *) &computationHeight);
    }
    else if( whichKernel == 2 )//convolveConstant
    {
        ciErrNum |= clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void *) &cl_inputImage); //ptr to mem buf on card    
        ciErrNum |= clSetKernelArg(ckKernel, 1, sizeof(cl_mem), (void *) &cl_outputImage); //ptr to mem buf on card
        ciErrNum |= clSetKernelArg(ckKernel, 2, sizeof(cl_sampler), (void *) &cl_inputSampler); //ptr to mem buf on card   
        ciErrNum |= clSetKernelArg(ckKernel, 3, sizeof(cl_sampler), (void *) &cl_outputSampler); //ptr to mem buf on card    
        ciErrNum |= clSetKernelArg(ckKernel, 4, sizeof(cl_mem), (void *) &cl_imageWidth);
        ciErrNum |= clSetKernelArg(ckKernel, 5, sizeof(cl_mem), (void *) &cl_imageHeight);
        ciErrNum |= clSetKernelArg(ckKernel, 6, sizeof(cl_mem), (void *) &cl_filter); //ptr to mem buf on card
        ciErrNum |= clSetKernelArg(ckKernel, 7, sizeof(cl_mem), (void *) &cl_filterWidth);    
        ciErrNum |= clSetKernelArg(ckKernel, 8, sizeof(unsigned int), (void *) &computationWidth);
        ciErrNum |= clSetKernelArg(ckKernel, 9, sizeof(unsigned int), (void *) &computationHeight);
    }
    else if( whichKernel == 3 )//convolveGloballMemConstant  -- does not work
    {        
//        ciErrNum |= clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void *) &cl_inputDataBuffer); //ptr to mem buf on card    
//        ciErrNum |= clSetKernelArg(ckKernel, 1, sizeof(cl_mem), (void *) &cl_inputImage); //ptr to mem buf on card    
//        ciErrNum |= clSetKernelArg(ckKernel, 2, sizeof(cl_mem), (void *) &cl_outputImage); //ptr to mem buf on card
//        ciErrNum |= clSetKernelArg(ckKernel, 3, sizeof(cl_sampler), (void *) &cl_inputSampler); //ptr to mem buf on card   
//        ciErrNum |= clSetKernelArg(ckKernel, 4, sizeof(cl_sampler), (void *) &cl_outputSampler); //ptr to mem buf on card    
//        ciErrNum |= clSetKernelArg(ckKernel, 5, sizeof(cl_mem), (void *) &cl_imageWidth);
//        ciErrNum |= clSetKernelArg(ckKernel, 6, sizeof(cl_mem), (void *) &cl_imageHeight);
//        ciErrNum |= clSetKernelArg(ckKernel, 7, sizeof(cl_mem), (void *) &cl_filter); //ptr to mem buf on card
//        ciErrNum |= clSetKernelArg(ckKernel, 8, sizeof(cl_mem), (void *) &cl_filterWidth);     
//        ciErrNum |= clSetKernelArg(ckKernel, 9, sizeof(unsigned int), (void *) &computationWidth);
//        ciErrNum |= clSetKernelArg(ckKernel, 10, sizeof(unsigned int), (void *) &computationHeight);
        
        
        ciErrNum |= clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void *) &cl_inputDataBuffer); //ptr to mem buf on card    
        ciErrNum |= clSetKernelArg(ckKernel, 1, sizeof(cl_mem), (void *) &cl_inputImage); //ptr to mem buf on card    
        ciErrNum |= clSetKernelArg(ckKernel, 2, sizeof(cl_mem), (void *) &cl_outputImage); //ptr to mem buf on card
        ciErrNum |= clSetKernelArg(ckKernel, 3, sizeof(cl_sampler), (void *) &cl_inputSampler); //ptr to mem buf on card   
        ciErrNum |= clSetKernelArg(ckKernel, 4, sizeof(cl_sampler), (void *) &cl_outputSampler); //ptr to mem buf on card    
        ciErrNum |= clSetKernelArg(ckKernel, 5, sizeof(cl_mem), (void *) &cl_imageWidth);
        ciErrNum |= clSetKernelArg(ckKernel, 6, sizeof(cl_mem), (void *) &cl_imageHeight);
        ciErrNum |= clSetKernelArg(ckKernel, 7, sizeof(cl_mem), (void *) &cl_filter); //ptr to mem buf on card
        ciErrNum |= clSetKernelArg(ckKernel, 8, sizeof(cl_mem), (void *) &cl_filterWidth);    
        ciErrNum |= clSetKernelArg(ckKernel, 9, sizeof(unsigned int), (void *) &computationWidth);
        ciErrNum |= clSetKernelArg(ckKernel, 10, sizeof(unsigned int), (void *) &computationHeight);
    }
    else if( whichKernel == 4 )//convolveConstantLocal
    {
        ciErrNum |= clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void *) &cl_inputImage); //ptr to mem buf on card    
        ciErrNum |= clSetKernelArg(ckKernel, 1, sizeof(cl_mem), (void *) &cl_outputImage); //ptr to mem buf on card
        ciErrNum |= clSetKernelArg(ckKernel, 2, sizeof(cl_sampler), (void *) &cl_inputSampler); //ptr to mem buf on card   
        ciErrNum |= clSetKernelArg(ckKernel, 3, sizeof(cl_sampler), (void *) &cl_outputSampler); //ptr to mem buf on card    
        ciErrNum |= clSetKernelArg(ckKernel, 4, sizeof(cl_mem), (void *) &cl_imageWidth);
        ciErrNum |= clSetKernelArg(ckKernel, 5, sizeof(cl_mem), (void *) &cl_imageHeight);
        ciErrNum |= clSetKernelArg(ckKernel, 6, sizeof(cl_mem), (void *) &cl_filter); //ptr to mem buf on card
        ciErrNum |= clSetKernelArg(ckKernel, 7, sizeof(cl_mem), (void *) &cl_filterWidth);    
        ciErrNum |= clSetKernelArg(ckKernel, 8, sizeof(unsigned int), (void *) &computationWidth);
        ciErrNum |= clSetKernelArg(ckKernel, 9, sizeof(unsigned int), (void *) &computationHeight);
        //ciErrNum |= clSetKernelArg(ckKernel, 10, sizeof(unsigned int), (void *) &computationHeight);//block dim
    }
    else if( whichKernel == 5 )//convolveGlobalMemLocal
    {
        ciErrNum |= clSetKernelArg(ckKernel, 0, sizeof(cl_mem), (void *) &cl_inputDataBuffer); //ptr to mem buf on card    
        ciErrNum |= clSetKernelArg(ckKernel, 1, sizeof(cl_mem), (void *) &cl_inputImage); //ptr to mem buf on card    
        ciErrNum |= clSetKernelArg(ckKernel, 2, sizeof(cl_mem), (void *) &cl_outputImage); //ptr to mem buf on card
        ciErrNum |= clSetKernelArg(ckKernel, 3, sizeof(cl_sampler), (void *) &cl_inputSampler); //ptr to mem buf on card   
        ciErrNum |= clSetKernelArg(ckKernel, 4, sizeof(cl_sampler), (void *) &cl_outputSampler); //ptr to mem buf on card    
        ciErrNum |= clSetKernelArg(ckKernel, 5, sizeof(int), (void *) &imageWidth);
        ciErrNum |= clSetKernelArg(ckKernel, 6, sizeof(int), (void *) &imageHeight);
        ciErrNum |= clSetKernelArg(ckKernel, 7, sizeof(cl_mem), (void *) &cl_filter); //ptr to mem buf on card
        ciErrNum |= clSetKernelArg(ckKernel, 8, sizeof(int), (void *) &filterWidth);    
        ciErrNum |= clSetKernelArg(ckKernel, 9, sizeof(unsigned int), (void *) &computationWidth);
        ciErrNum |= clSetKernelArg(ckKernel, 10, sizeof(unsigned int), (void *) &computationHeight);
    }
    
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    
    std::cout << "done with kernal and prog setup" << std::endl;

    //so kernel not yet launched
    
    // init timer 1 for fps measurement 
    shrDeltaT(1);  
    
    // Start main GLUT rendering loop for processing and rendering, 
	// or otherwise run No-GL Q/A test sequence
    shrLog("\n%s...\n", bQATest ? "No-GL test sequence" : "Standard GL Loop"); 
    if(!bQATest) 
    {
        glutMainLoop();
    }
    
    // Normally unused return path
    Cleanup(EXIT_SUCCESS);
    
}

void createImageDataBuffers()
{
//    // create VBO
//    unsigned int size = sizeof( float) * 4 * imageWidth * imageHeight; 
//
//        
//    if(!bQATest)
//    {
//            // create buffer object
//        glGenBuffers(1, vbo);
//        glBindBuffer(GL_ARRAY_BUFFER, *vbo);
//            
//        // initialize buffer object
//        glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
//            
//#ifdef GL_INTEROP
//        // create OpenCL buffer from GL VBO
//        vbo_cl = clCreateFromGLBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, *vbo, NULL);
//#else
//        // create standard OpenCL mem buffer
//        vbo_cl = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, size, NULL, &ciErrNum);
//#endif
//        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
//    }
//    else 
//    {
//        // create standard OpenCL mem buffer
//        vbo_cl = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY, size, NULL, &ciErrNum);
//        shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
//    }
    
}

void initFilter( int filterWidth )
{
      
    //edge detect filter- 3x3
    if( filterWidth == 3 )
    {
        filter[0] = 0.0;
        filter[1] = 1.0;
        filter[2] = 0.0;
        filter[3] = 1.0;
        filter[4] = -4.0;
        filter[5] = 1.0;
        filter[6] = 0.0;
        filter[7] = 1.0;
        filter[8] = 0.0;
    }
    
//    for(int i = 0; i < 5; i++)
//    {
//        for(int j = 0; j < 5; j++)
//        {
//            if( i == 0 || i == 4 || j == 0 || j == 4)
//                filter[i*5+j] = 0;
//            else
//                filter[i*5+j] = 1;
//            
//        }
//    }
    
    if( filterWidth == 5 )
    {
        
        
        for(int i = 0; i < 5; i++)
        {
            for(int j = 0; j < 5; j++)
            {
                filter[i*5+j] = -1.0;            
            }
        }
        
        filter[2*5 + 2] = 24.0;  
        
//        for(int i = 0; i < 5; i++)
//        {
//            for(int j = 0; j < 5; j++)
//            {
//                    filter[i*5+j] = 0.0;            
//            }
//        }
//            
//        filter[0*5 + 2] = -1.0;
//        filter[1*5 + 1] = -1.0;
//        filter[1*5 + 2] = -2.0;
//        filter[1*5 + 3] = -1.0;
//        filter[2*5 + 0] = -1.0;
//        filter[2*5 + 1] = -2.0;
//        filter[2*5 + 2] = 16.0;
//        filter[2*5 + 3] = -2.0;
//        filter[2*5 + 4] = -1.0;
//        filter[3*5 + 1] = -1.0;
//        filter[3*5 + 2] = -2.0;
//        filter[3*5 + 3] = -1.0;
//        filter[4*5 + 2] = -1.0;
    }
    
    if( filterWidth == 7 )
    {
        
        for(int i = 0; i < 7; i++)
        {
            for(int j = 0; j < 7; j++)
            {
                filter[i*7+j] = -1.0;            
            }
        }
        
        filter[3*7 + 3] = 48.0;      
    }
    
    
}


// Initialize GL
//*****************************************************************************
void InitGL(int* argc, char** argv)//set up gl
{
    
    eyeX = (float)(maxX-minX)/2;
    eyeY = (float)(maxY-minY)/2;
    eyeZ = (float)2700.0;//750.0;
    minEyeX = -eyeX;//(float)minX;
    minEyeY = -eyeY;//(float)minY;
    minEyeZ = 50.0;
    minEyeX = eyeX;//(float)maxX;
    maxEyeY =  eyeY;//(float)maxY;
    maxEyeZ = eyeZ*5.0;//750.0; 
    
    deltaX = 0.0;
    deltaY = 0.0;
    deltaZ = 0.0;
    
    // initialize GLUT 
    glutInit(argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowPosition (glutGet(GLUT_SCREEN_WIDTH)/2 - window_width/2, 
                            glutGet(GLUT_SCREEN_HEIGHT)/2 - window_height/2);
    glutInitWindowSize(window_width, window_height);
    iGLUTWindowHandle = glutCreateWindow("GPU Programming Final");
#if !(defined (__APPLE__) || defined(MACOSX))
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_GLUTMAINLOOP_RETURNS);
#endif
    
    // register GLUT callback functions
    glutDisplayFunc(DisplayGL);
    glutKeyboardFunc(KeyboardGL);
    glutMouseFunc(mouse);
    glutMotionFunc(motion);
    //glutMouseWheelFunc(mouseWheel);
	glutTimerFunc(REFRESH_DELAY, timerEvent,0);
    glutSpecialFunc( processSpecialKeys );
    
	// initialize necessary OpenGL extensions
    glewInit();
    GLboolean bGLEW = glewIsSupported("GL_VERSION_2_0 GL_ARB_pixel_buffer_object"); 
    shrCheckErrorEX(bGLEW, shrTRUE, pCleanup);
    
    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);
    
    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 11000.0);
    gluLookAt(eyeX, eyeY, eyeZ,    eyeX, eyeY, 0,    0, 1, 0);
    
    return;
}

//// Run the OpenCL part of the computation
////*****************************************************************************

void runKernel()
{
 	size_t szGlobalWorkSize[] = {imageWidth, imageHeight};//(numPeople+numMilitary+numUFOs)/computationHeight, computationHeight};
    
    //addition for workgroups
    size_t szLocalWorkSize[2];
    // set up execution configuration
    szLocalWorkSize[0] = BLOCK_DIM;
    szLocalWorkSize[1] = BLOCK_DIM;

    
    ciErrNum = CL_SUCCESS;
    
#ifdef GL_INTEROP   
    // map OpenGL buffer object for writing from OpenCL
    glFinish();
    
    
//    ciErrNum = clEnqueueAcquireGLObjects(cqCommandQueue, 1, &cl_, 0,0,0);
//    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
//    
//    ciErrNum = clEnqueueAcquireGLObjects(cqCommandQueue, 1, &laser_cl, 0,0,0);
//    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    
    //added- but trying it without
//    ciErrNum = clEnqueueAcquireGLObjects(cqCommandQueue, 1, &cl_imageOutput, 0,0,0);
//    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    
#endif
    
    cl_event eventGlobal;
    cl_int errcode_ret;
    
    if( whichKernel == 1 )
        ciErrNum |= clSetKernelArg(ckKernel, 11, sizeof(unsigned int), &anim);//set final argument- counter
    else if( whichKernel == 3 )
        ciErrNum |= clSetKernelArg(ckKernel, 11, sizeof(unsigned int), &anim);//set final argument- counter
    else if( whichKernel == 5 )
        ciErrNum |= clSetKernelArg(ckKernel, 11, sizeof(unsigned int), &anim);//set final argument- counter        
    
    // execute the kernel
    if( whichKernel < 4 )
        ciErrNum |= clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 2, NULL, szGlobalWorkSize, NULL, 0, 0, &eventGlobal );
    else
        ciErrNum |= clEnqueueNDRangeKernel(cqCommandQueue, ckKernel, 2, NULL, szGlobalWorkSize, szLocalWorkSize, 0, 0, &eventGlobal );
    cl_ulong end, start;
    
    // lets do some profiling
    errcode_ret = clWaitForEvents(1, &eventGlobal);
    oclCheckError(errcode_ret, CL_SUCCESS);
    errcode_ret = clGetEventProfilingInfo(eventGlobal, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, 0);
    errcode_ret |= clGetEventProfilingInfo(eventGlobal, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, 0);
    oclCheckError(errcode_ret, CL_SUCCESS);
    fprintf(stderr, "Global kernel time: %0.3f ms\n",(end-start)*1.0e-6f);
    
    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    
    
#ifdef GL_INTEROP
    // unmap buffer object
//    ciErrNum = clEnqueueReleaseGLObjects(cqCommandQueue, 1, &vbo_cl, 0,0,0);
//    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
//    
//    ciErrNum = clEnqueueReleaseGLObjects(cqCommandQueue, 1, &vbo_cl, 0,0,0);
//    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
    
    //added but trying without
//    ciErrNum = clEnqueueReleaseGLObjects(cqCommandQueue, 1, &cl_inputImage, 0,0,0);
//    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);

    clFinish(cqCommandQueue);
#else
    
//    
//    // Explicit Copy 
//    // map the PBO to copy data from the CL buffer via host
//    glBindBufferARB(GL_ARRAY_BUFFER, vbo);   
//    
//    // map the buffer object into client's memory
//    void* ptr = glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY_ARB);
//    
//    unsigned int size = ((numPeople+numMilitary) * 8 * sizeof( float) * vertPerBoid) + (numUFOs * 8 * sizeof(float) * vertPerBoid); //4 position, 4 color  
//    ciErrNum = clEnqueueReadBuffer(cqCommandQueue, vbo_cl, CL_TRUE, 0, size, ptr, 0, NULL, NULL);
//    
//    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
//    
//    glUnmapBufferARB(GL_ARRAY_BUFFER); 
//    
//    // Explicit Copy 
//    // map the PBO to copy data from the CL buffer via host
//    glBindBufferARB(GL_ARRAY_BUFFER, laserBuf);   
//    
//    // map the buffer object into client's memory
//    void* ptr2 = glMapBufferARB(GL_ARRAY_BUFFER, GL_WRITE_ONLY_ARB);
//    
//    unsigned int size2 = ((numMilitary) * 8 * sizeof( float) * 2) + (numUFOs * 8 * sizeof(float) * 2); //4 position, 4 color  
//    ciErrNum = clEnqueueReadBuffer(cqCommandQueue, laser_cl, CL_TRUE, 0, size2, ptr2, 0, NULL, NULL);
//    
//    shrCheckErrorEX(ciErrNum, CL_SUCCESS, pCleanup);
//    
//    glUnmapBufferARB(GL_ARRAY_BUFFER); 
//    
//    glBindBufferARB(GL_ARRAY_BUFFER, vbo);   


#endif
    
}


// Display callback
//*****************************************************************************

void DisplayGL() //computation linked to graphics refresh- better to have running separately
{
    framesPerSecond();    
        
    anim += 1;
    
    // run OpenCL kernel to generate vertex positions
    runKernel();//will put info in the vertex buffer
    //draw what comes back
    
    // set view matrix
    glClear(GL_COLOR_BUFFER_BIT);
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    //gluLookAt(eyeX, eyeY, eyeZ,    eyeX, eyeY, 0,    0, 1, 0);
    //gluLookAt((maxX-minX)/2, (maxY-minY)/2, 750,    (maxX-minX)/2, (maxY-minY)/2, 0,    0, 1, 0);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);//rotate with mouse
    glRotatef(rotate_y, 0.0, 1.0, 0.0);
    glTranslatef( deltaX, deltaY, deltaZ );//eyeX, eyeY, eyeZ );    
    //------DRAW TEXTURE---------
    glBindTexture(GL_TEXTURE_2D, outputImage);
    
    glEnable( GL_TEXTURE_2D );
    
    //draw map    
    glBegin( GL_POLYGON );
    glNormal3f(0.0, 0.0, 1.0);
    glTexCoord2i(0, 0);
	glVertex3f( minX, minY, 0.0f );
	
	glTexCoord2i(1, 0);//texWidth, texHeight);
	glVertex3f(  maxX, minY, 0.0f );
	
	glTexCoord2i(1, 1);//texWidth, 0 );
	glVertex3f( maxX,  maxY, 0.0f );
	
	glTexCoord2i(0, 1);
	glVertex3f( minX,  maxY, 0.0f );
	glEnd();
    
    glDisable( GL_TEXTURE_2D );
    
    glFlush();
        
    // flip backbuffer to screen
    glutSwapBuffers();
    glutPostRedisplay();
}

void timerEvent(int value)
{
    glutPostRedisplay();
	glutTimerFunc(REFRESH_DELAY, timerEvent,0);
}

// Keyboard events handler
//*****************************************************************************
void KeyboardGL(unsigned char key, int x, int y)
{
    switch(key) 
    {
        case '\033': // escape quits 0
            // Cleanup up and quit
            bNoPrompt = shrTRUE;
	        Cleanup(EXIT_SUCCESS);
            break;
        case '+' :
            deltaZ += 100.0;
            break;
        case '=' :
            deltaZ += 100.0;
            break;
        case '-' :
            deltaZ -= 100.0;
            break;
        case '_' :
            deltaZ -= 100.0;
            break;  
    }
    
    
    if( deltaZ > maxEyeZ )
        deltaZ= maxEyeZ;
    else if( deltaZ < minEyeZ )
        deltaZ = minEyeZ;

}

void processSpecialKeys(int key, int x, int y) {
    
	switch(key) {
		case GLUT_KEY_UP :
            deltaY -= 100.0;
            break;
		case GLUT_KEY_DOWN :
            deltaY += 100.0;
            break;
		case GLUT_KEY_RIGHT :
            deltaX -= 100.0;
            break;
        case GLUT_KEY_LEFT :
            deltaX += 100.0;
            break;
	}
//    
//    if( deltaX < minEyeX )
//        deltaX = minEyeX;
//    else if( deltaX > maxEyeX )
//        deltaX = maxEyeX;
//    
//    if( deltaY < minEyeY )
//        deltaY = minEyeY;
//    else if( deltaY > maxEyeY )
//        deltaY = maxEyeY;
    
    
}

// Mouse event handlers
//*****************************************************************************

//turned off mouse induced movement for now
void mouse(int button, int state, int x, int y)
{
    // Wheel reports as button 3(scroll up) and button 4(scroll down)
    if ((button == 3) || (button == 4)) // It's a wheel event
    {
        // Each wheel event reports like a button click, GLUT_DOWN then GLUT_UP
        if (state == GLUT_UP) return; // Disregard redundant GLUT_UP events
            fprintf(stderr, "HI  Scroll %s At %d %d\n", (button == 3) ? "Up" : "Down", x, y);
        fprintf(stderr, "Hello!!!!!!!!! Scroll %s At %d %d\n", (button == 3) ? "Up" : "Down", x, y);
    }
    else
    {  // normal button event
        fprintf(stderr, "Button %s At %d %d\n", (state == GLUT_DOWN) ? "Down" : "Up", x, y);
        //printf("Button %s At %d %d\n", (state == GLUT_DOWN) ? "Down" : "Up", x, y);
    }
    
//    if (state == GLUT_UP )
//	{
//		if ( button == GLUT_WHEEL_UP )
//		{
//			fprintf(stderr, "Wheel Up\n");
//		}
//		else if( button == GLUT_WHEEL_DOWN )
//		{
//			fprintf(stderr, "Wheel Down\n");
//		}
//	}
    
}

//void mouseWheel(int button, int dir, int x, int y)
//{
//    if (dir > 0)
//    {
//        std::cout << "scroll in" << std::endl;
//    }
//    else
//    {
//        // Zoom out
//        std::cout << "scroll out" << std::endl;
//    }
//    
//    return;
//}


void motion(int x, int y)
{
//    float dx, dy;
//    dx = (float)(x - mouse_old_x);
//    dy = (float)(y - mouse_old_y);
//
//    if (mouse_buttons & 1) {
//        rotate_x += dy * 0.2f;
//        rotate_y += dx * 0.2f;
//    } 
//
//    mouse_old_x = x;
//    mouse_old_y = y;
    
}


// Function to clean up and exit
//*****************************************************************************
void Cleanup(int iExitCode)
{
    // Cleanup allocated objects
    shrLog("\nStarting Cleanup...\n\n");
	if(ckKernel)clReleaseKernel(ckKernel); 
    if(cpProgram)clReleaseProgram(cpProgram);
    if(cqCommandQueue)clReleaseCommandQueue(cqCommandQueue);
    if(cxGPUContext)clReleaseContext(cxGPUContext);
    if(cPathAndName)free(cPathAndName);
    if(cSourceCL)free(cSourceCL);
    if(cdDevices)delete(cdDevices);

    // finalize logs and leave
    shrQAFinish2(bQATest, *pArgc, (const char **)pArgv, (iExitCode == 0) ? QA_PASSED : QA_FAILED ); 
    if (bQATest || bNoPrompt)
    {
        shrLogEx(LOGBOTH | CLOSELOG, 0, "%s Exiting...\n", cExecutableName);
    }
    else 
    {
        shrLogEx(LOGBOTH | CLOSELOG, 0, "%s Exiting...\nPress <Enter> to Quit\n", cExecutableName);
        #ifdef WIN32
            getchar();
        #endif
    }
    exit (iExitCode);
}


/** loadTexture
 *     loads a png file into an opengl texture object, using cstdio , libpng, and opengl.
 * 
 *     \param filename : the png file to be loaded
 *     \param width : width of png, to be updated as a side effect of this function
 *     \param height : height of png, to be updated as a side effect of this function
 * 
 *     \return GLuint : an opengl texture id.  Will be 0 if there is a major error,
 *                                     should be validated by the client of this function.
 * 
 */
GLuint loadTexture(const string filename, int &width, int &height) //PNG LOADER
{ 
    //header for testing if it is a png
    png_byte header[8];
    
    //open file as binary
    FILE *fp = fopen(filename.c_str(), "rb");
    if (!fp) {
        return TEXTURE_LOAD_ERROR;
    }
    
    //read the header
    fread(header, 1, 8, fp);
    
    //test if png
    int is_png = !png_sig_cmp(header, 0, 8);
    if (!is_png) {
        fclose(fp);
        return TEXTURE_LOAD_ERROR;
    }
    
    //create png struct
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL,
                                                 NULL, NULL);
    if (!png_ptr) {
        fclose(fp);
        return (TEXTURE_LOAD_ERROR);
    }
    
    //create png info struct
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, (png_infopp) NULL, (png_infopp) NULL);
        fclose(fp);
        return (TEXTURE_LOAD_ERROR);
    }
    
    //create png info struct
    png_infop end_info = png_create_info_struct(png_ptr);
    if (!end_info) {
        png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp) NULL);
        fclose(fp);
        return (TEXTURE_LOAD_ERROR);
    }
    
    //png error stuff, not sure libpng man suggests this.
    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
        fclose(fp);
        return (TEXTURE_LOAD_ERROR);
    }
    
    //init png reading
    png_init_io(png_ptr, fp);
    
    //let libpng know you already read the first 8 bytes
    png_set_sig_bytes(png_ptr, 8);
    
    // read all the info up to the image data
    png_read_info(png_ptr, info_ptr);
    
    //variables to pass to get info
    int bit_depth, color_type;
    png_uint_32 twidth, theight;
    
    // get info about png
    png_get_IHDR(png_ptr, info_ptr, &twidth, &theight, &bit_depth, &color_type,
                 NULL, NULL, NULL);
    
    //update width and height based on png info
    width = twidth;
    height = theight;
    
    std::cout << "width = " << width << " height = " << height << endl;
    
    // Update the png info struct.
    png_read_update_info(png_ptr, info_ptr);
    
    // Row size in bytes.
    int rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    
    // Allocate the image_data as a big block, to be given to opengl
    //png_byte *image_data = new png_byte[rowbytes * height];
    image_data = new png_byte[rowbytes * height];
    if (!image_data) {
        //clean up memory and close stuff
        png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
        fclose(fp);
        return TEXTURE_LOAD_ERROR;
    }
    
    //row_pointers is for pointing to image_data for reading the png with libpng
    png_bytep *row_pointers = new png_bytep[height];
    if (!row_pointers) {
        //clean up memory and close stuff
        png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
        delete[] image_data;
        fclose(fp);
        return TEXTURE_LOAD_ERROR;
    }
    // set the individual row_pointers to point at the correct offsets of image_data
    for (int i = 0; i < height; ++i)
        row_pointers[height - 1 - i] = image_data + i * rowbytes;
    
    //read the png into image_data through row_pointers
    png_read_image(png_ptr, row_pointers);
    
    //Now generate the OpenGL texture object
    GLuint texture;
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D,0, GL_RGBA, width, height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, (GLvoid*) image_data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);//GL_LINEAR);
    
    
    //clean up memory and close stuff
    png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
    //delete[] image_data;
    delete[] row_pointers;
    fclose(fp);
    
    return texture;
}



int frame = 0;
int theTime = 0;
int timebase = 0;

void framesPerSecond()
{
    frame++;
	theTime=glutGet(GLUT_ELAPSED_TIME);
    
	if (theTime - timebase > 1000) 
    {
        //fprintf(stderr, "frames per second: %4.2f ms\n",frame*1000.0/(theTime-timebase));
        
        //print this and counts to window frame
        char str[300];
        sprintf( str, "GPU Programming Final:  Frames per second: %4.2f\n", frame*1000.0/(theTime-timebase) );
        glutSetWindowTitle(str);
	 	timebase = theTime;
		frame = 0;
	}
}
