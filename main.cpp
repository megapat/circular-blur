
#include <iostream>
#include <CImg.h>

#include "opencl.h"

#define NON_OPTIMIZED
//#define READ_ALIGNED
//#define READ4

constexpr unsigned WGX(16);
constexpr unsigned WGY(16);

OpenCL ocl(DEVICE_GPU);

// This function takes a positive integer and rounds it up to
// the nearest multiple of another provided integer
unsigned int roundUp(unsigned value, unsigned multiple) 
{
    // Determine how far past the nearest multiple the value is
    auto remainder = value % multiple;
    // Add the difference to make the value a multiple
    if(remainder != 0) 
    {
        value += (multiple - remainder);
    }
    
    return value;
}

#define KERNEL_SOURCE(source) #source

template <typename Image>
int blur_image(Image const& inputImage, Image& outputImage)
{
    static const std::string kernel_source = KERNEL_SOURCE(
        __kernel void convolution(__global float* imageIn, 
                                  __global float* imageOut,
                                  __constant float* filter,
                                  int rows,
                                  int cols,
                                  int filterWidth,
                                  __local float* localImage,
                                  int localHeight,
                                  int localWidth)
        {
            // Determine the amount of padding for this filter
            int filterRadius = filterWidth / 2;
            int padding = filterRadius * 2;
            
            // Determine the size of the workgroup output region
            int groupStartCol = get_group_id(0)*get_local_size(0);
            int groupStartRow = get_group_id(1)*get_local_size(1);
            
            // Determine the local ID of each work-item
            int localCol = get_local_id(0);
            int localRow = get_local_id(1);

            // Determine the global ID of each work-item. work-items
            // representing the output region will have a unique global
            // ID
            int globalCol = groupStartCol + localCol;
            int globalRow = groupStartRow + localRow;

            // Cache the data to local memory
            // Step down rows
            for (int i = localRow; i < localHeight; i += get_local_size(1)) 
            {
                int curRow = groupStartRow + i;
                // Step across columns
                for (int j = localCol; j < localWidth; j += get_local_size(0)) 
                {
                    int curCol = groupStartCol + j;
                    
                    // Perform the read if it is in bounds
                    if (curRow < rows && curCol < cols)
                    {
                        localImage[i*localWidth + j] = imageIn[curRow*cols+curCol];
                    }
                }
            }
            
            barrier(CLK_LOCAL_MEM_FENCE);

            // Perform the convolution
            if (globalRow < rows-padding && globalCol < cols-padding) 
            {
                // Each work-item will filter around its start location
                //(starting from the filter radius left and up)
                float sum = 0.0f;
                int filterIdx = 0;
                // Not unrolled
                for (int i = localRow; i < localRow+filterWidth; i++) 
                {
                    int offset = i*localWidth;
                    for (int j = localCol; j < localCol+filterWidth; j++)
                    {
                        sum += localImage[offset+j] * filter[filterIdx++];
                    }
                }
                
                /*
                // Inner loop unrolled
                for (int i = localRow; i < localRow+filterWidth; i++) 
                {
                    int offset = i*localWidth+localCol;
                    sum += localImage[offset++] * filter[filterIdx++];
                    sum += localImage[offset++] * filter[filterIdx++];
                    sum += localImage[offset++] * filter[filterIdx++];
                    sum += localImage[offset++] * filter[filterIdx++];
                    sum += localImage[offset++] * filter[filterIdx++];
                    sum += localImage[offset++] * filter[filterIdx++];
                    sum += localImage[offset++] * filter[filterIdx++];
                }
                */
                
                // Write the data out
                imageOut[(globalRow+filterRadius)*cols + (globalCol+filterRadius)] = sum;
            }

            return;
        }
    );
    
    int ret = CL_SUCCESS;
    
    try {
    
    int imgw = inputImage.width();
    int imgh = inputImage.height();
    
    int dataSize = imgh*imgw*sizeof(float);
    
#ifdef NON_OPTIMIZED
    int devw = imgw;
#else // READ_ALIGNED jj READ4
    int devw = roundUp(imgw, WGX);
#endif
    int devh = imgh;
    
    int devDataSize = imgh*devw*sizeof(float);
    
    // 45 degree motion blur
    float filter[49] =
    {
        0, 0, 0, 0, 0, 0.0145, 0,
        0, 0, 0, 0, 0.0376, 0.1283, 0.0145,
        0, 0, 0, 0.0376, 0.1283, 0.0376, 0, 
        0, 0, 0.0376, 0.1283, 0.0376, 0, 0, 
        0, 0.0376, 0.1283, 0.0376, 0, 0, 0, 
        0.0145, 0.1283, 0.0376, 0, 0, 0, 0, 
        0, 0.0145, 0, 0, 0, 0, 0
    };
    
    int filterWidth = 7;
    int filterRadius = filterWidth/2;
    int paddingPixels = (int)(filterWidth/2) * 2;
    
    auto context = ocl.context();
    auto devices = context.getInfo<CL_CONTEXT_DEVICES>();
    
    if (devices.empty())
    {
        std::cerr << "ERROR: OpenCL => device not found" << std::endl;
        return -1;
    }
    
    auto device = devices.front();
    
    //cl::Program::Sources programSource(1, std::make_pair(kernel_source.data(), kernel_source.size()));
    cl::Program program(context, kernel_source);
    program.build(devices);
    
    for (auto& dev : devices)
    {
        std::string build_output = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
        std::cout << "BUILD INFO: " << build_output << std::endl;
    }
    
    cl::CommandQueue queue(context, device);
    
    cl::Buffer devInputImage(context, CL_MEM_READ_ONLY, devDataSize);
    cl::Buffer devOutputImage(context, CL_MEM_WRITE_ONLY, devDataSize);
    cl::Buffer devFilter(context, CL_MEM_READ_ONLY, 49*sizeof(float));

#ifdef NON_OPTIMIZED  
    queue.enqueueWriteBuffer(devInputImage, CL_TRUE, 0, devDataSize, inputImage.data());
#else // READ_ALIGNED jj READ4
    size_t buffer_origin[3] {0, 0, 0};
    size_t host_origin[3] {0, 0, 0};
    size_t region[3] { devw*sizeof(float), imgh, 1};
    
    queue.clEnqueueWriteBufferRect(devInputImage, CL_TRUE, buffer_origin, host_origin, region,
        devw*sizeof(float), 0, imgw*sizeof(float), 0), inputImage.data());
#endif
    queue.enqueueWriteBuffer(devFilter, CL_TRUE, 0, 49*sizeof(float), filter);
    
#if defined NON_OPTIMIZED || defined READ_ALIGNED
    cl::Kernel kernel(program, "convolution");
#else // READ4
    cl::Kernel kernel(program, "convolution_read4");
#endif
    
    // Selected workgroup size is 16x16
    // When computing the total number of work-items, the
    // padding work-items do not need to be considered
    auto totalWorkItemsX = roundUp(imgw - paddingPixels, WGX);
    auto totalWorkItemsY = roundUp(imgh - paddingPixels, WGY);
    // Size of a workgroup
    cl::NDRange localSize {WGX, WGY};
    // Size of the NDRange
    cl::NDRange globalSize {totalWorkItemsX, totalWorkItemsY};
    // The amount of local data that is cached is the size of the
    // workgroups plus the padding pixels
#if defined NON_OPTIMIZED || defined READ_ALIGNED
    int localWidth = localSize[0] + paddingPixels;
#else // READ4
    // Round the local width up to 4 for the read4 kernel
    int localWidth = roundUp(localSize[0]+paddingPixels, 4);
#endif
    int localHeight = localSize[1] + paddingPixels;
    // Compute the size of local memory (needed for dynamic allocation)
    size_t localMemSize = (localWidth * localHeight * sizeof(float));
    // Set the kernel arguments
    kernel.setArg(0, devInputImage);
    kernel.setArg(1, devOutputImage);
    kernel.setArg(2, devFilter);
    kernel.setArg(3, devh);
    kernel.setArg(4, devw);
    kernel.setArg(5, filterWidth);
    kernel.setArg(6, localMemSize, nullptr);
    kernel.setArg(7, localHeight);
    kernel.setArg(8, localWidth);
    
    // Execute the kernel
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, globalSize, localSize);
    queue.finish();
    
    // Read back the output image
#ifdef NON_OPTIMIZED
    queue.enqueueReadBuffer(devOutputImage, CL_TRUE, 0, devDataSize, outputImage.data());
#else // READ_ALIGNED jj READ4
    // Begin reading output from (3,3) on the device
    // (for 7x7 filter with radius 3)
    buffer_origin[0] = 3*sizeof(float);
    buffer_origin[1] = 3;
    buffer_origin[2] = 0;
    // Read data into (3,3) on the host
    host_origin[0] = 3*sizeof(float);
    host_origin[1] = 3;
    host_origin[2] = 0;
    // Region is image size minus padding pixels
    region[0] = (imageWidth-paddingPixels)*sizeof(float);
    region[1] = (imageHeight-paddingPixels);
    region[2] = 1;
    // Perform the read
    queue.enqueueReadBufferRect(devOutputImage, CL_TRUE, buffer_origin, host_origin, region, deviceWidth*sizeof(float), 0, imageWidth*sizeof(float), 0, outputImage);
#endif
    
    }
    catch (cl::Error err)
    {
        std::cerr << "ERROR: OpenCL => " << err.what() << std::endl;
        ret = err.err();
    }
    
    return ret;
}

int main(int argc, char **argv) 
{
    using namespace cimg_library;
    
    std::cout << "Hello, world!" << std::endl;
    
    if (argc != 2)
    {
        std::cerr << "ERROR: filename missing" << std::endl;
        return -1;
    }

#if 1    
    if (ocl.init(PLATFORM_AMD))
    {
        std::cerr << "ERROR: Cannot init OpenCL" << std::endl;
        return -1;
    }
#endif    
    //auto context = ocl.context();
    std::string fname(argv[1]);
    const unsigned char red[] = { 255,0,0 }, green[] = { 0,255,0 }, blue[] = { 0,0,255 };

    typedef CImg<float> ImageType;
    
    try
    {
        ImageType image(fname.c_str()), visu(500, 400, 1, 3, 0);
        std::cout << image.data()[0] << std::endl;
        std::cout << image.data()[1] << std::endl;
        std::cout << image.data()[2] << std::endl;
        
        ImageType oimage(image.width(), image.height(), 1, 3, 255.0f);
        oimage.draw_text(10, 10, "Blur test with OpenCL", green);
        
        if (blur_image(image, oimage))
        {
            std::cerr << "Error: image blur" << std::endl;
            return -1;
        }
        
        CImgDisplay main_disp(image,"Click a point");
        CImgDisplay draw_disp(visu,"Intensity profile");
        CImgDisplay blur_disp(oimage, "Blured");
        
        while (!(main_disp.is_closed() || 
                 blur_disp.is_closed() || 
                 draw_disp.is_closed())) 
        {
            main_disp.wait();
            if (main_disp.button() && main_disp.mouse_y() >= 0) 
            {
                int y = main_disp.mouse_y();
                visu.fill(0).draw_graph(image.get_crop(0, y, 0, 0, image.width() - 1, y, 0, 0), red, 1, 1, 0, 255, 0);
                visu.draw_graph(image.get_crop(0, y, 0, 1, image.width()-1, y, 0, 1), green, 1, 1, 0, 255, 0);
                visu.draw_graph(image.get_crop(0, y, 0, 2, image.width()-1, y, 0, 2), blue, 1, 1, 0, 255, 0).display(draw_disp);
            }
        }
    }
    catch (CImgInstanceException const& err)
    {
        std::cerr << "ERROR: Cannot load image" << std::endl;
    }
    
    return 0;
}
