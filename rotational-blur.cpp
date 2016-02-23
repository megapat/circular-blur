
#include "opencl.h"

static constexpr double Epsilon = (1.0e-15);
static constexpr unsigned PixelSize = 16;

template <typename Ptr, typename T>
inline constexpr bool aligned(Ptr p)
{
    return bool(size_t(p) & (sizeof(T) - 1));
}

int rotational_blur(cl::Context& context, float* image, int width, int heigth, const float angle)
{
    int length = width * heigth;
    
    // todo add constant
    cl::Buffer devInputImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, length * PixelSize, image);
    
    float* filteredImage = new float[length];
    cl::Buffer devInputImage(context, CL_MEM_WRITE_ONLY, length * PixelSize);
    
    
    delete [] filteredImage;
}