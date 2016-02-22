
#include <iostream>

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

enum PlatformType
{
    PLATFORM_AMD,
    PLATFORM_NVIDIA,
    PLATFORM_INTEL,
    PLATFORM_UNKNOWN,
};

enum DeviceType : cl_device_type
{
    DEVICE_DEFAULT = CL_DEVICE_TYPE_DEFAULT,
    DEVICE_CPU     = CL_DEVICE_TYPE_CPU,
    DEVICE_GPU     = CL_DEVICE_TYPE_GPU,
    DEVICE_ALL     = CL_DEVICE_TYPE_ALL,
};

static const char* AMD_PLATFORM_ID = "Advanced Micro Devices, Inc.";

struct OpenCL
{
    OpenCL(DeviceType type = DEVICE_DEFAULT) :
        type_ (type),
        platform_ ()
    {
    }
    
    int init(PlatformType type);
    
    cl::Platform platform() const
    {
        return platform_;
    }
    
    cl::Context context() const
    {
        cl_context_properties cps[3] =
        {
            CL_CONTEXT_PLATFORM,
            (cl_context_properties)platform_(),
            0
        };
        return cl::Context(type_, cps);
    }
    
private:
    DeviceType type_;
    cl::Platform platform_;
};

int OpenCL::init(PlatformType type)
{
    int ret = CL_INVALID_PLATFORM;
    
    auto getPlatform = [](std::string const& id)
    {
        if (!id.compare(AMD_PLATFORM_ID))
            return PLATFORM_AMD;
        
        return PLATFORM_UNKNOWN;
    };
    
    try  
    {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty())
        {
            //std::cout << "Platform size 0\n";
            return -1;
        }
        
        for (auto& platform : platforms)
        {
            if (type == getPlatform(platform.getInfo<CL_PLATFORM_VENDOR>()))
            {
                platform_ = platform;
                ret = CL_SUCCESS;
                break;
            }
            //std::cout << "Platform name: " << vendor << std::endl;
            
            
//             cl_context_properties cps[3] =
//             {
//                 CL_CONTEXT_PLATFORM,
//                 (cl_context_properties)platform(),
//                 0
//             };
//             
//             contexts_.emplace_back(type, cps);
//             auto& context = contexts_.back();
//             
//             for (auto& device : context.devices())
//             {
//                 std::string deviceName = device.getInfo<CL_DEVICE_NAME>();
//                 std::cout << "Device: " << deviceName << std::endl;
//             }
        }   
    }
    catch (cl::Error err)
    {
        ret = err.err();  
    }

    return ret;
}
