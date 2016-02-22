/*  Image Convolution algorithms */


//Version1 BASIC CONVOLUTION WITH TEXTURE MEMORY
__kernel void convolve( read_only image2d_t inputImage, write_only image2d_t outputImage, sampler_t inputImageSampler, sampler_t outputImageSampler, int imageWidth, int imageHeight, __global float* filter, int filterWidth, unsigned int width, unsigned int height )
{
    unsigned int x = get_global_id(0);//get my global id in both dim
    unsigned int y = get_global_id(1);
    int2 coord = (int2)(x, y);

    int halfFilterWidth = filterWidth/2;
    
    //make sure in bounds
    if( x >= halfFilterWidth && y >= halfFilterWidth && x <= imageWidth-1-halfFilterWidth && y <= imageHeight-1-halfFilterWidth )
    {
        float4 outPixel = (float4)(0.0, 0.0, 0.0, 0.0);
        int offsetX = -filterWidth/2;//offset from x into input image ( apply filter to neighbors )
        int offsetY = -filterWidth/2;//offset from y into input image
        for( int row = 0; row < filterWidth; row++)//apply all rows and cols of filter to image
        {
            offsetX += row; 
            for(int col = 0; col < filterWidth; col++)
            {
                offsetY += col;
                coord = (int2) ( x + offsetX, y + offsetY );
                float4 sampledPix = read_imagef( inputImage, inputImageSampler, coord );
                
                outPixel += sampledPix * filter[row*filterWidth+col]; //apply filter to neighboring pixels, and sum

            }
            offsetY = -filterWidth/2;
        }
        
        //write out result
        coord = (int2)(x, y);
        write_imagef( outputImage, coord, (float4)(outPixel.x, outPixel.y, outPixel.z, 1.0) );
    }
    else if( x >= 0 && y >= 0 && x < imageWidth && y < imageHeight ) //if in picture bounds but out of convole bounds- (eg. filter can't look at neighbors) make it black
    {
        coord = (int2)(x, y);
        write_imagef( outputImage, coord, (float4)(0.0, 0.0, 0.0, 1.0) );
    }

} 


// CONVOLUTION WITH GLOBAL MEMORY 
__kernel void convolveGloballMem( __global float4* inputImageDataBuf, read_only image2d_t inputImage, write_only image2d_t outputImage, sampler_t inputImageSampler, sampler_t outputImageSampler, int imageWidth, int imageHeight, __global float* filter, int filterWidth, unsigned int width, unsigned int height, unsigned int counter )
{
    unsigned int x = get_global_id(0);//get my global id in both dim
    unsigned int y = get_global_id(1);
    int2 coord = (int2)(x, y);
    
    int halfFilterWidth = filterWidth/2;
    float4 outPixel = (float4)(0.0, 0.0, 0.0, 0.0);
    
    //init- copy from input texture to global float mem 
    if( counter < 2  && x >= 0 && y >= 0 && x < imageWidth && y < imageHeight )
    {       
        float4 pix = read_imagef( inputImage, inputImageSampler, coord );
        inputImageDataBuf[x*imageWidth+y] = pix;
        outPixel = pix;
    }
    
    //make sure in bounds
    if( counter >= 2 && x >= halfFilterWidth && y >= halfFilterWidth && x <=imageWidth-1-halfFilterWidth && y <= imageHeight-1-halfFilterWidth )
    {
        int offsetX = -halfFilterWidth;//-filterWidth/2;//offset from x into input image ( apply filter to neighbors )
        int offsetY = -halfFilterWidth;//-filterWidth/2;//offset from y into input image
        for( int row = 0; row < filterWidth; row++)//apply all rows and cols of filter to image
        {
            offsetX += row; 
            for(int col = 0; col < filterWidth; col++)
            {
                offsetY += col;
                coord = (int2) ( x + offsetX, y + offsetY );
                float4 sampledPix = inputImageDataBuf[ coord.x*imageWidth+coord.y ];//read_imagef( inputImage, inputImageSampler, coord );
                outPixel += sampledPix * filter[row*filterWidth+col]; //apply filter to neighboring pixels, and sum
                
            }
            offsetY = -halfFilterWidth;//-filterWidth/2;
        }
        
        //write out result
        coord = (int2)(x, y);
        write_imagef( outputImage, coord, (float4)(outPixel.x, outPixel.y, outPixel.z, 1.0) );
    }
    else if( counter >= 2 &&x >= 0 && y >= 0 && x < imageWidth && y < imageHeight ) //if in picture bounds but out of convole bounds- (eg. filter can't look at neighbors) make it black
    {
        coord = (int2)(x, y);
        write_imagef( outputImage, coord, (float4)(0.0, 0.0, 0.0, 1.0) );
    }
} 


//Version 3 CONSTANT MEMORY with Textures
__kernel void convolveConstant( read_only image2d_t inputImage, write_only image2d_t outputImage, sampler_t inputImageSampler, sampler_t outputImageSampler, __constant int* imageWidth,  __constant int* imageHeight, __constant float* filter, __constant int* filterWidth, unsigned int width, unsigned int height )
{
    unsigned int x = get_global_id(0);//get my global id in both dim
    unsigned int y = get_global_id(1);
    int2 coord = (int2)(x, y);
    
    int halfFilterWidth = filterWidth[0]/2;
    
    //make sure in bounds
    if( x >= halfFilterWidth && y >= halfFilterWidth && x <= imageWidth[0]-1-halfFilterWidth && y <= imageHeight[0]-1-halfFilterWidth )
    {
        float4 outPixel = (float4)(0.0, 0.0, 0.0, 0.0);
        int offsetX = -filterWidth[0]/2;//offset from x into input image ( apply filter to neighbors )
        int offsetY = -filterWidth[0]/2;//offset from y into input image
        for( int row = 0; row < filterWidth[0]; row++)//apply all rows and cols of filter to image
        {
            offsetX += row; 
            for(int col = 0; col < filterWidth[0]; col++)
            {
                offsetY += col;
                coord = (int2) ( x + offsetX, y + offsetY );
                float4 sampledPix = read_imagef( inputImage, inputImageSampler, coord );
                
                outPixel += sampledPix * filter[row*filterWidth[0]+col]; //apply filter to neighboring pixels, and sum
                
            }
            offsetY = -filterWidth[0]/2;
        }
        
        //write out result
        coord = (int2)(x, y);
        write_imagef( outputImage, coord, (float4)(outPixel.x, outPixel.y, outPixel.z, 1.0) );
    }
    else if( x >= 0 && y >= 0 && x < imageWidth[0] && y < imageHeight[0] ) //if in picture bounds but out of convole bounds- (eg. filter can't look at neighbors) make it black
    {
        coord = (int2)(x, y);
        write_imagef( outputImage, coord, (float4)(0.0, 0.0, 0.0, 1.0) );
    }
    
} 

//
//__kernel void convolveConstantLocal( read_only image2d_t inputImage, write_only image2d_t outputImage, sampler_t inputImageSampler, sampler_t outputImageSampler, __constant int* imageWidth,  __constant int* imageHeight, __constant float* filter, __constant int* filterWidth, unsigned int width, unsigned int height )
//{
//    
//    //determine the amount of padding for this filter
//    int filterRadius = (filterWidth[0]/2); 
//    int padding = filterRadius*2; 
//    
//    //determine the size of the workgroup output region
//    int groupStartCol = get_group_id(0)*get_local_size(0); 
//    int groupStartRow = get_group_id(1)*get_local_size(1); 
//    
//    //determine the local id of each work item
//    int localCol = get_local_id(0);
//    int localRow = get_local_id(1);
//    
//    //determine the global id of each work item.  
//    int globalCol = groupStartCol + localCol;
//    int globalRow = groupStartRow + localRow;
//    
//    //local width and height- note these were before passed in as params
//    int localWidth = get_local_size(0)+padding;
//    int localHeight = get_local_size(1)+padding;
//    
//    __local float4 localImage[localWidth][localHeight];//Identification of this workgroup
//
//    
//    //step down rows
//   for(int i = localRow; i < localHeight; i+=get_local_size(1) )
//    {
//        int currRow = groupStartRow + i;
//        for(int j = localCol; j < localWidth; j+=get_local_size(0) )
//        {
//            int currCol = groupStartCol + j;
//            
//            //perform reads if in bounds
//            if( currRow < imageHeight[0] && currCol < imageWidth[0] )
//            {
//                localImage[i*localWidth+j] = readImage
//            }
//        }
//    }
//    
//}

#define BLOCK_DIM 16
//TEXTURE MEMORY AND LOCAL MEMORY
__kernel void convolveConstantLocal( read_only image2d_t inputImage, write_only image2d_t outputImage, sampler_t inputImageSampler, sampler_t outputImageSampler, __constant int* imageWidth,  __constant int* imageHeight, __constant float* filter, __constant int* filterWidth, unsigned int width, unsigned int height )
{
    int w = filterWidth[0];
    int wBy2 = w>>1; //w divided by 2
    
    //Goes up to 7x7 filters 
    __local float4 P[BLOCK_DIM+6][BLOCK_DIM+6];//Identification of this workgroup
    
    int i = get_group_id(0);
    int j = get_group_id(1); //Identification of work-item
    int idX = get_local_id(0);
    int idY = get_local_id(1);
    int ii = i*BLOCK_DIM + idX; // == get_global_id(0);
    int jj = j*BLOCK_DIM + idY; // == get_global_id(1);
    
    int2 coords = (int2)(ii, jj);
    //Reads pixels
    P[idX][idY] = read_imagef(inputImage, inputImageSampler, coords);
    
    //Needs to read extra elements for the filter in the borders
    if (idX < w)
    {    
        coords.x = ii + BLOCK_DIM; 
        coords.y = jj;
        P[idX + BLOCK_DIM][idY] = read_imagef(inputImage, inputImageSampler, coords);
    }
    
    if (idY < w)
    {   
        coords.x = ii; 
        coords.y = jj + BLOCK_DIM;
        P[idX][idY + BLOCK_DIM] = read_imagef(inputImage, inputImageSampler, coords);
    }
    if( idX < w && idY < w )
    {
        coords.x = ii + BLOCK_DIM; 
        coords.y = jj + BLOCK_DIM;
        P[idX + BLOCK_DIM][idY + BLOCK_DIM] = read_imagef(inputImage, inputImageSampler, coords);
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int x = get_global_id(0);
    int y = get_global_id(1); 
    float4 convPix = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float4 temp;
    float4 pix;
    
    if( x >= 0 && y >= 0 && x < imageWidth[0]-filterWidth[0] && y < imageHeight[0]-filterWidth[0] ) //if in picture bounds 
    {
        //Computes convolution
        for (int ix = 0; ix < w; ix++)
        {
            for (int jy = 0; jy < w; jy++)
            {
                pix = P[ix+idX][jy+idY];//(float4)((float)P[ix+idX][jy+idY].x, (float)P[ix][jy].y, (float)P[ix][jy].z, (float)P[ix][jy].w);
                convPix += pix * filter[ix + w*jy];
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);

        //pix = (float4)(1.0, 0.0, 0.0, 1.0);
        coords.x = x + (w>>1); 
        coords.y = y + (w>>1);
        pix = (float4)(convPix.x, convPix.y, convPix.z, 1.0);
        write_imagef(outputImage, coords, pix);

    }
}

 

// CONVOLUTION WITH GLOBAL MEMORY and LOCAL MEM
__kernel void convolveGloballMemLocal( __global float4* inputImageDataBuf, read_only image2d_t inputImage, write_only image2d_t outputImage, sampler_t inputImageSampler, sampler_t outputImageSampler, int imageWidth, int imageHeight, __global float* filter, int filterWidth, unsigned int width, unsigned int height, unsigned int counter )
{
    //init- copy from input texture to global float mem 
    if( counter < 2  )
    {       
        unsigned int x = get_global_id(0);//get my global id in both dim
        unsigned int y = get_global_id(1);
        int2 coord = (int2)(x, y);
        
        if( x >= 0 && y >= 0 && x < imageWidth && y < imageHeight )
        {
            float4 pix = read_imagef( inputImage, inputImageSampler, coord );
            inputImageDataBuf[x*imageWidth+y] = pix;
        }        
    }
    
    if( counter >= 2 )
    {
        int w = filterWidth;
        int wBy2 = w>>1; //w divided by 2
        
        //Goes up to 7x7 filters 
        __local float4 P[BLOCK_DIM+6][BLOCK_DIM+6];//Identification of this workgroup
        
        int i = get_group_id(0);
        int j = get_group_id(1); //Identification of work-item
        int idX = get_local_id(0);
        int idY = get_local_id(1);
        int ii = i*BLOCK_DIM + idX; // == get_global_id(0);
        int jj = j*BLOCK_DIM + idY; // == get_global_id(1);
        
        int2 coords = (int2)(ii, jj);
        //Reads pixels
        P[idX][idY] = inputImageDataBuf[ coords.x*imageWidth+coords.y ];//read_imagef(inputImage, inputImageSampler, coords);
        
        //Needs to read extra elements for the filter in the borders
        if (idX < w)
        {    
            coords.x = ii + BLOCK_DIM; 
            coords.y = jj;
            P[idX + BLOCK_DIM][idY] = inputImageDataBuf[ coords.x*imageWidth+coords.y ];//read_imagef(inputImage, inputImageSampler, coords);
        }
        
        if (idY < w)
        {   
            coords.x = ii; 
            coords.y = jj + BLOCK_DIM;
            P[idX][idY + BLOCK_DIM] = inputImageDataBuf[ coords.x*imageWidth+coords.y ];//read_imagef(inputImage, inputImageSampler, coords);
        }
        if( idX < w && idY < w )
        {
            coords.x = ii + BLOCK_DIM; 
            coords.y = jj + BLOCK_DIM;
            P[idX + BLOCK_DIM][idY + BLOCK_DIM] = inputImageDataBuf[ coords.x*imageWidth+coords.y ];//read_imagef(inputImage, inputImageSampler, coords);
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        int x = get_global_id(0);
        int y = get_global_id(1); 
        float4 convPix = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        float4 temp;
        float4 pix;
        
        if( x >= 0 && y >= 0 && x < imageWidth-filterWidth && y < imageHeight-filterWidth ) //if in picture bounds 
        {
            //Computes convolution
            for (int ix = 0; ix < w; ix++)
            {
                for (int jy = 0; jy < w; jy++)
                {
                    pix = P[ix+idX][jy+idY];//(float4)((float)P[ix+idX][jy+idY].x, (float)P[ix][jy].y, (float)P[ix][jy].z, (float)P[ix][jy].w);
                    convPix += pix * filter[ix + w*jy];
                }
            }
            
            barrier(CLK_LOCAL_MEM_FENCE);
            
            //pix = (float4)(1.0, 0.0, 0.0, 1.0);
            coords.x = x + (w>>1); 
            coords.y = y + (w>>1);
            pix = (float4)(convPix.x, convPix.y, convPix.z, 1.0);
            write_imagef(outputImage, coords, pix);
            
        }
    }
} 





//DIFFERENT VERSION OF CONVOLUTION
//  works, but what is the point... 
__kernel void anotherConvolveConstant( read_only image2d_t inputImage, write_only image2d_t outputImage, sampler_t inputImageSampler, sampler_t outputImageSampler, __constant int* imageWidth,  __constant int* imageHeight, __constant float* filter, __constant int* filterWidth, unsigned int width, unsigned int height )
{
    int w = filterWidth[0];
    int x = get_global_id(0);
    int y = get_global_id(1); 
    float4 convPix = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float4 temp;
    float4 pix;
    int2 coords = (int2)(0, 0);
    
    if( x >= 0 && y >= 0 && x < imageWidth[0]-filterWidth[0] && y < imageHeight[0]-filterWidth[0] ) //if in picture bounds 
    {
        for (int i = 0; i < w; i++)
        {
            for (int j = 0; j < w; j++)
            {        
                coords.x = x+i; 
                coords.y = y+j;
                pix = read_imagef(inputImage, inputImageSampler, coords);
                convPix += pix * filter[i + w*j];
            }
            
        }
        
        coords.x = x + (w>>1); 
        coords.y = y + (w>>1);
        pix = (float4)(convPix.x, convPix.y, convPix.z, 1.0);
        write_imagef(outputImage, coords, pix);
    }
}




//Version 5 CONSTANT MEMORY with Textures and Local memory
//not working
//__kernel void convolveConstantMyLocal( read_only image2d_t inputImage, write_only image2d_t outputImage, sampler_t inputImageSampler, sampler_t outputImageSampler, __constant int* imageWidth,  __constant int* imageHeight, __constant float* filter, __constant int* filterWidth, unsigned int width, unsigned int height )
//{
//    int w = filterWidth[0];
//    int wBy2 = w>>1; //w divided by 2
//    
//    int halfFilterWidth = filterWidth[0]/2;
//    
//    //idx into local buffer  : 3 is the current maximum filter size
//    int localMemX = 1 + localX;//filterWidth[0] + localX;
//    int localMemY = 1 + localY;//filterWidth[0] + localY;
//    
//    __local float4 localMem[BLOCK_DIM+2][BLOCK_DIM+2]; 
//    
//    //Read pixel at current coordinate into local mem
//    int2 coords = (int2)(x, y);
//    localMem[localMemX][localMemY] = read_imagef(inputImage, inputImageSampler, coords);
//    
//    //If necessary, read extra pixels
//    if ( localX == 0 ) 
//    {   
//        int minY = 0;
//        int maxY = 0;
//        if( localY == 0 )
//        {
//            minY = -halfFilterWidth;
//            maxY = 0;//-1; 
//        }
//        else if( localY == BLOCK_DIM-1 )
//        {
//            minY = 0;
//            maxY = halfFilterWidth;
//        }
//        for(int i = -halfFilterWidth; i <= 0; i++ )//read pixels to the right
//        {
//            for(int j = minY; j <= maxY; j++ )
//            {
//                coords = (int2)(x+i, y+j);
//                localMem[localMemX+i][localMemY+j] = read_imagef(inputImage, inputImageSampler, coords);
//            }
//        }
//    }
//    if( localY == 0 )
//    {
//        for(int i = -halfFilterWidth; i <= 0; i++ )//read pixels to the right
//        {
//            coords = (int2)(x, y+i);
//            localMem[localMemX][localMemY+i] = read_imagef(inputImage, inputImageSampler, coords);
//        }
//    }
//    if( localX == BLOCK_DIM-1 )
//    {
//        int minY = 0;
//        int maxY = 0;
//        if( localY == 0 )
//        {
//            minY = -halfFilterWidth;
//            maxY = 0;//-1; 
//        }
//        else if( localY == BLOCK_DIM-1 )
//        {
//            minY = 0;//1;
//            maxY = halfFilterWidth; 
//        }
//        for(int i = 0; i <= halfFilterWidth; i++ )//read pixels to the right
//        {
//            for(int j = minY; j <= maxY; j++)
//            {
//                coords = (int2)(x+i, y+j);
//                localMem[localMemX+i][localMemY+j] = read_imagef(inputImage, inputImageSampler, coords);
//            }
//        }
//    }
//    if( localY == BLOCK_DIM-1 )
//    {
//        for(int i = 0; i <= halfFilterWidth; i++ )//read pixels to the right
//        {
//            coords = (int2)(x, y+i);
//            localMem[localMemX][localMemY+i] = read_imagef(inputImage, inputImageSampler, coords);
//        }
//    }
//    
//    barrier(CLK_LOCAL_MEM_FENCE);
//    
//    float4 outPixel = (float4)(0.0, 0.0, 0.0, 0.0);
//    
//    //make sure in bounds
//    if( x >= halfFilterWidth && y >= halfFilterWidth && x <= imageWidth[0]-1-halfFilterWidth && y <= imageHeight[0]-1-halfFilterWidth )
//    {
//        int offsetX = -filterWidth[0]/2;//offset from x into input image ( apply filter to neighbors )
//        int offsetY = -filterWidth[0]/2;//offset from y into input image
//        for( int row = 0; row < filterWidth[0]; row++)//apply all rows and cols of filter to image
//        {
//            offsetX += row; 
//            for(int col = 0; col < filterWidth[0]; col++)
//            {
//                offsetY += col;
//                float4 sampledPix = localMem[localMemX+offsetX][localMemY+offsetY];//read_imagef( inputImage, inputImageSampler, coord );
//                
//                outPixel += sampledPix * filter[row*filterWidth[0]+col]; //apply filter to neighboring pixels, and sum
//                
//            }
//            offsetY = -filterWidth[0]/2;
//        }
//        
//        //outPixel = localMem[localMemX][localMemY];
//    }
//    
//    barrier(CLK_LOCAL_MEM_FENCE);
//    
//    coords = (int2)(x, y);
//    write_imagef( outputImage, coords, (float4)(outPixel.x, outPixel.y, outPixel.z, 1.0) );
//    
//} 





//Version2 CONVOLUTION WITH GLOBAL MEMORY

//CAN'T GET THIS TO WORK
//Version below (commented out) is actual version
//this version was just a test- code is identical to convolveGlobalMem
//but does not work if input is constant mem
//even though I never use the constant mem
__kernel void convolveGloballMemConstant( __global float4* inputImageDataBuf, read_only image2d_t inputImage, write_only image2d_t outputImage, sampler_t inputImageSampler, sampler_t outputImageSampler,  __constant int* imageWidth,  __constant int* imageHeight, __constant float* filter, __constant int* filterWidth, unsigned int width, unsigned int height, unsigned int counter )
{
    unsigned int x = get_global_id(0);//get my global id in both dim
    unsigned int y = get_global_id(1);
    int2 coord = (int2)(x, y);
    
    int iw = 512;
    int ih = 512;
    int fw = 3;
    
    int halfFilterWidth = fw/2;//filterWidth/2;
    float4 outPixel = (float4)(0.0, 0.0, 0.0, 0.0);
    
    //init- copy from input texture to global float mem 
    if( counter < 2  && x >= 0 && y >= 0 && x < iw && y < ih )//imageWidth && y < imageHeight )
    {       
        float4 pix = read_imagef( inputImage, inputImageSampler, coord );
        inputImageDataBuf[x*iw+y] = pix;//imageWidth+y] = pix;
        outPixel = pix;
    }
    
    //make sure in bounds
    if( counter >= 2 && x >= halfFilterWidth && y >= halfFilterWidth && x <= iw-1-halfFilterWidth && y <= ih-1-halfFilterWidth )//imageWidth-1-halfFilterWidth && y <= imageHeight-1-halfFilterWidth )
    {
        int offsetX = -halfFilterWidth;//-filterWidth/2;//offset from x into input image ( apply filter to neighbors )
        int offsetY = -halfFilterWidth;//-filterWidth/2;//offset from y into input image
        for( int row = 0; row < fw; row++)//filterWidth; row++)//apply all rows and cols of filter to image
        {
            offsetX += row; 
            for(int col = 0; col < fw; col++)//filterWidth; col++)
            {
                offsetY += col;
                coord = (int2) ( x + offsetX, y + offsetY );
                float4 sampledPix = inputImageDataBuf[ coord.x*iw+coord.y ];//*imageWidth+coord.y ];//read_imagef( inputImage, inputImageSampler, coord );
                outPixel += sampledPix * filter[row*fw+col];//*filterWidth+col]; //apply filter to neighboring pixels, and sum
                
            }
            offsetY = -halfFilterWidth;//-filterWidth/2;
        }
        
        //write out result
        coord = (int2)(x, y);
        write_imagef( outputImage, coord, (float4)(outPixel.x, outPixel.y, outPixel.z, 1.0) );
    }
    else if( counter >= 2 &&x >= 0 && y >= 0 && x < iw && y < ih )//imageWidth && y < imageHeight ) //if in picture bounds but out of convole bounds- (eg. filter can't look at neighbors) make it black
    {
        coord = (int2)(x, y);
        write_imagef( outputImage, coord, (float4)(0.0, 0.0, 0.0, 1.0) );
    }
} 




////Global memory with constant data
//__kernel void convolveGloballMemConstant(__global float4* inputImageDataBuf, read_only image2d_t inputImage, write_only image2d_t outputImage, sampler_t inputImageSampler, sampler_t outputImageSampler,  __constant int* imageWidth,  __constant int* imageHeight, __constant float* filter, __constant int* filterWidth, unsigned int width, unsigned int height, unsigned int counter )
//{  
//    
//    unsigned int x = get_global_id(0);//get my global id in both dim
//    unsigned int y = get_global_id(1);
//    int2 coord = (int2)(x, y);
//    
//    int halfFilterWidth = (int)filterWidth[0]/2;
//    float4 outPixel = (float4)(0.0, 0.0, 0.0, 0.0);
//    //init- copy from input texture to global float mem 
//    if( counter < 2  && x >= 0 && y >= 0 && x < imageWidth[0] && y < imageHeight[0] )
//    {       
//        float4 pix = read_imagef( inputImage, inputImageSampler, coord );
//        inputImageDataBuf[x*imageWidth[0]+y] = pix;
//        outPixel = pix;
//    }
//    
//    //make sure in bounds
//    if( counter >= 2 && x >= halfFilterWidth && y >= halfFilterWidth && x <= imageWidth[0]-1-halfFilterWidth && y <= imageHeight[0]-1-halfFilterWidth )
//    {
//        int offsetX = -filterWidth[0]/2;//offset from x into input image ( apply filter to neighbors )
//        int offsetY = -filterWidth[0]/2;//offset from y into input image
//        for( int row = 0; row < filterWidth[0]; row++)//apply all rows and cols of filter to image
//        {
//            offsetX += row; 
//            for(int col = 0; col < filterWidth[0]; col++)
//            {
//                offsetY += col;
//                coord = (int2) ( x + offsetX, y + offsetY );
//                float4 sampledPix = inputImageDataBuf[ coord.x*imageWidth[0]+coord.y];//imageWidth[0]+coord.y ];//read_imagef( inputImage, inputImageSampler, coord );
//                outPixel += sampledPix * filter[row*filterWidth[0]+col]; //apply filter to neighboring pixels, and sum
//            }
//            offsetY = -filterWidth[0]/2;
//        }
//        
//        
//        if( imageWidth[0] == 512 )
//            outPixel = (float4)(1.0, 0.0, 0.0, 1.0);
//        
//        
////        //write out result
//        coord = (int2)(x, y);
//        write_imagef( outputImage, coord, (float4)(outPixel.x, outPixel.y, outPixel.z, 1.0) );
//    }
//    else if( counter >= 2 &&x >= 0 && y >= 0 && x < imageWidth[0] && y < imageHeight[0] ) //if in picture bounds but out of convole bounds- (eg. filter can't look at neighbors) make it black
//    {
//        coord = (int2)(x, y);
//        write_imagef( outputImage, coord, (float4)(0.0, 0.0, 0.0, 1.0) );
//    }
//    
//    
//} 


//Version6 CONSTANT MEMORY with float4 buffer and Local memory
//__kernel void convolveGlobalMemConstantLocal(  __global float4* inputImageDataBuf, read_only image2d_t inputImage, write_only image2d_t outputImage, sampler_t inputImageSampler, sampler_t outputImageSampler, __constant int* imageWidth,  __constant int* imageHeight, __constant float* filter, __constant int* filterWidth, unsigned int width, unsigned int height , unsigned int counter )
//{
//    int groupX = get_group_id(0);
//    int groupY = get_group_id(1);  
//    int localX = get_local_id(0);
//    int localY = get_local_id(1);
//    int x = get_global_id(0); 
//    int y = get_global_id(1); 
//    
//    //init- copy from input texture to global float mem 
//    if( counter < 2  && x >= 0 && y >= 0 && x < imageWidth[0] && y < imageHeight[0] )
//    {       
//        float4 pix = read_imagef( inputImage, inputImageSampler, coord );
//        inputImageDataBuf[x*imageWidth[0]+y] = pix;
//        outPixel = pix;
//    }
//    else 
//    {
//        int halfFilterWidth = filterWidth[0]/2;
//        
//        //idx into local buffer  : 3 is the current maximum filter size
//        int localMemX = 1 + localX;//filterWidth[0] + localX;
//        int localMemY = 1 + localY;//filterWidth[0] + localY;
//        
//        __local float4 localMem[BLOCK_DIM+2][BLOCK_DIM+2]; 
//        
//        //Read pixel at current coordinate into local mem
//        int2 coords = (int2)(x, y);
//        localMem[localMemX][localMemY] = read_imagef(inputImage, inputImageSampler, coords);
//        
//        //If necessary, read extra pixels
//        if ( localX == 0 ) 
//        {   
//            int minY = 0;
//            int maxY = 0;
//            if( localY == 0 )
//            {
//                minY = -halfFilterWidth;
//                maxY = -1; 
//            }
//            else if( localY == BLOCK_DIM-1 )
//            {
//                minY = 1;
//                maxY = halfFilterWidth;
//            }
//            for(int i = -halfFilterWidth; i < 0; i++ )//read pixels to the right
//            {
//                for(int j = minY; j <= maxY; j++ )
//                {
//                    coords = (int2)(x+i, y+j);
//                    localMem[localMemX+i][localMemY+j] = read_imagef(inputImage, inputImageSampler, coords);
//                }
//            }
//        }
//        if( localY == 0 )
//        {
//            for(int i = -halfFilterWidth; i < 0; i++ )//read pixels to the right
//            {
//                coords = (int2)(x, y+i);
//                localMem[localMemX][localMemY+i] = read_imagef(inputImage, inputImageSampler, coords);
//            }
//        }
//        if( localX == BLOCK_DIM-1 )
//        {
//            int minY = 0;
//            int maxY = 0;
//            if( localY == 0 )
//            {
//                minY = -halfFilterWidth;
//                maxY = -1; 
//            }
//            else if( localY == BLOCK_DIM-1 )
//            {
//                minY = 1;
//                maxY = halfFilterWidth; 
//            }
//            for(int i = 1; i <= halfFilterWidth; i++ )//read pixels to the right
//            {
//                for(int j = minY; j <= maxY; j++)
//                {
//                    coords = (int2)(x+i, y+j);
//                    localMem[localMemX+i][localMemY+j] = read_imagef(inputImage, inputImageSampler, coords);
//                }
//            }
//        }
//        if( localY == BLOCK_DIM-1 )
//        {
//            for(int i = 1; i <= halfFilterWidth; i++ )//read pixels to the right
//            {
//                coords = (int2)(x, y+i);
//                localMem[localMemX][localMemY+i] = read_imagef(inputImage, inputImageSampler, coords);
//            }
//        }
//        
//        barrier(CLK_LOCAL_MEM_FENCE);
//        
//        float4 outPixel = (float4)(0.0, 0.0, 0.0, 0.0);
//        
//        //make sure in bounds
//        if( x >= halfFilterWidth && y >= halfFilterWidth && x <= imageWidth[0]-1-halfFilterWidth && y <= imageHeight[0]-1-halfFilterWidth )
//        {
//            int offsetX = -filterWidth[0]/2;//offset from x into input image ( apply filter to neighbors )
//            int offsetY = -filterWidth[0]/2;//offset from y into input image
//            for( int row = 0; row < filterWidth[0]; row++)//apply all rows and cols of filter to image
//            {
//                offsetX += row; 
//                for(int col = 0; col < filterWidth[0]; col++)
//                {
//                    offsetY += col;
//                    float4 sampledPix = localMem[localMemX+offsetX][localMemY+offsetY];//read_imagef( inputImage, inputImageSampler, coord );
//                    
//                    outPixel += sampledPix * filter[row*filterWidth[0]+col]; //apply filter to neighboring pixels, and sum
//                    
//                }
//                offsetY = -filterWidth[0]/2;
//            }
//            
//            //outPixel = localMem[localMemX][localMemY];
//        }
//        
//        barrier(CLK_LOCAL_MEM_FENCE);
//        
//        coords = (int2)(x, y);
//        write_imagef( outputImage, coords, (float4)(outPixel.x, outPixel.y, outPixel.z, 1.0) );
//    }
//    
//} 


__kernel void initKernel( __global float4* inputImageDataBuf, read_only image2d_t inputImage, sampler_t inputImageSampler, int imageWidth, int imageHeight )
{
    unsigned int x = get_global_id(0);//get my global id in both dim
    unsigned int y = get_global_id(1);
    int2 coord = (int2)(x, y);
    float4 outPixel = (float4)(0.0, 0.0, 0.0, 0.0);
    
    //init- copy from input texture to global float mem 
    if( x >= 0 && y >= 0 && x < imageWidth && y < imageHeight )
    {       
        float4 pix = read_imagef( inputImage, inputImageSampler, coord );
        inputImageDataBuf[x*imageWidth+y] = pix;
        outPixel = pix;
    }
}



