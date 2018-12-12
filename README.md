# FImgP (Fast Image Processing Library)
FImgP (Fast Image Processing Library) aims to be a modern C++ library for image processing in CUDA devices.
The main reason for existence of this project, is that the OpenCV API is written in old C++, meaning that a lot of the meta programming and functional programming capabilities that modern C++ can offer, are wasted.

### Features
One of the core features of FImgP, is its functional programming capabilities, as an example of these, is the apply_pixel functional, better explained with an example:
```
struct mult_add
{
  static __forceinline__ float_3 operator()(float_3 p1,float_3 p2,float mval)
  {
      float_3 result;
      result[0]=p1[0]*mval+p2[0];
      result[1]=p1[1]*mval+p2[1];
      result[2]=p1[2]*mval+p2[2];
      return result;
  }
};
...
apply_function_to_pixel<float_3,mult_add<float_3>>(dest,src1,src2,rows,cols,dest_pitch,src1_pitch,src2_pitch,DEFAULT_STREAM,0.5f);
```
In this case, the apply functional will apply the mult_add function object to two input images and store the result back. One important thing to take notice in this example, is that the mval parameter in mult_add is specified as a variadic argument in the apply functional, with a value of 0.5f.

So as you can see the apply functional creates an abstraction layer over CUDA, meaning that the programmer does not have to write CUDA code, enabling the programmer to write more high level code, with out necessarily knowing the internals of the CUDA language.

## Installation
As all the main components are C++ template classes there is no need to compile anything before use.

### System Requirements
***Hardware requirements***
* CUDA capable device with compute capability 3.5 or higher.

***Software requirements***
* OpenCV 3.0 or later for IO operations.
* CUDA 9.0 or higher.
* GCC 6 if using in combination with CUDA 9.0.
* GCC 7 if using in combination with CUDA 10.0.
* GNU Make & bash to compile the examples.

## Examples
In the examples folder there are two examples:
* 4K merge, which merges two videos with different perspectives into a single output video. Showing how to load and write a video and perform histogram matching. A sample output of this example is: https://youtu.be/d1KHoSzh-pw
* Cell contrast, which takes a series of tif images and produce a video with enhanced contrast. Showing how to load an image, perform histogram matching, and the use of the apply functional. A sample output of this example is: https://youtu.be/iXQBbHwBcs4

To compile these examples you need to change in the makefiles the OpenCV path, the CUDA path, and the device architecture.
## Notice
The library as it is now, will be soon deprecated in favor of a new API which will be released around January-February 2019, with a more robust and user friendly API, some speed improvements and greater variety of functions and algorithms.

## License
For more information about the license check the LICENSE file.
