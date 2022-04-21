//
//  speed_up_vadim.h
//  OpenCV
//
//  Created by Zihao Mu on 2022/4/19.
//

#ifndef speed_up_vadim_h
#define speed_up_vadim_h

//#ifdef __ARM_NEON
//#include <arm_neon.h>
//#endif
//
//#ifdef __ARM_NEON
//enum { FX_CONV_MR=8, FX_CONV_NR=12 };
//#elif defined __AVX__
//enum { FX_CONV_MR=6, FX_CONV_NR=8 };
//#else
//enum { FX_CONV_MR=6, FX_CONV_NR=4 };
//#endif

#include "../../precomp.hpp"
namespace cv { namespace dnn {



//static void convolution_vadim(InputArray _input, OutputArray _output, InputArray _colKernel, const std::vector<float>& bias, const std::vector<float>& reluslope, int kernelW, int kernelH, int stride_w, int stride_h, int dilation_w, int dilation_h)
//{
//    
//}

}} // namespace cv::dnn

#endif /* speed_up_vadim_h */
