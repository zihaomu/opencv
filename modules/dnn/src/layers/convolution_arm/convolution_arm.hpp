// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2022, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_CONVOLUTION_ARM_HPP
#define OPENCV_CONVOLUTION_ARM_HPP

#include "../../precomp.hpp"
#include "convolution_sgemm.hpp"
#include "convolution_sgemm_pack4.hpp"
#include "convolution_sgemm_pack8.hpp"
#include "convolution_pack4.hpp"
#include "convolution_pack8.hpp"

#include "fast_conv_2d.hpp"
#include "conv_2d_3x3s1_winograd.hpp"

//#include "convolution_sgemm_pack8.hpp"
#if __ARM_NEON

// some optimized branch

#endif

#endif //OPENCV_CONVOLUTION_ARM_HPP
