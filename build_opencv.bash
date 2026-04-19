cmake -DWITH_EIGEN=OFF \
    -DBUILD_opencv_dnn=OFF \
    -DWITH_OPENCL=OFF \
    -DWITH_CAROTENE=OFF \
    -DWITH_IPP=OFF \
    -DWITH_KLEIDICV=OFF \
    -DBUILD_opencv_imgcodecs=OFF \
    -DENABLE_LIBJPEG_TURBO_SIMD=OFF \
    ..

make -j8