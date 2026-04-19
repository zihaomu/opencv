cmake -DWITH_EIGEN=OFF \
    -DBUILD_opencv_dnn=OFF \
    -DWITH_OPENCL=OFF \
    -DWITH_CAROTENE=OFF \
    -DWITH_IPP=OFF \
    -DWITH_KLEIDICV=OFF \
    -DBUILD_opencv_imgcodecs=OFF \
    -DWITH_JPEG=OFF \
    ..

make -j8