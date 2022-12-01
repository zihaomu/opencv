# ----------------------------------------------------------------------------
#  Detect frameworks that may be used by 3rd-party libraries as well as OpenCV
# ----------------------------------------------------------------------------

# --- HPX ---
if(WITH_HPX)
  find_package(HPX REQUIRED)
  ocv_include_directories(${HPX_INCLUDE_DIRS})
  set(HAVE_HPX TRUE)
endif(WITH_HPX)

# --- GCD ---
if(APPLE AND NOT HAVE_TBB)
  set(HAVE_GCD 1)
else()
  set(HAVE_GCD 0)
endif()

# --- Concurrency ---
if(MSVC AND NOT HAVE_TBB AND NOT OPENCV_DISABLE_THREAD_SUPPORT)
  set(_fname "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/concurrencytest.cpp")
  file(WRITE "${_fname}" "#if _MSC_VER < 1600\n#error\n#endif\nint main() { return 0; }\n")
  try_compile(HAVE_CONCURRENCY "${CMAKE_BINARY_DIR}" "${_fname}")
  file(REMOVE "${_fname}")
else()
  set(HAVE_CONCURRENCY 0)
endif()

# --- OpenMP ---
if(WITH_OPENMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    set(HAVE_OPENMP 1)
  else() # manually set openmp lib and header file.
    message(STATUS "OpenMP: Trying to setup OpenMP manually.")
    set(OpenMP_DIR "" CACHE PATH "Path to OpenMP installation, which contains OpenMP's header and library files.")
    set(OpenMP_libomp_LIBRARY "" CACHE PATH "Path to libomp.")

    if(NOT OpenMP_DIR)
      message(STATUS "OpenMP: Please manually set OpenMP_DIR to enable OpenMP!")
    else()
      find_library(OpenMP_libomp_LIBRARY NAMES omp PATHS "${OpenMP_DIR}/lib" NO_DEFAULT_PATH)
      find_path(OpenMP_INCLUDE_DIR NAMES omp.h PATHS "${OpenMP_DIR}/include" NO_DEFAULT_PATH)

      if(NOT OpenMP_libomp_LIBRARY)
        message(STATUS "OpenMP: Failed to find libomp in ${OpenMP_DIR}/lib. Turning off OpenMP.")
      endif()

      if(NOT OpenMP_INCLUDE_DIR)
        message(STATUS "OpenMP: Failed to find omp.h in ${OpenMP_DIR}/include. Turning off OpenMP.")
      else()
        ocv_include_directories(${OpenMP_INCLUDE_DIR})
      endif()

      if(OpenMP_libomp_LIBRARY AND OpenMP_INCLUDE_DIR)
        set(HAVE_OPENMP 1)
      else()
        set(HAVE_OPENMP 0)
      endif()
    endif()
  endif()

  if(HAVE_OPENMP)
    if((NOT OpenMP_CXX_FOUND) AND (NOT OPENMP_FOUND))
      if(IOS OR APPLE)
        set(OpenMP_CXX_FLAGS "-Xclang -fopenmp")
        set(OpenMP_C_FLAGS "-Xclang -fopenmp")
      else()
        set(OpenMP_CXX_FLAGS "-fopenmp")
        set(OpenMP_C_FLAGS "-fopenmp")
      endif()
    endif()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  endif()
endif()
MARK_AS_ADVANCED(OpenMP_INCLUDE_DIR)

ocv_clear_vars(HAVE_PTHREADS_PF)
if(WITH_PTHREADS_PF AND HAVE_PTHREAD)
  set(HAVE_PTHREADS_PF 1)
else()
  set(HAVE_PTHREADS_PF 0)
endif()
