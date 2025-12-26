# Install script for directory: /data/dahu/mlsys/MNN

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "TRUE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include" TYPE DIRECTORY FILES "/data/dahu/mlsys/MNN/transformers/llm/engine/include/" FILES_MATCHING REGEX "/[^/]*\\.hpp$")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/MNN" TYPE FILE FILES
    "/data/dahu/mlsys/MNN/include/MNN/MNNDefine.h"
    "/data/dahu/mlsys/MNN/include/MNN/Interpreter.hpp"
    "/data/dahu/mlsys/MNN/include/MNN/HalideRuntime.h"
    "/data/dahu/mlsys/MNN/include/MNN/Tensor.hpp"
    "/data/dahu/mlsys/MNN/include/MNN/ErrorCode.hpp"
    "/data/dahu/mlsys/MNN/include/MNN/ImageProcess.hpp"
    "/data/dahu/mlsys/MNN/include/MNN/Matrix.h"
    "/data/dahu/mlsys/MNN/include/MNN/Rect.h"
    "/data/dahu/mlsys/MNN/include/MNN/MNNForwardType.h"
    "/data/dahu/mlsys/MNN/include/MNN/AutoTime.hpp"
    "/data/dahu/mlsys/MNN/include/MNN/MNNSharedContext.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/MNN/expr" TYPE FILE FILES
    "/data/dahu/mlsys/MNN/include/MNN/expr/Expr.hpp"
    "/data/dahu/mlsys/MNN/include/MNN/expr/ExprCreator.hpp"
    "/data/dahu/mlsys/MNN/include/MNN/expr/MathOp.hpp"
    "/data/dahu/mlsys/MNN/include/MNN/expr/NeuralNetWorkOp.hpp"
    "/data/dahu/mlsys/MNN/include/MNN/expr/Optimizer.hpp"
    "/data/dahu/mlsys/MNN/include/MNN/expr/Executor.hpp"
    "/data/dahu/mlsys/MNN/include/MNN/expr/Module.hpp"
    "/data/dahu/mlsys/MNN/include/MNN/expr/NeuralNetWorkOp.hpp"
    "/data/dahu/mlsys/MNN/include/MNN/expr/ExecutorScope.hpp"
    "/data/dahu/mlsys/MNN/include/MNN/expr/Scope.hpp"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/data/dahu/mlsys/MNN/project/android/build/libMNN.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libMNN.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libMNN.so")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/data/dahu/android-ndk-r29/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libMNN.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/data/dahu/mlsys/MNN/project/android/build/source/backend/opencl/cmake_install.cmake")
  include("/data/dahu/mlsys/MNN/project/android/build/express/cmake_install.cmake")
  include("/data/dahu/mlsys/MNN/project/android/build/tools/cv/cmake_install.cmake")
  include("/data/dahu/mlsys/MNN/project/android/build/tools/audio/cmake_install.cmake")
  include("/data/dahu/mlsys/MNN/project/android/build/tools/converter/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/data/dahu/mlsys/MNN/project/android/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
