# Minimal build of usd-brep SMLib + SM_API (Apache-2.0) as static libraries,
# plus the SSI evidence runner. No TBB (upstream leaves SM_USE_TBB undefined),
# no Packman/premake/USD.
cmake_minimum_required(VERSION 3.20)
project(ssi_evidence CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(UP ${CMAKE_SOURCE_DIR}/upstream/source)
file(GLOB SMLIB_SRC ${UP}/SMLib/src/*.cpp)
file(GLOB SMAPI_SRC ${UP}/SM_API/src/*.cpp)

# Same defines / suppressions as upstream's macOS release premake config.
# SSI_SM_TRACE=ON instead uses upstream's debug defines (SM_DEBUG_CODE=1), which make
# SMLib print the file/line of every error return; used only to locate error origins.
option(SSI_SM_TRACE "Build SMLib with SM_DEBUG_CODE error tracing" OFF)
if(SSI_SM_TRACE)
  set(SM_DEFS _DEBUG NOMINMAX SM_DEBUG_CODE=1 SM_GRAPHICS_CALLBACKS=1)
else()
  set(SM_DEFS NDEBUG NOMINMAX SM_GRAPHICS_CALLBACKS=1)
endif()
set(SM_OPTS -Wall -Wno-missing-braces -Wno-inconsistent-missing-override
    -Wno-overloaded-virtual -Wno-unused-but-set-variable -Wno-unused-variable
    -Wno-parentheses-equality -w)

add_library(SMLib STATIC ${SMLIB_SRC})
target_include_directories(SMLib PUBLIC ${UP}/SMLib/inc)
target_compile_definitions(SMLib PRIVATE ${SM_DEFS})
target_compile_options(SMLib PRIVATE ${SM_OPTS})

add_library(SM_API STATIC ${SMAPI_SRC})
target_include_directories(SM_API PUBLIC ${UP}/SM_API/inc ${UP}/SMLib/inc)
target_compile_definitions(SM_API PRIVATE ${SM_DEFS} SM_API_EXPORTS)
target_compile_options(SM_API PRIVATE ${SM_OPTS})
target_link_libraries(SM_API PUBLIC SMLib)

if(EXISTS ${CMAKE_SOURCE_DIR}/runner/ssi_runner.cpp)
  add_executable(ssi_runner runner/ssi_runner.cpp)
  target_compile_definitions(ssi_runner PRIVATE ${SM_DEFS})
  target_link_libraries(ssi_runner PRIVATE SM_API)
endif()
