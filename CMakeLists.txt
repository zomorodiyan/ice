cmake_minimum_required(VERSION 3.16.3)

set(This orange)

project(${This} CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # to create position independent executables or shared libraries
enable_testing()
add_subdirectory(test)

set(Headers
  main.hpp
)
set(Sources
  main.cpp
)
add_library(${PROJECT_NAME} STATIC ${Sources} ${Headers}) # to use for gtest
#add_executable(${PROJECT_NAME} main.cpp)


# Download all the submodules
find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")

# Update somodules as needed
  option(GIT_SUBMODULE "Check submodules during build" ON)
  if(GIT_SUBMODULE)
    message(STATUS "Submodule update")
    execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    RESULT_VARIABLE GIT_SUBMOD_RESULT)
    if(NOT GIT_SUBMOD_RESULT EQUAL "0")
      message(FATAL_ERROR "git submodule update --init failed with
      ${GIT_SUBMOD_RESULT}, please checkout submodules")
    endif()
  endif()
endif()

# Check all the sobmodules
if(NOT EXISTS "${PROJECT_SOURCE_DIR}/external/gtest/CMakeLists.txt")
  message(FATAL_ERROR "the gtest submodule was not downloaded! GIT_SUBMODULE
  was turned off or failed. pleach checkout submodules.")
endif()

# Add google test library
add_subdirectory(external/gtest)
target_link_libraries(${PROJECT_NAME} PRIVATE gtest)
target_include_directories(${PROJECT_NAME} PRIVATE gtest/include)

# Add internal libraries (peeler, etc.)
add_subdirectory(peeler)
target_link_libraries(${PROJECT_NAME} PRIVATE peeler)
target_include_directories(${PROJECT_NAME} PRIVATE peeler/)
