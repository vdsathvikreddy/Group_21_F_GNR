cmake_minimum_required(VERSION 3.15)
project(conv_ops)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Get pybind11 include directory from pip installation
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(PYBIND11_CMAKE_DIR)
    # Use the found pybind11
    find_package(pybind11 CONFIG PATHS ${PYBIND11_CMAKE_DIR})
else()
    # Fallback: get include directory directly
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_include())"
        OUTPUT_VARIABLE PYBIND11_INCLUDE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    
    if(NOT PYBIND11_INCLUDE_DIR)
        message(FATAL_ERROR "pybind11 not found. Install with: pip install pybind11")
    endif()
    
    # Manual pybind11 setup
    add_library(pybind11::module INTERFACE IMPORTED)
    set_target_properties(pybind11::module PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${PYBIND11_INCLUDE_DIR};${Python3_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${Python3_LIBRARIES}"
    )
    
    # Define pybind11_add_module manually
    function(pybind11_add_module target_name)
        add_library(${target_name} MODULE ${ARGN})
        target_link_libraries(${target_name} PRIVATE pybind11::module)
        set_target_properties(${target_name} PROPERTIES
            PREFIX "${PYTHON_MODULE_PREFIX}"
            SUFFIX "${PYTHON_MODULE_EXTENSION}"
        )
        if(MSVC)
            target_compile_definitions(${target_name} PRIVATE _USE_MATH_DEFINES)
        endif()
    endfunction()
endif()

# Add optimization flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
endif()

# Source files
set(SOURCES
    conv_ops.cpp
)

# Create the extension module
pybind11_add_module(conv_ops ${SOURCES})

# Link libraries
target_link_libraries(conv_ops PRIVATE)
