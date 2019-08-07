cmake_minimum_required(VERSION 3.5)
project(vnDataCollect)

set(CMAKE_CXX_STANDARD 14)

#### Find/Create libraries that will be used
set(SOURCE  # Set the source for the VN libraries
        src/attitude.cpp
        src/compositedata.cpp
        src/conversions.cpp
        src/criticalsection.cpp
        src/dllvalidator.cpp
        src/error_detection.cpp
        src/event.cpp
        src/ezasyncdata.cpp
        src/memoryport.cpp
        src/packet.cpp
        src/packetfinder.cpp
        src/port.cpp
        src/position.cpp
        src/searcher.cpp
        src/sensors.cpp
        src/serialport.cpp
        src/thread.cpp
        src/types.cpp
        src/util.cpp
        src/utilities.cpp
        src/vntime.cpp
        include/vn/exceptions.h
        include/vn/matrix.h
        include/vn/attitude.h
        include/vn/boostpython.h
        include/vn/dllvalidator.h
        include/vn/signal.h
        include/vn/error_detection.h
        include/vn/position.h
        include/vn/registers.h
        include/vn/utilities.h
        include/vn/memoryport.h
        include/vn/nocopy.h
        include/vn/compositedata.h
        include/vn/criticalsection.h
        include/vn/compiler.h
        include/vn/sensors.h
        include/vn/searcher.h
        include/vn/event.h
        include/vn/ezasyncdata.h
        include/vn/serialport.h
        include/vn/export.h
        include/vn/vector.h
        include/vn/vntime.h
        include/vn/packetfinder.h
        include/vn/conversions.h
        include/vn/types.h
        include/vn/int.h
        include/vn/thread.h
        include/vn/mock.h
        include/vn/port.h
        include/vn/util.h
        include/vn/consts.h
        include/vn/packet.h)
# Create a VN library from the source
add_library(libvncxx ${SOURCE})

#### Include the header files
include_directories(include)    # VectorNav headers are local in the project (ie. not installed into /usr/local/include)

add_executable(vnDataCollect main.cpp CSVWriter.h CSVWriter.cpp)

#### Link the libraries
target_link_libraries(vnDataCollect LINK_PRIVATE libvncxx)  # VectorNav API (not installed)
if (UNIX OR APPLE)
    target_link_libraries(vnDataCollect LINK_PUBLIC pthread)    # POSIX thread libraries
else()
    target_link_libraries(vnDataCollect LINK_PUBLIC Advapi32 SetupAPI)
endif()