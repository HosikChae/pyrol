# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/alphred/ALPHRED_V3/Util/vnDataCollect

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/alphred/ALPHRED_V3/Util/vnDataCollect

# Include any dependencies generated for this target.
include CMakeFiles/vnDataCollect.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/vnDataCollect.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/vnDataCollect.dir/flags.make

CMakeFiles/vnDataCollect.dir/main.cpp.o: CMakeFiles/vnDataCollect.dir/flags.make
CMakeFiles/vnDataCollect.dir/main.cpp.o: main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alphred/ALPHRED_V3/Util/vnDataCollect/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/vnDataCollect.dir/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vnDataCollect.dir/main.cpp.o -c /home/alphred/ALPHRED_V3/Util/vnDataCollect/main.cpp

CMakeFiles/vnDataCollect.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vnDataCollect.dir/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alphred/ALPHRED_V3/Util/vnDataCollect/main.cpp > CMakeFiles/vnDataCollect.dir/main.cpp.i

CMakeFiles/vnDataCollect.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vnDataCollect.dir/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alphred/ALPHRED_V3/Util/vnDataCollect/main.cpp -o CMakeFiles/vnDataCollect.dir/main.cpp.s

CMakeFiles/vnDataCollect.dir/main.cpp.o.requires:

.PHONY : CMakeFiles/vnDataCollect.dir/main.cpp.o.requires

CMakeFiles/vnDataCollect.dir/main.cpp.o.provides: CMakeFiles/vnDataCollect.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/vnDataCollect.dir/build.make CMakeFiles/vnDataCollect.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/vnDataCollect.dir/main.cpp.o.provides

CMakeFiles/vnDataCollect.dir/main.cpp.o.provides.build: CMakeFiles/vnDataCollect.dir/main.cpp.o


CMakeFiles/vnDataCollect.dir/CSVWriter.cpp.o: CMakeFiles/vnDataCollect.dir/flags.make
CMakeFiles/vnDataCollect.dir/CSVWriter.cpp.o: CSVWriter.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/alphred/ALPHRED_V3/Util/vnDataCollect/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/vnDataCollect.dir/CSVWriter.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/vnDataCollect.dir/CSVWriter.cpp.o -c /home/alphred/ALPHRED_V3/Util/vnDataCollect/CSVWriter.cpp

CMakeFiles/vnDataCollect.dir/CSVWriter.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/vnDataCollect.dir/CSVWriter.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/alphred/ALPHRED_V3/Util/vnDataCollect/CSVWriter.cpp > CMakeFiles/vnDataCollect.dir/CSVWriter.cpp.i

CMakeFiles/vnDataCollect.dir/CSVWriter.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/vnDataCollect.dir/CSVWriter.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/alphred/ALPHRED_V3/Util/vnDataCollect/CSVWriter.cpp -o CMakeFiles/vnDataCollect.dir/CSVWriter.cpp.s

CMakeFiles/vnDataCollect.dir/CSVWriter.cpp.o.requires:

.PHONY : CMakeFiles/vnDataCollect.dir/CSVWriter.cpp.o.requires

CMakeFiles/vnDataCollect.dir/CSVWriter.cpp.o.provides: CMakeFiles/vnDataCollect.dir/CSVWriter.cpp.o.requires
	$(MAKE) -f CMakeFiles/vnDataCollect.dir/build.make CMakeFiles/vnDataCollect.dir/CSVWriter.cpp.o.provides.build
.PHONY : CMakeFiles/vnDataCollect.dir/CSVWriter.cpp.o.provides

CMakeFiles/vnDataCollect.dir/CSVWriter.cpp.o.provides.build: CMakeFiles/vnDataCollect.dir/CSVWriter.cpp.o


# Object files for target vnDataCollect
vnDataCollect_OBJECTS = \
"CMakeFiles/vnDataCollect.dir/main.cpp.o" \
"CMakeFiles/vnDataCollect.dir/CSVWriter.cpp.o"

# External object files for target vnDataCollect
vnDataCollect_EXTERNAL_OBJECTS =

vnDataCollect: CMakeFiles/vnDataCollect.dir/main.cpp.o
vnDataCollect: CMakeFiles/vnDataCollect.dir/CSVWriter.cpp.o
vnDataCollect: CMakeFiles/vnDataCollect.dir/build.make
vnDataCollect: liblibvncxx.a
vnDataCollect: CMakeFiles/vnDataCollect.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/alphred/ALPHRED_V3/Util/vnDataCollect/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable vnDataCollect"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/vnDataCollect.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/vnDataCollect.dir/build: vnDataCollect

.PHONY : CMakeFiles/vnDataCollect.dir/build

CMakeFiles/vnDataCollect.dir/requires: CMakeFiles/vnDataCollect.dir/main.cpp.o.requires
CMakeFiles/vnDataCollect.dir/requires: CMakeFiles/vnDataCollect.dir/CSVWriter.cpp.o.requires

.PHONY : CMakeFiles/vnDataCollect.dir/requires

CMakeFiles/vnDataCollect.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/vnDataCollect.dir/cmake_clean.cmake
.PHONY : CMakeFiles/vnDataCollect.dir/clean

CMakeFiles/vnDataCollect.dir/depend:
	cd /home/alphred/ALPHRED_V3/Util/vnDataCollect && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/alphred/ALPHRED_V3/Util/vnDataCollect /home/alphred/ALPHRED_V3/Util/vnDataCollect /home/alphred/ALPHRED_V3/Util/vnDataCollect /home/alphred/ALPHRED_V3/Util/vnDataCollect /home/alphred/ALPHRED_V3/Util/vnDataCollect/CMakeFiles/vnDataCollect.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/vnDataCollect.dir/depend
