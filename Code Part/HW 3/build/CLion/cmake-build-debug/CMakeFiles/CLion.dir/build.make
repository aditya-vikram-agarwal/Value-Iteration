# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

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
CMAKE_COMMAND = /snap/clion/69/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /snap/clion/69/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/pthomas/Desktop/HW3/build/CLion

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/pthomas/Desktop/HW3/build/CLion/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/CLion.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/CLion.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/CLion.dir/flags.make

CMakeFiles/CLion.dir/home/pthomas/Desktop/HW3/main.cpp.o: CMakeFiles/CLion.dir/flags.make
CMakeFiles/CLion.dir/home/pthomas/Desktop/HW3/main.cpp.o: /home/pthomas/Desktop/HW3/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pthomas/Desktop/HW3/build/CLion/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/CLion.dir/home/pthomas/Desktop/HW3/main.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CLion.dir/home/pthomas/Desktop/HW3/main.cpp.o -c /home/pthomas/Desktop/HW3/main.cpp

CMakeFiles/CLion.dir/home/pthomas/Desktop/HW3/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CLion.dir/home/pthomas/Desktop/HW3/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/pthomas/Desktop/HW3/main.cpp > CMakeFiles/CLion.dir/home/pthomas/Desktop/HW3/main.cpp.i

CMakeFiles/CLion.dir/home/pthomas/Desktop/HW3/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CLion.dir/home/pthomas/Desktop/HW3/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/pthomas/Desktop/HW3/main.cpp -o CMakeFiles/CLion.dir/home/pthomas/Desktop/HW3/main.cpp.s

# Object files for target CLion
CLion_OBJECTS = \
"CMakeFiles/CLion.dir/home/pthomas/Desktop/HW3/main.cpp.o"

# External object files for target CLion
CLion_EXTERNAL_OBJECTS =

CLion: CMakeFiles/CLion.dir/home/pthomas/Desktop/HW3/main.cpp.o
CLion: CMakeFiles/CLion.dir/build.make
CLion: CMakeFiles/CLion.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/pthomas/Desktop/HW3/build/CLion/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable CLion"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/CLion.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/CLion.dir/build: CLion

.PHONY : CMakeFiles/CLion.dir/build

CMakeFiles/CLion.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/CLion.dir/cmake_clean.cmake
.PHONY : CMakeFiles/CLion.dir/clean

CMakeFiles/CLion.dir/depend:
	cd /home/pthomas/Desktop/HW3/build/CLion/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/pthomas/Desktop/HW3/build/CLion /home/pthomas/Desktop/HW3/build/CLion /home/pthomas/Desktop/HW3/build/CLion/cmake-build-debug /home/pthomas/Desktop/HW3/build/CLion/cmake-build-debug /home/pthomas/Desktop/HW3/build/CLion/cmake-build-debug/CMakeFiles/CLion.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/CLion.dir/depend

