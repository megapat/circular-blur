#
# GLSL example
#

OCL_UTILS=\
	/Developer/GPU\ Computing/shared/obj/x86_64/release/shrUtils.cpp.o \
	/Developer/GPU\ Computing/shared/obj/x86_64/release/cmd_arg_reader.cpp.o \
	/Developer/GPU\ Computing/shared/obj/x86_64/release/rendercheckGL.cpp.o \
	/Developer/GPU\ Computing/OpenCL/common/lib/liboclUtil_x86_64.a

project4: project4.o 
	g++ project4.o -framework GLUT -framework OpenGL -framework OpenCL -o project4 -lGLEW -L/opt/local/lib -lXmu -framework AppKit $(OCL_UTILS)  `pkg-config --libs --cflags libpng`

project4.o: project4.cpp
	g++ -c -o project4.o project4.cpp  -I/System/Library/Frameworks/OpenCL.framework/Headers/ -I/System/Library/Frameworks/GLUT.framework/Headers -I/System/Library/Frameworks/OpenGL.framework/Headers -I/opt/local/include -I/Developer/GPU\ Computing/OpenCL/common/inc -I/Developer/GPU\ Computing/shared/inc `pkg-config --cflags libpng`


clean:
	rm -rf project4.o project4

.PHONY: clean

