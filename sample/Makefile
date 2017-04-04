TARGET=resize_yuv420p
OBJS=resize_yuv420p.o
OPTFLAGS=-O2 -march=native
LDFLAGS=-L ../lib -L /usr/local/lib -L /opt/intel/compilers_and_libraries_2016.2.181/linux/ipp/lib/intel64
CXXFLAGS=-std=c++98 -Wall -Wextra -fno-exceptions -g $(OPTFLAGS) -I ../include -I /usr/local/include -I /opt/intel/compilers_and_libraries_2016.2.181/linux/ipp/include

.PHONY: all
all: $(TARGET)

.PHONY: clean
clean:
	$(RM) $(TARGET) $(OBJS)


$(TARGET): $(OBJS) ../lib/libiqo.a
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ -liqo -lopencv_core -lopencv_imgproc -lippcore -lippi -lipps
