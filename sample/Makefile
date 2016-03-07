TARGET=resize_yuv420p
OBJS=resize_yuv420p.o
OPTFLAGS=-O2 -march=native
LDFLAGS=-L ../lib
CXXFLAGS=-std=c++98 -Wall -Wextra -fno-exceptions -g $(OPTFLAGS) -I ../include

.PHONY: all
all: $(TARGET)

.PHONY: clean
clean:
	$(RM) $(TARGET) $(OBJS)


$(TARGET): $(OBJS) ../lib/libiqo.a
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ -liqo