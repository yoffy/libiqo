TARGET=libiqo.a
OBJS=IQOLanczosResizer.o
OPTFLAGS=-O3 -march=native -ffast-math -D NDEBUG
#OPTFLAGS=-O0 -march=native
CXXFLAGS=-std=c++98 -Wall -Wextra -fno-exceptions -g $(OPTFLAGS) -I ../include

.PHONY: all
all: $(TARGET)

.PHONY: clean
clean:
	$(RM) $(TARGET) $(OBJS)


.SUFFIX: .s
.cpp.s:
	$(CXX) $(CXXFLAGS) -S -o $@ $^

$(TARGET): $(OBJS)
	$(AR) -cr $@ $^
