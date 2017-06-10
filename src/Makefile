TARGET=libiqo.a
OBJS=IQOLanczosResizer.o
OPTFLAGS=-Ofast -march=haswell -D NDEBUG
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
