CXX      = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -static -static-libgcc -static-libstdc++
INCLUDES = -Isource/Lib
SRC      = \
	source/StaticCoder.cpp \
	source/Lib/Utils/global_logger.cpp \
    source/Lib/EncLib/CABACEncoder.cpp \
	source/Lib/EncLib/BinEncoder.cpp \
	source/Lib/DecLib/CABACDecoder.cpp \
	source/Lib/DecLib/BinDecoder.cpp \
	source/Lib/CommonLib/ContextModel.cpp \
	source/Lib/CommonLib/ContextModeler.cpp \
    source/Lib/Test/test_model_args.cpp

TARGET = StaticBAC

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
