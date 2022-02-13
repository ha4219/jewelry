.SUFFIXES : .cc .o

CXX = g++

INC_DIR = /home/dnatuna/jewelry/coinGenerator/coinGenerator
SRC_DIR = /home/dnatuna/jewelry/coinGenerator/coinGenerator

OPENCV = `pkg-config opencv --cflags --libs`
LIBS = -lglut -lGLU -lGL $(OPENCV)
CXXFLAGS = -g -Wall -std=c++14 -O2 -I$(INC_DIR) -I/usr/local/include/opencv4

SRCS = $(SRC_DIR)/coinGenerator.cpp $(SRC_DIR)/delaunay.cpp $(SRC_DIR)/etf.cpp $(SRC_DIR)/modelGen.cpp
OBJS = $(SRC_DIR)/coinGenerator.o $(SRC_DIR)/delaunay.o $(SRC_DIR)/etf.o $(SRC_DIR)/modelGen.o

PROG = coinGenerator

all : $(PROG)

$(PROG) : $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)
dep :
	gccmakedep $(INC_DIR) $(SRCS)
