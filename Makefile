CC=g++
CNN_DIR = ./cnn
EIGEN = ./eigen
CNN_BUILD_DIR=$(CNN_DIR)/build
INCS=-I$(CNN_DIR) -I$(CNN_BUILD_DIR) -I$(EIGEN)
LIBS=-L$(CNN_BUILD_DIR)/cnn/
FINAL=-lcnn -lboost_regex -lboost_serialization -lboost_program_options -lrt -lpthread
#FINAL=-lcnn -lcnncuda -lboost_regex -lboost_serialization -lboost_program_options -lcuda -lcudart -lcublas
CFLAGS=-std=c++11 -Ofast -g -march=native -pipe
#CFLAGS=-std=c++11 -Wall -pedantic -O0 -g -pipe -DDEBUG
BINDIR=bin
OBJDIR=obj
SRCDIR=src

.PHONY: clean
all: make_dirs $(BINDIR)/train $(BINDIR)/sample $(BINDIR)/loss $(BINDIR)/modes

make_dirs:
	mkdir -p $(OBJDIR)
	mkdir -p $(BINDIR)

include $(wildcard $(OBJDIR)/*.d)

$(OBJDIR)/%.o: $(SRCDIR)/%.cc
	$(CC) $(CFLAGS) $(INCS) -c $< -o $@
	$(CC) -MM -MP -MT "$@" $(CFLAGS) $(INCS) $< > $(OBJDIR)/$*.d

$(BINDIR)/train: $(addprefix $(OBJDIR)/, train.o mlp.o io.o morphlm.o utils.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/modes: $(addprefix $(OBJDIR)/, modes.o mlp.o io.o morphlm.o utils.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/sample: $(addprefix $(OBJDIR)/, sample.o mlp.o io.o morphlm.o utils.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

$(BINDIR)/loss: $(addprefix $(OBJDIR)/, loss.o mlp.o io.o morphlm.o utils.o)
	$(CC) $(CFLAGS) $(LIBS) $(INCS) $^ -o $@ $(FINAL)

clean:
	rm -rf $(BINDIR)/*
	rm -rf $(OBJDIR)/*
