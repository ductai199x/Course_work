SOURCE	:= jacobi_solver.c compute_gold.c
HEADER	:= $(wildcard *.h)
CC	:= gcc
TARGET	:= jacobi_solver.o
LINK	:= -fopenmp -lpthread -lm
CFLAGS	:= -O3 -Wall
STD		:=
REPORT	:= rpt.txt


all: clean $(TARGET) report

$(TARGET): $(SOURCE)
	$(CC) -o $(TARGET) $(SOURCE) $(HEADER) $(CFLAGS) $(STD) $(LINK)

report: $(TARGET)
	./$(TARGET) -s 512 -t 4 -r > $(REPORT)
	./$(TARGET) -s 512 -t 8 -r >> $(REPORT)
	./$(TARGET) -s 512 -t 16 -r >> $(REPORT)
	./$(TARGET) -s 512 -t 32 -r >> $(REPORT)
	./$(TARGET) -s 1024 -t 4 -r >> $(REPORT)
	./$(TARGET) -s 1024 -t 8 -r >> $(REPORT)
	./$(TARGET) -s 1024 -t 16 -r >> $(REPORT)
	./$(TARGET) -s 1024 -t 32 -r >> $(REPORT)
	./$(TARGET) -s 2048 -t 4 -r >> $(REPORT)
	./$(TARGET) -s 2048 -t 8 -r >> $(REPORT)
	./$(TARGET) -s 2048 -t 16 -r >> $(REPORT)
	./$(TARGET) -s 2048 -t 32 -r >> $(REPORT)
	cat rpt.txt

clean:
	rm -f $(TARGET)
