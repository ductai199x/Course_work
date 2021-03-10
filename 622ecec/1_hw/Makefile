SOURCE	:= saxpy.c
HEADER	:= $(wildcard *.h)
CC	:= gcc
TARGET	:= saxpy.o
LINK	:= -lpthread -lm
CFLAGS	:= -O3 -Wall
STD	:= c99
REPORT	:= rpt.txt


all: clean $(TARGET) report

$(TARGET): $(SOURCE)
	$(CC) -o $(TARGET) $(SOURCE) $(HEADER) $(CFLAGS) -std=$(STD) $(LINK)

report: $(TARGET)
	./$(TARGET) -e 4 -t 4 -r > $(REPORT)
	./$(TARGET) -e 4 -t 8 -r >> $(REPORT)
	./$(TARGET) -e 4 -t 16 -r >> $(REPORT)
	./$(TARGET) -e 6 -t 4 -r >> $(REPORT)
	./$(TARGET) -e 6 -t 8 -r >> $(REPORT)
	./$(TARGET) -e 6 -t 16 -r >> $(REPORT)
	./$(TARGET) -e 8 -t 4 -r >> $(REPORT)
	./$(TARGET) -e 8 -t 8 -r >> $(REPORT)
	./$(TARGET) -e 8 -t 16 -r >> $(REPORT)
	cat rpt.txt

clean:
	rm -f $(TARGET)
