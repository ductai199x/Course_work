SOURCE	:= $(wildcard *.c)
HEADER	:= $(wildcard *.h)
CC	:= gcc
TARGET	:= Main.o
LINK	:= -lm

all: clean $(TARGET)

$(TARGET): $(SOURCE)
	$(CC) -o $(TARGET) $(SOURCE) $(HEADER) $(LINK)

clean:
	rm -f $(TARGET)
