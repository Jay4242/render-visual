CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -pedantic
LDFLAGS = -lraylib -lGL -lm -lpthread -ldl -lrt -lX11

TARGET = render-visual

all: $(TARGET)

$(TARGET): render-visual.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

.PHONY: clean
clean:
	rm -f $(TARGET)
