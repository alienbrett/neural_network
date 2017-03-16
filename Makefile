CC = g++
CFLAGS = -std=c++11 -Wall -Wextra -Wshadow -pthread

main: main.cpp Network.o Vector_Storage.o
	$(CC) $(CFLAGS) -o main main.cpp Network.o Vector_Storage.o

Network.o: Network.cpp Network.h
	$(CC) $(CFLAGS) -c Network.cpp

Vector_Storage.o: Vector_Storage.cpp Vector_Storage.h
	$(CC) $(CFLAGS) -c Vector_Storage.cpp

clean:
	rm main *.o
