CC = g++
CFLAGS = -std=c++11 -Wall -Wextra -Wshadow -pthread

main: main.cpp Network.o VectorIO.o
	$(CC) $(CFLAGS) -o main main.cpp Network.o VectorIO.o

Network.o: Network.cpp Network.h
	$(CC) $(CFLAGS) -c Network.cpp

VectorIO.o: VectorIO.cpp VectorIO.h
	$(CC) $(CFLAGS) -c VectorIO.cpp

clean:
	rm main *.o
