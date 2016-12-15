OBJS = Turbofan_TASOPTc.o driver_Turbofan.o
CC = g++
CFLAGS = -c -larmadillo -fPIC
LFLAGS = -larmadillo 

Turbofan : $(OBJS)
	$(CC) $(LFLAGS) $(OBJS) -o Turbofan

Turbofan_TASOPTc.o	: Turbofan_TASOPTc.hpp
	$(CC) $(CFLAGS) Turbofan_TASOPTc.cpp

driver_Turbofan.o	: Turbofan_TASOPTc.hpp driver_Turbofan.hpp
	$(CC) $(CFLAGS) driver_Turbofan.cpp

clean :
	\rm *.o *~ p1