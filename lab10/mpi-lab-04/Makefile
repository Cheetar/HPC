CC := CC
LFLAGS :=
ALL := other-dmmmult blas-dmmmult laplace-seq

all : $(ALL)

other-dmmmult: other-dmmmult.cpp
	$(CC) -o $@ $<

#my-dmmmult: my-dmmmult.cpp
#	$(CC) -o $@ $<

blas-dmmmult: blas-dmmmult.cpp
	$(CC) -o $@ $<

laplace-seq: laplace-seq.o laplace-common.o
	$(CC) $(LFLAGS) -o $@ $^

laplace-common.o: laplace-common.cpp Makefile
	$(CC) -c $<

laplace-seq.o: laplace-seq.cpp Makefile
	$(CC) -c $<

clean :
	rm -f $(ALL)
