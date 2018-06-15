OBJECTS= constructs.o mtrand.o
WXFLAGS=`wx-config --cppflags --libs std,gl`
CPPFLAGS=-g3   -pthread -Wall $(WXFLAGS)


all: lamodel


cleanup:
	rm -f submit.sh.e*
	rm -f submit.sh.o*
