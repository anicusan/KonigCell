LIB_DEPS = konigcell2d.c
EXAMPLE_DEPS = example.c $(LIB_DEPS)


LIB_OUT = libkonigcell2d.so
EXAMPLE_OUT = kc2d_example


FLAGS = -Wall -Werror -Wextra -fPIC -I. -lm -O3


all: lib example


lib:
	$(CC) $(FLAGS) $(LIB_DEPS) -shared -o $(LIB_OUT)


example:
	$(CC) $(FLAGS) $(EXAMPLE_DEPS) -o $(EXAMPLE_OUT)


.PHONY: clean


clean:
	rm -f $(LIB_OUT) $(EXAMPLE_OUT)
