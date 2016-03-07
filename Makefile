.PHONY: all
all:
	$(MAKE) -C src
	mkdir -p lib
	mv src/libiqo.a lib
	$(MAKE) -C sample

.PHONY: clean
clean:
	$(MAKE) -C src clean
	$(MAKE) -C sample clean