all: konigcell2d konigcell3d


konigcell2d:
	$(MAKE) -C konigcell2d


konigcell3d:
	$(MAKE) -C konigcell3d


.PHONY: clean konigcell2d konigcell3d


clean:
	$(MAKE) -C konigcell2d clean
	$(MAKE) -C konigcell3d clean
