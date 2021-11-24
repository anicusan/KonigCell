all: konigcell2d konigcell3d


konigcell2d:
	make -C konigcell2d


konigcell3d:
	make -C konigcell3d


.PHONY: clean konigcell2d konigcell3d


clean:
	make -C konigcell2d clean
	make -C konigcell3d clean
