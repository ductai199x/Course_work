VIEWER=less

.PHONY: prob1 prob2 prob2-view

prob1: set.py test-set.py
	- chmod 755 set.py
	- chmod 755 test-set.py
	- ./test-set.py

prob2: set.py bfs_dfs.py
	- chmod 755 set.py
	- chmod 755 bfs_dfs.py
	- ./bfs_dfs.py

prob2-view: bfs_dfs.py
	- $(VIEWER) bfs_dfs.py
