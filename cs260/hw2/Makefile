VIEWER = less

.PHONY: run view clean

run : problem1.py problem2.py problem3.py view
	- chmod 755 problem1.py
	- chmod 755 problem2.py
	- chmod 755 problem3.py
	- chmod 755 node.py
	- ./problem3.py

view : problem1.py problem2.py
	- $(VIEWER) problem1.py
	- $(VIEWER) problem2.py

clean :
	- rm -rf *.pyc > /dev/null 2>&1
	- rm -rf *.py~ > /dev/null 2>&1
