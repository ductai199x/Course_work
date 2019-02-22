PYTHON = python
VIEWER = more

.PHONY: clean run view

run :  hello.py
	chmod 755 hello.py 
	$(PYTHON) hello.py

view : hello.py
	$(VIEWER) hello.py 

clean :
	- \rm $(outFile) > /dev/null 2>&1
	- \rm *.pyc > /dev/null 2>&1
