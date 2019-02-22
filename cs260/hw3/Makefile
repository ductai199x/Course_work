VIEWER = less

.PHONY: prob1 prob2 prob3 clean

prob1 : prob1.py
	- chmod 755 prob1.py
	- ./prob1.py $(arg)

prob2 : prob2.py
	- chmod 755 prob2.py
	- ./prob2.py $(arg)

prob3 : prob1.py prob2.py
	- chmod 755 prob1.py
	- chmod 755 prob2.py
	- ./prob1.py > 1.out
	- ./prob2.py > 2.out
	- gnuplot plot.plg

clean:
	- rm -r *~ > /dev/null 2>&1
	- rm 1.out > /dev/null 2>&1
	- rm 2.out > /dev/null 2>&1
	- rm 2.png > /dev/null 2>&1
