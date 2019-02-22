VIEWER = less

.PHONY: run test_post post view

run : test_post

test_post : post.py
	- python3 post.py < sample-post.input

post : post.py
	- python3 post.py

view: post.py
	- $(VIEWER) post.py
