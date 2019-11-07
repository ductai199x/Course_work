#!/usr/bin/env python3

import sys

class Node:
	"""Just a little wrapper for a node in a singly-linked list,
	essentially, a 2-tuple:
		1. the data (the payload, the reason for existing)
		2. the rest of the list

	No operations are provided
	"""

	def __init__( self, data, next=None ):
		self.data = data
		self.next = next

	def __str__( self ):
		return str( self.data )

	def echo( self ):
		print( self.__str__() )


def list2string( L ) :
	
	if L is None :
		rv = '()'
	else :
		rv = '('
		while L.next is not None :
			rv += L.__str__() + ','
			L = L.next
		rv += L.__str__() + ')'

	return rv

def main( argv=sys.argv ) :
	"Just a test driver - NOT part of the class"

	c = Node( 13 )
	print( "c is holding: " + list2string( c ))
	d = Node( 12, c )
	print(  "d is holding: " + list2string( d ))

if __name__ == "__main__" :
	# then this was NOT included in another file, so, run the test driver
	main()

