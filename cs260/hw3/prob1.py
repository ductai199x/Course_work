#!/usr/bin/env python

import sys
import timeit

def naive_fib(n):
    if n <= 1:
        return 1
    else:
        return naive_fib(n - 1) + naive_fib(n - 2)

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

def main():
    if len(sys.argv) > 1:
        fn = sys.argv[1]
        print naive_fib(fn)
    else:
        n = [i for i in range(5, 45, 5)]
        for fn in n:
            wrapped = wrapper(naive_fib, fn)
            print fn, timeit.timeit(wrapped, number=1)


if __name__ == "__main__":
    main()


