#!/usr/bin/env python

import sys
import timeit

memo_dict = {0: 1, 1: 1}

def memo_fib(n):
    if n in memo_dict:
        return memo_dict[n]
    else:
        fib = memo_fib(n-1) + memo_fib(n-2)
        memo_dict[n] = fib
        return fib

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

def main():
    global memo_dict
    if len(sys.argv) > 1:
        fn = sys.argv[1]
        memo_dict = {0: 1, 1: 1}
        print memo_fib(fn)
    else:
        n = [i for i in range(450, 1000, 50)]
        for fn in n:
            wrapped = wrapper(memo_fib, fn)
            memo_dict = {0: 1, 1: 1}
            print fn, timeit.timeit(wrapped, number=1000)


if __name__ == "__main__":
    main()


