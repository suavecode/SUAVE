from cProfile import Profile

# install using pip install line_profiler
from line_profiler import LineProfiler


'''
Profile the given function line by line
:param f: test function
:return: the profiled function, evaluated
'''


def line_profile(f):
    def profiled_func(*args, **kwargs):
        try:
            lp = LineProfiler()
            lp.add_function(f)
            lp.enable_by_count()
            return f(*args, **kwargs)
        finally:
            lp.print_stats()

    return profiled_func


'''
Profile the given function
:param f: test function
:return: the profiled function, evaluated
'''


def profile(f):
    def profiled_func(*args, **kwargs):
        p = Profile()
        try:
            # profile the input function
            p.enable()
            r = f(*args, **kwargs)
            p.disable()
            return r
        finally:
            p.print_stats()

    return profiled_func


'''
A test function
:return:
'''


def get_number():
    for x in xrange(5000000):
        yield x


'''
A test function
:return:
'''


@line_profile  # profile the decorated function, line-by-line
@profile  # profile the decorated function
def test_function():
    for x in get_number():
        i = x ^ x ^ x


if __name__ == '__main__':
    import sys
    sys.exit(test_function())
