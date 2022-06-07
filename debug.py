import sys
import time
import cProfile, pstats
from io import StringIO
import os

sys.stdin = open('input.txt', 'r')
sys.stdout = open('output.txt', 'w')
profileFile = "profile.prof"
profileFiletxt = "profile.txt"


def timit(func):
    """
            timit Decorator
Its a debug decorator to get number of function calls and time taken to execute the function.
timit significantly affects running time of functions.
Runtime may increase by 133%.
But time returned by timit has a relative error of 6-10 %.
Actual Example.
>||Time = 2593.29ms|| actual
>||Time = 3672.60ms|| just wrapping
>||Time = 4731.96ms|| wrapping with calls
>||Time = 6051.84ms|| wrapping with calls and time
>||Time = 2784.02ms|| time determined by timit.
    """

    def wrap(*args, **kwargs):
        wrap.call += 1
        val = func(*args, **kwargs)
        wrap.time = time.time() - wrap.time
        return val

    wrap.call = 0
    wrap.time = time.time()
    return wrap


store = (set, list, tuple, dict, zip)


def printrec(a, space=0, d=2, first=True, prefix="", line=False):
    under = line
    if isinstance(a, store):
        if isinstance(a, dict):
            for i in a.values():
                if isinstance(i, store): under = True; break
        else:
            for i in a:
                if isinstance(i, store): under = True; break
        if under:
            if isinstance(a, list):
                print(" " * (space) + prefix + '(' + str(len(a)) + ')' + '[')
                space += len(prefix)
                for i in a:
                    printrec(i, space=space + d, d=d, first=False, line=line)
                if first:
                    print(" " * (space) + ']')
                else:
                    print(" " * (space) + '],\n')

            elif isinstance(a, tuple):
                print(" " * (space) + prefix + '(' + str(len(a)) + ')' + '(')
                space += len(prefix)
                for i in a:
                    printrec(i, space=space + d, d=d, first=False, line=line)
                if first:
                    print(" " * (space) + ')')
                else:
                    print(" " * (space) + '),\n')

            elif isinstance(a, set):
                print(" " * (space) + prefix + '(' + str(len(a)) + ')' + '{')
                space += len(prefix)
                for i in a:
                    printrec(i, space=space + d, d=d, first=False, line=line)
                if first:
                    print(" " * (space) + '}')
                else:
                    print(" " * (space) + '},\n')

            elif isinstance(a, dict):
                print(" " * (space) + prefix + '(' + str(len(a)) + ')' + '{')
                space += len(prefix)
                for k, v in a.items():
                    printrec(v, space=space + d, d=d, first=False, prefix=f"{repr(k)}: ", line=line)
                if first:
                    print(" " * (space) + '}')
                else:
                    print(" " * (space) + '},\n')
            elif isinstance(a, zip):
                print(" " * (space) + prefix + '(' + str(len(tuple(a))) + ')' + '|<')
                space += len(prefix)
                for i in a:
                    printrec(i, space=space + d, d=d, first=False, line=line)
                if first:
                    print(" " * (space) + '>|')
                else:
                    print(" " * (space) + '>|,\n')
        else:
            print(f'{" " * (space) + prefix + "(" + str(len(a)) + ")"}{a}', end=",\n")
    else:
        print(f'{" " * (space + (1 if prefix == "" else 0)) + prefix}{repr(a)}', end=",\n")


def debug(**kwargs):
    """
            debug
A debug function to print variable and objects in a nice and understandable format.
@params: keyWordArguments
@params: line, to print every element of iterable object in new line
@print -> variable_name_passed : variable_referenced_value
For more understanding run debug.py and check examples in main.
    """
    print("_____________________")
    if 'line' in kwargs:
        line = kwargs['line']
        kwargs.pop('line')
    else:
        line = False

    for i, j in kwargs.items():
        print(">", end="")
        if isinstance(j, list) or isinstance(j, tuple):
            if j and isinstance(j[0], store):
                print(f"{i} = {type(j)}")
                printrec(j, space=1, line=line)
            else:
                print(f"{i} = {type(j)}")
                printrec(j, space=1, line=line)
        elif isinstance(j, set):
            print(f"{i} = {type(j)}")
            printrec(j, space=1, line=line)
        elif isinstance(j, dict):
            print(f"{i} = {type(j)}")
            printrec(j, space=1, line=line)
        elif isinstance(j, zip):
            print(f"{i} = (zip)")
            printrec(j, space=1, line=line)
        else:
            if i == 'Time':
                print(f"||{i} = {j}||")
            else:
                print(f"{i} = {j}, ({type(j)})")
    print("---------------------")


def pr(func):
    """
            pr Decorator
Its a debug decorator to print return value of function.
Use example.
When you are directly passing function values and may want to know what you have passed.
rather than making dummy variable to store function values and printing it, just decorate the function with this decorator.
    """

    def wrap(*args, **kwargs):
        val = func(*args, **kwargs)
        debug(func_name=func.__name__, returned=val)
        return val

    return wrap


@pr
def frmt(s, f=3, d=3):
    def frmt_(s, d):
        if len(s) <= d: return s
        return frmt_(s[:-d], d) + "," + s[-d:]

    if type(s) != str: s = str(s)
    if len(s) <= f: return s
    return frmt_(s[:-f], d=d) + "," + s[-f:]


@pr
def no2words(num):
    a = "One Two Three Four Five Six Seven Eight Nine Ten Eleven Twelve Thirteen " \
        "Fourteen Fifteen Sixteen Seventeen Eighteen Nineteen".split()
    b = "Twenty Thirty Forty Fifty Sixty Seventy Eighty Ninety".split()
    if num == 0: return 'Zero'

    def convert(no):
        if no == 0: return []
        if no < 20:
            return [a[no - 1]]
        elif no < 100:
            return [b[no // 10 - 2]] + convert(no % 10)
        elif no < 1000:
            return [a[no // 100 - 1]] + ['Hundred'] + convert(no % 100)
        s = []
        for i, j in enumerate("Billion Million Thousand".split()):
            if no >= 1000 ** (3 - i): s += convert(no // (1000 ** (3 - i))) + [j]
            no = no % 1000 ** (3 - i)
        s += convert(no)
        return s

    return " ".join(convert(num))


def header(heading, l=50):
    print("*" * l)
    print("*" * ((l - len(heading)) // 2) + heading + "*" * ((l - len(heading)) // 2))
    print("*" * l)


def profile(f, *args, **kwargs):
    pr = cProfile.Profile()
    pr.enable()
    # with cProfile.Profile() as pr:
    #     f(*args, **kwargs)
    f(*args, **kwargs)
    pr.disable()

    stats = pstats.Stats(pr)
    stats.dump_stats(profileFile)

    stream = StringIO()
    stats = pstats.Stats(profileFile, stream=stream)
    stats.sort_stats(pstats.SortKey.TIME)

    stats.print_stats()
    stream.seek(0)
    data = stream.read()

    base = os.path.dirname(sys.executable)
    cwd = os.getcwd()

    data = data.replace(base, "=")
    data = data.replace(cwd, "~")
    myText = open(profileFiletxt, 'w')
    myText.write(data)
    myText.close()


if __name__ == "__main__":
    x = 1
    y = '2'
    z = [1, 2, 3, 4]
    a = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
    c = {1: [1, 2, 3], 2: 3}
    d = [[[{'a': 2, 'b': 3}, (1, 2, 3), ['a', 'b', 'c']], 1, 2, 3, 4], {1: 1, 2: 2, 3: 3, 4: 4}, (1, 2, 3, 4), 'a']
    e = ['q', {'a': {1: '11', 2: 22, 3: 33, 4: {'r': 23, 's': 45}}, 'b': [1, 2, 3, 4], 1: ('asd', 'car')}, '2', 3]
    f = {'distance': 1024, 'name': 'danish', 'type': {'a': 1, 'b': 2, 'c': 3, 'd': [[1, 2, 3], [4, 5, 6]]},
         4: (1, 2, 3)}
    g = {'zip1': zip((1, 2, 3, 4), (6, 7, 8, 9))}
    debug(x=x, y=y, z=z, a=a, d=d, e=e, f=f, g=g, line=True)
    no2words(12345)  # @pr and no2words use
    frmt(123456789)  # @pr and frmt use

    # NOTE: remove pr from frmt and no2words if not required any info and only outputs