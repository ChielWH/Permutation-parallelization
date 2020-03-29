import math
import types
import pyximport
pyximport.install()
from itertools import product, permutations
from collections import Counter
from perm_by_index_optimized import permutation_by_index

# the optimized versions of the built-in permutations generator and of the permutation by index function
indexed_perm_opt = permutation_by_index
built_it_perm_opt = permutations


def indexed_perm_unopt(n, k):
    """get the permutation at index k for permutations of n players

    Args:
        n (int): The length of the set of players
        k (int): The index of the permutation

    Returns:
        TYPE: tuple (with index of the elements)
    """
    numbers = list(range(n))
    permutation = []
    k -= 1
    while n > 0:
        n -= 1
        # get the index of current digit
        index, k = divmod(k, math.factorial(n))
        permutation.append(numbers.pop(index))

    return tuple(permutation)


def built_it_perm_unopt(lst):
    """The pseudocode given for Python's C implementation of permutation
    from: http://svn.python.org/view/python/branches/py3k/Modules/itertoolsmodule.c?view=markup
    but it only works in python 2k :)"""
    pool = tuple(lst)
    n = len(pool)
    r = n
    indices = list(range(n))
    cycles = list(range(n - r + 1, n + 1))[::-1]
    yield tuple(pool[i] for i in indices[:r])
    while n:
        for i in reversed(range(r)):
            cycles[i] -= 1
            if cycles[i] == 0:
                indices[i:] = indices[i + 1:] + indices[i:i + 1]
                cycles[i] = n - i
            else:
                j = cycles[i]
                indices[i], indices[-j] = indices[-j], indices[i]
                yield tuple(pool[i] for i in indices[:r])
                break
        else:
            return


def product_permutations(iterable):
    pool = tuple(iterable)
    n = len(pool)
    for indices in product(range(n), repeat=n):
        if len(set(indices)) == n:
            yield tuple(pool[i] for i in indices)


def perm1(lst):
    yield tuple(lst)

    if len(lst) == 1:
        return

    n = len(lst) - 1
    while 1:
        j = n - 1
        while lst[j] >= lst[j + 1]:
            j -= 1
            if j == -1:
                return  # terminate
        l = n
        while lst[j] >= lst[l]:
            l -= 1
        lst[j], lst[l] = lst[l], lst[j]
        k = j + 1
        l = n
        while k < l:
            lst[k], lst[l] = lst[l], lst[k]
            k += 1
            l -= 1
        yield tuple(lst)


def perm2(lst):
    yield tuple(lst)
    if len(lst) == 1:
        return
    if len(lst) == 2:
        lst[0], lst[1] = lst[1], lst[0]
        yield tuple(lst)
        return

    n = len(lst) - 1
    while 1:
        # half the time, j = n-1, so we can just switch n-1 with n
        if lst[-2] < lst[-1]:
            lst[-2], lst[-1] = lst[-1], lst[-2]
        else:
            # and now we know that n-1 > n, so start j at n-2
            j = n - 2
            while lst[j] >= lst[j + 1]:
                j -= 1
                if j == -1:
                    return  # terminate
            l = n
            while lst[j] >= lst[l]:
                l -= 1
            lst[j], lst[l] = lst[l], lst[j]
            k = j + 1
            l = n
            while k < l:
                lst[k], lst[l] = lst[l], lst[k]
                k += 1
                l -= 1
        yield tuple(lst)


def perm3(lst):
    yield tuple(lst)

    if len(lst) == 1:
        return
    if len(lst) == 2:
        lst[0], lst[1] = lst[1], lst[0]
        yield tuple(lst)
        return

    n = len(lst) - 1
    while 1:
        # half the time, j = n-1, so we can just switch n-1 with n
        if lst[-2] < lst[-1]:
            lst[-2], lst[-1] = lst[-1], lst[-2]
        # let's special case the j = n-2 scenario too!
        elif lst[-3] < lst[-2]:
            if lst[-3] < lst[-1]:
                lst[-3], lst[-2], lst[-1] = lst[-1], lst[-3], lst[-2]
            else:
                lst[-3], lst[-2], lst[-1] = lst[-2], lst[-1], lst[-3]
        else:
            # and now we know that n-2 > n-1, so start j at n-3
            j = n - 3
            if j < 0:
                return
            y = lst[j]
            x = lst[-3]
            z = lst[-1]
            while y >= x:
                j -= 1
                if j < 0:
                    return  # terminate
                x = y
                y = lst[j]
            if y < z:
                lst[j] = z
                lst[j + 1] = y
                lst[n] = x
            else:
                l = n - 1
                while y >= lst[l]:
                    l -= 1
                lst[j], lst[l] = lst[l], y
                lst[n], lst[j + 1] = lst[j + 1], lst[n]
            k = j + 2
            l = n - 1
            while k < l:
                lst[k], lst[l] = lst[l], lst[k]
                k += 1
                l -= 1
        yield tuple(lst)


def perm4(lst):
    if max([lst.count(x) for x in lst]) > 1:
        raise "no repeated elements"
    yield tuple(lst)
    if len(lst) == 1:
        return

    n = len(lst) - 1
    c = [0 for i in range(n + 1)]
    o = [1 for i in range(n + 1)]
    j = n
    s = 0
    while 1:
        q = c[j] + o[j]
        if q >= 0 and q != j + 1:
            lst[j - c[j] + s], lst[j - q + s] = lst[j - q + s], lst[j - c[j] + s]
            yield tuple(lst)
            c[j] = q
            j = n
            s = 0
            continue
        elif q == j + 1:
            if j == 1:
                return
            s += 1
        o[j] = -o[j]
        j -= 1


def perm5(l):
    if len(l) == 1:
        yield tuple(l)
        return

    pop, insert, append = l.pop, l.insert, l.append

    def halfperm():
        ll = l
        llen = len(ll)
        if llen == 2:
            yield ll
            return
        aRange = range(llen)
        v = pop()
        for p in halfperm():
            for j in aRange:
                insert(j, v)
                yield ll
                del ll[j]
        append(v)

    for h in halfperm():
        yield tuple(h)
        h.reverse()
        yield tuple(h)
        h.reverse()

# making the functions returning an iterator instead of a generator (of size factorial(n))


def assert_permutations(perm_func):
    n = 7
    perms = perm_func(n)
    if isinstance(perms, types.GeneratorType):
        perms = list(perms)

    # lengths check
    assert all([len(perm) == n for perm in perms]), f'"{perm_func.__name__}" failed the lengths test,\n length must be {n}, first one was {len(perms[0])}'
    # unique elements check
    assert all([len(set(perm)) == n for perm in perms]), f'"{perm_func.__name__}" failed the unique elements test'
    # unique permutation check
    assert Counter(perms).most_common(1)[0][1] == 1, f'"{perm_func.__name__}" failed the unique permutations test'
    # completeness check
    assert len(perms) == math.factorial(n), f'"{perm_func.__name__}" failed the completeness test'
    print(f'\033[1m{perm_func.__name__:20}\033[0m correctly implemented!')


def indexed_unoptimized(n):
    return [indexed_perm_unopt(n, k) for k in range(1, math.factorial(n) + 1)]


def indexed_optimized(n):
    return [indexed_perm_opt(n, k) for k in range(1, math.factorial(n) + 1)]


def built_in_optimized(n):
    return list(built_it_perm_opt(list(range(n))))


def built_in_unoptimized(n):
    return list(built_it_perm_unopt(list(range(n))))


def product_perms(n):
    return list(product_permutations(range(n)))


def method1(n):
    return list(perm1(list(range(n))))


def method2(n):
    return list(perm2(list(range(n))))


def method3(n):
    return list(perm4(list(range(n))))


def method4(n):
    return list(perm5(list(range(n))))
