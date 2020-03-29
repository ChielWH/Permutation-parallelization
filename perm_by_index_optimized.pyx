import math

cpdef tuple permutation_by_index(int n, int k):
    """get the permutation at index k for permutations of n players

    Args:
        n (int): The length of the set of players
        k (int): The index of the permutation

    Returns:
        TYPE: tuple
    """
    cdef list numbers, permutation
    cdef int index
    numbers = list(range(n))
    permutation = []
    k -= 1
    while n > 0:
        n -= 1
        # get the index of current digit
        index, k = divmod(k, math.factorial(n))
        permutation.append(numbers.pop(index))

    return tuple(permutation)
