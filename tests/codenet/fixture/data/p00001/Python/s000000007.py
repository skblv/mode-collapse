import functools

print(functools.reduce(lambda left, right: left + right, map(int, input().split()), 0))
