import sys


def solve(T: int, X: int):
    print(T/X)
    return


def main():
    def iterate_tokens():
        for line in sys.stdin:
            for word in line.split():
                yield word
    tokens = iterate_tokens()
    T = int(next(tokens))
    X = int(next(tokens))
    solve(T, X)

if __name__ == '__main__':
    main()
