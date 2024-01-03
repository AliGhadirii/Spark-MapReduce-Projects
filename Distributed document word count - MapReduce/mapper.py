import sys

for line in sys.stdin:
    # to remove leading and trailing whitespace
    line = line.strip()
    # split the line into words
    words = line.split()
    for word in words:
        # output here will be the input for the Reduce step, i.e. the input for reducer.py
        print(word, 1)
