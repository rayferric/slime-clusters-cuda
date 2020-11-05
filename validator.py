import sys

# Read inputs line-by-line:
lines1 = open(sys.argv[1]).readlines()
lines2 = open(sys.argv[2]).readlines()

# Uniquify and sort:
sorted1 = sorted(list(set(lines1)))
sorted2 = sorted(list(set(lines2)))

# Compare:
sys.exit(0 if sorted1 == sorted2 else 1)