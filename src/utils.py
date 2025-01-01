def find_factor_pair(n):
    best = (1, n)
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            j = n // i
            # Prefer factor pairs with minimal difference (closer to square)
            if abs(i - j) < abs(best[0] - best[1]):
                best = (i, j)
    return best
