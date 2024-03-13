def levenshtein_distance(s1, s2):
    """
    Calculate the Levenshtein distance between two strings (s1 and s2).
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def lev_score(a, b):
    """
    Calculate the normalized Levenshtein score between two strings (a and b),
    which is 1 minus the Levenshtein distance divided by the length of the longer string.
    """
    lev_dist = levenshtein_distance(a, b)
    return 1 - lev_dist / max(len(a), len(b))
