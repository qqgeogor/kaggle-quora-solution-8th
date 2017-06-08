def try_divide(x, y, val=0.0):
    """ 
        Try to divide two numbers
    """
    if y != 0.0:
        val = float(x) / y
    return val

def get_jaccard(seq1, seq2):
    """Compute the Jaccard distance between the two sequences `seq1` and `seq2`.
    They should contain hashable items.
    
    The return value is a float between 0 and 1, where 0 means equal, and 1 totally different.
    """
    set1, set2 = set(seq1), set(seq2)
    return 1 - len(set1 & set2) / float(len(set1 | set2))

def get_dice(A,B):
    A, B = set(A), set(B)
    intersect = len(A.intersection(B))
    union = len(A) + len(B)
    d = try_divide(2*intersect, union)
    return d

def get_sorensen(seq1, seq2):
    """Compute the Sorensen distance between the two sequences `seq1` and `seq2`.
    They should contain hashable items.
    
    The return value is a float between 0 and 1, where 0 means equal, and 1 totally different.
    """
    set1, set2 = set(seq1), set(seq2)
    return 1 - (2 * len(set1 & set2) / float(len(set1) + len(set2)))

def get_count_q1_in_q2(seq1,seq2):
    set1, set2 = set(seq1), set(seq2)
    return len(set1 & set2)

def get_ratio_q1_in_q2(seq1,seq2):
    set1, set2 = set(seq1), set(seq2)
    try:
        return len(set1 & set2)/float(len(set1))
    except:
        return 0.0

def get_count_of_question(seq1):
    return len(seq1)

def get_count_of_unique_question(seq1):
    set1 = set(seq1)
    return len(set1)

def get_ratio_of_unique_question(seq1):
    set1 = set(seq1)
    try:
        return len(set1)/float(len(seq1))
    except:
        return 0.0

def get_count_of_digit(seq1):
    return sum([1. for w in seq1 if w.isdigit()])

def get_ratio_of_digit(seq1):
    try:
        return sum([1. for w in seq1 if w.isdigit()])/float(len(seq1))
    except:
        return 0.0
