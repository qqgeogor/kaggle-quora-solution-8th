
def pos_match(pos_1,pos_2):
    set1 = set(pos_1)
    set2 = set(pos_2)
    return  len(set1.intersection(set2)) * 2.0 / (len(pos_1)+len(pos_2))

def pos_match_num(pos_1,pos_2):
    set1 = set(pos_1)
    set2 = set(pos_2)
    return  len(set1.intersection(set2))

def pos_sub(pos_1,pos_2):
    set1 = set(pos_1)
    set2 = set(pos_2)
    return ((set1.issubset(set2)) | (set2.issubset(set1)))

def pos_diff(pos_1,pos_2):
    set1 = set(pos_1)
    set2 = set(pos_2)
    return len(set1.difference(set2))+len(set2.difference(set1))

def pos_same(pos_1,pos_2):
    set1 = set(pos_1)
    set2 = set(pos_2)
    return  int(set1==set2)

def pos_most_same(pos_1,pos_2):
    c_p_1 = {}
    c_p_2 = {}
    for i in pos_1:
        if i in c_p_1:
            c_p_1[i]= c_p_1[i]+1
        else:
            c_p_1.setdefault(i,0)
    for i in pos_2:
        if i in c_p_2:
            c_p_2[i]= c_p_2[i]+1
        else:
            c_p_2.setdefault(i,0)
    c_p_1 = list(sorted(c_p_1.items(),key=lambda x:x[1]))
    c_p_2 = list(sorted(c_p_2.items(), key=lambda x:x[1]))
    return c_p_1[len(c_p_1)-1]==c_p_2[len(c_p_2)-1]

def pos_fre_min_same(pos_1,pos_2):
    c_p_1 = {}
    c_p_2 = {}
    for i in pos_1:
        if i in c_p_1:
            c_p_1[i]= c_p_1[i]+1
        else:
            c_p_1.setdefault(i,0)
    for i in pos_2:
        if i in c_p_2:
            c_p_2[i]= c_p_2[i]+1
        else:
            c_p_2.setdefault(i,0)
    c_p_1 = list(sorted(c_p_1.items(),key=lambda x:x[1]))
    c_p_2 = list(sorted(c_p_2.items(), key=lambda x:x[1]))
    return c_p_1[0]==c_p_2[0]


def pos_len(pos_1,pos_2):
    set1 = set(pos_1)
    set2 = set(pos_2)
    return abs(len(set1)-len(set2))

def en_is_empty(pos_1,pos_2):
    if (len(pos_1)==0)&(len(pos_2)==0):
        return 1
    if len(pos_1)==0:
        return 2
    if len(pos_2)==0:
        return 3
    return 0
