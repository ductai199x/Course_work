# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
mem = {0: 0, 1: 1}

def fibonacci(n):
    if n in mem:
        return mem[n]
    else:
        mem[n] = fibonacci(n-1) + fibonacci(n-2)
        return mem[n]


N = 14
fibonacci(N)
print(f"N = {N}: {[mem[i] for i in range(0, N+1)]}")


# %%
mem = {}

def levenshtein(string1, string2):
    # corner cases
    if string1 == "":
        return len(string2)
    if string2 == "":
        return len(string1)


    cost = 0 if string1[-1] == string2[-1] else 1
       
    dist1 = string1[:-1] + string2
    dist2 = string1 + string2[:-1]
    dist3 = string1[:-1] + string2[:-1]

    if dist1 not in mem:
        mem[dist1] = levenshtein(string1[:-1], string2)
    if dist2 not in mem:
        mem[dist2] = levenshtein(string1, string2[:-1])
    if dist3 not in mem:
        mem[dist3] = levenshtein(string1[:-1], string2[:-1])
    res = min([mem[dist1]+1, mem[dist2]+1, mem[dist3]+cost])
    
    return res

s1 = "kitten"; s2 = "sitting"
print(f"str1: {s1}, str2: {s2}, dist={levenshtein(s1, s2)}")
s1 = "Python"; s2 = "Peithen"
print(f"str1: {s1}, str2: {s2}, dist={levenshtein(s1, s2)}")
