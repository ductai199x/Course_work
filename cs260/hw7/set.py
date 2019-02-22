#!/usr/bin/env python3

class Set:
    def __init__(self, values):
        self.values = values
        self.parents = self.Initialize(values)
    
    def Initialize(self, values):
        ret = [ -1 for i in range(0, len(values)) ]
        return ret
     
    def Find(self, s, value):
        idx = s.values.index(value)
        p = []
        while ( s.parents[idx] != -1 ):
            p.append(idx)
            idx = s.parents[idx]
            
        for i in p:
            s.parents[i] = idx
        
        return s.values[idx]

    def Find2(self, s, value):
        idx = s.values.index(value)
        if ( s.parents[idx] == -1 ):
            return value
        else:
            return self.Find2(s, s.values[s.parents[idx]])

    def Merge(self, s, value1, value2):
        set1 = self.Find(s, value1)
        set2 = self.Find(s, value2)

        idx1 = s.values.index(set1)
        idx2 = s.values.index(set2)

        # Check if loop -> exit
        if ( set1 != set2 ):
            s.parents[idx1] = idx2
    
    def Merge2(self, s, value1, value2):
        set1 = self.Find2(s, value1)
        set2 = self.Find2(s, value2)

        idx1 = s.values.index(set1)
        idx2 = s.values.index(set2)

        # Check if loop -> exit
        if ( set1 != set2 ):
            s.parents[idx1] = idx2

def main():
    A = [-186, 667, 887, -289, -374, -360, 121, -907, -594, 876]
    my_set = Set(A)

    my_set.Merge(my_set, -186, 667)
    my_set.Merge(my_set, 667, 876)
    print(my_set.parents)
 
if __name__ == "__main__":
    main()
