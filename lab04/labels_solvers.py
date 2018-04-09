import numpy as np

# Implementation of the classical Union-Find (UF)
class UF:
    def __init__(self, max_length):
        self.P_ = np.zeros((max_length), dtype=np.uint)
        # self.P_[0] = 0 # First label is for background pixels
        self.length_ = 1

    # Add new (sequential value) label to the equivalence array 
    def NewLabel(self):
        self.P_[self.length_] = self.length_
        self.length_ += 1
        return self.length_ - 1

    # Return the label associated to specified index
    def GetLabel(self, index):
        return self.P_[index]

    # Solve equivalence between label i and j
    def Merge(self, i, j):
        # Find root of label i
        while self.P_[i] < i:
            i = self.P_[i]

        # Find root of label j
        while self.P_[j] < j:
            j = self.P_[j]

        if i < j:
            self.P_[j] = i
            return i

        self.P_[i] = j
        return j

    # This function flattens the equivalences solver trees to exploit them in the second scan
    def Flatten(self):
    
        k = 1
        for i in range(1,self.length_):
            if self.P_[i] < i:
                self.P_[i] = self.P_[self.P_[i]]
            else:
                self.P_[i] = k
                k += 1

        return k