import Spn
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian
def main():

    f = [Categorical]
    for i in range(200):
        f.append(Gaussian)

    spn = Spn.Spn(train='train.csv',test='test.csv',zscore=4,features=f)
    
    spn.saveresults('output.csv')
    
if __name__ == "__main__":
    main()