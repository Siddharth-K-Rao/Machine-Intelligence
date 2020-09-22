from check import *
import sklearn as sk

def test_case():
    path=input("Input the path:")
    df = pd.read_csv(path)
    #df=df.iloc[:50,:]

    for i in range(0,len(df.columns)-1):
        try:
            print("get_information_gain for",end=" ")
            print(df.columns[i],end="=")
            print(get_information_gain(df,df.columns[i]))

        except:
            print("Test Case  for the function get_information_gain FAILED for",end=" ")
            print(df.columns[i])



if __name__=="__main__":
	test_case()
