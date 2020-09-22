from Assignment1 import *

def float_eq(a,b,eps=0.04):
    return abs(a-b) <= eps

def test_case():
    print("Testcase 1")
    df = pd.read_csv('Test.csv')
    print(float_eq(get_entropy_of_dataset(df),0.9709505944546686))
    print(float_eq(get_entropy_of_attribute(df, 'Sky'), 0.9509775004326937))
    print(float_eq(get_information_gain(df, 'Sky'), 0.01997309402197489))
    print(float_eq(get_entropy_of_attribute(df, 'Airtemp'), 0.6490224995673063))
    print(float_eq(get_information_gain(df, 'Airtemp'), 0.3219280948873623))
    print(float_eq(get_entropy_of_attribute(df, 'Humidity'), 0.9509775004326937))
    print(float_eq(get_information_gain(df, 'Humidity'), 0.01997309402197489))
    print(float_eq(get_entropy_of_attribute(df, 'Water'), 0.8))
    print(float_eq(get_information_gain(df, 'Water'), 0.17095059445466854))
    print(float_eq(get_entropy_of_attribute(df, 'Forecast'), 0.9509775004326937))
    print(float_eq(get_information_gain(df, 'Forecast'), 0.01997309402197489))
    print(get_selected_attribute(df)) #Airtemp
    print(get_selected_attribute(df)[1] == "Airtemp")
    print()

    print("Testcase 2")
    df = pd.read_csv('Test1.csv')
    print(float_eq(get_entropy_of_dataset(df), 0.9402859586706311))
    print(float_eq(get_entropy_of_attribute(df, 'Age'), 0.6324823551623816))
    print(float_eq(get_information_gain(df, 'Age'), 0.30780360350824953))
    print(float_eq(get_entropy_of_attribute(df, 'Income'), 0.9110633930116763))
    print(float_eq(get_information_gain(df, 'Income'), 0.02922256565895487))
    print(float_eq(get_entropy_of_attribute(df, 'Student'), 0.7884504573082896))
    print(float_eq(get_information_gain(df, 'Student'), 0.15183550136234159))
    print(float_eq(get_entropy_of_attribute(df, 'Credit_rating'), 0.8921589282623617))
    print(float_eq(get_information_gain(df, 'Credit_rating'), 0.04812703040826949))
    print(get_selected_attribute(df)) #Age
    print(get_selected_attribute(df)[1] == "Age")
    print()

    print("Testcase 3")
    df = pd.read_csv('Test2.csv')
    print(float_eq(get_entropy_of_dataset(df), 0.9852281360342515))
    print(float_eq(get_entropy_of_attribute(df,'salary'), 0.5156629249195446))
    print(float_eq(get_information_gain(df,'salary'), 0.46956521111470695))
    print(float_eq(get_entropy_of_attribute(df,'location'), 0.2857142857142857))
    print(float_eq(get_information_gain(df,'location'), 0.6995138503199658))
    print(get_selected_attribute(df)) #salary
    print(get_selected_attribute(df)[1]=="location")
    print()

    print("Testcase 4")
    df = pd.read_csv('Test3.csv')
    print(float_eq(get_entropy_of_dataset(df), 0.9709505944546686))
    print(float_eq(get_entropy_of_attribute(df,'toothed'), 0.963547202339972))
    print(float_eq(get_information_gain(df,'toothed'), 0.007403392114696539))
    print(float_eq(get_entropy_of_attribute(df,'breathes'), 0.8264662506490407))
    print(float_eq(get_information_gain(df,'breathes'), 0.1444843438056279))
    print(float_eq(get_entropy_of_attribute(df,'legs'), 0.4141709450076292))
    print(float_eq(get_information_gain(df,'legs'), 0.5567796494470394))
    print(get_selected_attribute(df)) #legs
    print(get_selected_attribute(df)[1] == "legs")


if __name__ == "__main__":
    test_case()
