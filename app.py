import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Creating side bar with options
menu = ["Home","Data Exploration","Data Modeling"]
choice = st.sidebar.selectbox("Menu",menu)

def menuChoiceHome():
    # On page load or clicking the Home selection
    if choice == "Home":
        # Writing title and subtitle
        st.write("""
                     # Credit Card Fraud Detection
                     """)
        # Writing GitHub Link
        st.write(
            "GitHub Repo [Click Here!](https://github.com/antoneev/CreditCardFraudDetection)")
        # Writing Data Link
        st.write(
            "Dataset [Click Here!](https://www.kaggle.com/mlg-ulb/creditcardfraud)")

def menuChoiceDE(df, dfScaled):
    # Clicking the Data Exploration
    # Displaying the page title
    st.title('Data Exploration')
    # st.balloons()

    # Display Dataset Checkbox
    if st.checkbox("Display Dataset"):
        # Enter number field
        number = st.number_input("Number of Rows to View",step=1)

        # Option dropdown
        option = st.selectbox(
        'Sort by Row?',
        ('Time', 'V1', 'V2','V3','V4','V5','V6','V7','V8','V9','V10',
         'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
         'V21','V22','V23','V24','V25','V26','V27','Amount','Class'))

        # Display Dataframe
        st.dataframe(df.head(number).sort_values(option))

    # Display Scaled Dataset Checkbox
    if st.checkbox("Display Scaled Dataset"):
        number = st.number_input("Number of Scaled Rows to View",step=1)

        # Option dropdown
        option = st.selectbox(
        'Sort by Row?',
        ('V1', 'V2','V3','V4','V5','V6','V7','V8','V9','V10',
         'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
         'V21','V22','V23','V24','V25','V26','V27','scaled_time','scaled_amount','Class'))

        # Display Dataframe
        st.dataframe(dfScaled.head(number).sort_values(option))

    # Display Dataset Info Checkbox
    if st.checkbox("Display Dataset Info"):
        # Display df.info()
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.write(s)

    # Display Dataset Summary Checkbox
    if st.checkbox("Display Dataset Summary"):
        # Display df.describe()
        st.text(df.describe())

    # Display if Dataset has any NaNs Checkbox
    if st.checkbox("Does the Dataset have any NaNs?"):
        # Saves False or True into answer
        answer = df.isnull().values.any()
        # Displays No if False was returned
        if answer == False:
            st.write("No")
        # Otherwise return Yes
        else:
            st.write("Yes")

    # Display Distribution of Target Variable Info Checkbox
    if st.checkbox("Distribution of Target Variable Information"):
        # Number of variables in Class feature
        st.write("Number of variables in the Class feature:", str(df['Class'].count()))
        # Number of non-fraud datapoints
        st.write("Number of non-fraud datapoints:", str(df['Class'].value_counts()[0]))
        # Number of fraud datapoints
        st.write("Number of fraud datapoints:", str(df['Class'].value_counts()[1]))
        # Percentage of non-fraud datapoints
        st.write('% of non-fraud datapoints: {:.2f}%.'.format((df['Class'].value_counts() / df['Class'].count())[0] * 100))
        # Percentage of fraud datapoints
        st.write('% of fraud datapoints: {:.2f}%.'.format((df['Class'].value_counts() / df['Class'].count())[1] * 100))

    # Display Distribution Graphs and Correlation (Images) Checkbox
    if st.checkbox("Display Distribution Graphs and Correlation"):
        # Display Target Distribution Graph
        st.image('imgs/distTargetVar.png')
        # Display Amount Distribution Graph
        st.image('imgs/distAmountVar.png')
        # Display Time Distribution Graph
        st.image('imgs/distTimeVar.png')
        # Display Correlection Heatmap
        st.image('imgs/corr.png')

def menuChoiceDM(df, scaledDf):
    # Displaying the page title
    st.title('Data Modeling')

    X = df.drop('Class', axis=1)
    y = df['Class']

    training_features, test_features, training_target, test_target \
    = train_test_split(X, y, test_size=0.20, random_state=30)

    # Random Under Sampling
    x_train, x_val, y_train, y_val = train_test_split(training_features, training_target,
                                                      test_size=.10,
                                                      random_state=12)

    rus = RandomUnderSampler(random_state=42)
    x_train_res, y_train_res = rus.fit_resample(x_train, y_train)

    dfModels = pd.read_csv('GridSearchCV.csv')
    best_estimators = {'decision_tree': DecisionTreeClassifier(max_depth=3, min_samples_leaf=5), \
                        'svc': svm.SVC(C=1), \
                        'knn': KNeighborsClassifier(n_neighbors=3), \
                        'logistic_regression': LogisticRegression(C=0.1), \
                        'random_forest': RandomForestClassifier(max_depth=8, max_features='sqrt', n_estimators=200)}

    if st.checkbox("Training/Test Feature Info"):
        st.write("Size of training features:", training_features.shape)
        st.write("Size of training target:", training_target.shape)
        st.write("Size of test features:", test_features.shape)
        st.write("Size of test target:", test_target.shape)

        train_unique_label, train_counts_label = np.unique(training_target, return_counts=True)
        test_unique_label, test_counts_label = np.unique(test_target, return_counts=True)

        st.markdown('## Label Distributions: 0 - nonfraud 1 - fraud')
        trainData = train_counts_label / len(training_target)
        st.write("Train data non-fraud distribution", trainData[0])
        st.write("Train data fraud distribution", trainData[1])
        testData = test_counts_label / len(test_target)
        st.write("Test data non-fraud distribution", testData[0])
        st.write("Test data fraud distribution", testData[1])

    if st.checkbox("Random Under Sampling Info"):
        st.markdown("## Size before Random Under Sampling")
        st.write("Size of training features:", x_train.shape)
        st.write("Size of training target:", y_train.shape)

        st.markdown("## Size after Random Under Sampling")
        st.write("Size of training features:", x_train_res.shape)
        st.write("Size of training target:", y_train_res.shape)

        (unique, counts) = np.unique(y_train_res, return_counts=True)
        frequencies = np.asarray((unique, counts)).T
        st.write("Number of non-fraud datapoints:", frequencies[0][1])
        st.write("Number of fraud datapoints:", frequencies[1][1])

        st.image('imgs/RUSBarChart.png')

    if st.checkbox("Hyperparameter Tuning"):
        st.dataframe(dfModels)
        st.write(best_estimators)

    if st.checkbox("Training/Validation Model Tuning"):
        option = st.selectbox(
            'Select Algorithm',
            ('Select Option','Decision Tree', 'SVM', 'KNN', 'Logistic Regression', 'Random Forest'))

        if option == "Decision Tree":
            descisionTree = best_estimators['decision_tree']
            descisionTree_Train = descisionTree.fit(x_train_res, y_train_res)
            descisionTree_predTrain = descisionTree_Train.predict(x_train_res)
            st.markdown("## Training Data")
            st.text(classification_report(y_train_res, descisionTree_predTrain))

            descisionTree_Val = descisionTree.fit(x_val, y_val)
            descisionTree_predVal = descisionTree_Val.predict(x_val)
            st.markdown("## Validation Data")
            st.text(classification_report(y_val, descisionTree_predVal))
        elif option == "SVM":
            SVC = best_estimators['svc']
            SVC_Train = SVC.fit(x_train_res, y_train_res)
            SVC_predTrain = SVC_Train.predict(x_train_res)
            st.markdown("## Training Data")
            st.text(classification_report(y_train_res, SVC_predTrain))

            SVC_Val = SVC.fit(x_val, y_val)
            SVC_predVal = SVC_Val.predict(x_val)
            st.markdown("## Validation Data")
            st.text(classification_report(y_val, SVC_predVal))
        elif option == "KNN":
            KNN = best_estimators['knn']
            KNN_Train = KNN.fit(x_train_res, y_train_res)
            KNN_predTrain = KNN_Train.predict(x_train_res)
            st.markdown("## Training Data")
            st.text(classification_report(y_train_res, KNN_predTrain))

            KNN_Val = KNN.fit(x_val, y_val)
            KNN_predVal = KNN_Val.predict(x_val)
            st.markdown("## Validation Data")
            st.text(classification_report(y_val, KNN_predVal))
        elif option == "Logistic Regression":
            log_reg = best_estimators['logistic_regression']
            log_regTrain = log_reg.fit(x_train_res, y_train_res)
            log_reg_predTrain = log_regTrain.predict(x_train_res)
            st.markdown("## Training Data")
            st.text(classification_report(y_train_res, log_reg_predTrain))

            log_regVal = log_reg.fit(x_val, y_val)
            log_reg_predVal = log_regVal.predict(x_val)
            st.markdown("## Validation Data")
            st.text(classification_report(y_val, log_reg_predVal))
        elif option == "Random Forest":
            RandForest = best_estimators['random_forest']
            RandForestTrain = RandForest.fit(x_train_res, y_train_res)
            RandForestpredTrain = RandForestTrain.predict(x_train_res)
            st.markdown("## Training Data")
            st.text(classification_report(y_train_res, RandForestpredTrain))

            RandForestVal = RandForest.fit(x_val, y_val)
            RandForestpredVal = RandForestVal.predict(x_val)
            st.markdown("## Validation Data")
            st.text(classification_report(y_val, RandForestpredVal))
        else:
            st.markdown("# Please select an option")

    if st.checkbox("Testing Data"):
        RandForest = best_estimators['random_forest']
        RandForestTest = RandForest.fit(test_features, test_target)
        RandForestpredTest = RandForestTest.predict(test_features)

        st.text(classification_report(test_target, RandForestpredTest))
        st.write("Model accuracy:", round(accuracy_score(test_target, RandForestpredTest) * 100, 2), "%")

        plt.figure(figsize=(50, 20))
        _ = tree.plot_tree(RandForestTest.estimators_[0], feature_names=test_features.columns, filled=True)

def main():

    # Read non-scaled data
    df = pd.read_csv('creditcard.csv')
    # Read scaled data
    dfScaled = pd.read_csv('creditcard-scaleddata.csv')

    # Calls Home Function
    if choice == "Home":
        menuChoiceHome()

    # Calls Data Exploration Function
    if choice == "Data Exploration":
        menuChoiceDE(df, dfScaled)

    # Calls Data Modeling Function
    if choice == "Data Modeling":
        menuChoiceDM(df, dfScaled)

if __name__ == '__main__':
	main()