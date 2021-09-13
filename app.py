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
from sklearn.metrics import accuracy_score

# Creating side bar with options
menu = ["Home","Data Exploration","Existing Data Modeling","New Data Modeling"]
choice = st.sidebar.selectbox("Menu",menu)

def dataSplit(df):
    # Splitting Data into Target Variable vs. All Variables
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Splitting data into 20% test vs training features
    training_features, test_features, training_target, test_target \
        = train_test_split(X, y, test_size=0.20, random_state=30)

    # Splitting data into 10% validation vs train features
    x_train, x_val, y_train, y_val = train_test_split(training_features, training_target,
                                                      test_size=.10,
                                                      random_state=12)

    # Random Under Sampling
    rus = RandomUnderSampler(random_state=42)
    x_train_res, y_train_res = rus.fit_resample(x_train, y_train)

    # Retuning all variables
    return training_features, test_features, training_target, test_target, x_train, x_val, y_train, y_val, \
           x_train_res, y_train_res

def menuChoiceHome():
    # On page load or clicking the Home selection
    # Display balloons
    st.balloons()
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
        st.text(s)

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
        # Displaying info
        st.write("Number of variables in the Class feature:", df['Class'].count())
        st.write("Number of non-fraud datapoints:", df['Class'].value_counts()[0])
        st.write("Number of fraud datapoints:", df['Class'].value_counts()[1])
        st.write('% of non-fraud datapoints:', round((df['Class'].value_counts() / df['Class'].count())[0] * 100,2), "%")
        st.write('% of fraud datapoints:', round((df['Class'].value_counts() / df['Class'].count())[1] * 100,2), "%")

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

def menuChoiceEDM(df):
    # Displaying the page title
    st.title('Data Modeling')

    # Returning all training/val/test data
    training_features, test_features, training_target, test_target, x_train, x_val, y_train, y_val, x_train_res,\
    y_train_res = dataSplit(df)

    # Calling the GridSearchCV csv - This is done to save time rather than loading multiple times
    dfModels = pd.read_csv('GridSearchCV.csv')
    # Saving best estimators found from the GridSearchCV search
    best_estimators = {'decision_tree': DecisionTreeClassifier(max_depth=3, min_samples_leaf=5), \
                        'svc': svm.SVC(C=1), \
                        'knn': KNeighborsClassifier(n_neighbors=3), \
                        'logistic_regression': LogisticRegression(C=0.1), \
                        'random_forest': RandomForestClassifier(max_depth=8, max_features='sqrt', n_estimators=200)}

    # Display Training/Test Feature Info
    if st.checkbox("Training/Test Feature Info"):
        # Displaying info
        st.write("Size of training features:", training_features.shape)
        st.write("Size of training target:", training_target.shape)
        st.write("Size of test features:", test_features.shape)
        st.write("Size of test target:", test_target.shape)

        # Returning the count of datapoints in each section
        train_unique_label, train_counts_label = np.unique(training_target, return_counts=True)
        test_unique_label, test_counts_label = np.unique(test_target, return_counts=True)

        # Using returning datapoints to calculate the length of data in each data pool
        st.markdown('## Label Distributions: 0 - nonfraud 1 - fraud')
        trainData = train_counts_label / len(training_target)
        st.write("Train data non-fraud distribution", trainData[0])
        st.write("Train data fraud distribution", trainData[1])
        testData = test_counts_label / len(test_target)
        st.write("Test data non-fraud distribution", testData[0])
        st.write("Test data fraud distribution", testData[1])

    # Displaying Random Under Sampling Info
    if st.checkbox("Random Under Sampling Info"):
        # Displaying info before RUS
        st.markdown("## Size before Random Under Sampling")
        st.write("Size of training features:", x_train.shape)
        st.write("Size of training target:", y_train.shape)

        # Displaying info after RUS
        st.markdown("## Size after Random Under Sampling")
        st.write("Size of training features:", x_train_res.shape)
        st.write("Size of training target:", y_train_res.shape)

        # Displaying variables in each data pool
        (unique, counts) = np.unique(y_train_res, return_counts=True)
        frequencies = np.asarray((unique, counts)).T
        st.write("Number of non-fraud datapoints:", frequencies[0][1])
        st.write("Number of fraud datapoints:", frequencies[1][1])

        # Displaying distribution image
        st.image('imgs/RUSBarChart.png')

    # Displaying Hyperparameter Tuning Info
    if st.checkbox("Hyperparameter Tuning"):
        # Display models comparison
        st.dataframe(dfModels)
        # Display models best estimators
        st.write(best_estimators)

    # User modeling using best estimators
    if st.checkbox("Training/Validation Model Tuning"):
        # Select model
        option = st.selectbox(
            'Select Algorithm',
            ('Select Option','Decision Tree', 'SVM', 'KNN', 'Logistic Regression', 'Random Forest'))

        # Decision Tree selected
        if option == "Decision Tree":
            # Display model running
            st.info("Model is running... :runner:")
            # Calling the best estimators
            descisionTree = best_estimators['decision_tree']
            # Fitting the model
            descisionTree_Train = descisionTree.fit(x_train_res, y_train_res)
            # Predicting the model
            descisionTree_predTrain = descisionTree_Train.predict(x_train_res)
            # Output of data
            st.markdown("## Training Data")
            st.text(classification_report(y_train_res, descisionTree_predTrain))
            st.write("Model accuracy:", round(accuracy_score(y_train_res, descisionTree_predTrain) * 100, 2), "%")

            # Fitting the model
            descisionTree_Val = descisionTree.fit(x_val, y_val)
            # Predicting the model
            descisionTree_predVal = descisionTree_Val.predict(x_val)
            # Output of data
            st.markdown("## Validation Data")
            st.text(classification_report(y_val, descisionTree_predVal))
            st.write("Model accuracy:", round(accuracy_score(y_val, descisionTree_predVal) * 100, 2), "%")
            # Display model done running
            st.success("Model is done running... :tada:")

        # SVM selected
        elif option == "SVM":
            # Display model running
            st.info("Model is running... :runner:")
            # Calling the best estimators
            SVC = best_estimators['svc']
            # Fitting the model
            SVC_Train = SVC.fit(x_train_res, y_train_res)
            # Predicting the model
            SVC_predTrain = SVC_Train.predict(x_train_res)
            # Output of data
            st.markdown("## Training Data")
            st.text(classification_report(y_train_res, SVC_predTrain))
            st.write("Model accuracy:", round(accuracy_score(y_train_res, SVC_predTrain) * 100, 2), "%")

            # Fitting the model
            SVC_Val = SVC.fit(x_val, y_val)
            # Predicting the model
            SVC_predVal = SVC_Val.predict(x_val)
            # Output of data
            st.markdown("## Validation Data")
            st.text(classification_report(y_val, SVC_predVal))
            st.write("Model accuracy:", round(accuracy_score(y_val, SVC_predVal) * 100, 2), "%")
            # Display model done running
            st.success("Model is done running... :tada:")

        # KNN selected
        elif option == "KNN":
            # Display model running
            st.info("Model is running... :runner:")
            # Calling the best estimators
            KNN = best_estimators['knn']
            # Fitting the model
            KNN_Train = KNN.fit(x_train_res, y_train_res)
            # Predicting the model
            KNN_predTrain = KNN_Train.predict(x_train_res)
            # Output of data
            st.markdown("## Training Data")
            st.text(classification_report(y_train_res, KNN_predTrain))
            st.write("Model accuracy:", round(accuracy_score(y_train_res, KNN_predTrain) * 100, 2), "%")

            # Fitting the model
            KNN_Val = KNN.fit(x_val, y_val)
            # Predicting the model
            KNN_predVal = KNN_Val.predict(x_val)
            # Output of data
            st.markdown("## Validation Data")
            st.text(classification_report(y_val, KNN_predVal))
            st.write("Model accuracy:", round(accuracy_score(y_val, KNN_predVal) * 100, 2), "%")
            # Display model done running
            st.success("Model is done running... :tada:")

        # Logistic Regression selected
        elif option == "Logistic Regression":
            # Display model running
            st.info("Model is running... :runner:")
            # Calling the best estimators
            log_reg = best_estimators['logistic_regression']
            # Fitting the model
            log_regTrain = log_reg.fit(x_train_res, y_train_res)
            # Predicting the model
            log_reg_predTrain = log_regTrain.predict(x_train_res)
            # Output of data
            st.markdown("## Training Data")
            st.text(classification_report(y_train_res, log_reg_predTrain))
            st.write("Model accuracy:", round(accuracy_score(y_train_res, log_reg_predTrain) * 100, 2), "%")

            # Fitting the model
            log_regVal = log_reg.fit(x_val, y_val)
            # Predicting the model
            log_reg_predVal = log_regVal.predict(x_val)
            # Output of data
            st.markdown("## Validation Data")
            st.text(classification_report(y_val, log_reg_predVal))
            st.write("Model accuracy:", round(accuracy_score(y_val, log_reg_predVal) * 100, 2), "%")
            # Display model done running
            st.success("Model is done running... :tada:")

        # Random Forest selected
        elif option == "Random Forest":
            # Display model running
            st.info("Model is running... :runner:")
            # Calling the best estimators
            RandForest = best_estimators['random_forest']
            # Fitting the model
            RandForestTrain = RandForest.fit(x_train_res, y_train_res)
            # Predicting the model
            RandForestpredTrain = RandForestTrain.predict(x_train_res)
            # Output of data
            st.markdown("## Training Data")
            st.text(classification_report(y_train_res, RandForestpredTrain))
            st.write("Model accuracy:", round(accuracy_score(y_train_res, RandForestpredTrain) * 100, 2), "%")

            # Fitting the model
            RandForestVal = RandForest.fit(x_val, y_val)
            # Predicting the model
            RandForestpredVal = RandForestVal.predict(x_val)
            # Output of data
            st.markdown("## Validation Data")
            st.text(classification_report(y_val, RandForestpredVal))
            st.write("Model accuracy:", round(accuracy_score(y_val, RandForestpredVal) * 100, 2), "%")
            # Display model done running
            st.success("Model is done running... :tada:")

        else:
            # Display instructions
            st.markdown("# Please select an option. Models are using there training/validation data split.")

    # Displaying the test data on the best training model
    if st.checkbox("Testing Data"):
        # Display model running
        st.info("Model is running... :runner:")
        # Calling the best estimators
        RandForest = best_estimators['random_forest']
        # Fitting the model
        RandForestTest = RandForest.fit(test_features, test_target)
        # Predicting the model
        RandForestpredTest = RandForestTest.predict(test_features)

        # Output of data
        st.text(classification_report(test_target, RandForestpredTest))
        st.write("Model accuracy:", round(accuracy_score(test_target, RandForestpredTest) * 100, 2), "%")
        # Display model done running
        st.success("Model is done running... :tada:")

def menuChoiceADM(df):
    # Returning all training/val/test data
    training_features, test_features, training_target, test_target, x_train, x_val, y_train, y_val, x_train_res, y_train_res = dataSplit(df)

    # Algorithm selection
    option = st.selectbox(
        'Select Algorithm',
        ('Select Option', 'Decision Tree', 'SVM', 'KNN', 'Logistic Regression', 'Random Forest'))

    # Decision Tree selected
    if option == "Decision Tree":
        # Display documentation link
        st.write(
            "Decision Tree Documentation [Click Here!](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)")

        # Possible parameters to tune. Current placeholders are the default values
        criterion = st.text_input("criterion", "gini")
        max_depth = st.number_input("max_depth", min_value= 1, step=1)
        min_samples_leaf = st.number_input("min_samples_leaf", 1, step=1)

        # Connecting the running of the model to a button
        if st.button("Model Algorithm") == True:
            # Display model running
            st.info("Model is running... :runner:")
            # Pass in parameters
            model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
            # Fit model
            modelFit = model.fit(test_features, test_target)
            # Predict model
            modelPred = modelFit.predict(test_features)
            # Display model output
            st.markdown("## Test Data")
            st.text(classification_report(test_target, modelPred))
            st.write("Model accuracy:", round(accuracy_score(test_target, modelPred) * 100, 2), "%")
            # Display model done running
            st.success("Model is done running... :tada:")

    # SVM is selected
    elif option == "SVM":
        # Display documentation link
        st.write(
            "SVM Documentation [Click Here!](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)")

        # Possible parameters to tune. Current placeholders are the default values
        C = st.number_input("C", 1.0)
        kernel = st.text_input("kernel", "rbf")

        # Connecting the running of the model to a button
        if st.button("Model Algorithm") == True:
            # Display model running
            st.info("Model is running... :runner:")
            # Pass in parameters
            model = svm.SVC(C=C, kernel=kernel)
            # Fit model
            modelFit = model.fit(test_features, test_target)
            # Predict model
            modelPred = modelFit.predict(test_features)
            # Display model output
            st.markdown("## Test Data")
            st.text(classification_report(test_target, modelPred))
            st.write("Model accuracy:", round(accuracy_score(test_target, modelPred) * 100, 2), "%")
            # Display model done running
            st.success("Model is done running... :tada:")

    # KNN Selected
    elif option == "KNN":
        # Display documentation link
        st.write(
            "KNN Documentation [Click Here!](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)")

        # Possible parameters to tune. Current placeholders are the default values
        n_neighbors = st.number_input("n_neighbors", 5, step=1)
        algorithm = st.text_input("algorithm", "auto")

        # Connecting the running of the model to a button
        if st.button("Model Algorithm") == True:
            # Display model running
            st.info("Model is running... :runner:")
            # Pass in parameters
            model = KNeighborsClassifier(n_neighbors=n_neighbors,algorithm=algorithm)
            # Fit model
            modelFit = model.fit(test_features, test_target)
            # Predict model
            modelPred = modelFit.predict(test_features)
            # Display model output
            st.markdown("## Test Data")
            st.text(classification_report(test_target, modelPred))
            st.write("Model accuracy:", round(accuracy_score(test_target, modelPred) * 100, 2), "%")
            # Display model done running
            st.success("Model is done running... :tada:")

    # Logistic Regression selected
    elif option == "Logistic Regression":

        # Display documentation link
        st.write(
            "Logistic Regression Documentation [Click Here!](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)")

        # Possible parameters to tune. Current placeholders are the default values
        C = st.number_input("C", 1.0)
        penalty = st.text_input("penalty", 'l2')

        # Connecting the running of the model to a button
        if st.button("Model Algorithm") == True:
            # Display model running
            st.info("Model is running... :runner:")
            # Pass in parameters
            model = LogisticRegression(C=C, penalty=penalty)
            # Fit model
            modelFit = model.fit(test_features, test_target)
            # Predict model
            modelPred = modelFit.predict(test_features)
            # Display model output
            st.markdown("## Test Data")
            st.text(classification_report(test_target, modelPred))
            st.write("Model accuracy:", round(accuracy_score(test_target, modelPred) * 100, 2), "%")
            # Display model done running
            st.success("Model is done running... :tada:")

    # Random Forest selected
    elif option == "Random Forest":
        # Display documentation link
        st.write(
            "Random Forest Documentation [Click Here!](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)")

        # Possible parameters to tune. Current placeholders are the default values
        n_estimators = st.number_input("n_estimators", 100, step=1)
        max_features = st.text_input("max_features", 'auto')
        max_depth = st.number_input("max_depth", min_value=1, step=1)
        criterion = st.text_input("criterion", 'gini')

        # Connecting the running of the model to a button
        if st.button("Model Algorithm") == True:
            # Display model running
            st.info("Model is running... :runner:")
            # Pass in parameters
            model = RandomForestClassifier(n_estimators=n_estimators,max_features=max_features,max_depth=max_depth,criterion=criterion)
            # Fit model
            modelFit = model.fit(test_features, test_target)
            # Predict model
            modelPred = modelFit.predict(test_features)
            # Display model output
            st.markdown("## Test Data")
            st.text(classification_report(test_target, modelPred))
            st.write("Model accuracy:", round(accuracy_score(test_target, modelPred) * 100, 2), "%")
            # Display model done running
            st.success("Model is done running... :tada:")

    else:
        # Display instructions
        st.markdown("# Please select an option. Models are using there test data split.")

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

    # Calls Existing Data Modeling Function
    if choice == "Existing Data Modeling":
        menuChoiceEDM(dfScaled)

    # Calls New Data Modeling Function
    if choice == "New Data Modeling":
        menuChoiceADM(dfScaled)

if __name__ == '__main__':
	main()