#--Group Members
#-1 Amanor Teinor – 22258276---
#-2.Joseph Harvey-Ewusi – 22253143---
#-3.Kwadwo Jectey Nyarko – 11410422---
#-4.Anael K. Djentuh - 22252467---
#-5.Princess Awurabena Frimpong- 22254024--
import numpy as np ## 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split,KFold,cross_val_score ##Cross validation
from sklearn.preprocessing import MinMaxScaler ## Scaling Algorithm

from sklearn.tree import DecisionTreeClassifier ## Decision Tree Classisfier
from sklearn.ensemble import RandomForestClassifier ## Loading of Random Forest Classifier


from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,classification_report,confusion_matrix,roc_curve,auc ## classification validation metrics
from sklearn.model_selection import cross_val_score
import streamlit as st ## Loading of streamlit 
from PIL import Image
from streamlit import dataframe
from unicodedata import numeric
import shap ## To help explain each feature importance to the churn analysis
from fpdf import FPDF
import io
import tempfile
import os

logo = Image.open("data/assets/FOXTECH LOGO.jpeg") ### Loading of Company Logo
st.image(logo,caption="",width=300)

#Looading data at the Global level
try:
    dataset = pd.read_csv("data/Customer-Churn.csv")
except FileNotFoundError as e :
    st.error(f"User, check if there is error loading dataset{e}")
    st.stop()

##creating a copy of data
dataset_1 = dataset.copy() 

##Function to preprocess data.
def preprocess_churn_dataset(df):
    df = df.copy() ## Creating a copy of the dataa

    # Handle missing TotalCharges values by dropping them
    df['TotalCharges'] = df['TotalCharges'].replace(" ", np.nan)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors ='coerce')
    df = df.dropna(subset=['TotalCharges'])

    # Create 'Family' feature
    df['Family'] = np.where((df['Partner'] == 'No') & (df['Dependents'] == 'No'), 'No', 'Yes')

    # Create 'OnlineServices' feature
    df['OnlineServices'] = np.where((df['OnlineSecurity'] == 'No') & (df['OnlineBackup'] == 'No'), 'No', 'Yes')

    # Create 'StreamingServices' feature
    df['StreamingServices'] = np.where((df['StreamingTV'] == 'No') & (df['StreamingMovies'] == 'No'), 'No', 'Yes')

    # Encode 'Churn' column
    df['Churn_Num'] = df['Churn'].map({'Yes': 1, 'No': 0}).astype(int)

    # Mapping for categorical encoding
    mappings = {
        'Gender_Num': {'Female': 1, 'Male': 0},
        'Family_Num': {'Yes': 1, 'No': 0},
        'PhoneService_Num': {'Yes': 1, 'No': 0},
        'MultipleLines_Num': {'Yes': 1, 'No': 0, 'No phone service': 2},
        'InternetService_Num': {'DSL': 0, 'Fiber optic': 1, 'No': 2},
        'OnlineServices_Num': {'Yes': 1, 'No': 0},
        'DeviceProtection_Num': {'Yes': 1, 'No': 0, 'No internet service': 2},
        'StreamingServices_Num': {'Yes': 1, 'No': 0},
        'TechSupport_Num': {'Yes': 1, 'No': 0, 'No internet service': 2},
        'Contract_Num': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
        'PaperlessBilling_Num': {'Yes': 1, 'No': 0},
        'PaymentMethod_Num': {
            'Electronic check': 0,
            'Mailed check': 1,
            'Bank transfer (automatic)': 2,
            'Credit card (automatic)': 3
        }
    }
    ## Renaming  and reordering of columns
    column_sources = {
        'Gender_Num': 'gender',
        'Family_Num': 'Family',
        'PhoneService_Num': 'PhoneService',
        'MultipleLines_Num': 'MultipleLines',
        'InternetService_Num': 'InternetService',
        'OnlineServices_Num': 'OnlineServices',
        'DeviceProtection_Num': 'DeviceProtection',
        'StreamingServices_Num': 'StreamingServices',
        'TechSupport_Num': 'TechSupport',
        'Contract_Num': 'Contract',
        'PaperlessBilling_Num': 'PaperlessBilling',
        'PaymentMethod_Num': 'PaymentMethod'
    }

    for new_col, mapping in mappings.items():
        source_col = column_sources[new_col]
        df[new_col] = df[source_col].map(mapping).fillna(-1).astype(int)
    # Create tenure bins
    df['TenureRange'] = pd.cut(df['tenure'], 5)
    df.loc[df['tenure'] <= 8, 'TenureCat'] = 0
    df.loc[(df['tenure'] > 8) & (df['tenure'] <= 15), 'TenureCat'] = 1
    df.loc[(df['tenure'] > 15) & (df['tenure'] <= 30), 'TenureCat'] = 2
    df.loc[(df['tenure'] > 30) & (df['tenure'] <= 45), 'TenureCat'] = 3
    df.loc[(df['tenure'] > 45) & (df['tenure'] <= 60), 'TenureCat'] = 4
    df.loc[df['tenure'] > 60, 'TenureCat'] = 5

    # Create MonthlyCharges bins
    df['MonthlyChargesRange'] = pd.cut(df['MonthlyCharges'], 5)
    df.loc[df['MonthlyCharges'] <= 20, 'MonthlyChargesCat'] = 0
    df.loc[(df['MonthlyCharges'] > 20) & (df['MonthlyCharges'] <= 40), 'MonthlyChargesCat'] = 1
    df.loc[(df['MonthlyCharges'] > 40) & (df['MonthlyCharges'] <= 60), 'MonthlyChargesCat'] = 2
    df.loc[(df['MonthlyCharges'] > 60) & (df['MonthlyCharges'] <= 80), 'MonthlyChargesCat'] = 3
    df.loc[(df['MonthlyCharges'] > 80) & (df['MonthlyCharges'] <= 100), 'MonthlyChargesCat'] = 4
    df.loc[df['MonthlyCharges'] > 100, 'MonthlyChargesCat'] = 5

# Scale all numeric columns
    scaler = MinMaxScaler()
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

## Function to prepare the data for the various models
def prepare_churn_data_for_modeling(df):
        """
        Prepares a copy of the churn dataset by:
        - Dropping original categorical and non-required columns
        - Keeping encoded/numerical features
        - Reordering columns
        - Renaming columns for readability
        Returns a cleaned dataframe ready for classification.
        """

        # Make a copy to preserve original
        df_cleaned = df.copy()

        # Columns to drop (original categorical or unused features) and keeping their equivalent numeric column
        columns_to_drop = [
            'gender', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
            'PaymentMethod', 'TotalCharges', 'Churn', 'Family', 'OnlineServices', 'StreamingServices', 'TenureRange',
            'MonthlyChargesRange', 'TenureCat',
        ]
        df_cleaned.drop(columns=columns_to_drop, axis=1, inplace=True)

        # Desired column order (only encoded/numerical features retained)
        final_columns = [
             'Gender_Num', 'SeniorCitizen', 'Family_Num', 'PhoneService_Num',
            'MultipleLines_Num', 'InternetService_Num', 'OnlineServices_Num', 'DeviceProtection_Num',
            'TechSupport_Num', 'StreamingServices_Num', 'Contract_Num', 'PaperlessBilling_Num',
            'PaymentMethod_Num', 'MonthlyCharges', 'Churn_Num'
        ]
        df_cleaned = df_cleaned[final_columns]

        # Rename columns to readable names
        rename_dict = {
            'Gender_Num': 'Gender',
            'Family_Num': 'Family',
            'PhoneService_Num': 'PhoneService',
            'MultipleLines_Num': 'MultipleLines',
            'InternetService_Num': 'InternetService',
            'OnlineServices_Num': 'OnlineServices',
            'DeviceProtection_Num': 'DeviceProtection',
            'TechSupport_Num': 'TechSupport',
            'StreamingServices_Num': 'StreamingServices',
            'Contract_Num': 'Contract',
            'PaperlessBilling_Num': 'PaperlessBilling',
            'PaymentMethod_Num': 'PaymentMethod',
            'MonthlyCharges': 'MonthlyCharges',
            'Churn_Num': 'Churn'
        }
        df_cleaned.rename(columns=rename_dict, inplace=True)

        return df_cleaned

processed_dataset = preprocess_churn_dataset(dataset) ## Call of function and storing it in a variable

##Using the preparation function defined globally
prepared_data = prepare_churn_data_for_modeling(processed_dataset)

##Function to get the most important features
def get_top_10_features_by_model(model_name: str):
    if model_name == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=5, random_state=40)
    else:
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=40)

    X = prepared_data.drop(columns='Churn')
    y = prepared_data['Churn']
    model.fit(X, y)

    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    return feature_importances.sort_values(ascending=False).head(10).index.tolist()

## Creating variable for categorical columns
categorical_cols = [col for col in processed_dataset.columns
                            if col not in ['customerID', 'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges','Family_Num','Gender_Num','Churn_Num','PhoneService_Num','MultipleLines_Num','InternetService_Num','OnlineServices_Num','DeviceProtection_Num','StreamingServices_Num','TechSupport_Num','Contract_Num','PaperlessBilling_Num','PaymentMethod_Num','TenureCat','TenureRange','MonthlyChargesRange','MonthlyChargesCat']]

## Creating variable for numerical columns
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

##Page 1
def page1():
    st.header("Dataset")
    if st.checkbox("Show Raw Data"):
        st.write(dataset) ## Raw dataset

    if st.checkbox("Processed Data"):
        st.write(processed_dataset) ## Processed dataset
##Page 3
def page2():
    st.header("Exploratory Data Analysis")
  
    st.subheader("Summary Statistics")
    ##Summary statistics of Numeric Columns
    if st.checkbox("Summary Statistics of Numeric Columns"):
        st.write(processed_dataset[numeric_cols].describe())

    ##Summary statistics of Categorical Columns
    if st.checkbox("Summary Statistics of Categorical Columns"):
        st.write(processed_dataset.describe(include='object'))

    st.subheader("Univariate Analysis (Single Column View)")

    if st.checkbox("Visualizing Each Categorical Column"):

        # Dropdown for column selection
        selected_col = st.selectbox("Select a categorical column", categorical_cols)

        # Plot the selected column
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=processed_dataset, x=selected_col, palette='Set2', ax=ax)
        ax.set_title(f"Count Plot of {selected_col}")
        ax.tick_params(axis='x', rotation=45)

        #Streamlit plot
        st.pyplot(fig)

    
    if st.checkbox("Visualizing Numeric Columns"):


        # Chart selection
        chart_type = st.radio("Choose Chart Type", ['Histogram', 'Distribution (KDE)'])
        selected_col = st.selectbox("Select Numeric Column", numeric_cols)

        # Plotting both Histogram and KDE plots. Creates an option for user to select plot type.
        fig, ax = plt.subplots(figsize=(10, 5))

        if chart_type == 'Histogram':
            processed_dataset[selected_col].plot.hist(
                color='DarkBlue', alpha=0.7, bins=50, title=selected_col, ax=ax
            )
            ax.set_xlabel(selected_col)

        elif chart_type == 'Distribution (KDE)':
            sns.histplot(
                data=processed_dataset, x=selected_col, kde=True, color='teal', ax=ax
            )
            ax.set_title(f'Distribution Plot of {selected_col}')
            ax.set_xlabel(selected_col)

        # Show plot in Streamlit
        st.pyplot(fig)

    st.subheader("Bivariate Analysis")
    if st.checkbox("Checking for Outliers Using Boxplot on Continous Variables"):
        st.subheader('BoxPlot Analysis of  Numeric Variables : MonthlyCharges,TotalCharges and Tenure')

        selected_col = st.selectbox("Select Numeric Column", numeric_cols,key='boxplot_select')# Dropdown for column selection

        ##Plotting
        fig, ax = plt.subplots(figsize=(8, 4))

        sns.boxplot(data = processed_dataset,x=selected_col, palette='Set2', ax=ax)

        ax.set_title(f'BoxPlot of {selected_col}')
        st.pyplot(fig)


    if st.checkbox("Show Tenure Distribution By Churn"):
        fighist = sns.FacetGrid(processed_dataset, col='Churn_Num')
        fighist.map(plt.hist, 'tenure', bins=20,color='skyblue')


        st.pyplot(fighist.fig)

    if st.checkbox("Show TotalCharges Distribution By Churn"):
        fighist = sns.FacetGrid(processed_dataset, col='Churn_Num')
        fighist.map(plt.hist, 'TotalCharges', bins=20,color='skyblue')
        st.pyplot(fighist.fig)

    if st.checkbox('Showing the Relationship Between Categorical Variables and Churn'):
        selected_col = st.selectbox("Select a categorical column", categorical_cols,key='distrubution_plots')

        fig,ax =plt.subplots(figsize=(10,5))

        sns.countplot(data=processed_dataset,x=selected_col,hue='Churn_Num',palette='Set3',ax=ax)
        ax.set_title(f'{selected_col} vs Churn')
        ax.set_ylabel('Count')
        ax.set_xlabel(selected_col)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    if st.checkbox("Correlation Matrix"):
            fig,ax=plt.subplots(figsize=(10,10))
            selected_cols_2 = ['tenure', 'TotalCharges', 'MonthlyCharges', 'Churn_Num']
            corr_matrix= processed_dataset[selected_cols_2].corr()  ##Calculating of correlation among columns

            
            ##Plotting the Correlation Matrix
            sns.heatmap(corr_matrix,annot=True,cmap='coolwarm',fmt='.2f',ax =ax)
            ax.set_title('Correlation Matrix')
            st.pyplot(fig)
            st.markdown(
                """
                🔍 **Insight:** `TotalCharges` shows a **weak negative correlation** with churn.
                This implies that customers with **higher total charges (i.e., long-term or high-paying customers)** 
                are **less likely to churn**.
                """
            )



##age 3
def page3():
    st.header("Classification (with Feature Selection)")

    st.markdown("""
        This section splits the dataset, trains the selected model on all features,
        identifies the top 10 most important features, then retrains using only those features.
    """)
    # --- Model Selection ---
    model_options = ['Decision Tree', 'Random Forest']
    selected_model = st.selectbox("Select a classification model", model_options)

    if selected_model == 'Decision Tree':
        model = DecisionTreeClassifier(max_depth=5, random_state=40)
    else:
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=40)

    # --- Full Dataset ---
    full_X = prepared_data.drop(columns='Churn')
    full_y = prepared_data['Churn']

    # --- Initial Train-Test Split (using all features) ---
    X_train_full, X_test_full, y_train, y_test = train_test_split(full_X, full_y, test_size=0.2, random_state=40)

    # --- Initial Training to get Feature Importances ---
    model.fit(X_train_full, y_train)
    importances = pd.Series(model.feature_importances_, index=full_X.columns)
    top_10_features = importances.sort_values(ascending=False).head(10).index.tolist()
    st.success(f"Top 10 Most Important Features: {', '.join(top_10_features)}")

    # --- Filter only Top 10 Features for Final Training ---
    X_train = X_train_full[top_10_features]
    X_test = X_test_full[top_10_features]

    # --- Retrain using only top 10 features ---
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # --- Metrics ---
    acc = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, full_X[top_10_features], full_y, cv=10)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    ##Save metrics to session state
    prefix = selected_model.replace(" ","_")
    st.session_state[f"{prefix}_accuracy"] = acc
    st.session_state[f"{prefix}_f1"] = f1
    st.session_state[f"{prefix}_precision"] = precision
    st.session_state[f"{prefix}_recall"] = recall
    st.session_state[f"{prefix}_cv_scores"] = cv_scores

    ## Display Metrics
    st.metric("Test Accuracy", f"{acc:.2%}")
    with st.expander("10-Fold Cross-Validation (Top 10 Features)"):
        st.write(f"CV Mean Accuracy: **{cv_scores.mean():.2%}**")
        st.write("All Fold Scores:", cv_scores)

    st.subheader("📋 Detailed Metrics Table")
    metrics_df = pd.DataFrame({
        "Metric": ["F1 Score", "Precision", "Recall"],
        "Score": [f1, precision, recall]
    })
    metrics_df["Score"] = metrics_df["Score"].apply(lambda x: f"{x:.2%}")
    st.dataframe(metrics_df)

    # --- Confusion Matrix ---
    cm = confusion_matrix(y_test, y_pred)
    st.subheader("Confusion Matrix")
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'], ax=ax_cm)
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("Actual")
    st.pyplot(fig_cm)

    # --- Classification Report ---
    st.subheader("Classification Report")
    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).transpose()
    st.dataframe(report_df.style.format("{:.2f}").background_gradient(cmap='YlGnBu'))

    # --- ROC Curve & AUC ---
    st.subheader("ROC Curve & AUC")
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    fig_roc, ax_roc = plt.subplots()
    ax_roc.plot(fpr, tpr, color='darkorange', label=f"ROC Curve (AUC = {roc_auc:.2f})")
    ax_roc.plot([0, 1], [0, 1], linestyle='--', color='navy')
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("Receiver Operating Characteristic")
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

    # --- Feature Importance Plot ---
    st.subheader("Top 10 Feature Importances")
    fig_feat, ax_feat = plt.subplots()
    importances[top_10_features].sort_values().plot(kind='barh', ax=ax_feat, color='teal')
    ax_feat.set_title("Top 10 Important Features")
    st.pyplot(fig_feat)


def page4():
    st.title("Customer Churn Prediction")
    st.write("Fill in the customer details to predict churn.") 

    # Model selection
    model_option = st.selectbox("Select Model", ["Decision Tree", "Random Forest"])

    # --- Get Top 10 Features Dynamically ---
    top_10_features = get_top_10_features_by_model(model_option)

    if model_option == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=5, random_state=40)
    else:
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=40)

    # Prepare training data and fit model
    x = prepared_data[top_10_features]
    y = prepared_data['Churn']
    model.fit(x, y)

    # --- Form Layout ---
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        # Feature input widgets dynamically
        inputs = {}
        for i, feature in enumerate(top_10_features):
            with (col1 if i % 2 == 0 else col2):
                if feature == 'OnlineServices':
                    inputs[feature] = st.selectbox(feature, ['Yes', 'No'])
                elif feature in ['TechSupport', 'DeviceProtection', 'StreamingServices']:
                    inputs[feature] = st.selectbox(feature, ['Yes', 'No', 'No internet service'])

                elif feature == 'PaymentMethod':
                    inputs[feature] = st.selectbox(feature, [
                        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
                    ])
                elif feature == 'InternetService':
                    inputs[feature] = st.selectbox(feature, ['DSL', 'Fiber optic', 'No'])
                elif feature == 'Contract':
                    inputs[feature] = st.selectbox(feature, ['Month-to-month', 'One year', 'Two year'])
                elif feature == 'Family':
                    inputs[feature] = st.selectbox(feature, ['Yes', 'No'])
                elif feature == 'PaperlessBilling':
                    inputs[feature] = st.selectbox(feature, ['Yes', 'No'])
                elif feature == 'MonthlyCharges':
                    inputs[feature] = st.number_input("MonthlyCharges", min_value=0.0, max_value=500.0, value=70.0)
                elif feature == 'SeniorCitizen':
                    inputs[feature] = st.selectbox(feature, ['Yes', 'No'])
                elif feature == 'Gender':
                    inputs[feature] = st.selectbox(feature,['Male','Female'])
                elif feature == 'PhoneService':
                    inputs[feature] =st.selectbox(feature,['Yes','No'])

        ##Submit Button.
        submit = st.form_submit_button("Predict") 

    if submit:
        # --- Construct Input ---
        input_df = pd.DataFrame([inputs])

        # Apply same encoding mappings
        mappings = {
            'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
            'OnlineServices': {'Yes': 1, 'No': 0},
            'TechSupport': {'Yes': 1, 'No': 0, 'No internet service': 2},
            'PaymentMethod': {
                'Electronic check': 0,
                'Mailed check': 1,
                'Bank transfer (automatic)': 2,
                'Credit card (automatic)': 3
            },
            'DeviceProtection': {'Yes': 1, 'No': 0, 'No internet service': 2},
            'InternetService': {'DSL': 0, 'Fiber optic': 1, 'No': 2},
            'Family': {'Yes': 1, 'No': 0},
            'PaperlessBilling': {'Yes': 1, 'No': 0},
            'SeniorCitizen':{'Yes':1,'No':0},
            'Gender':{'Male':0,'Female':1},
            'PhoneService':{'Yes':1,'No':0}
        }

        for col, mapping in mappings.items():
            if col in input_df.columns:
                input_df[col] = input_df[col].map(mappings[col]).fillna(-1).astype(int)

        input_encoded = input_df[top_10_features]

        # --- Predict ---
        pred = model.predict(input_encoded)[0]
        prob = model.predict_proba(input_encoded)[0][1]

        st.success(f"Prediction: **{'Churn' if pred == 1 else 'No Churn'}**")
        st.metric("Churn Probability", f"{prob:.2%}")

        if pred == 1:
            st.warning("⚠️ This customer is likely to churn.") ## Output for prediction(when churn =1)
        else:
            st.balloons()
            st.info("✅ This customer is likely to stay.") ## Output for prediction(when churn =0)

        fig, ax = plt.subplots(figsize=(5, 0.4))
        ax.barh([''], [prob], color='orange' if prob > 0.5 else 'green')
        ax.set_xlim(0, 1)
        ax.set_xticks(np.linspace(0, 1, 5))
        ax.set_yticks([])
        st.pyplot(fig)


        # --- SHAP Explanation ---
        st.subheader("🔍 Why this prediction? (SHAP Explanation)")

        # Explain prediction
        explainer = shap.Explainer(model, x)
        shap_values = explainer(input_encoded)
        shap_row = shap_values[0]

        # Create dict of feature name → float SHAP value
        impact_dict = {
            feature: float(np.ravel(value)[0])  # ensure scalar float
            for feature, value in zip(input_encoded.columns, shap_row.values)
        }

        # Sort by absolute impact
        sorted_impact = sorted(impact_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)

        # Display top 5 contributors
        for feature, impact in sorted_impact[:5]:
            direction = "increases" if impact > 0 else "decreases"
            st.markdown(
                f"- **{feature}** → {direction} churn risk by **{abs(impact * 100):.1f}%**"
            )

        # --- SHAP Bar Plot (Manual Matplotlib Version) ---
        st.subheader("🔢 SHAP Feature Impact")

        # Get top 10 sorted features
        top_features = list(dict(sorted(impact_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)).items())[:10]

        # Create bar plot using matplotlib
        features = [f for f, _ in top_features]
        impacts = [i for _, i in top_features]
        colors = ['green' if i < 0 else 'red' for i in impacts]

        fig, ax = plt.subplots()
        ax.barh(features[::-1], impacts[::-1], color=colors[::-1])
        ax.set_xlabel("SHAP Value (Impact on Churn Prediction)")
        ax.set_title("Top 10 SHAP Feature Contributions")
        st.pyplot(fig)

        # --- Emoji Mapping for Key Features ---
        emoji_map = {
            'Contract': '📅',
            'TechSupport': '🛠️',
            'OnlineServices': '🌐',
            'DeviceProtection': '🔐',
            'PaymentMethod': '💳',
            'InternetService': '📡',
            'PaperlessBilling': '📄',
            'MonthlyCharges': '💰',
            'Family': '👨‍👩‍👧‍👦',
            'PhoneService': '📞',
            'SeniorCitizen': '👴',
            'Gender': '⚧️',
            'StreamingServices': '📺'
        }

        # --- High-Risk Feature Warning ---
        if pred == 1 and prob > 0.5:
            high_risk_features = [(f, v) for f, v in sorted_impact[:5] if v > 0]
            if high_risk_features:
                risky_labels = [
                    f"{emoji_map.get(f, '⚠️')} **{f}**" for f, _ in high_risk_features
                ]
                st.error(
                    f"🚨 **Risk Factors Increasing Churn:** {' | '.join(risky_labels)}"
                )
   
        class PDF(FPDF):
            def header(self):
                # Add FOXTECH logo
                self.image("data/assets/FOXTECH LOGO.jpeg", x=10, y=8, w=30)
                self.set_font("Arial", "B", 16)
                self.cell(0, 10, "Customer Churn Prediction Report", ln=True, align="C")
                self.ln(12)

            def add_prediction_section(self, prob):
                self.set_font("Arial", "", 12)
                self.multi_cell(0, 8, f"Churn Probability: {prob:.2%}")
                self.ln(5)

            def add_shap_explanation(self, sorted_impact):
                self.set_font("Arial", "B", 12)
                self.cell(0, 8, "SHAP Explanation (Top 5 Features):", ln=True)
                self.set_font("Arial", "", 11)
                for feature, impact in sorted_impact[:5]:
                    direction = "increases" if impact > 0 else "decreases"
                    self.multi_cell(
                        0, 6,
                        f"- {feature} {direction} churn risk by {abs(impact * 100):.1f}%"
                    )
                self.ln(5)

            def add_shap_plot(self, fig):
                # Save fig to a temp image file
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
                    fig.savefig(tmpfile.name, format="png", bbox_inches='tight')
                    tmpfile_path = tmpfile.name

                epw = self.w - self.l_margin - self.r_margin
                self.image(tmpfile.name, x=10, w=epw - 20)


                os.remove(tmpfile_path)
                self.ln(5)

        # --- 2. Create the PDF Report ---
        pdf = PDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.add_prediction_section(prob)
        pdf.add_shap_explanation(sorted_impact)
        pdf.add_shap_plot(fig)

        # --- 3. Save to memory for Streamlit download ---
        pdf_bytes = io.BytesIO()
        pdf_output_str = pdf.output(dest='S').encode('latin-1')
        pdf_bytes.write(pdf_output_str)
        pdf_bytes.seek(0)

        # --- 4. Show download button ---
        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_bytes,
            file_name="churn_prediction_report.pdf",
            mime="application/pdf"
        )

##Page 5
def page5():
    st.header("📊 Interpretation and Conclusions")

    st.markdown("""
    This section summarizes the results from our churn prediction models.
    We compare how different models performed and which features were most predictive of churn.
    """)

    # --- Load and compare top 10 features from both models ---
    dt_features = get_top_10_features_by_model("Decision Tree")
    rf_features = get_top_10_features_by_model("Random Forest")

    #---Column Layout-----
    st.subheader("🔍 Top Predictive Features")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Decision Tree - Top 10 Features**")
        st.write(dt_features)

    with col2:
        st.markdown("**Random Forest - Top 10 Features**")
        st.write(rf_features)

    # --- Feature Importance Plots ---
    def plot_feature_importance(model_name):
        if model_name == "Decision Tree":
            model = DecisionTreeClassifier(max_depth=5, random_state=40)
        else:
            model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=40)

        X = prepared_data.drop(columns="Churn")
        y = prepared_data["Churn"]
        model.fit(X, y)
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)

        fig, ax = plt.subplots()
        importances.tail(10).plot(kind="barh", ax=ax, color='skyblue')
        ax.set_title(f"Top 10 Feature Importances - {model_name}")
        st.pyplot(fig)

    st.subheader("📈 Feature Importance Visualization")

    st.markdown("**Decision Tree**")
    plot_feature_importance("Decision Tree")

    st.markdown("**Random Forest**")
    plot_feature_importance("Random Forest")


    # --- Performance Summary Table ---
    st.subheader("📋 Model Performance Comparison")
    
    # Retrieve metrics from page3 or fallback to N/A
    def get_metric(model, metric):
        key = model.replace(" ", "_") + f"_{metric}"
        if key in st.session_state:
          value = st.session_state[key]
          if isinstance(value,(np.ndarray,list)):
              return f"{np.mean(value):.2%}"
          else:
              return f"{value:.2%}"
        return "N/A"
        
  #--Summary Table---#
    summary_data = {
        "Model": ["Decision Tree", "Random Forest"],
        "Accuracy (CV)": [get_metric("Decision Tree", "cv_scores"), get_metric("Random Forest", "cv_scores")],
        "Test Accuracy": [get_metric("Decision Tree", "accuracy"), get_metric("Random Forest", "accuracy")],
        "F1 Score": [get_metric("Decision Tree", "f1"), get_metric("Random Forest", "f1")],
        "Precision": [get_metric("Decision Tree", "precision"), get_metric("Random Forest", "precision")],
        "Recall": [get_metric("Decision Tree", "recall"), get_metric("Random Forest", "recall")],
        "Strengths": [
            "Interpretable, easy to visualize",
            "Better accuracy, handles complexity well"
        ],
        "Weaknesses": [
            "May overfit, less stable",
            "Harder to interpret, slower to train"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df)

    # --- Conclusions ---
    st.subheader("🧾 Final Insights")

    st.markdown("""
    - Features like `Contract`, `TechSupport`, and `OnlineServices` consistently showed up as **strong predictors of churn**.
    - The **Random Forest model** outperformed Decision Tree in terms of accuracy and generalization, thanks to ensemble learning.
    - However, the **Decision Tree model** offers greater interpretability, which can be useful for explaining churn decisions to business stakeholders.
    - For operational deployment, **Random Forest** is preferred due to its robustness — while **Decision Tree** may be better for internal rule-based decisions or exploratory analysis.
    """)


pages = {
   "Dataset":page1,
    "Exploratory Data Analysis" :page2,
    "Classification" : page3,
    "Prediction" : page4,
    "Interpretation" : page5
 }

##Creating the  sidebar with  selectionbox
select_page=st.sidebar.selectbox("select page",list(pages.keys()))

##Display the page when clicked
pages[select_page]()


