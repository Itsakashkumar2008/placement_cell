import pandas as pd
from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
a=pd.read_csv("placement.csv")
print("total duplicate row are ",a.duplicated().sum())
print("total row with miss value ",a.isna().any(axis=1).sum())
a=a.drop_duplicates()
a = a.dropna(subset=['cgpa','internship','projects','tech_skill','placement'])
a=a.reset_index(drop=True)
a.index=range(1,len(a)+1)
a.to_csv("clean_placement.csv",index=False)
files.download("clean_placement.csv")
b=pd.read_csv("clean_placement.csv")
b['placement'] = b['placement'].astype(str).str.strip().str.title().map({'Placed': 1, 'Not Placed': 0})
for col in ['internship', 'tech_skill']:
    if col in b.columns and b[col].dtype == 'object':
        b[col] = b[col].astype(str).str.strip().str.title().map({'Yes': 1, 'No': 0})
X = b[['cgpa','internship','projects','tech_skill']].astype(float)
y = b['placement'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)
model1=LogisticRegression()
model1.fit(X_train, y_train)
predictions1=model1.predict(X_test)
accuracy1=accuracy_score(y_test, predictions1)
precision1=precision_score(y_test, predictions1)
recall1=recall_score(y_test, predictions1)
f1_1=f1_score(y_test, predictions1)
model2=RandomForestClassifier(random_state=17)
model2.fit(X_train, y_train)
predictions2=model2.predict(X_test)
accuracy2=accuracy_score(y_test, predictions2)
precision2=precision_score(y_test, predictions2)
recall2=recall_score(y_test, predictions2)
f1_2=f1_score(y_test, predictions2)
print("LOGISTIC REGRESSION MODEL SCORES")
print("Accuracy Score  : ",accuracy1*100,"%")
print("Precision Score : ",precision1*100,"%")
print("Recall Score    : ",recall1*100,"%")
print("F1 Score        : ",f1_1*100,"%")
print("RANDOM FOREST MODEL SCORES")
print("Accuracy Score  : ",accuracy2*100,"%")
print("Precision Score : ",precision2*100,"%")
print("Recall Score    : ",recall2*100,"%")
print("F1 Score        : ",f1_2*100,"%")
k=input("want to see placement detail if yes then write yes for no write no:").strip().lower()
if k=="yes":
  try:
    student_name = input("enter student name: ").strip()
    user_cgpa = float(input("Enter CGPA: "))
    user_intern = input("Has the student done an Internship? (Yes/No or 1/0): ").strip().title()
    user_intern = 1 if user_intern in ['Yes', '1'] else 0
    user_projects = int(input("Enter number of projects completed: "))
    user_tech = float(input("Enter technical skill score: "))
    user_student_data = pd.DataFrame([{
            'cgpa': user_cgpa,
            'internship': user_intern,
            'projects': user_projects,
            'tech_skill': user_tech
        }])
    user_prediction = model2.predict(user_student_data)
    print("eligibility result for ",student_name.upper())
    if user_prediction == 1:
      print(" Congratulations! ",student_name," is ready for placement.")
    else:
      print(student_name,"is NOT YET READY for placement. Needs improvement.")
  except:
    print("invalid input")
elif k=="no":
  print("thank you")
else:
  print("invalid input")
