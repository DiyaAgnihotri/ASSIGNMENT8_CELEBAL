import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_and_preprocess_train():
    df = pd.read_csv('Training Dataset.csv')
    df.fillna({
        'Gender': df['Gender'].mode()[0],
        'Married': df['Married'].mode()[0],
        'Dependents': df['Dependents'].mode()[0],
        'Self_Employed': df['Self_Employed'].mode()[0],
        'LoanAmount': df['LoanAmount'].median(),
        'Loan_Amount_Term': df['Loan_Amount_Term'].mode()[0],
        'Credit_History': df['Credit_History'].mode()[0]
    }, inplace=True)

    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
    df['Dependents'] = df['Dependents'].replace('3+', '3').astype(int)

    df_rag = df.copy() 
    for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    X = df.drop(columns=['Loan_ID', 'Loan_Status'])
    y = df['Loan_Status']
    return X, y, df_rag

def load_and_preprocess_test():
    df = pd.read_csv('Test Dataset.csv')
    df.fillna({
        'Gender': df['Gender'].mode()[0],
        'Married': df['Married'].mode()[0],
        'Dependents': df['Dependents'].mode()[0],
        'Self_Employed': df['Self_Employed'].mode()[0],
        'LoanAmount': df['LoanAmount'].median(),
        'Loan_Amount_Term': df['Loan_Amount_Term'].mode()[0],
        'Credit_History': df['Credit_History'].mode()[0]
    }, inplace=True)

    df['Dependents'] = df['Dependents'].replace('3+', '3').astype(int)
    for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    loan_ids = df['Loan_ID']
    X_test = df.drop(columns=['Loan_ID'])
    return X_test, loan_ids

def train_and_predict(X, y, X_test):
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42, use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    preds = model.predict(X_test)
    preds = ['Y' if p == 1 else 'N' for p in preds]
    return model, preds

def row_to_text(row):
    gender = 'Male' if row['Gender'] == 0 else 'Female'
    married = 'married' if row['Married'] == 1 else 'not married'
    education = 'Graduate' if row['Education'] == 0 else 'Not Graduate'
    self_employed = 'self-employed' if row['Self_Employed'] == 1 else 'not self-employed'
    if isinstance(row['Property_Area'], str):
        property_area = row['Property_Area']
    else:
        property_area_map = {0: 'Urban', 1: 'Rural', 2: 'Semiurban'}
        property_area = property_area_map.get(row['Property_Area'], 'Unknown')
    loan_status = 'approved' if row['Loan_Status'] == 1 else 'rejected'

    return (
        f"A {gender} applicant who is {married}, has {row['Dependents']} dependents, "
        f"is a {education} and {self_employed}. Their income is ₹{row['ApplicantIncome']} with "
        f"a co-applicant income of ₹{row['CoapplicantIncome']}. They requested a loan of ₹{row['LoanAmount']} "
        f"for {row['Loan_Amount_Term']} months, credit history: {row['Credit_History']}, "
        f"property area: {property_area}. Loan was {loan_status}."
    )

def prepare_rag(df_raw):
    df_raw.fillna('unknown', inplace=True)
    df_raw['rag_text'] = df_raw.apply(row_to_text, axis=1)
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    embeddings = model.encode(df_raw['rag_text'].tolist(), show_progress_bar=False)
    return df_raw['rag_text'].tolist(), embeddings, model, df_raw 


def answer_query(query, rag_texts, rag_embeddings, embedder, top_k=3, df_raw=None):
    q_embed = embedder.encode([query])
    scores = cosine_similarity(q_embed, rag_embeddings)[0]
    top_indices = np.argsort(scores)[-top_k:][::-1]
    context = "\n".join([rag_texts[i] for i in top_indices])

    summary = ""
    conclusion = ""
    if df_raw is not None:
        query_lower = query.lower()
        checks = {
            'self-employed': 'Self_Employed',
            'married': 'Married',
            'graduate': 'Education',
            'not graduate': 'Education',
            'male': 'Gender',
            'female': 'Gender',
            'urban': 'Property_Area',
            'rural': 'Property_Area',
            'semiurban': 'Property_Area'
        }

        for keyword, column in checks.items():
            if keyword in query_lower:
                summary += "\n\n📊 **Statistical Summary**:\n"
                approval_by_group = df_raw.groupby(column)['Loan_Status'].agg(['count', 'sum'])
                approval_by_group['rate'] = approval_by_group['sum'] / approval_by_group['count'] * 100
                sorted_rates = approval_by_group['rate'].sort_values(ascending=False)

                for cat, rate in sorted_rates.items():
                    summary += f"- {cat}: {rate:.2f}% approval\n"

                highest = sorted_rates.index[0]
                lowest = sorted_rates.index[-1]
                if keyword in str(highest).lower():
                    conclusion = f"\n✅ **Conclusion**: Yes, {keyword} applicants are more likely to get approved."
                elif keyword in str(lowest).lower():
                    conclusion = f"\n❌ **Conclusion**: No, {keyword} applicants are less likely to get approved."
                else:
                    conclusion = f"\nℹ️ **Conclusion**: No clear effect of being {keyword} on approval rate."
                break

    return f"Based on similar applicants:\n{context}{summary}{conclusion}"

if __name__ == "__main__":
    X_train, y_train, df_raw = load_and_preprocess_train()
    X_test, loan_ids = load_and_preprocess_test()
    model, preds = train_and_predict(X_train, y_train, X_test)

    submission = pd.DataFrame({
        'Loan_ID': loan_ids,
        'Loan_Status': preds
    })
    submission.to_csv("/content/submission.csv", index=False)

    rag_texts, rag_embeddings, embedder, df_full = prepare_rag(df_raw)
    sample_q = "What kind of people are more likely to get approved?"
    print("\n🧠 Sample Answer:\n", answer_query(sample_q, rag_texts, rag_embeddings, embedder, df_raw=df_full))
