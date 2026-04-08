## Preprocessing
# Necessary Preprocessing For Regression
- It is necessary to drop all rows for which the BMI column is N/A, as this is the target variable we are looking to predict
- The 'id' column must be dropped, it carries no predictive value and could confuse the model
- Columns labeled as 'other' in the gender column are dropped, as there is only 1 row with this label, and the third option creates unneeded complexity. A single row will not allow the model to learn anything meaningful from 'other' gender.
- Binary categorical features ('gender', 'ever_married', 'residence_type') must be encoded as 0/1
- Categorical features such as 'work_type' and 'smoking_status' must be one-hot encoded

# Necessary Preprocessing For Classification
- The 'id' column must be dropped, it carries no predictive value and could confuse the model
- Columns labeled as 'other' in the gender column are dropped, as there is only 1 row with this label, and the third option creates unneeded complexity. A single row will not allow the model to learn anything meaningful from 'other' gender.
- 'ever_married', 'residence_type', and 'gender' will be binary encoded
- Non-binary categorical features such as 'work_type' and 'smoking_status' must be one-hot encoded
- Because we are not predicting BMI now, we can impute its value when missing for better predicitons


# Nonessential, but useful, Preprocessing
- We may want to remove stroke from the data for regression, as it could leak signal
- 'Residence_type' should be lowercased, as no other column labels are capitalized
- For classification, we may want to resample so that stroke '1' represents more of the data, otherwise always guessing 0 will give high accuracy