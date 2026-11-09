# SAD Mini-Analysis and Laboratory Reflection

## A. SAD mini-analysis

| SAD element | Answer |
|---|---|
| Problem | Academic risk is often noticed after poor results. The prototype explores whether historical student information can help identify learners who may need additional academic support. |
| User | Faculty member or academic analyst. |
| Inputs | Student attributes and academic indicators such as study time, absences, past failures, and previous period grades G1 and G2. |
| Process | Validate data → preprocess missing and categorical/numerical values → apply the trained model → classify the student as at risk or not at risk. |
| Outputs | AT RISK / NOT AT RISK prediction, downloadable prediction results, and model evaluation summary. |
| Key limitation | The model is trained on one Portuguese secondary-school dataset. Its performance may not generalize to another institution or population. |

## B. Testing and reflection answers

### 1. What is the prediction target, and why is G3 excluded from the input features?
The target is whether a student is at risk of failing the final Mathematics grade: G3 below 10 is AT RISK, while G3 of 10 or higher is NOT AT RISK. G3 is excluded from the input features because it is the value the model is supposed to predict. Including it would reveal the answer to the model (target leakage) and make the evaluation misleading.

### 2. What does the model's recall for the 'At risk' class tell you?
Recall for the At risk class measures the proportion of all students who are actually at risk that the model correctly identifies. A higher recall means fewer at-risk students are missed. It is important because missed students may not receive timely academic support. Recall should still be considered together with precision and F1-score.

### 3. Why might a model trained on this dataset perform differently on students from your institution?
The dataset represents students from a particular Portuguese secondary-school context. Students at another institution may have different backgrounds, grading practices, teaching methods, resources, attendance patterns, and academic conditions. These differences can change the relationships between input indicators and final grades, so performance may decrease when the model is used on a different population.

### 4. What action should a faculty member take after seeing an 'AT RISK' prediction?
The faculty member should review the student's academic record and circumstances, verify that the input data is accurate, and consider speaking with the student or offering tutoring and other appropriate support. The prediction should be treated as an early warning, not as a diagnosis or an automatic basis for punishment or exclusion.

## C. Evaluation results — complete after running the model

Run `python train_model.py` and copy the actual values printed by the program:

| Metric | Actual result |
|---|---:|
| Accuracy | To be filled from execution |
| Precision — At risk | To be filled from execution |
| Recall — At risk | To be filled from execution |
| F1-score — At risk | To be filled from execution |

Do not submit invented metric values.
