# Basic ML Modelling with Tensorflow
## Business Explanation
This project includes a machine learning model developed to predict the likelihood of customers churning in a bank.

## Dataset
### CSV File:
- Total Features : 10
- Total Row : 10000
- CSV File Size : 837.42 kB
### Feature Explanation Table
| Feature           | Description                                                                                                             |
|-------------------|-------------------------------------------------------------------------------------------------------------------------|
| RowNumber         | Sequential record number with no impact on the analysis or outcome.                                                    |
| CustomerId        | Unique identifier assigned to each customer, devoid of influence on customer attrition.                                 |
| Surname           | Family name of the customer, considered irrelevant in assessing the likelihood of them leaving the bank.                |
| CreditScore       | Numeric value reflecting the customer's creditworthiness, influencing the probability of churn; higher scores indicate lower likelihood.  |
| Geography         | The geographical location of the customer, contributing to the analysis as it may affect their decision to leave the bank. |
| Gender            | Exploration of whether gender plays a role in customer churn.                                                             |
| Age               | A relevant factor, as older customers tend to exhibit greater loyalty and are less prone to leaving the bank compared to younger counterparts. |
| Tenure            | The duration in years that the customer has been a client of the bank, generally associated with increased loyalty.       |
| Balance           | A crucial indicator of customer churn; higher account balances correlate with lower likelihood of departure.              |
| NumOfProducts     | The count of products purchased by the customer from the bank, influencing their overall engagement.                      |
| IsActiveMember    | Indicator of an active customer, with active members less likely to leave the bank.                                      |
| EstimatedSalary   | Similar to balance, lower salaries are associated with higher probabilities of customer churn.                           |
| Exited            | Binary indicator representing whether the customer left the bank (1) or not (0).                                         |
| Complain          | Binary indicator denoting whether the customer has lodged a complaint (1) or not (0).                                    |
| Satisfaction Score| Score provided by the customer reflecting their satisfaction with the bank's complaint resolution process.                 |
| Card Type         | The type of card held by the customer, potentially impacting their engagement and loyalty.                                |
| Points Earned     | The points earned by the customer through credit card usage, contributing to the overall customer profile.                |

## Result :
**Model :** LinearClassifier (Tensorflow) <br>
**Accuracy :** %99
