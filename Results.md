`run eda.py`
### Output
*First 5 rows of the dataset*
    Date            Closing Price   Opening Price   Low Price   Volume       Rate of Change %
- 2024-08-01          156.3          159.3  ...      156.1      79.15M           -2.56%
- 2024-07-31          160.4          158.2  ...      158.1     173.91M            1.07%
- 2024-07-30          158.7          158.8  ...      158.0     138.14M           -0.63%
- 2024-07-29          159.7          158.7  ...      158.4     126.28M            1.14%
- 2024-07-26          157.9          159.3  ...      157.9     155.08M           -0.13%

[5 rows x 7 columns]

Data types and missing values:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 9202 entries, 0 to 9201
Data columns (total 7 columns):
 #   Column            Non-Null   Count      Dtype         
---  ------            ---------  -----      -----         
 0.   Date              9202    non-null   datetime64[ns]
 1.   Closing Price     9202    non-null   float64       
 2.   Opening Price     9202    non-null   float64       
 3.  High Price         9202    non-null   float64       
 4.   Low Price         9202    non-null   float64       
 5.   Volume            9202    non-null   object        
 6.   Rate of Change %  9202    non-null   object        
- dtypes: datetime64[ns](1), float64(4), object(2)
- memory usage: 503.4+ KB
- None

*Descriptive statistics:*
                                  Date  Closing Price  ...   High Price    Low Price
- count                           9202    9202.000000  ...  9202.000000  9202.000000
- mean   2005-10-21 08:44:04.642469248      92.180961  ...    93.176451    91.330146
- min              1987-02-12 00:00:00      33.000000  ...    33.200000    32.200000
- 25%              1996-06-06 06:00:00      52.000000  ...    52.800000    51.500000
- 50%              2005-10-11 12:00:00      85.100000  ...    86.050000    84.200000
- 75%              2015-03-04 18:00:00     110.800000  ...   111.900000   109.275000
- max              2024-08-01 00:00:00     305.900000  ...   311.800000   303.900000
- std                              NaN      50.452228  ...    51.049837    50.087405

[8 rows x 5 columns]

Missing values in each column:
- Date                0
- Closing Price       0
- Opening Price       0
- High Price          0
- Low Price           0
- Volume              0
- Rate of Change %    0
- dtype: int64

![Figure_1](https://github.com/user-attachments/assets/d69dc6d1-bca2-4678-ba69-fdac1c479207)
![Figure_2 1](https://github.com/user-attachments/assets/381e961a-daee-41bc-bdf6-e3bff4ed88fc)
![Figure_3](https://github.com/user-attachments/assets/22eb47c2-c7d6-4048-af6b-ee105de02838)

`run data_prep.py`
### Output
Training data shape: torch.Size([7361, 1, 3])
Test data shape: torch.Size([1841, 1, 3])

`run model.py`
### Output
- Epoch [10/200], Loss: 0.0312
- Epoch [20/200], Loss: 0.0231
- Epoch [30/200], Loss: 0.0182
- Epoch [40/200], Loss: 0.0159
- Epoch [50/200], Loss: 0.0146
- Epoch [60/200], Loss: 0.0130
- Epoch [70/200], Loss: 0.0113
- Epoch [80/200], Loss: 0.0095
- Epoch [90/200], Loss: 0.0076
- Epoch [100/200], Loss: 0.0057
- Epoch [110/200], Loss: 0.0040
- Epoch [120/200], Loss: 0.0025
- Epoch [130/200], Loss: 0.0014
- Epoch [140/200], Loss: 0.0006
- Epoch [150/200], Loss: 0.0002
- Epoch [160/200], Loss: 0.0001
- Epoch [170/200], Loss: 0.0000
- Epoch [180/200], Loss: 0.0000
- Epoch [190/200], Loss: 0.0000
- Epoch [200/200], Loss: 0.0000
*Confusion Matrix:*
[[2761]]
**Accuracy:** 100.00%

![2761](https://github.com/user-attachments/assets/bed99b4c-5166-4e44-b3f9-f999663285a7)

`run evalve.py`
### Output
- Epoch [10/200], Loss: 0.1062
- Epoch [20/200], Loss: 0.0806
- Epoch [30/200], Loss: 0.0582
- Epoch [40/200], Loss: 0.0391
- Epoch [50/200], Loss: 0.0241
- Epoch [60/200], Loss: 0.0145
- Epoch [70/200], Loss: 0.0099
- Epoch [80/200], Loss: 0.0085
- Epoch [90/200], Loss: 0.0078
- Epoch [100/200], Loss: 0.0071
- Epoch [110/200], Loss: 0.0063
- Epoch [120/200], Loss: 0.0056
- Epoch [130/200], Loss: 0.0048
- Epoch [140/200], Loss: 0.0042
- Epoch [150/200], Loss: 0.0035
- Epoch [160/200], Loss: 0.0029
- Epoch [170/200], Loss: 0.0023
- Epoch [180/200], Loss: 0.0018
- Epoch [190/200], Loss: 0.0014
- Epoch [200/200], Loss: 0.0010
*Mean Squared Error: 87.95034663283606*

![200](https://github.com/user-attachments/assets/4afdf413-ee81-45d2-87b8-8f395e42f24d)

`run improved.py`
### Output
*Mean Squared Error (XGBoost): 654.4639729729596*

![improved](https://github.com/user-attachments/assets/7c024336-6512-4b3b-ade4-228fe131f770)





