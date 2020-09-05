import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('D:\\Data Analytics\\titanic.csv', index_col = 0)

#I don't want to analyze Siblings/Spouses or Parents/Children isolatedly instead I will create new column which will indicate presence of family members
data['Family'] = (data['SibSp'] > 0) | (data['Parch'] > 0)
#Is passenger a child? sort data values into bins "child" and "adult"
data['AgeRange'] = pd.cut(data['Age'], [0,15,80], labels = ['child', 'adult'])
#Get rid of variables we are not going to use
data.drop(['Name', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis = 1, inplace = True)

  
#Summarise Variables and count Missing values
data.info()
round(data.describe(), 2)
data.isnull().sum().to_frame(name = 'missing').T

for i in ['Survived', 'Pclass', 'Sex', 'Embarked', 'Family', 'AgeRange']:
    plt.figure()
    plt.xlabel(i)
    data[i].value_counts().plot(kind = 'bar')
    plt.show()

#Further Data Exploration \n What is the survival rate by class, sex and age? What about combining these factors?

#First lets drop rows with missing values for Age
data_clean_age = data.dropna(subset = ['Age'])

#Let’s take a look at the distribution of passengers by age and fare, grouped by sex and class, and with survival information.

def scatter_plot_class(pclass):
    g = sns.FacetGrid(data_clean_age[data_clean_age['Pclass'] == pclass], 
                      col='Sex',
                      col_order=['male', 'female'],
                      hue='Survived', 
                      hue_kws=dict(marker=['v', '^']), 
                      size=6)
    g = (g.map(plt.scatter, 'Age', 'Fare', edgecolor='w', alpha=0.7, s=80).add_legend())
    g.fig.suptitle('CLASS {}'.format(pclass))
    
#plotted separately because the fare scale for the first class makes it difficult to visualize second and third class charts
scatter_plot_class(1)
scatter_plot_class(2)
scatter_plot_class(3)

#Conclusion: Females have higher chance of survivability, especially in Pclass 1 and 2. Children have higher chance of survival in Pclass 1 and 2 

#Survival rate by class, sex and age range:
survived_by_class = data_clean_age.groupby('Pclass')['Survived'].mean()
survived_by_class
plt.figure()
survived_by_class.plot.bar(color = '#CC6600')


survived_by_sex = data_clean_age.groupby('Sex')['Survived'].mean()
survived_by_sex
plt.figure()
survived_by_sex.plot.bar(color = '#CCCC00')

survived_by_age = data_clean_age.groupby('AgeRange')['Survived'].mean()
survived_by_age
plt.figure()
survived_by_age.plot.bar(color = '#99CC00')


survival_summary = pd.concat([data_clean_age.groupby(['Pclass', 'Sex', 'AgeRange'])['Survived'].mean(),
                              data_clean_age.groupby(['Pclass', 'Sex', 'AgeRange'])['Survived'].count()],
                              axis = 1)
survival_summary.columns = ['Survived', 'Count']
print(survival_summary)                         

#The following chart visualizes the results from the table                         
g = sns.factorplot(
    x='AgeRange', 
    y='Survived', 
    col='Pclass',
    row='Sex',
    data=data_clean_age,
    margin_titles=True, 
    kind="bar", 
    ci=None)                         
                         
#Conclusion: The highest survival rate is among children, particularly Classes 1 and 2. Also females in Class 3 have higher survival rate than males in any class, even 1. Which tells us the priority was to save females over rich males.

#Was there a difference in fare rates for males and females?
data_clean_age.groupby(['Sex'])['Fare'].mean()
#On average females had almost twice higher fare rate than males. Was this trend present in all Classes?
data_clean_age.groupby(['Pclass', 'Sex'])['Fare'].mean()

g= sns.factorplot(
        y = 'Fare',
        x = 'Sex',
        col = 'Pclass',
        data = data_clean_age,
        kind = 'bar')

#Conclusion: the fare was much higher for females in 1st class, and slightly higher for 2 and 3 classes

#Was there particular Embaking location charging females more?
data_clean_age.groupby(['Embarked', 'Pclass', 'Sex'])['Fare'].mean()
g= sns.factorplot(
        y = 'Fare',
        x = 'Sex',
        col = 'Pclass',
        row = 'Embarked',
        data = data_clean_age,
        kind = 'bar')

#Locations S had the largest difference in prices based on Sex, followed by C, where location Q had identical prices for Classes 1 and 2, and slightly higher fare for males in class 3 than females, which is unusual.

#What fraction of the passengers embarked on each port? Is there a difference in their survival rates?
data_clean_embarked = data.dropna(subset = ['Embarked'])
embarked_summary = data_clean_embarked.groupby('Embarked').mean()
embarked_summary['Count'] = data_clean_embarked['Embarked'].value_counts()
embarked_summary

fig, (axis1,axis2) = plt.subplots(1, 2, figsize=(14,6))

sns.countplot(x='Embarked', data=data_clean_embarked, order=['S','C','Q'], ax=axis1)
sns.barplot(x=embarked_summary.index, y='Survived', data=embarked_summary, order=['S','C','Q'], ax=axis2)

#Passengers embarked in location C had the higher chance of survival, this could be explained by the mean Class being lower than two other locations, indicating more rich people boarding in location C.

#Is the presence of a family member a good indicator for survival?
survived_by_family = data.groupby('Family')['Survived'].mean()
survived_by_family

#Presence of family indicates higher chance of survival, let's find out what other factors may have influenced that.
family_by_class = data.groupby('Pclass')['Family'].mean()
family_by_class

family_by_sex = data.groupby('Sex')['Family'].mean()
family_by_sex

family_by_age = data.groupby('AgeRange')['Family'].mean()
family_by_age

#Richer people were more likely to have family members, women and children were more likely to have family members. As discovered earlier these factors have higher survival rate.