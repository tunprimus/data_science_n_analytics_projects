import pandas as pd


def calculate_demographic_data(print_data=True):
    # Read data from file
    df = pd.read_csv("adult.data.csv")

    # How many of each race are represented in this dataset? This should be a Pandas series with race names as the index labels.
    df["count"] = 1
    race_count = df.groupby(["race"]).count()["count"]

    # What is the average age of men?
    average_age_men = round(df[df["sex"] == "Male"]["age"].mean(), 1) or round(df.loc[df["sex"] == "Male", "age"].mean(), 1)

    # What is the percentage of people who have a Bachelor's degree?
    percentage_bachelors = round(df["education"].value_counts(normalize=True)["Bachelors"] * 100, 1)

    # What percentage of people with advanced education (`Bachelors`, `Masters`, or `Doctorate`) make more than 50K?
    # What percentage of people without advanced education make more than 50K?

    # with and without `Bachelors`, `Masters`, or `Doctorate`
    # high_mask = df[(df["education"] == "Bachelors") | (df["education"] == "Masters") | (df["education"] == "Doctorate")]["education"]
    # higher_education = ((df["education"].value_counts(normalize=True)["Bachelors"] + df["education"].value_counts(normalize=True)["Masters"] + df["education"].value_counts(normalize=True)["Doctorate"]) * 100).round(1)

    
    higher_education = df[df["education"].isin(["Bachelors", "Masters", "Doctorate"])]
    lower_education = df[~df["education"].isin(["Bachelors", "Masters", "Doctorate"])]

    # percentage with salary >50K
    higher_education_rich = ((higher_education["salary"] == ">50K").value_counts(normalize=True) * 100).round(1)[True]
    lower_education_rich = ((lower_education["salary"] == ">50K").value_counts(normalize=True) * 100).round(1)[True]

    # What is the minimum number of hours a person works per week (hours-per-week feature)?
    min_work_hours = df["hours-per-week"].min()

    # What percentage of the people who work the minimum number of hours per week have a salary of >50K?
    # df[df["hours-per-week"] > min_work_hours].count()
    # df[df["hours-per-week"] == min_work_hours]
    # df[df["hours-per-week"] == min_work_hours].count()
    # df[df["hours-per-week"] > min_work_hours, df["salary"] == ">50K"]
    # df[df["hours-per-week"] == min_work_hours, df["salary"] == ">50K"]
    # df[(df["hours-per-week"] == min_work_hours) & (df["salary"] == ">50K")]["salary"].value_counts()
    # df[(df["hours-per-week"] == min_work_hours) & (df["salary"] != ">50K")]["salary"].value_counts()
    # df[df["hours-per-week"] == min_work_hours].count()["salary"]
    # rich_percentage = ((df[(df["hours-per-week"] == min_work_hours) & (df["salary"] == ">50K")]["salary"].value_counts() / df[df["hours-per-week"] == min_work_hours].count()["salary"]) * 100).round(1)
    
    rich_percentage = ((df[(df["hours-per-week"] == min_work_hours) & (df["salary"] == ">50K")]["salary"].count() / df[df["hours-per-week"] == min_work_hours].count()["salary"]) * 100).round(1)

    # What country has the highest percentage of people that earn >50K?
    # df.head()
    # df[df["salary"] == ">50K"]
    # df[df["salary"] == ">50K"].groupby(df["native-country"]).count()
    # df["salary"].groupby(df["native-country"]).count()
    # df["salary"].groupby(df["native-country"]).value_counts(normalize=True)
    # df[df["salary"] == ">50K"]["native-country"].value_counts() / df["native-country"].value_counts()
    (df[df["salary"] == ">50K"]["native-country"].value_counts() / df["native-country"].value_counts()).idxmax()
    highest_ipc_country = (df[df["salary"] == ">50K"]["native-country"].value_counts() / df["native-country"].value_counts())

    highest_earning_country = highest_ipc_country.idxmax()
    highest_earning_country_percentage = (highest_ipc_country[highest_earning_country] * 100).round(1)

    # Identify the most popular occupation for those who earn >50K in India.
    # df[(df["salary"] == ">50K") & (df["native-country"] == "India")]
    # df[(df["salary"] == ">50K") & (df["native-country"] == "India")]["occupation"]
    # df[(df["salary"] == ">50K") & (df["native-country"] == "India")]["occupation"].describe()
    # df[(df["salary"] == ">50K") & (df["native-country"] == "India")]["occupation"].describe()["top"]

    top_IN_occupation = df[(df["salary"] == ">50K") & (df["native-country"] == "India")]["occupation"].describe()["top"]



    # DO NOT MODIFY BELOW THIS LINE

    if print_data:
        print("Number of each race:\n", race_count) 
        print("Average age of men:", average_age_men)
        print(f"Percentage with Bachelors degrees: {percentage_bachelors}%")
        print(f"Percentage with higher education that earn >50K: {higher_education_rich}%")
        print(f"Percentage without higher education that earn >50K: {lower_education_rich}%")
        print(f"Min work time: {min_work_hours} hours/week")
        print(f"Percentage of rich among those who work fewest hours: {rich_percentage}%")
        print("Country with highest percentage of rich:", highest_earning_country)
        print(f"Highest percentage of rich people in country: {highest_earning_country_percentage}%")
        print("Top occupations in India:", top_IN_occupation)

    return {
        'race_count': race_count,
        'average_age_men': average_age_men,
        'percentage_bachelors': percentage_bachelors,
        'higher_education_rich': higher_education_rich,
        'lower_education_rich': lower_education_rich,
        'min_work_hours': min_work_hours,
        'rich_percentage': rich_percentage,
        'highest_earning_country': highest_earning_country,
        'highest_earning_country_percentage':
        highest_earning_country_percentage,
        'top_IN_occupation': top_IN_occupation
    }
