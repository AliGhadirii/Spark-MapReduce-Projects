from pyspark.sql import SparkSession
from pyspark.sql.functions import when, regexp_extract
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier,
    GBTClassifier,
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName("CC Project").getOrCreate()
sc = spark.sparkContext

# using spark dataframe instead of traditional RDDs, better performance
df_train = spark.read.csv("train.csv", header=True, inferSchema=True)
df_test = spark.read.csv("test.csv", header=True, inferSchema=True)

# filling Embarked with most repeated value which was 's'
df_train = df_train.fillna({"Embarked": "S"})

df_train = df_train.withColumn(
    "Title", regexp_extract(df_train["Name"], "([A-Za-z]+)\.", 1)
)

title_dic = {
    "Mr": "Mr",
    "Miss": "Miss",
    "Mrs": "Mrs",
    "Master": "Master",
    "Mlle": "Miss",
    "Major": "Mr",
    "Col": "Mr",
    "Sir": "Mr",
    "Don": "Mr",
    "Mme": "Miss",
    "Jonkheer": "Mr",
    "Lady": "Mrs",
    "Capt": "Mr",
    "Countess": "Mrs",
    "Ms": "Miss",
    "Dona": "Mrs",
    "Dr": "Mr",
    "Rev": "Mr",
}

df_train = df_train.replace(to_replace=title_dic, subset="Title")

df_train = df_train.withColumn(
    "Age",
    when((df_train["Age"].isNull()) & (df_train["Title"] == "Mr"), 33.02).otherwise(
        df_train["Age"]
    ),
)
df_train = df_train.withColumn(
    "Age",
    when((df_train["Age"].isNull()) & (df_train["Title"] == "Mrs"), 35.98).otherwise(
        df_train["Age"]
    ),
)
df_train = df_train.withColumn(
    "Age",
    when((df_train["Age"].isNull()) & (df_train["Title"] == "Master"), 4.57).otherwise(
        df_train["Age"]
    ),
)
df_train = df_train.withColumn(
    "Age",
    when((df_train["Age"].isNull()) & (df_train["Title"] == "Miss"), 21.86).otherwise(
        df_train["Age"]
    ),
)

# Combining Parch and SibSp
df_train = df_train.withColumn("FamilySize", df_train["Parch"] + df_train["SibSp"])

df_train = df_train.drop(
    "PassengerId", "Name", "SibSp", "Parch", "Ticket", "Cabin", "Title"
)

# Converting cateforical values to numerical values
stringIndex = StringIndexer(
    inputCols=["Sex", "Embarked"], outputCols=["SexNum", "EmbNum"]
)
stringIndex_model = stringIndex.fit(df_train)
df_train = stringIndex_model.transform(df_train).drop("Sex", "Embarked")


# filling Fare with with the median of this feature in train data (14.45)
df_test = df_test.fillna({"Fare": 14.45})

df_test = df_test.withColumn(
    "Title", regexp_extract(df_test["Name"], "([A-Za-z]+)\.", 1)
)

df_test = df_test.replace(to_replace=title_dic, subset="Title")

# filling age feature based on what we calculated from train data
df_test = df_test.withColumn(
    "Age",
    when((df_test["Age"].isNull()) & (df_test["Title"] == "Mr"), 33.02).otherwise(
        df_test["Age"]
    ),
)
df_test = df_test.withColumn(
    "Age",
    when((df_test["Age"].isNull()) & (df_test["Title"] == "Mrs"), 35.98).otherwise(
        df_test["Age"]
    ),
)
df_test = df_test.withColumn(
    "Age",
    when((df_test["Age"].isNull()) & (df_test["Title"] == "Master"), 4.57).otherwise(
        df_test["Age"]
    ),
)
df_test = df_test.withColumn(
    "Age",
    when((df_test["Age"].isNull()) & (df_test["Title"] == "Miss"), 21.86).otherwise(
        df_test["Age"]
    ),
)

# Combining Parch and SibSp
df_test = df_test.withColumn("FamilySize", df_test["Parch"] + df_test["SibSp"])

df_test = df_test.drop(
    "PassengerId", "Name", "SibSp", "Parch", "Ticket", "Cabin", "Title"
)

# Converting cateforical values to numerical values
stringIndex = StringIndexer(
    inputCols=["Sex", "Embarked"], outputCols=["SexNum", "EmbNum"]
)
stringIndex_model = stringIndex.fit(df_test)
df_test = stringIndex_model.transform(df_test).drop("Sex", "Embarked")

vec_asmbl = VectorAssembler(inputCols=df_train.columns[1:], outputCol="features")
df_train = vec_asmbl.transform(df_train).select("features", "Survived")

df_train, df_val = df_train.randomSplit([0.7, 0.3])

evaluator = MulticlassClassificationEvaluator(
    labelCol="Survived", metricName="accuracy"
)
