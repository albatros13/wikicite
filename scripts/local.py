import findspark
import os

os.environ["PYSPARK_SUBMIT_ARGS"] = " --packages com.databricks:spark-xml_2.12:0.15.0 pyspark-shell"
os.environ["JAVA_HOME"] = "C:\Java"
findspark.init()

PROJECT_HOME = 'c:///users/natal/PycharmProjects/cite-classifications-wiki/'