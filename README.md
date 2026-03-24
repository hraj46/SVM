# SVM
Energy Data Analysis Using Support Vector Machine 
(SVM) 
 
## Project Description 
In recent years, global energy consumption has increased significantly due to rapid population growth, technological advancement, and rising living standards. This increasing demand directly impacts electricity costs, environmental sustainability, and the overall efficiency of power distribution systems. As a result, analyzing and predicting energy usage patterns has become a key priority for research institutions, industries, and smart home developers. 
Modern residential and commercial buildings are equipped with a variety of environmental sensors that record parameters such as temperature, humidity, wind speed, and outdoor weather conditions, along with detailed logs of appliance-level energy consumption. These rich and highfrequency datasets provide an excellent opportunity to apply machine learning techniques for detecting usage trends and forecasting consumption behavior. This project focuses on building a classification model using Support Vector Machine (SVM) to automatically categorize appliance energy consumption as either Normal or High based on various environmental factors. By learning from historical data, the model can identify situations where energy usage spikes unexpectedly and help in enhancing energy efficiency strategies. 
 
Objectives of the Project 
This project aims to: 
1.	Perform preprocessing and cleaning of an energy dataset collected from environmental sensors. 
2.	Conduct Exploratory Data Analysis (EDA) to uncover correlations and trends. 
3.	Binary a binary classification model using Support Vector Machine (SVM). 
4.	Predict whether energy consumption values fall under Normal (<300 Wh) or High (≥300 Wh). 
5.	Evaluate model performance using accuracy, precision, recall, and F1 score. 
6.	Visualize analytical findings using appropriate charts and dashboards. 
7.	Provide actionable insights to improve energy usage efficiency. 
 
Dataset Description 
The dataset used in this project contains readings from multiple indoor sensors, outdoor environmental parameters, and hourly appliance energy consumption values. 
Column 	Meaning 	Typical Range 	Type 	Importance 
T1 	Temperature in the living room (°C) 	18 – 26 	Numerical 	Higher T1 may increase energy use (AC/fan). 
RH_1 	Relative Humidity in living room (%) 	30 – 60 	Numerical 	High humidity can increase cooling or dehumidifier use. 
T2 	Temperature in the kitchen (°C) 	17 – 25 	Numerical 	Influences appliance heat and overall energy load. 
RH_2 	Relative Humidity in kitchen (%) 	35 – 65 	Numerical 	Affects cooling/heating balance in that area. 
T_out 	Outdoor temperature (°C) 	5 – 35 	Numerical 	Major factor affecting indoor cooling/heating needs. 
RH_out 	Outdoor humidity 
(%) 	20 – 80 	Numerical 	Impacts indoor air comfort and HVAC demand. 
	Appliances 	Energy consumption 	100 – 800 	Numerical 	This is the 
of appliances (Wh) 	prediction target for — Target Variable 	your ML model. 
 
 
Methodology 
A.	Data Preprocessing 
•	Removing missing or inconsistent records 
•	Converting raw data into structured format 
•	Creating the target label (Normal or High) 
•	Downsampling the dominant class to balance the dataset 
•	Selecting relevant environmental features 
 
B.	Exploratory Data Analysis (EDA) 
•	Distribution of appliance energy consumption 
•	Relationship between temperature/humidity and electricity usage 
•	Correlation matrix to identify the strongest influencing sensors 
•	Time-based consumption patterns 
 
C.	Model Development 
 
 
 
 
 
 
 
 
 
Source Code: 
import org.apache.spark.sql.SparkSession import org.apache.spark.sql.functions.{when, col} 
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler, StringIndexer} import org.apache.spark.ml.classification.LinearSVC import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator import org.apache.spark.ml.Pipeline 
import org.apache.spark.mllib.evaluation.MulticlassMetrics 
 
// Step 1: Start Spark session val spark = SparkSession.builder() 
  .appName("EnergySVM_Binary") 
  .master("local[*]") 
  .getOrCreate() 
 
// Step 2: Load and clean data val df = spark.read 
  .option("header", "true") 
  .option("inferSchema", "true") 
  .csv("C:/Users/Himanshu Raj/Downloads/Energy.csv") 
  .na.drop() 
 
// Step 3: Create binary label: Normal (<300) vs High (>=300) 
val binaryLabeled = df.withColumn("label", when(col("Appliances") < 300, "Normal").otherwise("High")) 
 
// Step 4: Downsample dominant class to balance 
val normal = binaryLabeled.filter(col("label") === "Normal").sample(0.5, seed = 42) val high = binaryLabeled.filter(col("label") === "High") 
val balanced = normal.union(high) 
 
// Step 5: Index label 
val indexer = new StringIndexer().setInputCol("label").setOutputCol("labelIndex") 
 
// Step 6: Assemble and scale features val assembler = new VectorAssembler() 
  .setInputCols(Array("T1", "RH_1", "T2", "RH_2", "T_out", "RH_out")) 
  .setOutputCol("features_unscaled") 
 
val scaler = new StandardScaler() 
  .setInputCol("features_unscaled") 
  .setOutputCol("features") 
 
// Step 7: Define SVM classifier val svm = new LinearSVC()   .setLabelCol("labelIndex") 
  .setFeaturesCol("features") 
  .setMaxIter(50) 
  .setRegParam(0.3) // Higher regularization to reduce overfitting 
 
// Step 8: Build pipeline 
val pipeline = new Pipeline().setStages(Array(indexer, assembler, scaler, svm)) 
 
// Step 9: Train-test split 
val Array(train, test) = balanced.randomSplit(Array(0.8, 0.2), seed = 42) 
 
// Step 10: Train model 
val model = pipeline.fit(train) 
 
// Step 11: Predict 
val predictions = model.transform(test) 
 
// Step 12: Show predictions 
predictions.select("label", "labelIndex", "prediction").show(20, truncate = false) 
 
// Step 13: Evaluate accuracy 
val evaluator = new MulticlassClassificationEvaluator() 
  .setLabelCol("labelIndex") 
  .setPredictionCol("prediction") 
  .setMetricName("accuracy") 
 
val accuracy = evaluator.evaluate(predictions) 
println(s"Model Accuracy: ${accuracy * 100}%") 
 
// Step 14: Confusion matrix and metrics 
val predictionAndLabels = predictions.select("prediction", "labelIndex") 
  .rdd.map(row => (row.getDouble(0), row.getDouble(1))) 
 
val metrics = new MulticlassMetrics(predictionAndLabels) 
 
println("Confusion Matrix:") 
println(metrics.confusionMatrix) 
 
println(s"Precision: ${metrics.weightedPrecision}") println(s"Recall: ${metrics.weightedRecall}") println(s"F1 Score: ${metrics.weightedFMeasure}") 
  
  
 
 
 
 
 
 
 
Output 
  
Accuracy: 85.71% 
The results demonstrate that the Support Vector Machine model is effective for binary classification of household energy consumption. With high accuracy and strong precisionrecall values, the model successfully distinguishes between Normal and High appliance usage levels. The confusion matrix further confirms the model’s reliability, showing a balanced detection of both categories. These outputs indicate that SVM is a suitable technique for energy usage prediction and can support decision-making in energy monitoring systems. 
 
 
 
 
 
 
 
Dashboard:  
  
 
My dashboard visualizes the key environmental and appliancerelated variables that affect energy consumption. The charts help me understand the distribution of temperature and humidity, their relationships, and their influence on appliance load. The correlation heatmap validates feature selection for the SVM model. Overall, these visual insights support accurate prediction and deepen domain understanding. 
 
Conclusion 
The project demonstrates that Support Vector Machines are highly effective for classifying energy consumption levels using environmental sensor data. This ML-driven approach can help households, industries, and smart building systems understand and manage energy demand more efficiently.
