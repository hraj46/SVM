// Libraries
import org.apache.spark.sql.SparkSession 
import org.apache.spark.sql.functions.{when, col} 
import org.apache.spark.ml.feature.{VectorAssembler, StandardScaler, StringIndexer} 
import org.apache.spark.ml.classification.LinearSVC 
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator 
import org.apache.spark.ml.Pipeline 
import org.apache.spark.mllib.evaluation.MulticlassMetrics

// Step 1: Start Spark session 
val spark = SparkSession.builder() 
.appName("EnergySVM_Binary") 
.master("local[*]") 
.getOrCreate()

// Step 2: Load and clean data 
val df = spark.read 
.option("header", "true") 
.option("inferSchema", "true") 
.csv("C:/Users/HimanshuRaj/Downloads/Energy.csv") 
.na.drop()

// Step 3: Create binary label: Normal (<300) vs High (>=300) 
val binaryLabeled = df.withColumn("label", 
when(col("Appliances") < 300, "Normal").otherwise("High"))

// Step 4: Down sample dominant class to balance 
val normal = binaryLabeled.filter(col("label") === "Normal").sample(0.5, seed = 42) 
val high = binaryLabeled.filter(col("label") === "High") 
val balanced = normal.union(high) 

// Step 5: Index label 
val indexer = new StringIndexer().setInputCol("label").setOutputCol("labelIndex")

// Step 6: Assemble and scale features 
val assembler = new VectorAssembler() 
.setInputCols(Array("T1", "RH_1", "T2", "RH_2", "T_out", "RH_out")) 
.setOutputCol("features_unscaled") 
val scaler = new StandardScaler() 
.setInputCol("features_unscaled") 
.setOutputCol("features")

// Step 7: Define SVM classifier 
val svm = new LinearSVC() 
.setLabelCol("labelIndex") 
.setFeaturesCol("features") 
.setMaxIter(50) 
.setRegParam(0.3)  // Higher regularization to reduce overfitting

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
println(s"Precision: ${metrics.weightedPrecision}") 
println(s"Recall: ${metrics.weightedRecall}") 
println(s"F1 Score: ${metrics.weightedFMeasure}") 