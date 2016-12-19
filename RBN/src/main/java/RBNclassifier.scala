import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import scala.io.Source
import java.io._
import util.matching.Regex
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.clustering.KMeans

import scala.math._
import java.lang.Float

object RBNclassifier {
  def main(args: Array[String]): Unit = {
    
    val modified_file = new PrintWriter(new File("D:/data/cal_data.txt" ))
    System.setProperty("hadoop.home.dir", "D:/BSC IT/L4S1/ANN/hadoop-2.6.4/bin");
    val spark = SparkSession
      .builder
      .appName("Tt").master("local")
      .getOrCreate()
    
val libsvm_dataset = spark.read.format("libsvm").load("D:/data/fdata.txt")

// applying k means clustering, dividing into 5 clusters
val kmeans = new KMeans().setK(5).setSeed(1L)
val model = kmeans.fit(libsvm_dataset)

// Evaluate clustering by computing Within Set Sum of Squared Errors.
val WSSSE = model.computeCost(libsvm_dataset)
println(s"Within Set Sum of Squared Errors = $WSSSE")

// Shows the result.
println("Cluster Centers: ")
model.clusterCenters.foreach(println)

var clusterarray:Array[Vector] = new Array[Vector](5)

val formatted_source = scala.io.Source.fromFile("D:/data/fdata.txt")

// Trains a k-means model.
    var psi_value:Double =0 
    var cluster_no:Int = 0;
    var line_of_array:Array[String] = new Array[String](10)
    for (line <- formatted_source.getLines()) {
    cluster_no =1
      line_of_array = line.split(" ")
      modified_file.write(line_of_array(0)+" ")
      for(singlecluster  <-model.clusterCenters){        
        psi_value = cal_transformation_matrix(line,singlecluster)
        modified_file.write(cluster_no+":"+psi_value+" ")
        cluster_no = cluster_no+1;
      }
      modified_file.write("\n")
     }
   modified_file.close()

    
val data = spark.read.format("libsvm").load("D://data//cal_data.txt")

    // Split the data into train and test
    val splits = data.randomSplit(Array(0.5, 0.5), seed = 1234L)
    val train = splits(0)
    val test = splits(1)
    
    // specify layers for the neural network:
    // input layer of size 5 (features), one intermediate of size 4
    // and output of size 5 (classes)
    val layers = Array[Int](5, 4, 5)

    // create the trainer and set its parameters
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100)

    // train the model
    val classify_model1 = trainer.fit(train)
    //model.save("modeldata")
    // compute accuracy on the test set
    val result = classify_model1.transform(test)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
      result.rdd.saveAsTextFile("D:/data/result1.txt")
    println("Test set accuracy = " + evaluator.evaluate(predictionAndLabels))
    
    // train the model
    val classify_model2 = trainer.fit(test)
    //model.save("modeldata")
    // compute accuracy on the test set
    val result2 = classify_model2.transform(train)
    val predictionAndLabels2 = result2.select("prediction", "label")
    val evaluator2 = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")
      result2.rdd.saveAsTextFile("D:/data/result2.txt")
    println("Train set accuracy = " + evaluator2.evaluate(predictionAndLabels2))
    
    spark.stop()
    
}
  
    def cal_transformation_matrix(line_data:String,centroid_vector:Vector ):Double={
      var data_deviation:Double = 0
      var line_info:Array[String] = new Array[String](10)
      var cluster_midpoint:Array[String] = new Array[String](2)
      line_info = line_data.split(" ")
      for(i <- 0 until 9){
        cluster_midpoint = line_info(i+1).split(":")
        data_deviation = data_deviation+exp(-1*(pow((centroid_vector.apply(i)-cluster_midpoint(1).toDouble),2)/0.02))
      }
   
    return data_deviation; 
   }
  
}