//Import Spark packages
import org.apache.spark.sql.SparkSession
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.feature.{PCA, PCAModel, StandardScaler, VectorAssembler}
import org.apache.spark.ml.Pipeline

object Main {
  def main(args: Array[String]): Unit = {

    //Initialing context
    val conf = new SparkConf().setAppName("tedPCA").setMaster("local[*]")
    val sc = SparkContext.getOrCreate(conf)
    val spark = SparkSession
      .builder()
      .appName("tedPCA")
      .getOrCreate()

    //Make dataframe colnames case sensitive
    spark.sql("set spark.sql.caseSensitive=true")

    //Read file
    val iris = spark.read
      .format("csv")
      .option("header","true")
      .option("delimiter",",")
      .option("inferSchema","true")
      .load("/home/christopher/Documents/TED/PCA/iris.csv")

    //Show input
    iris.show()

    //Getting numerical colnames
    val numerical = iris.schema
      .fields
      .filter(field => field.dataType.toString.equals("DoubleType"))
      .map(_.name.toString)

    //Print colnames
    numerical.foreach(println(_))

    //Setting vector assembler
    val assembler = new VectorAssembler()
      .setInputCols(numerical)
      .setOutputCol("features")

    //Setting normalizer
    val scaler = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithMean(true)
      .setWithStd(true)

    //Setting PCA creation
    val pca = new PCA()
      .setInputCol("scaledFeatures")
      .setOutputCol("pcaFeatures")
      .setK(numerical.length)

    //Building pipeline
    val pipeline = new Pipeline().setStages(Array(assembler, scaler, pca))

    //Fit pca by applying all previous procedures
    val pcaFit = pipeline.fit(iris)

    //Creates the principal components to the dataset based on the fit
    val output = pcaFit.transform(iris)

    //Shows all steps
    output.show(false)

    //Gets PC coeficients. Each column corresponds to a PC, i.e., it's a eigenvector
    val pcaCoef = pcaFit.stages(2).asInstanceOf[PCAModel].pc

    //Print coefs matrix.
    println(pcaCoef.toString(numerical.length, Int.MaxValue))

    //How to get pca variability?


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                               Bonus
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    val trainRatio = 0.8

    val Array(trainingSet, testSet) = iris.randomSplit(Array[Double](trainRatio, 1 - trainRatio), seed = 1123581321)
    val (train, test) = (trainingSet.toDF, testSet.toDF)

    //Fit pca by applying all previous procedures
    val trainFit = pipeline.fit(train)

    //Creates the principal components to the dataset based on the fit
    val testOutput = trainFit.transform(test)

    //Shows all steps
    testOutput.show(false)

    //Gets PC coeficients. Each column corresponds to a PC, i.e., it's a eigenvector
    val trainCoef = trainFit.stages(2).asInstanceOf[PCAModel].pc

    //Print coefs matrix.
    println(trainCoef.toString(numerical.length, Int.MaxValue))


    spark.stop()
    sc.stop()
  }
}

