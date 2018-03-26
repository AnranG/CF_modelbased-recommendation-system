
import java.io.{File, PrintWriter}

import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}


object CF_ModelBased {

  def main(args: Array[String]): Unit = {
    // Load and parse the data

    val conf = new SparkConf()
    conf.setAppName("CF_ModelBased")
    conf.setMaster("local[*]")
    conf.set("spark.executor.memory","1g")
    conf.set("driver-memory","4g")
    conf.set("executor-cores","2")

    val sc = new SparkContext(conf)



    var total_data = sc.textFile("data/video_small_num.csv").cache().map(line => line.split(","))
    val total_header = total_data.first()
    total_data = total_data.filter(_ (0) != total_header(0))

    val total = total_data.map(line => ((line(0).toInt, line(1).toInt), line(2).toDouble))


    var test_data = sc.textFile("data/video_small_testing_num.csv").map(line => line.split(","))
    val test_header = test_data.first()
    test_data = test_data.filter(_ (0) != test_header(0))

    val test = test_data.map(line => ((line(0).toInt, line(1).toInt), line(2).toDouble))


    val training_rdd = total.subtract(test).map(line => Array(line._1._1,line._1._2,line._2))


    val testing_rdd = test.map(line => Array(line._1._1,line._1._2,line._2))
      //.collect()






    val ratings = training_rdd.map { case Array(user, item, rate) =>
      Rating(user.toInt, item.toInt, rate.toDouble)
    }

    // Build the recommendation model using ALS
    val rank = 5
    val numIterations = 10
    val model = ALS.train(ratings, rank, numIterations, 0.5,-1,1)

    // Evaluate the model on rating data
    val ratings_testing = testing_rdd.map { case Array(user, product, rate) =>
      Rating(user.toInt, product.toInt, rate.toDouble)
    }


    val usersProducts = ratings_testing.map { case Rating(user, product, rate) =>
      (user, product)
    }


    val predictions = model.predict(usersProducts)

    val predictions_after_scale =   scale_rating(predictions)
        .map { case Rating(user, product, rate) =>
        ((user, product), rate)
      }

    val ratesAndPreds = ratings_testing.map { case Rating(user, product, rate) =>
      ((user, product), rate)
    }.join(predictions_after_scale).sortBy(x=>(x._1._1,x._1._2))
    val MSE = ratesAndPreds.map { case ((user, product), (r1, r2)) =>
      val err = (r1 - r2)
      err * err
    }.mean()
    println(s"Mean Squared Error = $MSE")

    val output_file = new File("data/ModelBasedCF_prediction.txt")
    val out = new PrintWriter(output_file)
    for (elem <- ratesAndPreds.collect()){

      out.write(elem._1._1 + "," + elem._1._2 +","+ elem._2._2 +"\n")

    }

    out.close()


    val RMSE = math.sqrt(MSE)

    val diff= ratesAndPreds.map { case ((user, product), (r1, r2)) => math.abs(r1 - r2)}

    var num1=0
    var num2=0
    var num3=0
    var num4=0
    var num5=0
    for ( x <- diff.collect) {
      x match {
        case x if (x>=0 && x<1) => num1+=1;
        case x if (x>=1 && x<2)=> num2+=1;
        case x if (x>=2 && x<3) => num3+=1;
        case x if (x>=3 && x<4) => num4+=1;
        case x if (x>=4 ) => num5+=1;
      }
    }

    println(">=0 and <1:"+ num1)
    println(">=1 and <2:"+ num2)
    println(">=2 and <3:"+ num3)
    println(">=3 and <4:"+ num4)
    println(">=4 :"+ num5)
    println("RMSE = " + RMSE)



  }


  def scale_rating(test_predict: RDD[Rating]): RDD[Rating] ={
    val min=test_predict.map {
      case Rating(user, product, rate) =>
        rate
    }.min()

    val max=test_predict.map {
      case Rating(user, product, rate) =>
        rate
    }.max()

    val interval=max-min

    test_predict.map{
      case Rating(user, product, rate) =>

        val r= ((rate-min)/interval)*4+1

        Rating(user, product, r)
    }
  }
}
