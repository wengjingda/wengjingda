---
layout:     post
title:      "分类算法以及分类流程简述"
subtitle:   ""
date:       2016-10-01 
author:     "wengjingda"
header-img: "img/post-bg-rwd.jpg"
catalog: true
tags:
    - 数据挖掘
    - spark
	- lda
---


一、MLlib简介
---------

MLlib是Spark的机器学习（ML）库。旨在简化机器学习的工程实践工作，并方便扩展到更大规模。MLlib由一些通用的学习算法和工具组成，包括分类、回归、聚类、协同过滤、降维等，同时还包括底层的优化原语和高层的管道API。

MLllib目前分为两个代码包：

 - spark.mllib 包含基于RDD的原始算法API
 - spark.ml 提供了基于DataFrames 高层次的API，可以用来构建机器学习管道

spark开发者推荐使用spark.ml，因为基于DataFrames的API更加的通用而且灵活。不过他们也会继续支持spark.mllib包。 用户可以放心使用，spark.mllib还会持续地增加新的功能。不过需要注意，如果新的算法能够适用于机器学习管道的概念，就应该将其放到 spark.ml包中，如：特征提取器和转换器。

[聚类](http://spark.apache.org/docs/1.6.2/mllib-clustering.html)（clustering）方法包括：
 - KMEANS聚类（K-means）
 - 二分KMEANS聚类（Bisecting k-means）
 - 隐含狄利克雷分布（Latent Dirichlet allocation (LDA)）
 - 混合高斯分布聚类（Gaussian Mixture Model (GMM)）
 - 幂迭代聚类 (PIC)
 - 流式KMEANS聚类（Streaming k-means）

接下来以利用java api实现spark中的lda算法作为引入，而其他聚类算法的实现过程可参照官方文档

**实验中平台及语言版本为：**  
**spark->1.6.2**  
**jdk->1.8**  
**hadoop->2.6**  

-------------------


二、LDA实例
---------

lda的原理及实现细则已经有很多优秀的博客阐述，再次不累述。

**利用java调用spark中的lda模型代码如下：**

``` java
import scala.Tuple2;

import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.clustering.DistributedLDAModel;
import org.apache.spark.mllib.clustering.LDA;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.SparkConf;

public class JavaLDAExample {
	public static void main(String[] args) {
		SparkConf conf = new SparkConf().setAppName("JavaLdaExample").setMaster("local");
		conf.set("spark.testing.memory", "2147480000");
		JavaSparkContext sc = new JavaSparkContext(conf);

		// 1 加载数据，返回的数据格式为：JavaRDD<Vector>
		// 其中每一个Vector代表文档的词向量
		String path = "/usr/local/spark1.6.2/data/mllib/sample_lda_data.txt";
		JavaRDD<String> data = sc.textFile(path);
		JavaRDD<Vector> parsedData = data.map(new Function<String, Vector>() {
			public Vector call(String s) {
				String[] sarray = s.trim().split(" ");
				double[] values = new double[sarray.length];
				for (int i = 0; i < sarray.length; i++)
					values[i] = Double.parseDouble(sarray[i]);
				return Vectors.dense(values);
			}
		});

		// 2 转换数据，spark要求的数据格式为 JavaRDD<long, Vector>
		// Long类型指代词向量的索引（唯一标识号）
		// 此处利用zipWithIndex()将vector与vector在RDD的索引组成键值对，并通过swap()调转索引及词向量
		JavaPairRDD<Long, Vector> corpus = JavaPairRDD
				.fromJavaRDD(parsedData.zipWithIndex().map(new Function<Tuple2<Vector, Long>, Tuple2<Long, Vector>>() {
					public Tuple2<Long, Vector> call(Tuple2<Vector, Long> doc_id) {
						return doc_id.swap();
					}
				}));
		corpus.cache(); // 缓存

		
		// 3 建立模型，设置训练参数，训练模型
		/**
		 * k: 主题数，或者聚类中心数 
		 * DocConcentration：文章分布的超参数(Dirichlet分布的参数)，必需>1.0
		 * TopicConcentration：主题分布的超参数(Dirichlet分布的参数)，必需>1.0 
		 * MaxIterations：迭代次数
		 * Seed：随机种子 
		 * CheckpointInterval：迭代计算时检查点的间隔
		 * Optimizer：优化计算方法，目前支持"em", "online"
		 
		 可在创建lda模型时创建
		 ldaModel = new lda().setK(5).setSeed(5).setDocConcentration(5);
		 
		 也可以通过lda模型获取
		 ldaModel.getK();
		 
		 */

		DistributedLDAModel ldaModel = (DistributedLDAModel) new LDA().setK(3).run(corpus);

		// 打印主题分布矩阵
		System.out.println("Learned topics (as distributions over vocab of " + ldaModel.vocabSize() + " words):");
		Matrix topics = ldaModel.topicsMatrix();
		for (int topic = 0; topic < 3; topic++) {
			System.out.print("Topic " + topic + ":");
			for (int word = 0; word < ldaModel.vocabSize(); word++) {
				System.out.print(" " + topics.apply(word, topic));
			}
			System.out.println();
		}

		ldaModel.save(sc.sc(), "myLDAModel"); // 缓存模型
		DistributedLDAModel sameModel = DistributedLDAModel.load(sc.sc(), "myLDAModel"); // 读取模型
	}
}
```

---
**LDA模型说明：**

spark中**DistributedLDAMode**继承自**LDAModel**，其中**LDAModel**主要用来初始化LDA模型，而**DistributedLDAModel**中提供运算的结果

**DistributedLDAMode比较重要的函数有：**

1、**topTopicsPerDocument(int k):**
对于每一个输入的文档，按权重排序，返回前k个主题。

Parameters:  
　　k - (undocumented)
Returns:  
　　RDD of (doc ID, topic indices, topic weights)

---
2、**topicDistributions():**
得到文档下的主题分布。

Returns:  
　　RDD of (document ID, topic distribution) pairs

---
3、 	**topicsMatrix():**
得到主题下的词条分布。

Returns:  
　　Matrix:  
　　　　Matrix(word_id, topic_id)表示特定主题下特定单词的权重

可用 weight = Matrix.apply(word_id, topic_id) 获得

 ---
 4、**describeTopics( int k):**
 按权重排序，返回各个主题下前k个词
 
Parameters:  
 　　maxTermsPerTopic - 需要返回前几个词
Returns:  
　　Array over topics. Each topic is represented as a pair of matching arrays: (term indices, term weights in topic). Each topic's terms are sorted in order of decreasing weight.

---
**可能遇到的问题：**

1、"A master URL must be set in your configuration" 缺乏配置主机
解决办法：创建SparkConf对象时设置主机 

```
new SparkConf().setMaster("local");
```

2、"Failed to connect to master"主机配置错误
解决方法：请参照一下spark配置资料

3、" java.lang.IllegalArgumentException: System memory 468189184 must be at least 4.718592E8 "JVM申请的memory不够导致无法启动SparkContext
解决办法：

(1). 自己的源代码处，可以在conf之后加上：

    SparkConf conf = new SparkConf().setAppName("JavaLDAExample").setMaster("local");
	conf.set("spark.testing.memory", "2147480000");

(2). 可以在Eclipse的Run Configuration处，有一栏是Arguments，下面有VMarguments，在下面添加下面一行(值也是只要大于512m即可)

-Dspark.testing.memory=1073741824

三、参考文档
-------

[转自自己的csdn博客](http://blog.csdn.net/csdn595075652/article/details/52718467)   
[spark配置整理资料](http://blog.csdn.net/baiyangfu_love/article/details/40537087)  
[spark配置指南](http://www.open-open.com/lib/view/open1418265814995.html)  
[spark配置官方资料（1.6.2版本）](http://spark.apache.org/docs/1.6.2/configuration.html)  
[spark-mllib数据类型](http://www.itnose.net/detail/6431042.html)  
[spark-mllib-lda源码解析](http://blog.csdn.net/sunbow0/article/details/47662603)  
[spark-mllib-ldamodel类文档](http://spark.apache.org/docs/1.6.2/api/java/org/apache/spark/mllib/clustering/LDA.html)  
[spark-mllib-DistributedLDAModel类文档](http://spark.apache.org/docs/1.6.2/api/java/org/apache/spark/mllib/clustering/DistributedLDAModel.html#topicsMatrix%28%29)  

