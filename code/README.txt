代码目录

preprocessing
	data_preprocessing.py 
word2vec.py 
生成词图的edgelist.py
生成文本二部图的edgelist.py
生成文本向量.py
LDA编码.py
auxiliary package
	node2vec_python 
	node2vec_C++
	(两个都是斯坦福大学的代码包，python版本运行速度慢，但是可以自行修改代码，
	加入早停词；而C++版本速度快)
实验代码
	LOG.py
	
	(验证文本节点向量的有效性)
	LDA编码_搭其它模型.py
	文本向量_搭其他模型.py
	
	(验证词节点向量的有效性)
	词向量_搭其它模型.py
	节点向量_搭其它模型.py
	词向量_节点向量_融合_取平均_搭其它模型.py
	
	词向量_CNN_搭其它模型.py
	节点向量_CNN_搭其它模型.py
	词向量_节点向量_融合_CNN_搭其它模型.py
	
	整体模型比较
		ft.py
		textcnn.py
		rf_lr_xgb_词向量_节点词向量_节点文本向量.py
		our_method.py
		

实验整体顺序
1、预处理
2、分别生成词和文本两幅图的edgelist
3、调用辅助包，生成向量
4、对比实验
	4.1、对比词节点向量的有效性，通过CNN或者词向量取平均，两种方式生成样本向量，然后用三种分类器对比
	4.2、对比文本节点向量的有效性，对比LDA编码
	4.3、对比整个模型的有效性，对比FastText，TextCNN
	

注意：CNN的模型均需要GPU来运行