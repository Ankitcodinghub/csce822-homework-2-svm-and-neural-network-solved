# csce822-homework-2-svm-and-neural-network-solved
**TO GET THIS SOLUTION VISIT:** [CSCE822 Homework 2-SVM and Neural Network Solved](https://www.ankitcodinghub.com/product/csce822-homework-2-svm-and-neural-network-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;93304&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;0&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;0&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;0\/5 - (0 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CSCE822 Homework 2-SVM and Neural Network Solved&quot;,&quot;width&quot;:&quot;0&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 0px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            <span class="kksr-muted">Rate this product</span>
    </div>
    </div>
<div class="page" title="Page 1">
<div class="layoutArea">
<div class="column">
Problem 1: Classifier evaluation measures

Given the following confusion matrix, manually calculate all the following performance measures of the classier

Accuracy, precision and recall (for each class), AUC score, true positive rate, false positive rate, specificity, sensitivity

Check your understanding by taking this test:

https://developers.google.com/machine-learning/crash-course/classification/check-your- understanding-accuracy-precision-recall

put down how many questions you answered correctly.

Describe the property of the classifiers that have the following ROC curves.

</div>
</div>
<table>
<tbody>
<tr>
<td>
<div class="layoutArea">
<div class="column">
(a)

</div>
</div>
</td>
<td>
<div class="layoutArea">
<div class="column">
(b)

</div>
</div>
</td>
</tr>
</tbody>
</table>
</div>
<div class="page" title="Page 2">
<table>
<tbody>
<tr>
<td>
<div class="layoutArea">
<div class="column">
(c)

</div>
</div>
</td>
<td>
<div class="layoutArea">
<div class="column">
(d)

</div>
</div>
</td>
</tr>
</tbody>
</table>
<div class="layoutArea">
<div class="column">
https://developers.google.com/machine-learning/crash-course/classification/check-your- understanding-roc-and-auc

Problem 2: Classification of e-tailer customers (Real-world problem) using Support vector machines. You can use weka or Scikit-learn python programming.

Objectives: E-commerce Customer Identification (Raw). Try to get the best performance using preprocessing, feature selection, data balancing, and parameter tuning.

The task involves binary classification to determine customers of the e-tailer. The training data contains 334 variables for a known set of 10000 customers and non- customers with a ratio of 1:10, respectively. The test data consists of a set of examples and is drawn from the same distribution as the training set.

Data: The feature data is train.csv and the label data is train_label.csv with corresponding labels for the records in train.csv. The test.csv is the test data.

Preprocessing steps to do:

<ul>
<li>You may use excel or write a simple script to merge the feature data file with label data file and save as csv file, then you can import into weka system.</li>
<li>Missing values: Check if there are any missing values inside the dataset, if so, use Weka’s missing value estimation filter to estimate the missing values to make the data complete</li>
<li>Normalization: since the features have very different value ranges, apply weka’s normalization procedure to make them comparable.</li>
<li>Attribute/Feature selection: Since there are 334 features in the dataset, it may be useful to use some feature/attribute selection to reduce the dataset before training classifiers. Select one method</li>
</ul>
</div>
</div>
</div>
<div class="page" title="Page 3">
<div class="layoutArea">
<div class="column">
(weka-&gt;filters-&gt;supervised-&gt;attribute-&gt;attributeSelection) to do feature selection.

Describe your selected method and explain how it works briefly.

<ul>
<li>Hint1: after you import the merged csv file into weka, the class label 1/0 is
regarded as numeric value rather than nominal labels. You need to use the weka-&gt;filter-&gt;unsupervised-&gt;attribute-&gt;numeric2Nominal filter to convert that column to nominal class. (you need to specify which column is your class label to apply this conversion) Also note that weka take first line as feature names!! So need to add a line of feature names.
</li>
<li>Hint2: The dataset is a severely unbalanced dataset. You may want to balance the data before training the classifier.</li>
<li>Hint3: if your training data has been applied a set of normalization or feature selection, you need to do the same with test dataset, otherwise the feature values are not consistent, and you will get absurd results on test data.</li>
<li>Hint5: The best AUC value for this problem is 0.6821. See what u can get. Experiments to do:</li>
</ul>
1) Experiments on the training dataset

You will need to build a classifier using a SVM algorithms to classify the data into customers and non-customers and evaluate their performance.

<ul>
<li>Pick one decision tree algorithm from Weka such as J48graft and describe it. (there are many decision tree algorithms)</li>
<li>Explain pre-processing filters in the table below. Run your decision tree algorithm with the default parameters. This is to learn how the preprocessing affects performance.</li>
<li>Write down the corresponding performance measures for class 1 (customer) in the following table for each processing</li>
<li>All measures are based on 10-fold cross-validation results (except the last row). Put your results in Table 1 (below)</li>
</ul>
2) Use your best classifier you trained in step one, predict the class labels for the test dataset test10000.csv. Save your prediction labels into the predict.csv file.

Write a program to calculate precision, recall, MCC (check the definition here

http://en.wikipedia.org/wiki/Receiver_operating_characteristic#Further_interpreta tions) using the true labels in the test10000_label.csv and the predicted labels in your predict.csv file.

</div>
</div>
</div>
<div class="page" title="Page 4">
<div class="layoutArea">
<div class="column">
Table 1. Comparing performances of classifiers on test dataset

</div>
</div>
<div class="layoutArea">
<div class="column">
Algorithm performance SVM (10-fold CV)

Result on test

data

Report requirement:

</div>
<div class="column">
Precision

</div>
<div class="column">
Recall

</div>
<div class="column">
MCC ROC area

</div>
</div>
<div class="layoutArea">
<div class="column">
1) Describe the preprocessing methods you used in the above experiments: missing value estimation, normalization, attribute selection, random forest

2) Report the performance results in Table 1

3) Submit the program to calculate the performance measures: Precision, Recall,

MCC from two label files. References on unbalanced data handling

<ol>
<li>https://medium.com/james-blogs/handling-imbalanced-data-in-classification-problems- 7de598c1059f</li>
<li>https://towardsdatascience.com/handling-imbalanced-datasets-in-machine-learning- 7a0e84220f28</li>
<li>https://medium.com/strands-tech-corner/unbalanced-datasets-what-to-do- 144e0552d9cd</li>
<li>https://www.analyticsvidhya.com/blog/2017/03/imbalanced-classification-problem/</li>
</ol>
Problem 3: Regression using deep neural networks.

The problem here is to develop a regression model that can beat a theory model. Attached thermal-data.xlsx contain a dataset for material thermal conductivity.

</div>
</div>
</div>
<div class="page" title="Page 5">
<div class="layoutArea">
<div class="column">
Develop a deep neural network regression program to predict the thermal conductivity (y- exp) using the all the features before it. (V,M,n,np,B,G,E,v,H,B’,G’,ρ,vL,vS,va,Θe,γel,γes,γe,A,).

Report the MSE, RMSE, MAE, R2 of 10-fold cross-validation.

Compare the MSE, RMSE, MAE, R2 of the theoretical model using the values in column y-theory

Try to tune your parameters of the models to achieve the best performance.

Plot the final scatter plot for your best model/result. The better the points are around the diagonal line the better your model is.

Cross-validation example code can be found here

https://www.analyticsvidhya.com/blog/2018/05/improve-model-performance-cross-validation- in-python-r/

</div>
</div>
</div>
