# LT2212 V20 Assignment 2

The a2.py uses Truncated SVD reduction and bonus part uses PCA. For building a np array I used CountVectorizer with max_df = 0.8 (I asked specificaly Asad if I can use it for better performance and he said yes. If case you don't want me to use it, there is a manual count with helper functions commented in the sumbitted code - I kept it only in part bonus as both a2 and bonus are almost the same).

It takes really long time for me to run the classifiers on the whole data. 

Please find below the stats for reduced data = 4000 files. I used NB and KNeigbours classifier. 
Since both of those rely on actual values, not abstract features, both of them show better accuracy when run without reduction.
Reducing dimentionality lowers accuracy more for NB, then for Kniegh. PCA works better for KNEigh.
However, I just for curiousity tried to run it with SVC insted of Kneigh. 
Then I noticed that in case of SVC, reduction actualy increases accuracy, but when run without reduction, the accuracy score is pretty low (around 19%).

I didn't notice big differende betweem SVD and PCA reduction. 

## With TruncatedSVD

Clf - Gaussian NB

without reduction: accuracy 74% (precision, recall and F1 score from 51 to 91 % per label)

with reduction accuracy: 

50% - 41%
25% - 60%
10% - 62%
5%  - 64% 

Classifier 2 = KNeighbours 

without reduction: accuracy 70% (precision, recall, f1-score 44-91% per label)

with reduction accuracy: 

50% -  64%     
25% -  60%     
10% - 51%
5% - 56%

## With PCA part bonus
Only accuracy:

Clf - NB

50% -  33 %
25% - 57%
10% - 61 %
5%  - 62%

Classifier 2 = KNeighbours 

50% - 69%
25% - 52%
10% - 55%
5%  - 60%



