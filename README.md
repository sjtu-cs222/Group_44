**Introduction of this project**

In this League of Legends World Tournament, KFC and the Tournament used deep learning to predict the strength of the hero lineup and the winning percentages of different periods. We envision whether such a winning rate prediction can be applied to the daily games of our average players. If we use match data from different regions in different regions, can we establish a real-time winning rate prediction for ordinary games? 

We used the crawler to collect the game data of the different leagues of the League of Legends statistics website, and then use the python implementation of xgboost to build our training test model, cross-validate and test our data. The Xgboost model achieves an accuracy of about 65\% on the test data. In view of the increased investigation accuracy as the number of samples increases, the model may perform better in the future as more game data increases.

**How to use**

You need to have:

xgboost

sklearn

pandas

matplotlib

numpy

sqlite3

Run analysis.py

**Notes**

analysis.py——Code of data reshapes split, and training testing of module. Implement of winning rate prediction.
parsing_utils.py——Establish and read of database which saved as database.
scrape_utils.py——Source of match data, and implement of parser, parsing match data from several network stations.

databases——Our data.



