DATA\_discussion3: Recommendation System and Human Bias
================
Yun Mai
June 29, 2017

The recommender system is designed to reduce the amount of time for the user to search the relevant items and help business to maximize the purchase intention. One the one hand, it is believed that increasing the accuracy of recommendation system could help in retaining the customers' satisfaction and loyalty, on the other hand, the results could be biased and even discriminate as pointed out in the Evan Estol's talk. Google search results are affected by user's previous search history and user profile.

The recommender system or machine learning model is not as objective as we thought as machine learning algorithms learn from people's behaviors. It is reported that algorithms could reinforce human prejudices. For example, a deep learning model for college applications selects against minorities. I think the reason that could lead to the human bias of the prediction/ recommendation is from a technical aspect. First of all, the data could be bias. For example, if a learning algorithm extracts statistical patterns of the training data bearing certain social biases to such as race, gender, etc. the results will be biased. Second, results derived from a smaller population will not as good as that from a larger population and the statistical pattern fit the majority will not appropriate to the minority population. Other factors such as culture difference and other potential complexities will cause the difficulty of the discovery of specific statistical pattern for different groups. In the deep learning process, decision making is based on certain features determined by the neural net. It is hard to identify the human bias when it happens as people do not know how neural net make decisions.

According to the collaborative filtering and content-based recommendation techniques learned from DATA 643, I think it is better to account for user and item biases when building the model. If we keep track of the average user bias and item bias and predict the difference from that average, the recommender system will work better.

Reference:

1.  Moritz Hardt, How big data is unfair

2.  Claire Cain Miller, 2015, The New York Times, When Algorithms Discriminate.