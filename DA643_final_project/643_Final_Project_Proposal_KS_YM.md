DATA643\_Final\_Project\_Proposal
================
Yun Mai, Kelly Shaffer
July 9, 2017

Restaurant Recommendation System based on Yelp data
===================================================

### Introduction

As modern consumers, we greatly benefit from restaurant recommendation applications. It is so convenient to get a list of restaurants that match our preferences without much clicking, comparing, and browsing through a long list of reviews for each single business.

In this project, we want to apply the algorithms to develop predictive models learned from the DATA643 course "of "Current Topic of Data Science - Recommendation System"" to build a restaurant recommendation system that suggests the most suitable restaurant for users. If time permits, we will build an application.

### Motivation

It is very common that we hang out with families, friends, and coworkers when comes to lunch or dinner time. As the users of recommendation applications, we care more about how we will like a restaurant. We will tend to have happier experiences when the prediction of the recommendation system is as good as what it says. As there is a completed and big data set of user and restaurants reviews, we want to see whether we can use the latest techniques to make good predictions. In the data set, there are not only reviews but also relevant information of users and restaurants that allow us to do more complicated computation, which might lead to the construction of a better model.

### Dataset and Algorithms

In this project, we will use a Yelp Dataset Challenge round 9 from yelp website. The dataset has 4.1M reviews and 947K tips by 1M users for 144K businesses; 1.1M business attributes, e.g. hours, parking availability, ambience; and aggregated check-ins over time for each of the 125K businesses. The data includes diverse sets of cities: Edinburgh in U.K.; Karlsruhe in Germany; Montreal and Waterloo in Canada; Pittsburgh, Charlotte, Urbana-Champaign, Phoenix, Las Vagas, Madison, and Cleveland in U.S.

In the Yelp dataset there is more information other than only ratings, so we can not only use content-based algorithm but also collaborative filtering algorithms. Location of the restaurant is an important factor to do the recommendation, so the location will be considered so the similarity between the distance and similarity between user/items will be combined. Other algorithms like alternative linear squares and singular value decomposition will also be used to build the prediction models.

Because there are three criteria in reviews: funny, useful, and cool, the rating will be calculated as follows:

*R* : *U**s**e**r**s* × *I**t**e**m**s* → *R*<sub>0</sub> × *R*<sub>1</sub> × ...*R*<sub>*k*</sub>

*R*<sub>0</sub> is the set of possible overall rating values, and *R*<sub>*i*</sub> represents the possible rating values for each individual criterion i (i = 1,..,k), typically on some numeric scale.

The prediction results of single-criteria collaborative filtering algorithm and multi-criteria collaborative filtering algorithms will be compared to decide which approach is better.

The implementation and evaluation will be performed in R and Apache Spark. At last, if time permits, an application will be built with the Shiny package.

### Reference:

1.  Blanca Vargas-Govea, Gabriel González-Serna, Rafael Ponce-Medellín. Effects of relevant contextual features in the performance of a restaurant recommender system.CARS,( 2011)
2.  Mengqi Yu, Meng Xue, Wenjia Ouyang. Restaurants Review Star Prediction for Yelp Dataset.Conference Proceedings (2015).
3.  Gediminas Adomavicius, YoungOk Kwon. New Recommendation Techniques for Multi-Criteria Rating Systems. IEEE Intelligent Systems 22-3 (2017).
4.  Jun Zeng, Feng Li, Haiyang Liu, Junhao Wen, Sachio Hirokawa. A Restaurant Recommender System Based on User Preference and Location in Mobile Environment. Advanced Applied Informatics (IIAI-AAI), 2016 5th IIAI International Congress.
