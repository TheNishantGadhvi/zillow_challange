# Zillow Kaggle Competition
### Understanding the Problem
Arguably the most important step, this is probably the biggest differentiator between a mediocre data scientist and a good data scientist. Spend some time reading about this problem and the data [here](https://www.kaggle.com/c/zillow-prize-1) to get started. 

I'll summarize the key points for you:

Your target variable is the log error between Zillow's estimated value for the property (Zestimate) and the actual sales price. This is significant because your model will be picking up only whatever signal Zillow's internal model for predicting home value isn't picking up on.

logerror=log(Zestimate)−log(SalePrice)
 
There are two types of files you need to worry about: 
properties which has attributes (attributes) of the properties you're trying to predict on and
train which has the sale date ( transactiondate) and  logerror for the sale. 
Your model will be of the form:

logerror≈f(transactiondate, attributes)+ϵ
 
The evaluation metric will be Mean Absolute Error, or the average absolute value of the difference between the predicted logerror and the true logerror.
 
Your goal is to predict the  logerror for 6 transaction dates: 
October, November, and December of both 2016 and 2017. 

Note that the train data comes from 2016, but we're asked to predict on 2017 data.

### Getting Organized
Organizing your code and data is a crucial first step to any data science project. There are several reasons for this:

- Reproducibility:
You will undoubtedly need to revisit and revise portions of your code. If your project is useful enough, somebody else will probably need to run your code too.

- Production Readiness:
The code you'd write to get a quick answer to your business/data science question would look much different than production-ready code. Even if your code won't go to production (say, if you plan to simply upload the final model to your organizations deployment platform), you'll need to ensure your pipeline is constructed in such a way to be able to handle (i.e. score) new data as it comes in. For example, a common mistake some data scientists make is writing their preprocessing step to be dependent on receiving a batch of data (like a tune or test set) and failing to account for the fact that in production the preprocessing will be done one record at a time.

- Avoiding Mistakes:
Writing clean code and staying organized will help you avoid time-consuming and costly mistakes.

### Bonus: Setting up your Compute Environment

If you want somebody else to be able to run your code, you may want to consider the compute environment or computer it will run on. I'm using [Conda](https://www.anaconda.com/distribution/) environment which is an ideal setup for data science jobs. Docker is helpful if you want to make sure your code will run on somebody else's machine regardless of the operating system they're running, which version of Python they have installed, and which package versions they're using. Oversimplifying a bit, Anaconda makes it easy to ensure that whoever runs your code will do so with the same versions of the packages you used to avoid time-consumming errors due to differences in the underlying Python code base.
