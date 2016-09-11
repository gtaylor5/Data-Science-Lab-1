
# coding: utf-8

# # Module 2 Lab - Probability
# 
# ## Directions
# 1. Show all work/steps/calculations. If it is easier to write it out by hand, do so and submit a scanned PDF in addition to this notebook. Otherwise, generate a Markdown cell for each answer.
# 2. You must submit to **two** places by the deadline:
#     1. In the Lab section of the Course Module where you downloaded this file from, and
#     2. In your Lab Discussion Group, in the forum for the appropriate Module.
# 3. You may use any core Python libraries or Numpy/Scipy. **Additionally, code from the Module notebooks and lectures is fair to use and modify.** You may also consult Stackoverflow (SO). If you use something from SO, please place a comment with the URL to document the code.

# In[4]:

import numpy as np
import random as py_random
import numpy.random as np_random
import time


# ## Manipulating and Interpreting Probability
# 
# Given the following *joint probability distribution*, $P(A|B)$, for $A$ and $B$,
# 
# 
# |    | a1   | a2   |
# |----|------|------|
# | **b1** | 0.37 | 0.16 |
# | **b2** | 0.23 | ?    |
# 
# Answer the following questions.
# 
# **1\. What is $P(A=a2, B=b2)$?**

# **2\. If I observe events from this probability distribution, what is the probability of seeing (a1, b1) then (a2, b2)?**

# **3\. Calculate the marginal probability distribution, $P(A)$.**

# **4\. Calculate the marginal probability distribution, $P(B)$.**

# **5\. Calculate the conditional probability distribution, $P(A|B)$.**

# **6\. Calculate the conditional probability distribution, $P(B|A)$.**

# **7\. Does $P(A|B) = P(B|A)$? What do we call the belief that these are always equal?**

# **8\. Does $P(A) = P(A|B)$? What does that mean about the independence of $A$ and $B$?**

# **9\. Using $P(A)$, $P(B|A)$, $P(B)$ from above, calculate,**
# 
# $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$
# 
# Does it match your previous calculation for $P(A|B)$?
# 
# If we let A = H (some condition, characteristic, hypothesis) and B = D (some data, evidence, a test result), then how do we interpret each of the following: $P(H)$, $P(D)$, $P(H|D)$, $P(D|H)$, $P(H, D)$?

# ## Generating Samples from Probability Distributions
# 
# ### Reproducible Random Numbers
# Before you begin working with random numbers in any situation, in Data Science, as opposed to Machine Learning, it is desirable to set the random seed and record it. We do this for several reasons:
# 
# 1. For reproducible research, we need to record the random seed that was used to generate our results.
# 2. For sharing with others, if our text said there was some result, and the user re-runs the notebook, we want to get the same results.
# 3. If we are creating a model, and we accidentally generate the best model ever, we want to be able to build it again.
# 
# Unfortunately, Python's library and Numpy's library do not share seeds so if you need to set the appropriate one. Additionally, they take slightly different arguments (Python can take any Hashable object; Numpy only takes ints). Fortunately, the name of the function is the same.
# 
# ```
# numpy.random.seed([N]) # Numpy library
# random.seed([N]) # Python core
# ```
# 
# You have several options.
# 
# 1. Call the appropriate `seed` function with a value of your choice, probably some Integer, like 27192759.
# 2. Run:
# 
# ```
#     int( time.time())
# ```
# 
# to print out a value you can use in either case. Do not just feed `int( time.time())` into the seed function. The whole point is to make the seed a constant.
# 
# You should set the seed before answering each question. It doesn't necessarily have to be a different value.
# 
# ### Questions
# 
# **1\. A trick coin has a probability of heads, $\theta=0.67$. Simulate 25 coin tosses from this Binomial distribution (25 Bernoulli trials).**

# In[4]:

from random import random
p = 0.67
[1 if random() < p else 0 for i in xrange(25)]


# **2\. Using $P(A, B)$ above, write a function `my_sample` that takes the distribution and the number of desired samples and returns a list of events from the distribution in the form `[("a1", "b2"), ("a1", "b1"), ...]`.**

# In[72]:

from random import random
def my_sample(samples):
    distrib = {.37 : ("a1", "b1"),.23 : ("a1", "b2"),.16 : ("a2", "b1"), .24 : ("a2", "b2")}
    events = []
    ranNums = []
    for i in xrange(samples):
        ranNum = random()
        ranNums.append(ranNum)
        if ranNum <=.16:
            events.append(distrib[.16])
        elif ranNum > .16 and ranNum <= (.23+.16):
            events.append(distrib[.23])
        elif ranNum > (.23+.16) and ranNum <= (.23+.16+.24):
            events.append(distrib[.24])
        elif ranNum > (.23+.16+.24) and ranNum <= (.23+.16+.24+.37):
            events.append(distrib[.37])
    print(events)

my_sample(25)



# The **mean** is a measure of central tendency. Exactly what that means will await another module but for now, you simply need to use `np.mean()` when asked to calculate the mean of a sample.
# 
# The **coefficient of variation**, $v$, is a dimensionless measure of the variability of a distribution. It allows you to compare how disperse two or more distributions might be, even if their means and **standard deviation**s are in completely different units. The definition is:
# 
# $v = |\frac{\sigma}{\mu}|$
# 
# With a little algebra, you can also calculate a desired *standard deviation* given a coefficient of variation.
# 
# Using the appropriate Numpy libraries and functions (or lecture materials),

# **3\. Generate 25 samples for $X_1$ from a normal distribution with $\mu=32.5$ and $v=0.01$. Calculate the mean value of $X_1$. How far off is it (percent)?**

# In[79]:

import numpy
from numpy import random
m = 32.5
v = .01
sigma = (m*v)
x1 = [random.normal(m, sigma) for i in xrange(25) ]
mean = numpy.mean(x1)
print(mean)
print("{}{}".format(((mean - 32.5)/32.5)*100, " % difference"))


# **4\. Generate 25 samples for $X_2$ from a normal distribution with $\mu=32.5$ and $v=0.05$. Calculate the mean value of $X_2$. How far off is it (percent)?**

# In[64]:

import numpy
from numpy import random
m = 32.5
v = .05
sigma = (m*v)
x2 = [random.normal(m, sigma) for i in xrange(25) ]
mean = numpy.mean(x2)
print(mean)
print("{}{}".format(((mean - 32.5)/32.5)*100, " % difference"))


# **5\. Generate 25 samples for $X_3$ from a normal distribution with $\mu=32.5$ and $v=0.10$. Calculate the mean value of $X_3$. How far off is it (percent)?**

# In[78]:

import numpy
from numpy import random
m = 32.5
v = .1
sigma = (m*v)
x3 = [random.normal(m, sigma) for i in xrange(25) ]
mean = numpy.mean(x3)
print(mean)
print("{}{}".format(((mean - 32.5)/32.5)*100, " % difference"))


# **6\. From a Systems Thinking perspective, how might we interpret the different variabilities of the variables $X_1$, $X_2$, and $X_3$? Could our estimates of $\bar{x}$ (the technical symbol for the sample mean), improve if we got more samples? Do we need more samples for all the $X$'s?**

# In the context of probability, the differences in variabilities means there is a tighter (or looser) correlation between certain outcomes and a particular event. Adding more samples would not improve the estimate. The variation is essentially fixed. Adding more samples would certainly give an answer that is less ambiguous or closer to the variation value. We would only need more samples for X2 and X3, namely the variables that have higher variability coeffients.

# **7\. We now want to sample a system that has two characteristics (variables) that are related. One is boolean ($X_1$)and the other is a normally distributed, numeric variable ($X_2$). The probability of $X_1$ being "true" (or 1) is $\theta=0.25$.**
# 
# If $X_1 = 0$, then $X_2$ is normally distributed with $\mu=32.5$ and $v=0.05$.
# If $X_1 = 1$, then $X_2$ is normally distributed with $\mu=39.1$ and $v=0.10$.
# 
# Generate 100 samples and calculate the mean value of $X_1$ and $X_2$.

# In[87]:

import numpy
from numpy import random
from random import random
p = 0.25
X1 = []
X2 = []
for i in xrange(100):
    rand = random()
    if rand >= p:
        X1.append(0)
        m = 32.5
        v = .05
        sigma = (m*v)
        X2.append(numpy.random.normal(m,sigma))
    elif rand < p:
        X1.append(1)
        m = 39.1
        v = .1
        sigma = (m*v)
        X2.append(numpy.random.normal(m,sigma))

meanX1 = numpy.mean(X1)
meanX2 = numpy.mean(X2)
print(meanX1)
print(meanX2)     

