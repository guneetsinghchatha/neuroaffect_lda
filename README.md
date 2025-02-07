## This paper is based on my final research during the class of Topic Modelling at UofT 



## Latent Emotional Dynamics in Psychedelic Microdosing Communities: A Neurosemantic Analysis of Affective Patterns Through LDA-Based Topic Modeling and Computational Psychometrics




Guneet Singh Chatha

Abstract
This study evaluates the microdosing report topics discussed on the subreddit r/microdosing since it’s inception on 7th January 2014 to 29th October 2021 to analyse why people are using microdosing and what insights can be drawn from the users. Multiple models were tested such
as LDA, LSA and NMF to identify topics discussed on the subreddit where NMF gave us the best results
Keywords: topic models, microdosing,


1.Introduction
This study analyses seven years of Posts and comments on the subreddit r/microdosing using LDA, LSA, NMF and Biterm to understand people’s opinion on the topic and more insights into what users feel on the microdosing experience.
The results from the analysis show that
1. Depression and Anxiety are the most self-treated diseases that people try microdosing
2. 3. for
Some common topics observed were dosage protocol, stamets stack and tolerance
The common themes on the subreddit were around microdosing psychedelics for feeling
something
4. NMF was the best approach for tackling short-text data
2.Background
2.1 Microdosing
Microdosing psychedelics is the practice of consuming very low, sub-hallucinogenic doses of a
psychedelic substance, such as lysergic acid diethylamide (LSD) or psilocybin-containing
mushrooms. According to media reports, microdosing has grown in popularity, yet the scientific
literature contains minimal research on this practice. There has been limited reporting on
adverse events associated with microdosing, and the experiences of microdosers in community
samples have not been categorized and till date only one research has looked into insights from
reddit to derive outcomes from self-reported users. Multiple survey based researches in a
controlled lab environment have been conducted indicating benefits for mental health and
addiction. Typical doses can be small as one twentieth of a typical recreational dose, sometimes
even less/ Psychedelics have typically been associated with marked alterations in cognition,
affect, perception and neurophysiology. Individuals who take psychedelics describe profound
changes in visual and auditory perception, accompanied by vivid imaginative experience and
intense emotions. However this is not the case with microdosing as it involves a subthreshold
dose. That is individuals aim to identify a dose at which they do not feel high and only gain the
positive identifiable acute drug effects.
Despite the reported lack of acute effects of microdosing, proponents claim a wide variety of
psychological creativity, productivity, social ability, focus and well being. Microdosing is thus a
curious phenomenon on the one had advocates deny experiencing the alterations in
consciousness that characterise typical recreational doses, yet people claim significant
psychological benefits from them. Many models using GTM and linear mixed-effects model
have already been conducted in this domain but no study has looked at analysing the entire
corpus of experiences as shared on the subreddit r/microdosing.
2.2 Subreddit r/microdosing
Reddit is an American social news aggregation, web content rating, and discussion website.
Registered members submit content to the site such as links, text posts, images, and videos,
which are then voted up or down by other members. Posts are organized by subject into user-
created boards called "communities" or "subreddits", which cover a variety of topics such as
news, politics, religion, science, movies, video games, music, books, sports, fitness, cooking,
pets, and image-sharing. Submissions with more upvotes appear towards the top of their
subreddit and, if they receive enough upvotes, ultimately on the site's front page. Although
there are strict rules prohibiting harassment, it still occurs, and Reddit administrators moderate
the communities and close or restrict them on occasion. Moderation is also conducted by
community-specific moderators, who are not considered Reddit employees.
The subreddit r/microdosing has boomed in popularity in recent times with posts increasing by
more than 200% post the covid lockdown as observed in early 2020. On the subreddit
r/microdosing people share their experiences with microdosing and ask questions related to
the domain. Our exploratory data analysis covers more features about the subreddit however
the general theme by analysing the subreddit flairs show topics such as –
1. Reports of people on multiple substances
2. Questions and answers about the dosage and dosing protocol
3. Research Questions
Research Question 1: Can topic modelling show us the range of emotions and feelings that
people are dealing with on microdosing
Research Question 2: What is the best model for analysing short text shared on reddit posts
4. Methods
4.1 Data Collection
Data collection was performed by creating a script that captures all posts and comments on the
subreddit. Reddit’s official API could not be used as it lacks a method for downloading archived
posts and posts from a specific timeframe, only posts that are categorised using reddit’s new
filters such as new, best and hot can be downloaded from the official API. The script created
made use of the unofficial Pushshift API to download all posts and comments posted in the
subreddit r/microdosing.
4.2 Data Cleaning
All posts that did not contain English words were removed and excluded from this analysis.
Empty and archived posts without any content were also removed since they shared images
and memes associated with the topic.
After all posts were stored into a csv file all un-necessary columns were removed and the
following column headers were saved into the final post dataframe -
1. Post Title
2. Post self text
3. Post ID
For the comment dataset only the comment ID and comment payload text was saved to the
final comment dataset.
4.3 Data Pre-processing
The dataset was then sorted by time of post /comment creation and concatenated into the
comment trees. Comment trees were concatenated to the relevant posts to generate the final
dataset. The final dataset was then moved through multiple iterations to remove stop words
and lemmatized into the final text set through using parts of speech tagging.
5. Analysis and Results
5.1 Descriptive Statistics
The final dataset included a total of 30,960 posts and 340,835 comments. However as all
comment trees were merged into the original posts by concatenating with a whitespace the
final dataset included a total of 30,960 posts. Multiple descriptive statistics tests with their
visualisations have been shown in section 5.2
5.2 Exploratory Data Visualisations
Making sense of the dataset
Understanding our data through bigrams and trigrams gives us some level of clarity to the top
most discussed words in the subreddit. Unigrams by itself don’t provide us with enough context
to understand the dataset but looking at bigrams and trigrams we can see that people are
talking about their feelings and questions related to dosage and protocol
Figure 1.1 – The top 30 Unigrams in the dataset
Figure 1.2 – The top 30 Bigrams in the dataset
Figure 1.3 – The top 30 Trigrams in the dataset
The sudden increase in posts can be observed after covid lockdowns in early 2020 resulting in a
dip then a stark increase towards the end of 2020 contributing to the increasing questions
regarded to first time users.
Figure 1.4 – Time-series of posts
5. Models
5.1 LDA
latent Dirichlet allocation (LDA) is a generative statistical model that allows sets of observations
to be explained by unobserved groups that explain why some parts of the data are similar. For
example, if observations are words collected into documents, it posits that each document is a
mixture of a small number of topics and that each word's presence is attributable to one of the
document's topics. LDA is an example of a topic model and belongs to the machine learning
field and in a wider sense to the artificial intelligence field. LDA imagines a fixed set of topics.
Each topic represents a set of words. And the goal of LDA is to map all the documents to the
topics in a way, such that the words in each document are mostly captured by those imaginary
topics. We want to maximise topic coherence and minimize the Jaccard similarity so the
distance between the lines is the shortest.
Figure 5.1.1 – Coherence and Jaccard for LDA
As the dataset is sparse and contains short-text LDA is not the best fit for our model. However
after multiple iterations the ideal number of topics was found out to be 5.
The performance metrics as observed for our LDA model when number of topics was 5 were as
follows -
Coherence Jaccard Similarity 0.2167
0.5134
The top 10 words with their probabilities in each topic are –
For topic 1
like 0.015
Dose 0.011
Feel 0.010
Make 0.008
Work 0.008
Good 0.008
Time 0.007
Know 0.007
Experience 0.006
Effect 0.006
For Topic 2
Dose Feel Like Time Make Help Really Good Well Effect 0.015
0.012
0.010
0.008
0.007
0.007
0.006
0.006
0.006
0.006
For Topic 3
Dose Like Time Feel Experience Trip Effect Good 0.011
0.011
0.010
0.010
0.007
0.006
0.006
0.006
Find 0.006
Work 0.006
For Topic 4
Dose 0.011
Feel 0.011
Like 0.010
Good 0.010
Make 0.007
Time 0.006
Work 0.006
Experience 0.006
Really 0.006
Help 0.006
For Topic 5
Dose 0.011
Like 0.009
Time 0.009
Help 0.008
Feel 0.008
Want 0.008
Know 0.007
Good 0.006
Also 0.006
Make 0.006
5.2 NMF
Non-negative matrix factorization is a group of algorithms in multivariate analysis and linear
algebra where a matrix V is factorized into (usually) two matrices W and H, with the property
that all three matrices have no negative elements. This non-negativity makes the resulting
matrices easier to inspect. NMF is useful when there are many attributes and the attributes are
ambiguous or have weak predictability. By combining attributes, NMF can produce meaningful
patterns, topics, or themes.
Just like we calculated the shortest distance points between coherence and jaccard scores for
LDA we tried multiple iterations to assess that the ideal number of topics was found to be 8 for
NMF.
Figure 5.2.1 – Coherence and Jaccard for NMF
Just like we calculated the shortest distance points between coherence and Jaccard scores for
LDA we tried multiple iterations to assess that the ideal number of topics was found to be 7 for
NMF.
The performance metrics as observed were –
Coherence 0.1897
Jaccard Similarity 0.3212
From our 7 NMF topics the words with highest value are listed below -
6. Evaluation of Results
Metric Jaccard Coherence
LDA (5 topic) 0.2167 0.5134
NMF (7 topic) 0.1897 0.3212
- By referring to metrics alone we see that the 7 topic solution for NMF provides the best
- By referring to metrics alone we see that the 7 topic solution for NMF provides the best
evaluation
- After interpreting clusters from both models it is clear that NMF has the greatest
potential in creating topic clusters for our datasetpotential in creating topic clusters for our dataset
1. 2. 3. 4. 5. 6. Topic 1 : like, help, feel, think, thing, anxiety, make, life, good, really
The first topics involves all the feelings associated with microdosing. Words like anxiety,
feeling and thinking have to do with human cognition and consciousness.
[… Focus, Creativity, no anxiety, and what depression ….]
[…myself do create a good day for myself and life feels…]
[… I really feel good, can think clearly and want to let…]
Topic 2 : water, tab, vodka, bottle, distilled, use, dissolve, solution, distil
These topics contain insights into volumetric dosing of LSD using vodka or distilled water
as a medium for diluting LSD doses. (chlorine degrades LSD structure)
[… Volumetric dosing style Finland, our most known vodka.….]
[…Vodka or distilled water to make volumetric doses…]
[… tab needs to dissolve into a solution with distilled …]
Topic 3 : microdosing, question, advice, truffle, dosing
This topic covers most of the questions and advice for people experimenting with
microdosing. The most common flair associated with these topics was “newbie advice”
[… Advice on using magic truffles for microdosing.….]
[…methadone as a short term detox and dosing advice…]
[… Microdosing truffles (and microdosing shrooms) …]
Topic 4 : day, week, md, tolerance, schedule, protocol
The topics here are associated with the dosage schedule. Due to tolerance microdosing
cannot be done everyday and people here are discussing the protocol for dosing
[…daily .2, makes sense to me you'd build tolerance.….]
[…What protocol would be best for a TBI patient that is…]
[… everyone does it with a consistent week schedule…]
Topic 5 : capsule, mushroom, dry, scale, cap, powder, grind
This topic cluster looks into the proper method of consuming psilocybin using grinded
mushroom powders into capsules
[…and turned everything into this powder before.….]
[…ground them up, and capsulated them, and how long did these capsule…]
[… while making my capsules (with a very fine grind/powder…]
Topic 6 : LSD, 1p, dmt, 1cp, drug, try
These topics include all isomers of other drugs such as 1p and 1cp LSD and other
psychedelic substances such as DMT
[…Microdose 4-AcO-DMT vs 1P-LSD.….]
[…the optimal 1cp LSD dosage using Fadiman protocol…]
[… Then moved to 1cp and started trying w 10mcg little …]
7. Topic 7: lions, mane, niacin, stack,stamets, flush
Paul stamets is a famous mycologist who suggested the stamets stack that also includes
inhibitors such as niacin and lions mane for increasing cognition and neurogenesis.
[…times with Niacin I experienced the flush physically.….]
[…the stamets stack with the lions mane and niacin and I just wanted…]
[… take lions mane every day. Niacin sometimes, not always.…]
7. Discussion
Research Question 1: Can topic modelling show us the range of emotions and feelings that
people are dealing with
Our topic model clearly highlights anxiety and depression are the most common diseases that
people are using microdosing for and our bigrams and trigrams reflect that outcome. However
this cannot be taken into full consideration as there is no way to isolate microdosing reports
from the subreddit. The subreddit has recently added flairs so it’s easy to tag a post as a report.
Since most of the people on this subreddit use it as a platform to ask questions and feedback
our dataset contains a lot of noise.
Research Question 2: What is the best model for analysing short text shared on reddit posts
For short-text data that is high in count NMF provides us with the best approach as our LDA
model topics were not easily comprehended.
8. Limitations
1. 2. 3. The ideal approach for reddit data would involve running hierarchical topic models. As
we are merging comments and subcomments into one master post file the topics might
get divided into subtopics and a clear hierarchy is needed as sometimes discussions
sway away from the real topic.
Biterm library was also attempted however extremely long run times were observed
from it.
A better way to tackle the research question would have been understanding the topics
over time and study them as they changed pre and post covid.
9. Conclusion
Topic models provide us with a great overview for understanding both structured and
unstructured text. For our analysis NMF yielded better results than LDA due to the text density.
We cannot completely understand all the emotions associated with microdosing until we can
isolate the noise and focus only on microdosing reports.
