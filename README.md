# F.U.S.E.Box

**F**inetune-able **U**niversal **S**entence **E**ncoder tool**Box**. Work in Progress. 

> My employer, [Rasa](https://rasa.com/), gives me education days every year to try something new. This year, I've decided to build a sentence encoder. The goal of this project is to create a variant of the [Universal Sentence Encoder](https://arxiv.org/pdf/1803.11175.pdf) that is lightweight to run, fast to train and easy to finetune. The main trick will involve multi-task learning on different datasets. It's likely many lessons will be learned from this work, which I intend to share on Github Pages in the form of a Diary.

### Status:

There's a couple of reasons why this didn't work. The main lessons I learned is that you can also just train a bunch of models on a bunch of datasets, take the vector of all predictions and this'd be a more useful representation for sentiment than anything else. 
