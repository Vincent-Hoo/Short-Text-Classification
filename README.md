# Short-Text-Classification

### Project Overview

This research work aimed to address the problem of short text classification. Unlike conventional documents which are descriptive, short texts face problem of sparse feature and keyword orientation. The key to this problem is how to add extra information so that the semantics can be enriched. We enrich the short text in two levels: word level and text level, both of which are implemented by network embedding algorithm. After semantics is enriched, we employ convolution neural network to address the classification task.

### Acknowledgements

The motivation of conducting this research is simple. One of my senior groupmate, Lu Ma, has graduated from his master program and I have read his thesis. I think this idea is great and under his encouragement, I undertook this project, conduct experiments again and write it into English-version paper.

### My thought

This is my first research work which is completely finished all by myself. I have been reading scientific paper all the time, but this is my first time to write one myself. It's extremely difficult at the very first beginning and I have to imitate the writing style of other papers, like the architecture, the expression of terminology. But after the paper is done, I feel relieved and satisfied even though the paper is not perfect.

As for the experiment phase, the process is miserable. Writing the code is easy, because there're lots of existing packages that I can find in GitHub. The most miserable part is tuning parameters. In this project, there're dozens of hyper-parameters. Searching in grid is impossible. Moreover, since I invoke some packages from other projects, my experiment is not continuous. I have to first generate some files from the packages, then run some codes using the generated files. What's more, I have to name the file very carefully, adding anything in the name that can help me remember the hyper-parameters, such as node2vec-dim100-len15-num20. When doing comparison experiments, I found our model is not quite satisfactory, which reveals the fact that some ideas that you think are good might be not good at all.  Tracing the reason behind is very tricky, it might be the datasets, might be the hyper-parameters. Finally I lost my patience and submit a relatively reasonable results in my paper.

Conducting scientific research is not as easy as I thought before. Now I admire our pioneers more than ever, because I have undergone the process they have been through dozens of years before.  No matter what, at least I experienced it myself and tried to learn how to conduct research and write papers