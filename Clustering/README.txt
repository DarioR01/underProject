Follows clustering

Both vader and textBlob sentiment analysis gives poor results

Attempted to automatically summarise the text using ADD with the idea that with lesser world, the major sentiment would underline the hidden bias, 
however sentiment was not preserved and actually scored tended to converge towards 0.

textBlob performs even worse with his subjectivity score than pure sentiment analysis with a stunning recall of approx. 75% but a terrible
35% precision and an f1 score of about 50%.

The above is because textBlob is unable to label the underline bias of news since it is based on a bayesian classifier, and therefore not good in semantic and context.

textBlob performs better when bias is not contextualised