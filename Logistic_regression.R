d1=data(mtcars)
library(caTools)
split<- sample.split(mtcars,SplitRatio=0.7)
training<- subset(mtcars,split==1)
testing<-  subset(mtcars,split==0)
model <- glm(vs ~ wt+disp,training,family="binomial")
data <-data.frame(wt=3.170,disp=351)
answer<- predict(model,data,type="response")
answer

