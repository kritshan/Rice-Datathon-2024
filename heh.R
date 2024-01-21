library(ggplot2)

setwd('C:/Users/solki/OneDrive/Desktop/Datathon')
data = read.csv('comparison.csv')

rf_mean = mean(data$Random.Forest)
rf_sd = sd(data$Random.Forest)
gb_mean = mean(data$Gradient.Boosting)
gb_sd = sd(data$Gradient.Boosting)

t.test(data$Gradient.Boosting, data$Random.Forest,
       alternative='two.sided')

df = data.frame(Method = c(rep("Random Forest", 50), rep("Gradient Boosting", 50)),
                Value = c(data$Random.Forest[1:50], data$Gradient.Boosting[1:50]))

p = ggplot(df, aes(x=Method, y=Value)) + 
  geom_boxplot(fill=c('#FF4949', '#30E7ED')) +
  xlab("Regression Method") +
  ylab("RMSE Value") +
  ggtitle("RMSE Value Achieved vs. Regression Method")
  

p