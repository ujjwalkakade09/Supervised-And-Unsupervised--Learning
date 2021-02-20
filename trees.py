from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

x=[[120,70,44],[150,80,42],[60,50,36],[100,65,38],[110,70,40],[105,68,38],[95,80,39],[100,60,35]]

y=['male','male','female','female','male','male','female','male']

clf=tree.DecisionTreeClassifier()
clf2 = RandomForestClassifier()
clf3=AdaBoostClassifier()

clf= clf.fit(x,y)

clf2=clf2.fit(x,y)

clf3=clf3.fit(x,y)


prediction= clf.predict([[105,75,39]])
prediction2=clf2.predict([[105,75,39]])
prediction3= clf3.predict([[105,75,39]])
print(prediction)
print(prediction2)
print(prediction3)
