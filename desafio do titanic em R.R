#### Entendimento ####
# leitura dos arquivos
train = read.csv('train (2).csv')
test = read.csv('test.csv')

# visualizando as primeiras linhas
head(train, 3)

# estrutura do df
str(train)

# dados estatísticos do df
summary(train)

#### Tratamento ####
# manipulação e limpeza dos dados
library(dplyr)
library(tidyr)

# eliminando colunas desnecessárias
train = select(train, -Cabin, -Name, -Ticket) # pegando todas colunas menos a Cabin, name e ticket
test = select(test, -Cabin, -Name, -Ticket)

# verificando linhas vazias
any(is.na(train))

# contando valores nan por coluna
colSums(is.na(train))

# tirar a média da coluna Age
media_train = mean(train$Age, na.rm =T)
media_test = mean(test$Age, na.rm=T)

# botar media nos valores nan
train[is.na(train$Age), "Age"] = media_train
test[is.na(test$Age), "Age"] = media_test

# verificar se restam valores nan
any(is.na(train$Age))
any(is.na(test$Age))

# verificar valores distintos
distinct(train, Embarked)

# contar valores distintos
table(train$Embarked)

# substituindo valores vazios de embarked pela moda
train[train$Embarked=="", "Embarked"] = "S"

# verificando novamente
table(train$Embarked)

# calcular a moda
#install.packages('DescTools')
library(DescTools)

Mode(train$Embarked)

# ajeitando datatype das colunas de treino
train$Survived = factor(train$Survived)
train$Pclass = factor(train$Pclass)
train$Sex = factor(train$Sex)
train$Embarked = factor(train$Embarked)

# verificando valores vazios em treino
is.na(test)

colSums(is.na(test))

# tratando coluna Fare em test
media_fare = mean(test$Fare, na.rm = T)

test[is.na(test$Fare), 'Fare'] = media_fare

# verificando se deu certo
colSums(is.na(test))

# verificando se embarked contem valores ''
table(test$Embarked)

# ajeitando datatype das colunas de test
test$Pclass = factor(test$Pclass)
test$Sex = factor(test$Sex)
test$Embarked = factor(test$Embarked)

#### Visualizacao ####
library(ggplot2)

# histograma de idades
ggplot(train, aes(Age)) + geom_histogram(bins=20, fill='blue', alpha=.4)

# separando por quem sobreviveu
plot1 = ggplot(train, aes(Age)) + geom_histogram(bins=20, aes(fill=Survived), alpha=.8, color='white')
plot1

# separando por genero
plot1 + facet_grid(Sex ~.)

# separando por classe
plot2 = plot1 + facet_grid(Sex ~ Pclass) + theme_bw()

# usando o plotly (graficos navegáveis)
library(plotly)

ggplotly(plot2)

#### Separando em treino e teste ####

# separando em treino e teste
library(caTools)

# declarar seed aleatoria (random_state)
set.seed(42)

sample = sample.split(train$Survived, .7)

treino = subset(train, sample==T)
teste = subset(train, sample==F)

#### Arvore de decisao ####
library(rpart)

str(treino)

arvore = rpart(Survived ~., method="class", data=treino[,-1]) # selecionando todos menos a coluna survived e eliminando coluna id do treino

# visualizando a arvore
library(rpart.plot)

prp(arvore)

# predict
ypred = predict(arvore, teste, type = 'class')

# salvar em uma nova coluna
teste$ypred = ypred

# matriz de confusão
table(teste$Survived, teste$ypred)

# retirar coluna ypred
teste = select(teste, -ypred)

# usando a biblioteca caret para a matriz de confusão
library(caret)
confusionMatrix(teste$Survived, ypred)

#### RandomForest ####
library(randomForest)

rf = randomForest(Survived~., method='class', data=treino[,-1])

ypredrf = predict(rf, teste, type='class')

confusionMatrix(teste$Survived, ypredrf)

#### Regressão Logística ####
reg.log = glm(Survived~., family=binomial(), data=treino[,-1])

# predict
ypredrl = predict(reg.log, teste, type='response') # pegar probabilidades

ypredrl = ifelse(ypredrl > .5,1,0)

# convertendo datatype
ypredrl = factor(ypredrl)

# matriz de confusão
confusionMatrix(teste$Survived, ypredrl)

#### KNN ####
library(class)

# convertendo textos pq esse modelo n aceita formato string
treino2 = treino
teste2 = teste

# convertendo sexo
treino2$isMale = ifelse(treino2$Sex == "male", 1, 0)
teste2$isMale = ifelse(teste2$Sex == "male", 1, 0)

# deletando coluna original
treino2 = select(treino2, -Sex)
teste2 = select(teste2, -Sex)

# tratando a coluna Embarked (parece o get_dummies/one-hot encoding)
treino2$n = 1
treino2 %>% pivot_wider(names_from = Embarked, values_from = n, values_fill = 0)

teste2$n = 1
teste2 %>% pivot_wider(names_from = Embarked, values_from = n, values_fill = 0)

treino2 = select(treino2, -Embarked, -n)
teste2 = select(teste2, -Embarked, -n)

# predict direto (knn tem aprendizagem preguiçosa)
ypredknn = knn(treino2[,-1], teste2[,-1], treino2$Survived, k=3)

# matriz de confusão
confusionMatrix(teste2$Survived, ypredknn)


#### Utilizando o melhor modelo nos dados da competição ####
head(test, 3)

# fazendo o predict
ypredfinal = predict(rf, test, method="class")

# add à base
test$Predicted = ypredfinal

# selecionando a base
base = select(test, PassengerId, ypredfinal)

# exportando
write.csv(base, 'result.csv', row.names = F) # row.names=F = index=False
