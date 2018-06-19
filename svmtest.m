%% Multi-Class SVM
%% Demo Begin
%% Initialize all to default
clc
% clear all
% close all
%% number of samples and Class initialization 
% nOfSamples= size(trainedSetHOG, 1); % 3410
% nOfClassInstance= size(unique(trainedSetClassesHOG, 'stable'), 1); % 62
Sample= trainedSetHOG; % trainedSetHOG
class = zeros(size(trainedSetClassesHOG));
%convert classes' names to doubles
parfor i = 1 : numel(trainedSetClassesHOG)
    class(i) = trainedSetClassesHOG{i,1};
end
%% SVM Classification
svm = SVMHelper;
Model=svm.train(Sample,class);
predict=svm.predict(Model,testObjects(1,:));
%[Model,predict] = svm.classify(Sample,class,testObjects);

classesTypes = cell(size(testObjects,1) , 1);
%convert doubles to classes' names 
parfor i = 1 : numel(predict)
    classesTypes{i} = char(predict(i));
end

% disp('class predict')
% disp([class predict])
%% Find Accuracy
% Accuracy=mean(class==predict)*100;
% fprintf('\nAccuracy =%d\n',Accuracy)


%Models = SVMModel.trainMult(trainedSetHOG, trainedSetClassesHOG);
%ClassesTypes = SVMModel.predictMult(Models, testObjects);