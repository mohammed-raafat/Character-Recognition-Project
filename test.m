%clear all;
clc;
[y,Fs] = audioread('done.wav');
ir = ImageReader;
cl = Classifier;

%load('trainHOG_4x4_Cells.mat');
%load('trainHOG_8x8_Cells.mat');
%load('train_4x4_Blocks.mat');
%load('train_8x8_Blocks.mat');

%train
%{
[dataClasses, imagePaths2D] = ir.read('dataset');
tic; %start stopwatch
%[trainedSetHOG, trainedSetClassesHOG] = cl.tr.TrainHOG(dataClasses, imagePaths2D, 120*120, [8 8]);
[trainedSetHOG, trainedSetClassesHOG] = cl.tr.TrainAsync(dataClasses, imagePaths2D, 120*120, [8 8], 1); % 1 = HOG
%[trainedSet, trainedSetClasses] = cl.tr.Train(dataClasses, imagePaths2D, 120*120, [4 4]);
%[trainedSet, trainedSetClasses] = cl.tr.TrainAsync(dataClasses, imagePaths2D, 120*120, [4 4], 0);
elapsedTrainingTimeMinutes = toc/60;
sound(y,Fs);
%}

%test
%{
tic; %start stopwatch
[testObjectsHOG, testObjectsPosHOG] = cl.getImgReadyHOG('testImgs/test4.jpg', 10, [8 8]);
I = imread('testImgs/test4.jpg');
for i=1:numel(testObjectsHOG(:,1))
	classTypeHOG = cl.weightedKNN(trainedSetHOG, trainedSetClassesHOG, testObjectsHOG(i,:), 3, 0);
    % Show the location of the objects in the original image
    I = insertObjectAnnotation(I, 'rectangle', testObjectsPosHOG{i}, classTypeHOG,'TextBoxOpacity',0.5,'FontSize',12);
end
imshow(imresize(I, 1));
elapsedClassificationTime = toc;
%}
%{
tic; %start stopwatch
[testObjects, testObjectsPos] = cl.getImgReady('testImgs/test4.jpg', 10, [4 4]);
I = imread('testImgs/test4.jpg');
for i=1:numel(testObjects(:,1))
	classType = cl.weightedKNN(trainedSet, trainedSetClasses, testObjects(i,:), 3, 0);
    % Show the location of the objects in the original image
    I = insertObjectAnnotation(I, 'rectangle', testObjectsPos{i}, classType,'TextBoxOpacity',0.5,'FontSize',12);
end
imshow(imresize(I, 1));
elapsedClassificationTime = toc;
%}

%test Async

%tic; %start stopwatch
[testObjects, testObjectsPos] = cl.getImgReadyHOG('testImgs/test4.jpg', 10, [8 8]);

%classesTypes = cl.weightedKNNAsync(trainedSetHOG, trainedSetClassesHOG, testObjects, 3, 0);

%[baySet, classes, classesProps] = cl.bh.getBayesianSet(trainedSetHOG, trainedSetClassesHOG, @normc);
%classesTypes = cl.bayesClassifyAsync(baySet, classes, classesProps, testObjects, @normc);

%svmModel = cl.svmTrain(trainedSetHOG, trainedSetClassesHOG);
%classesTypes = cl.svmClassifyAsync(svmModel, testObjects);

% I = imread('testImgs/test3.png');
% for i=1:numel(testObjects(:,1))
%     % Show the location of the objects in the original image
%     I = insertObjectAnnotation(I, 'rectangle', testObjectsPos{i}, classesTypes{i},'TextBoxOpacity',0.5,'FontSize',12);
% end
% imshow(imresize(I, 1));
% elapsedClassificationTime = toc;
%}