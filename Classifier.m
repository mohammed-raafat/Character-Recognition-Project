classdef Classifier
    % Classifier Add summary here
    
    properties(Constant)
        tr = Trainer;
        bh = BayesHelper;
    end
    properties(Access = private)
        svm = SVMHelper;
    end
    
    methods(Access = public)

        function [testObjects, testObjectsPositions] = getImgReady(self, ImgPath, noiseThreshold, blockSize)
            switch nargin
                case 3
                    blockSize = self.tr.defBlockSize;
                case 2
                    blockSize = self.tr.defBlockSize;
                    noiseThreshold = self.tr.defNoiseThreshold;
            end
            try
                [testObjects, ~, testObjectsPositions] = self.tr.Train({'Unknown'}, {{ImgPath}}, noiseThreshold, blockSize);
            catch
            end
        end
        
        function [testObjects, testObjectsPositions] = getImgReadyHOG(self, ImgPath, noiseThreshold, CellSize)
            switch nargin
                case 3
                    CellSize = self.tr.defBlockSize;
                case 2
                    CellSize = self.tr.defBlockSize;
                    noiseThreshold = self.tr.defNoiseThreshold;
            end
            try
                [testObjects, ~, testObjectsPositions] = self.tr.TrainHOG({'Unknown'}, {{ImgPath}}, noiseThreshold, CellSize);
            catch
            end
        end
        
        % k = 0 (means NN) , weights = 0 (means no weights)
        function [classType] = weightedKNN(~, data, dataClasses, testPattern, k, weights)
            [m,n] = size(data);  % get data matrix size
            % check for k & weights values
            if k == 0
                k = 1;
            end
            if weights == 0
                weights = ones(1,n);
            end
            % compute weighted Euclidean distances
            distances = zeros(m,1);
            for r=1:m
                rowDistance = 0;
                for c=1:n
                    rowDistance = rowDistance + weights(c)*((testPattern(c) - data(r,c))^2);
                end
                rowDistance = sqrt(rowDistance);
                distances(r) = rowDistance; % save the distance
            end
            % try to get the minimum k distances' classes by indexes
            kDistanceClasses = cell(k,1);
            for i=1:k
                [~, minimumDistanceIndex] = min(distances);
                kDistanceClasses{i} = dataClasses{minimumDistanceIndex};
                distances(minimumDistanceIndex) = Inf;
            end
            % return the most repeated class index
            y = unique(kDistanceClasses);
            n = zeros(length(y), 1);
            for iy = 1:length(y)
              n(iy) = length(find(strcmp(y{iy}, kDistanceClasses)));
            end
            [~, itemp] = max(n);
            classType = y(itemp);
            %classType = mode(kDistanceClasses);
        end
        
        function [classesTypes] = weightedKNNAsync(self, data, dataClasses, testPatterns, k, weights)
            parfor objIdx=1:numel(testPatterns(:,1))
                classType = self.weightedKNN(data, dataClasses, testPatterns(objIdx,:), k, weights);
                classesTypes{objIdx,1} = classType{1,1};
            end
        end
                
        function [classesTypes] = bayesClassifyAsync(self, baySet, classes, classesProps, testPatterns, normX)
            if nargin > 5
                testPatterns = normX(testPatterns); %normalize dataset
            end
            parfor objIdx=1:numel(testPatterns(:,1))
                classesTypes{objIdx,1} = self.bh.bayesClassify(baySet, classes, classesProps, testPatterns(objIdx,:));
            end
        end
            
        function [Model] = svmTrain(self, trainedSet, trainedSetClasses)
            Sample=trainedSet;
            class=zeros(size(trainedSetClasses));
            %convert classes' names to doubles
            parfor i = 1 : numel(trainedSetClasses)
                class(i) = trainedSetClasses{i,1};
            end
            Model=self.svm.train(Sample,class);
        end
        
        function [classType] = svmClassify(self, model, testPattern)
            predicted=self.svm.predict(model, testPattern);
            %convert double to class' name
            classType = char(predicted);
        end
        
        function [classesTypes] = svmClassifyAsync(self, model, testPatterns)
            predicted=self.svm.predict(model, testPatterns);
            classesTypes = cell(size(predicted));
            %convert doubles to classes' names 
            parfor i = 1 : numel(predicted)
                classesTypes{i} = char(predicted(i));
            end
        end
    end
end
