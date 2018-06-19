classdef BayesHelper
    % BayesHelper Add summary here
    properties(Constant)
        
    end
    
    methods(Access = public)
        % returns baySet as a matrix(m,n) m->count(classes)
        % n->count(features) & each element is a struct of (mean,variance)
        % values %parfor classIdx parfor featureIdx
        function [baySet, classes, classesProps] = getBayesianSet(self, dataSet, dataSetClasses, normX)
            if nargin > 3
                dataSet = normX(dataSet); %normalize dataset
            end
            [classes, classesProps] = self.calcClassProp(dataSetClasses);
            for classIdx=1:numel(classes)
                parfor featureIdx=1:numel(dataSet(1,:))
                    classfeatureCol = self.getClassFeatureCol(dataSet(:,featureIdx), dataSetClasses, classes{classIdx});
                    baySet(classIdx, featureIdx) = self.calcMeanVariance(classfeatureCol);
                end
            end
        end
        % calculate the propability that the new pattern belongs to each
        % class and returns the classType corresponding to the highest
        % propability value
        function [classType] = bayesClassify(self, baySet, classes, classesProps, newPattern)
            liklihoods = ones(size(classes));
            parfor classIdx=1:numel(classes)
                for featureIdx=1:numel(baySet(1,:))
                    s = baySet(classIdx, featureIdx);
                    x = newPattern(featureIdx);
                    if liklihoods(classIdx) == 0
                        liklihoods(classIdx) = 0.0001;
                    end
                    liklihoods(classIdx) = liklihoods(classIdx)*self.calcPartLiklihood(s.Mean, s.Variance, x);
                end
            end
            liklihoods = liklihoods.*cell2mat(classesProps);
            [~, maxClassLikIdx] = max(liklihoods);
            classType = classes{maxClassLikIdx};
        end
    end
    
    methods(Access = private)
        % get feature column for a specific class
        function classfeatureCol = getClassFeatureCol(~, dataSetFeatureCol, dataSetClasses, className)
            classfeatureCol = dataSetFeatureCol(ismember(dataSetClasses,className));
        end
        % calculate mean & variance for a feature column
        function structMeanVariance = calcMeanVariance(~, featureCol)
            mean = sum(featureCol)/numel(featureCol);
            variance = sum((featureCol-mean).^2)/(numel(featureCol)-1);
            structMeanVariance = struct('Mean', mean, 'Variance', variance);
        end
        % calculate the propability for each class by num of occurance in dataSetClasses
        function [classes, classesProps] = calcClassProp(~, dataSetClasses)
            classes=unique(dataSetClasses,'stable');
            classesProps = cellfun(@(x) sum(ismember(dataSetClasses,x))/numel(dataSetClasses),classes,'un',0);
        end
        % calculate a specific part for total liklihood along one feature
        % column mean & variance with the new value x
        function out = calcPartLiklihood(~, mean, variance, x)
            out = (exp(((x-mean)^2)/(2*variance))*sqrt(2*pi*variance))^-1;
        end
    end
end