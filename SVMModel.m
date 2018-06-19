% Demo model (under testing)

classdef SVMModel
    methods(Static)
        function Model = train(dataSet, dataSetClasses) % dataSetClasses Cell
            diffClasses = unique(dataSetClasses, 'stable');
            
            if numel(diffClasses) == 2
                
                Model.FirstClass = diffClasses{1};
                Model.SecondClass = diffClasses{2};
                
                dataSetClassesSign = ones(size(dataSetClasses));
                firstClassLogicalIndexes = cellfun(@(x) isequal(x,Model.FirstClass), dataSetClasses);
                dataSetClassesSign(firstClassLogicalIndexes) = -1;
                dataSet(firstClassLogicalIndexes,:) = dataSet(firstClassLogicalIndexes,:) * -1;
                try
                    data = horzcat(dataSet, dataSetClassesSign);
                catch
                    data = horzcat(dataSet, dataSetClassesSign');
                end
                MAX_ITERATIONS = size(data, 1)*10000;
                %initialize weight
                Model.Weight = data(1,:);
                ITERATIONS = 0;
                ACCEPTED_COUNT = 0;
                idx = 1;
                while ITERATIONS ~= MAX_ITERATIONS && ACCEPTED_COUNT < size(data, 1)
                    if  sum(Model.Weight .* data(idx,:)) < 0
                        % update weight
                        Model.Weight = Model.Weight + data(idx,:);
                        ACCEPTED_COUNT = -1;
                    end
                    ITERATIONS = ITERATIONS + 1;
                    ACCEPTED_COUNT = ACCEPTED_COUNT + 1;
                    idx = idx + 1;
                    if idx > size(data, 1)
                        idx = 1;
                    end
                end
                if ITERATIONS == MAX_ITERATIONS
                    Model.Weight = NaN;
                end
            else
                Model.Weight = NaN;
            end
        end
        
        function ClassesTypes = predict(Model, newPatterns)
            if (numel(Model.Weight) - size(newPatterns, 2)) == 1
                parfor i=1:size(newPatterns, 1)
                    result = sum((Model.Weight(1:end-1) .* newPatterns(i,:))) + Model.Weight(end);
                    if result < 0
                        ClassesTypes{i,1} = Model.FirstClass;
                    elseif result > 0
                        ClassesTypes{i,1} = Model.SecondClass;
                    else
                        ClassesTypes{i,1} = '';
                    end
                end
            else
                ClassesTypes = '';
            end
        end
        
        function [data, classes] = extract2classesData(dataSet, dataSetClasses, Class1, Class2)
            Class1LogicalIndexes = cellfun(@(x) isequal(x,Class1), dataSetClasses);
            Class2LogicalIndexes = cellfun(@(x) isequal(x,Class2), dataSetClasses);
            
            data = vertcat(dataSet(Class1LogicalIndexes, :), dataSet(Class2LogicalIndexes, :));
            classes = vertcat(dataSetClasses(Class1LogicalIndexes, :), dataSetClasses(Class2LogicalIndexes, :));
        end
        
        function Models = trainMult(dataSet, dataSetClasses)
            Models = cell(0,1);
            diffClasses = unique(dataSetClasses, 'stable');
            % different combinations = (C(C-1))/2 where C = # of diffClasses
            for class1Idx=1:numel(diffClasses)-1
                for class2Idx=class1Idx+1:numel(diffClasses)
                    [data, classes] = SVMModel.extract2classesData(dataSet,...
                        dataSetClasses, diffClasses{class1Idx}, diffClasses{class2Idx});
                    Models{end+1} = SVMModel.train(data, classes);
                end
            end
        end
        
        function ClassesTypes = predictMult(Models, newPatterns)
            predictionMatrix = cell(size(newPatterns, 1), 0);
            parfor modelIdx=1:numel(Models)
                predictedCol = SVMModel.predict(Models{modelIdx}, newPatterns);
                predictionMatrix = horzcat(predictionMatrix, predictedCol);
            end
            parfor i=1:size(newPatterns, 1)
                % get the most repeated prediction for this pattern
                predictionRow = predictionMatrix(i, :);
                diffPredictions = unique(predictionRow, 'stable');
                [~, itemp] = max(cellfun(@(x) isequal(x, predictionRow), diffPredictions));
                ClassesTypes{i,1} = diffPredictions(itemp);
            end
        end
    end
end