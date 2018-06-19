classdef Trainer < matlab.System
    % Trainer Add summary here
    
    % Pre-computed constants
    properties(Constant)
        imgSize = [320,320];
        objSize = [40,40];
        defBlockSize = [8 8];
        defNoiseThreshold = 120*120;
    end

    methods(Access = public)
        function enhancedBinaryImg = imenhance(self, rawImgPath, noiseThreshold)
            % image read
            img = imread(rawImgPath);
            % resize to fixed w & h
            %img = imresize(img, self.imgSize);
            % Convert to Gray image
            if size(img,3)==3 %RGB image
                img = rgb2gray(img);
            end
            % threshold or convert to binary image
            img = imbinarize(img);
            img = ~img; %negative
            % Remove all object containing fewer than noiseThreshold pixels
            if nargin < 3
                noiseThreshold = self.defNoiseThreshold;
            end
            enhancedBinaryImg = bwareaopen(img, noiseThreshold);
        end
        
        function [imgObjects, rectPositions] = extractObjects(self, enhancedBinaryImg)  
            %Segment all object in the image based on 8 connectivity
            objects = bwconncomp(enhancedBinaryImg,8);
            %foreach extracted object
            %initialize imgObjects as cell array
            imgObjects = cell(objects.NumObjects, 1);
            rectPositions = cell(objects.NumObjects, 1);
            parfor obj=1:objects.NumObjects %  Taha -TM
                %get colored pixels indexes column
                coloredPixelsIdx = objects.PixelIdxList(1,obj);
                %create a black image with the same enhancedBinaryImg size
                objImg = false(objects.ImageSize);
                %draw the white obj pixel by pixel
                for i=1:numel(coloredPixelsIdx)
                    %color the specified pixel by index
                    objImg(coloredPixelsIdx{i}) = 1;
                end
                %imgObjects{obj} = objImg;
                
                %get the BoundingBox for the objImg
                s = regionprops(objImg, 'BoundingBox');
                %get the bounding rectangle
                rect = s.BoundingBox;
                %crop & resize the extracted object then add to imgObjects array
                imgObjects{obj} = imresize(imcrop(objImg, rect), self.objSize);
                rectPositions{obj} = rect;
                
                %debug
                %imshow(imgObjects{obj});
            end        
        end
        
        function Centroid = getCentroid(~, imgObject)
            [m,n] = size(imgObject);
            X_hist=sum(imgObject,1); 
            Y_hist=sum(imgObject,2); 
            X=1:n; Y=1:m;
            if sum(X_hist) == 0
                centX = 0;
            else
                centX=sum(X.*X_hist)/sum(X_hist); 
            end
            if sum(Y_hist) == 0
                centY = 0;
            else
                centY=sum(Y'.*Y_hist)/sum(Y_hist);
            end
            Centroid = [centX centY];
            %get centroid pixel index
            %roundedX = round(centX);
            %roundedY = round(centY);
            % create helper indexer matrix
            %idxerMat = reshape(1:m*n, [m n]);
            %Centroid = idxerMat(roundedX, roundedY);
        end
        
        function Medoid = getMedoid(~, imgObject)
            imgObject = double(imgObject);
            % Get logical medoid matrix
            logical_med = imgObject==median(imgObject(:));
            % find medoids indexes [doubles]
            med_indexes = find(logical_med);
            if numel(med_indexes) == 0
                logical_med = ~logical_med;
                med_indexes = find(logical_med);
            end
            % find medoids indexes [(X,Y) pairs]
            [X,Y] = find(logical_med);
            % Get median value of med_indexes
            med_val = med_indexes(round(numel(med_indexes)/2));
            % Get median index of med_indexes
            med_index = find(med_indexes==med_val);
            medX = X(med_index);
            medY = Y(med_index);
            Medoid = [medX, medY];
            
            %[~, medX] = self.extractMedoidRow(imgObject');
            %[~, medY] = self.extractMedoidRow(imgObject);
            %Medoid = [medX medY];
            
            %get medoid pixel index
            %[m,n] = size(imgObject);
            % create helper indexer matrix
            %idxerMat = reshape(1:m*n, [m n]);
            %Medoid = round(idxerMat(medX, medY)/2);
        end
        
        function Perimeter = getPerimeter(~, imgObject)
            I=zeros(size(imgObject)); 
            I(2:end-1,2:end-1)=1;
            Perimeter = sum(reshape(imgObject.*I,1,[]));
        end
        
        function Area = getArea(~, imgObject)
            Area = 0;
            for i=1:numel(imgObject)
                if(imgObject(i))
                    Area = Area + 1;
                end
            end
        end
        
        function [dataSet, dataSetClasses, rectPositions] = Train(self, dataClasses, imagePaths2D, noiseThreshold, blockSize)
            dataSetClasses = cell(0,1);
            rectPositions = cell(0,1);
            dataSet_Initialized = 0;
            for classIdx = 1 : numel(dataClasses)
                classImgsPaths = imagePaths2D{classIdx};
                for classImgPathIdx = 1 : numel(classImgsPaths)
                    curImgPath = classImgsPaths{classImgPathIdx};
                    if nargin < 4
                        enhancedBinImg = self.imenhance(curImgPath);
                    else
                        enhancedBinImg = self.imenhance(curImgPath, noiseThreshold);
                    end
                    [imgObjs, imgObjsPositions] = self.extractObjects(enhancedBinImg);
                    rectPositions = vertcat(rectPositions, imgObjsPositions);
                    for objIdx = 1 : numel(imgObjs)
                        curObj = imgObjs{objIdx};
                        if nargin < 5
                            curObjSegms = self.segment(curObj);
                        else
                            curObjSegms = self.segment(curObj, blockSize);
                        end                   
                        numOfFeatures = 11;
                        if ~dataSet_Initialized %initialize for first time only
                            dataSet = zeros(0, numel(curObjSegms)*numOfFeatures);
                            dataSet_Initialized = 1;
                        end
                        colRange = 1:numOfFeatures;
                        [m,~] = size(dataSet);
                        %foreach object segment
                        for segIdx = 1:numel(curObjSegms)
                            curObjSegm = curObjSegms{segIdx};
                            featureVector = zeros(1, numOfFeatures);
                            % get all features & add to featureVector
                            featureVector(1,1:2) = self.getCentroid(curObjSegm);
                            featureVector(1,3:4) = self.getMedoid(curObjSegm);
                            featureVector(1,5) = self.getPerimeter(curObjSegm);
                            featureVector(1,6) = self.getArea(curObjSegm);
                            s = regionprops(curObjSegm,'Euler');
                            try
                                featureVector(1,7) = s.EulerNumber;
                            catch
                                featureVector(1,7) = 0;
                            end
                            s = regionprops(curObjSegm,'Extent');
                            try
                                featureVector(1,8) = s.Extent;
                            catch
                                featureVector(1,8) = 0;
                            end
                            s = regionprops(curObjSegm,'MajorAxisLength');
                            try
                                featureVector(1,9) = s.MajorAxisLength;                            
                            catch
                                featureVector(1,9) = 0;
                            end
                            s = regionprops(curObjSegm,'MinorAxisLength');
                            try
                                featureVector(1,10) = s.MinorAxisLength;
                            catch
                                featureVector(1,10) = 0;
                            end
                            s = regionprops(curObjSegm,'Orientation');
                            try
                                featureVector(1,11) = s.Orientation;
                            catch
                                featureVector(1,11) = 0;
                            end
                            % add featureVector to dataSet
                            if max(featureVector)
                                featureVector = abs(featureVector);
                                featureVector = featureVector/max(featureVector); %normalize
                            end
                            dataSet(m+1, colRange) = featureVector;
                            
                            colRange = colRange + numOfFeatures;
                        end
                        dataSetClasses{end+1,:} = dataClasses{classIdx};
                    end
                end              
            end
        end
        
        function imgSegments = segment(self, img, blockSize)
            [imgX,imgY] = size(img);
            if nargin < 3
                blockSize = self.defBlockSize;
            end
            blockX = blockSize(1);
            blockY = blockSize(2);
            imgSegments = cell(0, 1);
            if blockX <= imgX && blockY <= imgY
                x1 = 1; x2 = blockX;
                while x2 <= imgX
                    y1 = 1; y2 = blockY;
                    while y2 <= imgY
                        imgSegments{end+1} = img(x1:x2, y1:y2);
                        y1 = y2+1;
                        y2 = y2+blockY;
                        
                        if y1 <= imgY && y2 > imgY
                            y1 = y1-(y2-imgY);
                            y2 = imgY;
                        end
                        
                    end
                    x1 = x2+1;
                    x2 = x2+blockX;
                    
                    if x1 <= imgX && x2 > imgX
                        x1 = x1-(x2-imgX);
                        x2 = imgX;
                    end
                    
                end
            else
                imgSegments{1} = img;
            end
        end
        
        function [dataSet, dataSetClasses, rectPositions] = TrainHOG(self, dataClasses, imagePaths2D, noiseThreshold, CellSize)
            dataSetClasses = cell(0,1);
            rectPositions = cell(0,1);
            dataSet_Initialized = 0;
            for classIdx = 1 : numel(dataClasses)
                classImgsPaths = imagePaths2D{classIdx};
                for classImgPathIdx = 1 : numel(classImgsPaths)
                    curImgPath = classImgsPaths{classImgPathIdx};
                    if nargin < 4
                        enhancedBinImg = self.imenhance(curImgPath);
                    else
                        enhancedBinImg = self.imenhance(curImgPath, noiseThreshold);
                    end 
                    [imgObjs, imgObjsPositions] = self.extractObjects(enhancedBinImg);
                    rectPositions = vertcat(rectPositions, imgObjsPositions);
                    for objIdx = 1 : numel(imgObjs)
                        curObj = imgObjs{objIdx};
                        if nargin < 5
                            CellSize = self.defBlockSize;
                        end
                        hogFeatures = extractHOGFeatures(curObj,'CellSize', CellSize);
                        if ~dataSet_Initialized %initialize for first time only
                            dataSet = zeros(0, numel(hogFeatures));
                            dataSet_Initialized = 1;
                        end
                        dataSet(end+1,:) = hogFeatures;
                        dataSetClasses{end+1,:} = dataClasses{classIdx};
                    end
                end              
            end
        end
        
        function [dataSet, dataSetClasses, rectPositions] = TrainAsync(self, dataClasses, imagePaths2D, noiseThreshold, blockSize, isHOG)
            switch nargin
                case 5
                    isHOG = 1;
                case 4
                    isHOG = 1;
                    blockSize = self.defBlockSize;
                case 3
                    isHOG = 1;
                    blockSize = self.defBlockSize;
                    noiseThreshold = self.defNoiseThreshold;
            end
            if isHOG
                parfor classIdx = 1:numel(dataClasses)
                    [dataSetCell{classIdx,1}, dataSetClassesCell{classIdx,1}, rectPositionsCell{classIdx,1}] = self.TrainHOG({dataClasses{classIdx}}, {imagePaths2D{classIdx}}, noiseThreshold, blockSize);
                end
            else
                parfor classIdx = 1:numel(dataClasses)
                    [dataSetCell{classIdx,1}, dataSetClassesCell{classIdx,1}, rectPositionsCell{classIdx,1}] = self.Train({dataClasses{classIdx}}, {imagePaths2D{classIdx}}, noiseThreshold, blockSize);
                end
            end
            %Copy cell arrays to standard arrays
            dataSet = zeros(0, numel(dataSetCell{1,1}(1,:)));
            dataSetClasses = cell(0, 1);
            rectPositions = cell(0, 1);
            parfor classIdx = 1:numel(dataClasses)
                dataSet = vertcat(dataSet, dataSetCell{classIdx,1});
                dataSetClasses = vertcat(dataSetClasses, dataSetClassesCell{classIdx,1});
                rectPositions = vertcat(rectPositions, rectPositionsCell{classIdx,1});
            end
        end
    end
    
    methods(Access = private)
        function [MedoidVal, MedoidIdx] = extractMedoidRow(~, imgObject)
            [m,n] = size(imgObject);
            MedoidVal = zeros(1,n);
            lastRowDistance = 0;
            for r = 1:m
                rowDistance = 0;
                for c = 1:n
                    curEl = imgObject(r,c);
                    for internalRow = 1:m
                        rowDistance = rowDistance + abs(curEl - imgObject(internalRow, c));
                    end
                end

                if r == 1
                    MedoidVal = imgObject(1,:); % extract the first row
                    MedoidIdx = 1;
                    lastRowDistance = rowDistance; % save the last row distance
                else
                    if rowDistance < lastRowDistance
                        MedoidVal = imgObject(r,:); % extract the r's row
                        MedoidIdx = r;
                        lastRowDistance = rowDistance;
                    end
                end
            end
        end
    end
end
