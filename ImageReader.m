classdef ImageReader
    % ImageReader Add summary here
    % Public, tunable properties

    methods(Access = public)
        function [dataClasses, imagePaths2D] = read(~, DATASETPATH)
            % Perform one-time calculations, such as computing constants
            % Get all directories from the dataset folder
            DIRs = dir(DATASETPATH);
            % dataClasses: holds all classes names (each name is a folder name)
            dataClasses = cell(numel(DIRs)-2,1);
            % Cell array of arrays to hold all images' paths foreach class
            imagePaths2D = cell(size(dataClasses));
            % For each class like (A,B,C,...)
            parfor i=3:numel(DIRs) % 3 to skip . & .. roots
                % Get current class path
                CLASSDIRPATH = strcat( DATASETPATH , '\' , DIRs(i).name );
                dataClasses{i-2} = strrep(DIRs(i).name,'_',''); %to eliminate '_'
                Files = dir(CLASSDIRPATH);
                    imagePaths = cell(numel(Files)-2,1);
                    % for each image in this class
                    for j=3:numel(Files)
                        imagePaths{j-2} = strcat( CLASSDIRPATH, '\', Files(j).name);
                    end
                    imagePaths2D{i-2} = imagePaths;
            end
        end
    end
end
