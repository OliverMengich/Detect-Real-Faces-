
imds = imageDatastore('FacesFolder','IncludeSubfolders',true,'LabelSource','foldernames');
[training,test] = splitEachLabel(imds,0.6);

I = readimage(imds,72);
[hog,vis] = extractHOGFeatures(I,'CellSize',[27 27]);
%imshow(I),hold on; plot(vis);

numImages = numel(training.Files);
trainingFeatures = zeros(numImages, length(hog), 'single');


for i = 1:numImages
   I = readimage(training,i); 
    I = rgb2gray(I);
    I  = imresize(I,[284 283]);
    
      trainingFeatures(i, :) = extractHOGFeatures(I,'CellSize',[26 25]);  
     
end
   trainingLabels = training.Labels;
   
   
   queryImage = readimage(test,17);
   queryImage = rgb2gray(queryImage);
queryImage = imresize(queryImage,[284,283]);

queryFeatures = extractHOGFeatures(queryImage,'CellSize',[26 25]);
personLabel = predict(classifier,queryFeatures);
