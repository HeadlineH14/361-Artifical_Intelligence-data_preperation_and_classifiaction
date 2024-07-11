clear all, close all ,clc
load ("dataset-letters.mat");
i = dataset.images;
key = dataset.key;
l = dataset.labels;
sz = [28,28];
r = randperm(26000,12);
tab = zeros(26000,784);
%loops through for the amount of rows in i and adds it to the new tab
for k = 1:size(i,1)
    vec = i(k,:);
    tab(k,:) = vec;
end

%loops through 12 times and selects random letters from the samples 
for o = 1:12
    keys = key(:,l(r,:));
im = reshape(tab(r(o),:),sz);
subplot(3,4,o), imagesc(im),axis off,title (keys(:,o));
end

figure(1), colormap gray;
saveas(gcf,'data.png');


trainingDataNum = randperm(13000);
testingDataNum = 13000 + randperm(13000);

trainingData = i(trainingDataNum,:);

siz = size(im);
ker = [-1,-1,-1;-1,8,-1;-1,-1,-1];

cim = conv2(im,ker,'same');
figure(2);
subplot(311),imagesc(abs(cim)),axis image off;
tau = 0.15;
edgemap = abs(cim) >= tau;
subplot(313),imagesc(edgemap), axis image off;

mean(cim,'all');
mean(im,'all');
mean(ker,'all');





% own knn
features = i();
labels = categorical(l);
% Setup training features and labels


trfeatures = features(trainingDataNum,:);
trlabels = labels(trainingDataNum);
% Setup testing features and labels, and array for predictions

tefeatures = features(testingDataNum,:);
telabels = labels(testingDataNum);
tepredict = categorical.empty(size(tefeatures,1),0);
tepredict1 = categorical.empty(size(trfeatures,1),0);
% Setup k parameter
k = 1;
% Go through testing data to collect distance information and determine
% prediction
for u = 1:size(tefeatures,1)
    % Calculate distance of current testing sample from all training samples
    comp1 = trfeatures;
    comp2 = repmat(tefeatures(u,:),[size(trfeatures,1),1]);
    %l1 = dot(comp1,comp2)/norm(comp1)*norm(comp2);
    l1 = sum((comp1+comp1)+(comp2-comp2));
    l2 = sqrt(sum((comp1-comp2).^2,2));
    % Get minimum k row indices

    [~,ind1] = sort(l1);
    [~,ind] = sort(l2);
    ind = ind(1:k);
    ind1 = ind1(1:k);
    % get labels for testing data
    labs2 = telabels(ind1);
    labs = trlabels(ind);
    tepredict1(u,1) = mode(labs2);
    tepredict(u,1) = mode(labs);
end
% Calculate Accuracy
correct_predictions = sum(telabels==tepredict);
correct_predictions1 = sum(telabels==tepredict1);
accuracy = correct_predictions /size(telabels,1);
accuracy1 = correct_predictions1 / size(telabels,1);
% Confusion Matrix
figure(13)
confusionchart(telabels,tepredict);
title(sprintf('Accuracy=%.2f',accuracy));
saveas(gcf,'l2accuracy.png');
figure(14)
confusionchart(telabels,tepredict1);
title(sprintf('Accuracy=%.2f',accuracy1));
saveas(gcf,'l1accuracy.png');



%knn model
knnmodel = fitcknn(features,labels);
predictedKnn = predict(knnmodel, features);

corrrect_predictionsKnn = sum(labels == predictedKnn)
accuracy = corrrect_predictionsKnn/size(labels,1)
knn_resub_err = resubLoss(knnmodel)

%figure(4)
%knnmodelCM = confusionchart(labels,predictedKnn)


trfeaturesKnn = features(trainingDataNum,:);
tefeaturesKnn = features(testingDataNum,:);
trlabelsKnn = labels(trainingDataNum);
telabelsKnn = labels(testingDataNum);
knnmodel2Knn = fitcknn(trfeaturesKnn,trlabelsKnn);
predictedKnn = predict(knnmodel2Knn, tefeaturesKnn);
correct_predictionsKnn = sum(telabelsKnn == predictedKnn)
accuracyKnn = correct_predictionsKnn /size(telabelsKnn,1)
figure(20)
knnmodelCM = confusionchart(telabelsKnn,predictedKnn)
title(sprintf('Accuracy=%.2f',accuracyKnn));
saveas(gcf,'Knnaccuracy.png');



%tree decision
treeModel = fitctree(features,labels);
predictedTree = predict(treeModel, features);

corrrect_predictionsTree = sum(labels == predictedTree)
accuracy = corrrect_predictionsTree/size(labels,1)
tree_resub_err = resubLoss(treeModel)

%figure(4)
%treeModelCM = confusionchart(labels,predictedTree)


trfeaturesTree = features(trainingDataNum,:);
tefeaturesTree = features(testingDataNum,:);
trlabelsTree = labels(trainingDataNum);
telabelsTree = labels(testingDataNum);
knnmodel2Tree = fitctree(trfeaturesTree,trlabelsTree);
predictedTree = predict(knnmodel2Tree, tefeaturesTree);
correct_predictionsTree = sum(telabelsTree == predictedTree)
accuracyTree = correct_predictionsTree /size(telabelsTree,1)
figure(21)
treemodelCM = confusionchart(telabelsTree,predictedTree)
title(sprintf('Accuracy=%.2f',accuracyTree));
saveas(gcf,'Treeaccuracy.png');

