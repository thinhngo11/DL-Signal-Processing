%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ver 4: using type as data size 2 for 2000, 3 for 3000 and added Mod type
% variable
% 8/21/20 for all domains, types, and modes
% 5/26/20 all 100% at 10000 numframes Doppler 5G NR time-domain subframe-based input size = 15,736: 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% mode = 0; % generating data
mode = 0; % predicting data (5:1-5, 10:10, 15;15); 20: for statistics
type = 0; % 0: greater than or less; 1:equal or not
domain = 0; %0: freq, 1: time, 2: freq+time
scheme = 1; % 0: multi-class, 1: binary + merged binary

sps = 8;                % Samples per symbol

rng(1235)

spf_fs = 8568;  %612*14; %subframe frequency domain based prediction
spf_ts = 15376;
spf_tfs = spf_fs + spf_ts;
if domain == 0
   spf = spf_fs;
elseif domain == 1
   spf = spf_ts;
else
   spf = spf_tfs;
end
% spf = 15376;  %612*14; %subframe time domain based prediction

% dropoutRate = 0.5;
netWidth = 1;
filterSize = [1 sps];
poolSize = [1 2];
numHiddenUnits = 200;
WINDOWS = [1 2 3 4 5 10 15];
maxEpochs = 12;  %is significant
miniBatchSize = 64; %256;

% WINDOWS = [1 2 3 4 5];
% WINDOWS = [20];
numWin = length(WINDOWS);


rng('default');         % Configure random number generators

SNRs = [0, 10, 20];
% Dopplers = [0 20 40 60 80 100 120];
% Dopplers = [0 40 80 120 160 200];
% Dopplers = [0 40 80 120 160 200 240 280 320 360];
Dopplers = [0 50 100 150 200 250 300 350 400 450 500 550];
doppler_inc = Dopplers(2) - Dopplers(1);
Delays = [0 2 5]; % 0 25us
boundary = Dopplers(2 : length(Dopplers));
boundaryLabel = [-1, 1];
BoundLabelTypes = categorical(boundaryLabel);
numSNRTypes = length(SNRs);
DopplerTypes = categorical(Dopplers);
numDop = length(Dopplers);
numDelayTypes = length(Delays);
numBound = numDop - 1;

if scheme == 0
    numkk = 1;
    numLabel = numDop;
elseif type == 0
  numkk = numBound;
  rxTestPredAll = cell(numWin, numkk); 
  testAccuracyI = zeros(numWin, numkk); 
  numLabel = length(boundaryLabel);
else
  numkk = numDop;
  rxTestPredAll = cell(numWin, numkk); 
  testAccuracyI = zeros(numWin, numkk); 
  numLabel = length(boundaryLabel);
end

  modClassNet = [
  sequenceInputLayer([1 spf 2],'Name','input')    
  sequenceFoldingLayer('Name','fold')
  convolution2dLayer(filterSize, 16*netWidth, 'Padding', 'same', 'Name', 'CNN1')
  batchNormalizationLayer('Name', 'BN1')
  reluLayer('Name', 'ReLU1')
  maxPooling2dLayer(poolSize, 'Stride', [1 2], 'Name', 'MaxPool1')
  convolution2dLayer(filterSize, 24*netWidth, 'Padding', 'same', 'Name', 'CNN2')
  batchNormalizationLayer('Name', 'BN2')
  reluLayer('Name', 'ReLU2')
  maxPooling2dLayer(poolSize, 'Stride', [1 2], 'Name', 'MaxPool2')
  convolution2dLayer(filterSize, 32*netWidth, 'Padding', 'same', 'Name', 'CNN3')
  batchNormalizationLayer('Name', 'BN3')
  reluLayer('Name', 'ReLU3')
  maxPooling2dLayer(poolSize, 'Stride', [1 2], 'Name', 'MaxPool3')
  convolution2dLayer(filterSize, 48*netWidth, 'Padding', 'same', 'Name', 'CNN4')
  batchNormalizationLayer('Name', 'BN4')
  reluLayer('Name', 'ReLU4')
  maxPooling2dLayer(poolSize, 'Stride', [1 2], 'Name', 'MaxPool4')
  convolution2dLayer(filterSize, 64*netWidth, 'Padding', 'same', 'Name', 'CNN5')
  batchNormalizationLayer('Name', 'BN5')
  reluLayer('Name', 'ReLU5')
  maxPooling2dLayer(poolSize, 'Stride', [1 2], 'Name', 'MaxPool5')
%   convolution2dLayer(filterSize, 96*netWidth, 'Padding', 'same', 'Name', 'CNN6')
%   batchNormalizationLayer('Name', 'BN6')
%   reluLayer('Name', 'ReLU6')
%   averagePooling2dLayer([1 ceil(spf/32)], 'Name', 'AP1')
  sequenceUnfoldingLayer('Name','unfold')
  flattenLayer('Name','flatten')
  lstmLayer(numHiddenUnits,'OutputMode','last','Name','lstm')
  fullyConnectedLayer(numLabel, 'Name', 'FC1')
  softmaxLayer('Name', 'SoftMax')
  classificationLayer('Name', 'Output') ];

% testAccuracyS = zeros(length(WINDOWS), length(Dopplers));
% testAccuracyI = zeros(numWin, numkk); 

numFramesPerDopplerType = 3000;
percentTrainSamples = 80;
percentValidationSamples = 10;
percentTestSamples = 10;
numData = numFramesPerDopplerType*numDop; 
numValidation = numData * percentValidationSamples/100;
numTest = numData * percentTestSamples/100;
numTraining = numData * percentTrainSamples/100;
numwin = length(WINDOWS);
numTestperDop = numTest/numDop;

disp("mode=" + string(mode) + " type=" + string(type) + " domain=" + string(domain) + " scheme=" + string(scheme) + " numkk=" + string(numkk) + " numData=" + numData); 

validationFrequency = floor(numValidation/miniBatchSize);

if scheme == 0
    winstart = 1;
    winend = 7;
elseif mode == 5 || mode == 105
    winstart = 1;
    winend = 5;
elseif mode == 10 || mode == 110
    winstart = 6;
    winend = 6;
elseif mode == 15 || mode == 115
    winstart = 7;
    winend = 7;
end

if mode == 0
    % Generate the training data
    disp("Generate the training data");
%     [Data, Labels] = hGenerateTrainingData(numFramesPerDopplerType, DopplerTypes, boundary, kk);
    [Data, Labels] = hGenerateTrainingData(numFramesPerDopplerType, DopplerTypes, domain);
    Data = reshape(Data, [1 spf 2 numData]);
    save('NR5G' + string(domain) + '_DopplerN3000.mat', 'Data', 'Labels', '-v7.3');
elseif mode < 100 % pred if 5,10 or 15;  >100 compute pred statistics only
    disp("predicting");
    load('NR5G' + string(domain) + '_DopplerN3000.mat', 'Data', 'Labels');
    Labels = fixabug(Labels);
        
    valStart = 1;
    valEnd = numData * percentValidationSamples/100;
    rxValidation = Data(:,:,:,valStart:valEnd);
    rxValidationLabel = Labels(valStart:valEnd);

    % Split into test sets
    testStart = 1 + valEnd;
    testEnd = numData * percentTestSamples/100 + valEnd;
    rxTest = Data(:,:,:,testStart:testEnd);
    rxTestLabel = Labels(testStart:testEnd);

    % Split into training sets
    trainStart = 1 + testEnd;
    trainEnd = numData * percentTrainSamples/100 + testEnd;
    rxTraining = Data(:,:,:,trainStart:trainEnd);
    rxTrainingLabel = Labels(trainStart:trainEnd);

    rxTestPredAll = cell(numWin, numkk); 

%     parpool(numkk);
%     parfor kk = 1:numkk
  for kk = 1:numkk
        if scheme == 0
            rxValidationLabelkk = rxValidationLabel;
            rxTestLabelkk = rxTestLabel;
            rxTrainingLabelkk = rxTrainingLabel;
        else
            if type == 0
                Doppler = Dopplers(kk+1);  % matlab bug? error on index of 8
    %             Doppler = boundary(kk);
            else
                Doppler = Dopplers(kk);
            end
            rxValidationLabelkk = Doppler2BoundLabel(rxValidationLabel, Doppler, type);
            rxTestLabelkk = Doppler2BoundLabel(rxTestLabel, Doppler, type);
            rxTrainingLabelkk = Doppler2BoundLabel(rxTrainingLabel, Doppler, type);
        end
        
        div = 4;
        seqTrainingdiv = cell(numTraining/div, 1);
        seqValidationdiv = cell(numValidation/div, 1);

        for window = winstart:winend
            for trial = 1:div  % split data into div chunks to train div times due to out-of-memory err
                startTraining = (trial - 1) * numTraining/div + 1;
                stopTraining = trial * numTraining/div;
                seqIndex = zeros(WINDOWS(window), 1);
                rxTrainingLabelkkdiv = rxTrainingLabelkk(startTraining:stopTraining);
                rxTrainingdiv = rxTraining(:, :, :, startTraining:stopTraining);
                for i = 1:numTraining/div
                    seqTrainingdiv{i, 1} = zeros(1, spf, 2, WINDOWS(window));
                    cnt = 1;
                    seqIndex(cnt) = i;
                    for j = i+1:numTraining/div
                        if cnt == WINDOWS(window)
                            break
                        end
                        if rxTrainingLabelkkdiv(j) == rxTrainingLabelkkdiv(i)
                            cnt = cnt + 1;
                            seqIndex(cnt) = j;
                        end
                    end
                    if cnt ~= WINDOWS(window)
                        for j = 1:numTraining/div
                            if cnt == WINDOWS(window)
                                break
                            end
                            if rxTrainingLabelkkdiv(j) == rxTrainingLabelkkdiv(i)
                                cnt = cnt + 1;
                                seqIndex(cnt) = j;
                            end
                        end
                    end   
                    for k = 1:WINDOWS(window)
                        seqTrainingdiv{i, 1}(:, :, :, k) = rxTrainingdiv(:, :, :, seqIndex(k));
                    end
                end

                startValidation = (trial - 1) * numValidation/div + 1;
                stopValidation = trial * numValidation/div;
                seqIndex = zeros(WINDOWS(window), 1);
                rxValidationLabelkkdiv = rxValidationLabelkk(startValidation:stopValidation);
                rxValidationdiv = rxValidation(:, :, :, startValidation:stopValidation);
                for i = 1:numValidation/div
                    seqValidationdiv{i, 1} = zeros(1, spf, 2, WINDOWS(window));
                    cnt = 1;
                    seqIndex(cnt) = i;
                    for j = i+1:numValidation/div
                        if cnt == WINDOWS(window)
                            break
                        end
                        if rxValidationLabelkkdiv(j) == rxValidationLabelkkdiv(i)
                            cnt = cnt + 1;
                            seqIndex(cnt) = j;
                        end
                    end
                    if cnt ~= WINDOWS(window)
                        for j = 1:numValidation/div
                            if cnt == WINDOWS(window)
                                break
                            end
                            if rxValidationLabelkkdiv(j) == rxValidationLabelkkdiv(i)
                                cnt = cnt + 1;
                                seqIndex(cnt) = j;
                            end
                        end
                    end   
                    for k = 1:WINDOWS(window)
                        seqValidationdiv{i, 1}(:, :, :, k) = rxValidationdiv(:, :, :, seqIndex(k));
                    end
                end
             
%                [seqTraining, seqValidation, seqTest]=create_sequence(window, WINDOWS, spf, rxTraining, rxTrainingLabelkk, rxValidation, rxValidationLabelkk, rxTest, rxTestLabelkk); 
               options = trainingOptions('sgdm', ...
              'InitialLearnRate',2e-2, ...
              'MaxEpochs',maxEpochs, ...
              'MiniBatchSize',miniBatchSize, ...
              'Shuffle','every-epoch', ... 
              'Verbose',false, ...
              'ValidationPatience', 5, ...
              'ValidationData',{seqValidationdiv,rxValidationLabelkkdiv}, ...
              'ValidationFrequency',validationFrequency, ...
              'LearnRateSchedule', 'piecewise', ...
              'LearnRateDropPeriod', 9, ...
              'LearnRateDropFactor', 0.1, ...
              'ExecutionEnvironment', 'cpu');
            % analyzeNetwork(modClassNet{kk});

                trainNow = true;
                if trainNow == true
%                     disp("trial: " + string(trial));
                    if trial == 1
                        cnnLayers = layerGraph(modClassNet);
                        lgraph = connectLayers(cnnLayers,"fold/miniBatchSize","unfold/miniBatchSize");
                        [trainedNetS, info] = trainNetwork(seqTrainingdiv,rxTrainingLabelkkdiv,lgraph,options);
                        %disp("validation accuracy: " + string(info.FinalValidationAccuracy));
                    else
                        [trainedNetS, info] = trainNetwork(seqTrainingdiv,rxTrainingLabelkkdiv,layerGraph(trainedNetS),options);
                        %disp("validation accuracy: " + string(info.FinalValidationAccuracy));
                    end
                else
                %   load trainedModulationClassificationNetwork
                end
            end
            
            seqTest = cell(numTest, 1);
            seqIndex = zeros(WINDOWS(window), 1);
            for i = 1:numTest
                seqTest{i, 1} = zeros(1, spf, 2, WINDOWS(window));
                cnt = 1;
                seqIndex(cnt) = i;
                for j = i+1:numTest
                    if cnt == WINDOWS(window)
                        break
                    end
                    if rxTestLabelkk(j) == rxTestLabelkk(i)
                        cnt = cnt + 1;
                        seqIndex(cnt) = j;
                    end
                end
                if cnt ~= WINDOWS(window)
                    for j = 1:numTest
                        if cnt == WINDOWS(window)
                            break
                        end
                        if rxTestLabelkk(j) == rxTestLabelkk(i)
                            cnt = cnt + 1;
                            seqIndex(cnt) = j;
                        end
                    end
                end   
                for k = 1:WINDOWS(window)
                    seqTest{i, 1}(:, :, :, k) = rxTest(:, :, :, seqIndex(k));
                end
            end
    
            rxTestPred = classify(trainedNetS, seqTest);
            rxTestPredAll{window, kk} = rxTestPred;
            
            if scheme == 1  % binary prediction   
                nSEB = 0;
                testAccuracyI = 100 * mean(rxTestPred == rxTestLabelkk);
                for i = 1:numTest
                    label = DopplerTypes2num(rxTestLabel(i));
                    pred = bestFit(boundary(kk), rxTestPred(i), label);
                    if label == 0
                        if pred == 0
                            nSEB = nSEB + 10^-5;  % normalized square error
                        else
                            nSEB = nSEB + (pred - label) ^ 2 / pred ^ 2;
                        end
                    else 
                        nSEB = nSEB + (pred - label) ^ 2 / label ^ 2;
                    end                    
                end
                nMSEB = nSEB / numTest;
                disp("Window: " + WINDOWS(window) +" Doppler: "+boundary(kk)+" AccuracyI: "+testAccuracyI+" nMSEB: "+nMSEB);                
            else
                % Single-CNN classifier accuracy
                rxTestPredS = zeros(1.5 * numTestperDop, numDop); %the merged predict = bottom
                sum_square_errorS = zeros(numDop, 1);
                sumAccS = zeros(numDop, 1);
                cntAcc = zeros(numDop, 1);
                for i = 1:numTest
                    pred = DopplerTypes2num(rxTestPred(i));
                    label = DopplerTypes2num(rxTestLabel(i));
                    index = find(Dopplers == label);
                    sumAccS(index) = sumAccS(index) + double((label == pred));
                    cntAcc(index) = cntAcc(index) + 1;
                    rxTestPredS(cntAcc(index), index) = pred;
                    sum_square_errorS(index) = sum_square_errorS(index) + (pred - label) ^ 2;
                end
                for i = 1:numDop
                    MeanTestPredS = mean(rxTestPredS(1:cntAcc(i), i));
                    if Dopplers(i) == 0
                        if MeanTestPredS == 0
                            nMSES = 10^-5;
                        else
                            nMSES = (sum_square_errorS(i) / cntAcc(i)) / (MeanTestPredS ^ 2);
                        end
                    else 
                        nMSES = (sum_square_errorS(i) / cntAcc(i)) / (Dopplers(i) ^ 2);
                    end
                    testAccuracyS = 100 * sumAccS(i) / cntAcc(i);
                    disp("Window: " + WINDOWS(window) + " Doppler: "+Dopplers(i)+" AccuracyS: "+testAccuracyS+" MeanPredS: "+MeanTestPredS+" nMSES: "+nMSES);
                end            

%                 numTest = length(rxTestLabelkk);
%                 sum_square_errorS = 0;                
%                 for i = 1:numTest
%                     index = find(Dopplers == DopplerTypes2num(rxTestLabelkk(i)));
%                     sumAcc(index) = sumAcc(index) + double((DopplerTypes2num(rxTestLabelkk(i)) == DopplerTypes2num(rxTestPred(i))));
%                     cntAcc(index) = cntAcc(index) + 1;
%                     sum_square_errorS = sum_square_errorS + (label - rxTestPred(i)) ^ 2;                                   
%                 end
%                 for i = 1:numDop
%                     testAccuracyS = 100 * sumAcc(i) / cntAcc(i);
%                     disp("Swindow: " + window + " Doppler: " + i + " Accuracy: " + testAccuracyS);
%                 end
% %                 disp(" window: " + window + "Mean accuracyS: " + mean(testAccuracyS));
            end
            
        end

  end    
%     save('../../../work/bal718/NR5G_Doppler_predAll_' + string(mode) + '_' + string(type) + '_' + string(domain) + string(scheme) + '.mat',  'rxTestPredAll', 'rxTestLabel', '-v7.3');
    save('NR5G_Doppler_predAll_' + string(mode) + '_' + string(type) + '_' + string(domain) + string(scheme) + '.mat',  'rxTestPredAll', 'rxTestLabel', '-v7.3');
end

if mode > 0  % 5,10,15: pred & compute merged statistics; 105: statistics for 5, 11-: for 10, 115: for 15
    if mode > 100
        mode = mode - 100;
        load('../../../work/bal718/NR5G_Doppler_predAll_' + string(mode) + '_' + string(type) + '_' + string(domain) + string(scheme) + '.mat', 'rxTestPredAll', 'rxTestLabel');
    end
    disp('merged prediction');
    numTestperDop = numTest/numDop;
    for w = winstart:winend
        cntAcc = zeros(numDop, 1);        
        if scheme == 0  % multi-class 
            rxTestPredS = zeros(1.5 * numTestperDop, numDop); %the merged predict = bottom
            sum_square_errorS = zeros(numDop, 1);
            rxTestPred = rxTestPredAll{w, 1};
            sumAccS = zeros(numDop, 1);
            for i = 1:numTest
                pred = DopplerTypes2num(rxTestPred(i));
                label = DopplerTypes2num(rxTestLabel(i));
                index = find(Dopplers == label);
                sumAccS(index) = sumAccS(index) + double((label == pred));
                cntAcc(index) = cntAcc(index) + 1;
                rxTestPredS(cntAcc(index), index) = pred;
                sum_square_errorS(index) = sum_square_errorS(index) + (pred - label) ^ 2;
            end
        elseif type == 0 % less/greater  
            sumAccM = zeros(numDop, 1); 
            sum_square_errorM = zeros(numDop, 1);   
            rxTestPredM = zeros(1.5 * numTestperDop, numDop); % single merged prediction   
            sumAccMR = zeros(numDop, 1); 
            sum_square_errorMR = zeros(numDop, 1);   
            rxTestPredMR = zeros(1.5 * numTestperDop, numDop); %the best in merged range of bottom and top   
            
            rxTestPredB = rxTestLabel; % the best fit in the individual predicted range            
            for kk = 1:numkk
                nSEB = 0;               
                rxTestPred = rxTestPredAll{w, kk};
                for i = 1:numTest
                    label = DopplerTypes2num(rxTestLabel(i));
                    pred = bestFit(boundary(kk), rxTestPred(i), label);
                    rxTestPredB(i) = categorical(pred);
                    if label == 0
                        if pred == 0
                            nSEB = nSEB + 10^-5;  % normalized square error
                        else
                            nSEB = nSEB + (pred - label) ^ 2 / pred ^ 2;
                        end
                    else 
                        nSEB = nSEB + (pred - label) ^ 2 / label ^ 2;
                    end                    
                end
                testAccuracyB = 100 * mean(rxTestPredB == rxTestLabel);
                nMSEB = nSEB / numTest;
                disp("Window: " + WINDOWS(w) +" Doppler: "+boundary(kk)+" AccuracyB: "+testAccuracyB+" nMSEB: "+nMSEB);                
            end

            for i = 1:numTest
                top = 9999;
                bottom = -9999;       
                label = DopplerTypes2num(rxTestLabel(i));
                for kk = 1:numkk
                    rxTestPred = rxTestPredAll{w, kk};
                    if rxTestPred(i) == categorical(-1)
                        top = boundary(kk);
                        if top == min(boundary)
                            bottom = min(Dopplers);  % is the merged single pred target
                        else
                            bottom = boundary(kk-1);
                        end
                        break;
                    end
                end
                if top == 9999
                    top = max(Dopplers);
                    bottom = max(Dopplers);
                end
                index = find(Dopplers == label);
                cntAcc(index) = cntAcc(index) + 1;
                rxTestPredM(cntAcc(index), index) = bottom;
                sumAccM(index) = sumAccM(index) + double(label == bottom);
                sum_square_errorM(index) = sum_square_errorM(index) + (label - bottom) ^ 2;                
                sumAccMR(index) = sumAccMR(index) + double((label == bottom) || (label == top));
                if abs(label - top) > abs(label - bottom)
                   rxTestPredMR(cntAcc(index), index) = bottom;
                   sum_square_errorMR(index) = sum_square_errorMR(index) + (label - bottom) ^ 2;
                else
                   rxTestPredMR(cntAcc(index), index) = top;
                   sum_square_errorMR(index) = sum_square_errorMR(index) + (label - top) ^ 2;
                end
            end
%         else % for equal comparison - NOT GOOD
%             for i = 1:numTest
%                 pred = Dopplers(1) - 2; %when there is no equal=> mismatch
%                 for kk = 1:numkk
%                     rxTestPred = rxTestPredAll{w, kk};
%                     if rxTestPred(i) == categorical(-1)
%                         pred = Dopplers(kk);
%                         break;
%                     end
%                 end
%                 index = find(Dopplers == DopplerTypes2num(rxTestLabel(i)));
%                 sumAcc(index) = sumAcc(index) + double((DopplerTypes2num(rxTestLabel(i)) == pred));
%                 cntAcc(index) = cntAcc(index) + 1;
%                 rxTestPredM(cntAcc(index), index) = pred;
%                 sum_square_error(index) = sum_square_error(index) + (bottom - Dopplers(index)) ^ 2;
%             end
        end
        if scheme > 0  % binary
            for i = 1:numDop
                MeanTestPredM = mean(rxTestPredM(1:cntAcc(i), i));
                MeanTestPredMR = mean(rxTestPredMR(1:cntAcc(i), i));
                if Dopplers(i) == 0
                    if MeanTestPredM == 0
                        nMSEM = 10^-5;
                        nMSEMR = 10^-5;
                    else
                        nMSEM = (sum_square_errorM(i) / cntAcc(i)) / (MeanTestPredM ^ 2);
                        nMSEMR = (sum_square_errorMR(i) / cntAcc(i)) / (MeanTestPredMR ^ 2);
                    end
                else 
                    nMSEM = (sum_square_errorM(i) / cntAcc(i)) / (Dopplers(i) ^ 2);
                    nMSEMR = (sum_square_errorMR(i) / cntAcc(i)) / (Dopplers(i) ^ 2);
                end

                testAccuracyM = 100 * sumAccM(i) / cntAcc(i);
                testAccuracyMR = 100 * sumAccMR(i) / cntAcc(i);
                disp("Window: " + WINDOWS(w) +" Doppler: "+Dopplers(i)+" AccuracyM: "+testAccuracyM+" MeanPredM: "+MeanTestPredM+" nMSEM: "+nMSEM+" AccuracyMR: "+testAccuracyMR+" MeanPredMR: "+MeanTestPredMR+" nMSEMR: "+nMSEMR);
            end
        else
            for i = 1:numDop
                MeanTestPredS = mean(rxTestPredS(1:cntAcc(i), i));
                if Dopplers(i) == 0
                    if MeanTestPredS == 0
                        nMSES = 10^-5;
                    else
                        nMSES = (sum_square_errorS(i) / cntAcc(i)) / (MeanTestPredS ^ 2);
                    end
                else 
                    nMSES = (sum_square_errorS(i) / cntAcc(i)) / (Dopplers(i) ^ 2);
                end

                testAccuracyS = 100 * sumAccS(i) / cntAcc(i);
                disp("Window: " + WINDOWS(w) +" Doppler: "+Dopplers(i)+" AccuracyS: "+testAccuracyS+" MeanPredS: "+MeanTestPredS+" nMSES: "+nMSES);
            end            
        end
    end

end
% NR5G_Doppler
% mode=5 type=0 domain=1 scheme=0 numkk=1 numData=12000
% predicting
% Window: 1 Doppler: 0 AccuracyS: 1.0204 MeanPredS: 185.7143 nMSES: 1.6627
% Window: 1 Doppler: 50 AccuracyS: 43.3333 MeanPredS: 203.3333 nMSES: 18.9778
% Window: 1 Doppler: 100 AccuracyS: 0.90909 MeanPredS: 203.1818 nMSES: 3.6477
% Window: 1 Doppler: 150 AccuracyS: 7 MeanPredS: 221.5 nMSES: 1.3856
% Window: 1 Doppler: 200 AccuracyS: 0 MeanPredS: 217.8571 nMSES: 0.54911
% Window: 1 Doppler: 250 AccuracyS: 6.5217 MeanPredS: 245.1087 nMSES: 0.3387
% Window: 1 Doppler: 300 AccuracyS: 49.4737 MeanPredS: 260.5263 nMSES: 0.24181
% Window: 1 Doppler: 350 AccuracyS: 6.4516 MeanPredS: 262.3656 nMSES: 0.25258
% Window: 1 Doppler: 400 AccuracyS: 13.1313 MeanPredS: 281.8182 nMSES: 0.26641
% Window: 1 Doppler: 450 AccuracyS: 0 MeanPredS: 336.3208 nMSES: 0.21791
% Window: 1 Doppler: 500 AccuracyS: 2.9412 MeanPredS: 313.7255 nMSES: 0.27765
% Window: 1 Doppler: 550 AccuracyS: 30.0971 MeanPredS: 313.1068 nMSES: 0.3025
% Window: 2 Doppler: 0 AccuracyS: 32.6531 MeanPredS: 167.3469 nMSES: 1.8583
% Window: 2 Doppler: 50 AccuracyS: 20 MeanPredS: 189.4444 nMSES: 17.5
% Window: 2 Doppler: 100 AccuracyS: 15.4545 MeanPredS: 223.6364 nMSES: 3.9136
% Window: 2 Doppler: 150 AccuracyS: 7 MeanPredS: 197 nMSES: 0.8
% Window: 2 Doppler: 200 AccuracyS: 25.8929 MeanPredS: 225.4464 nMSES: 0.36998
% Window: 2 Doppler: 250 AccuracyS: 13.0435 MeanPredS: 279.3478 nMSES: 0.28696
% Window: 2 Doppler: 300 AccuracyS: 14.7368 MeanPredS: 313.6842 nMSES: 0.10351
% Window: 2 Doppler: 350 AccuracyS: 8.6022 MeanPredS: 330.1075 nMSES: 0.14373
% Window: 2 Doppler: 400 AccuracyS: 39.3939 MeanPredS: 325.7576 nMSES: 0.12042
% Window: 2 Doppler: 450 AccuracyS: 6.6038 MeanPredS: 400.9434 nMSES: 0.073841
% Window: 2 Doppler: 500 AccuracyS: 9.8039 MeanPredS: 377.451 nMSES: 0.13431
% Window: 2 Doppler: 550 AccuracyS: 19.4175 MeanPredS: 370.3883 nMSES: 0.1892
% Window: 3 Doppler: 0 AccuracyS: 37.7551 MeanPredS: 138.7755 nMSES: 2.575
% Window: 3 Doppler: 50 AccuracyS: 23.3333 MeanPredS: 178.8889 nMSES: 16.3556
% Window: 3 Doppler: 100 AccuracyS: 10 MeanPredS: 199.0909 nMSES: 2.8045
% Window: 3 Doppler: 150 AccuracyS: 10 MeanPredS: 226 nMSES: 0.95556
% Window: 3 Doppler: 200 AccuracyS: 47.3214 MeanPredS: 235.7143 nMSES: 0.35268
% Window: 3 Doppler: 250 AccuracyS: 5.4348 MeanPredS: 260.8696 nMSES: 0.26261
% Window: 3 Doppler: 300 AccuracyS: 30.5263 MeanPredS: 295.7895 nMSES: 0.14327
% Window: 3 Doppler: 350 AccuracyS: 15.0538 MeanPredS: 304.8387 nMSES: 0.16063
% Window: 3 Doppler: 400 AccuracyS: 22.2222 MeanPredS: 367.6768 nMSES: 0.082702
% Window: 3 Doppler: 450 AccuracyS: 22.6415 MeanPredS: 394.3396 nMSES: 0.13114
% Window: 3 Doppler: 500 AccuracyS: 6.8627 MeanPredS: 385.7843 nMSES: 0.16382
% Window: 3 Doppler: 550 AccuracyS: 19.4175 MeanPredS: 381.068 nMSES: 0.17781
% Window: 4 Doppler: 0 AccuracyS: 6.1224 MeanPredS: 163.7755 nMSES: 1.7034
% Window: 4 Doppler: 50 AccuracyS: 34.4444 MeanPredS: 166.6667 nMSES: 14.8444
% Window: 4 Doppler: 100 AccuracyS: 29.0909 MeanPredS: 127.7273 nMSES: 1.1977
% Window: 4 Doppler: 150 AccuracyS: 10 MeanPredS: 197.5 nMSES: 0.88556
% Window: 4 Doppler: 200 AccuracyS: 25 MeanPredS: 213.8393 nMSES: 0.24498
% Window: 4 Doppler: 250 AccuracyS: 42.3913 MeanPredS: 278.8043 nMSES: 0.20913
% Window: 4 Doppler: 300 AccuracyS: 4.2105 MeanPredS: 307.3684 nMSES: 0.11111
% Window: 4 Doppler: 350 AccuracyS: 22.5806 MeanPredS: 318.8172 nMSES: 0.13298
% Window: 4 Doppler: 400 AccuracyS: 57.5758 MeanPredS: 367.6768 nMSES: 0.090593
% Window: 4 Doppler: 450 AccuracyS: 15.0943 MeanPredS: 408.4906 nMSES: 0.07058
% Window: 4 Doppler: 500 AccuracyS: 30.3922 MeanPredS: 422.0588 nMSES: 0.092059
% Window: 4 Doppler: 550 AccuracyS: 9.7087 MeanPredS: 399.0291 nMSES: 0.15205
% Window: 5 Doppler: 0 AccuracyS: 35.7143 MeanPredS: 65.3061 nMSES: 2.8831
% Window: 5 Doppler: 50 AccuracyS: 27.7778 MeanPredS: 118.3333 nMSES: 9.5
% Window: 5 Doppler: 100 AccuracyS: 21.8182 MeanPredS: 160.9091 nMSES: 1.4364
% Window: 5 Doppler: 150 AccuracyS: 10 MeanPredS: 171 nMSES: 0.46222
% Window: 5 Doppler: 200 AccuracyS: 47.3214 MeanPredS: 209.8214 nMSES: 0.16183
% Window: 5 Doppler: 250 AccuracyS: 39.1304 MeanPredS: 238.0435 nMSES: 0.1513
% Window: 5 Doppler: 300 AccuracyS: 17.8947 MeanPredS: 302.1053 nMSES: 0.1076
% Window: 5 Doppler: 350 AccuracyS: 12.9032 MeanPredS: 311.828 nMSES: 0.12399
% Window: 5 Doppler: 400 AccuracyS: 22.2222 MeanPredS: 372.2222 nMSES: 0.10653
% Window: 5 Doppler: 450 AccuracyS: 13.2075 MeanPredS: 412.7358 nMSES: 0.1103
% Window: 5 Doppler: 500 AccuracyS: 23.5294 MeanPredS: 419.1176 nMSES: 0.096961
% Window: 5 Doppler: 550 AccuracyS: 14.5631 MeanPredS: 425.2427 nMSES: 0.095884
% Window: 10 Doppler: 0 AccuracyS: 30.6122 MeanPredS: 61.7347 nMSES: 2.0148
% Window: 10 Doppler: 50 AccuracyS: 47.7778 MeanPredS: 88.3333 nMSES: 3.7444
% Window: 10 Doppler: 100 AccuracyS: 39.0909 MeanPredS: 115.9091 nMSES: 0.43409
% Window: 10 Doppler: 150 AccuracyS: 14 MeanPredS: 175.5 nMSES: 0.14333
% Window: 10 Doppler: 200 AccuracyS: 26.7857 MeanPredS: 189.2857 nMSES: 0.11384
% Window: 10 Doppler: 250 AccuracyS: 53.2609 MeanPredS: 230.4348 nMSES: 0.07913
% Window: 10 Doppler: 300 AccuracyS: 40 MeanPredS: 295.2632 nMSES: 0.039474
% Window: 10 Doppler: 350 AccuracyS: 8.6022 MeanPredS: 326.8817 nMSES: 0.050691
% Window: 10 Doppler: 400 AccuracyS: 17.1717 MeanPredS: 376.7677 nMSES: 0.046717
% Window: 10 Doppler: 450 AccuracyS: 29.2453 MeanPredS: 456.1321 nMSES: 0.016888
% Window: 10 Doppler: 500 AccuracyS: 26.4706 MeanPredS: 437.7451 nMSES: 0.055588
% Window: 10 Doppler: 550 AccuracyS: 21.3592 MeanPredS: 446.6019 nMSES: 0.079676
% Window: 15 Doppler: 0 AccuracyS: 80.6122 MeanPredS: 19.3878 nMSES: 5.1579
% Window: 15 Doppler: 50 AccuracyS: 14.4444 MeanPredS: 69.4444 nMSES: 1.0333
% Window: 15 Doppler: 100 AccuracyS: 30.9091 MeanPredS: 81.8182 nMSES: 0.28182
% Window: 15 Doppler: 150 AccuracyS: 42 MeanPredS: 146.5 nMSES: 0.13
% Window: 15 Doppler: 200 AccuracyS: 14.2857 MeanPredS: 144.6429 nMSES: 0.16964
% Window: 15 Doppler: 250 AccuracyS: 44.5652 MeanPredS: 249.4565 nMSES: 0.052609
% Window: 15 Doppler: 300 AccuracyS: 21.0526 MeanPredS: 274.7368 nMSES: 0.082456
% Window: 15 Doppler: 350 AccuracyS: 30.1075 MeanPredS: 306.4516 nMSES: 0.10643
% Window: 15 Doppler: 400 AccuracyS: 16.1616 MeanPredS: 431.8182 nMSES: 0.028567
% Window: 15 Doppler: 450 AccuracyS: 11.3208 MeanPredS: 467.9245 nMSES: 0.03331
% Window: 15 Doppler: 500 AccuracyS: 17.6471 MeanPredS: 467.1569 nMSES: 0.052255
% Window: 15 Doppler: 550 AccuracyS: 16.5049 MeanPredS: 427.1845 nMSES: 0.13215

% NR5G_Doppler
% mode=5 type=0 domain=0 scheme=1 numkk=11 numData=12000
% predicting
% Window: 1 Doppler: 50 AccuracyI: 92 nMSEB: 0.078725
% Window: 2 Doppler: 50 AccuracyI: 91.75 nMSEB: 0.075903
% Window: 3 Doppler: 50 AccuracyI: 92.8333 nMSEB: 0.069427
% Window: 4 Doppler: 50 AccuracyI: 92.6667 nMSEB: 0.071671
% NR5G_Doppler
% mode=5 type=0 domain=0 scheme=0 numkk=1 numData=12000
% predicting
% Window: 1 Doppler: 0 AccuracyS: 28.5714 MeanPredS: 152.0408 nMSES: 2.0504
% Window: 1 Doppler: 50 AccuracyS: 7.7778 MeanPredS: 164.4444 nMSES: 14.6222
% Window: 1 Doppler: 100 AccuracyS: 14.5455 MeanPredS: 181.3636 nMSES: 2.9023
% Window: 1 Doppler: 150 AccuracyS: 21 MeanPredS: 231 nMSES: 1.24
% Window: 1 Doppler: 200 AccuracyS: 24.1071 MeanPredS: 236.6071 nMSES: 0.39063
% Window: 1 Doppler: 250 AccuracyS: 22.8261 MeanPredS: 252.1739 nMSES: 0.27391
% Window: 1 Doppler: 300 AccuracyS: 14.7368 MeanPredS: 316.8421 nMSES: 0.25673
% Window: 1 Doppler: 350 AccuracyS: 15.0538 MeanPredS: 336.0215 nMSES: 0.15668
% Window: 1 Doppler: 400 AccuracyS: 6.0606 MeanPredS: 389.3939 nMSES: 0.098643
% Window: 1 Doppler: 450 AccuracyS: 33.0189 MeanPredS: 440.0943 nMSES: 0.063242
% Window: 1 Doppler: 500 AccuracyS: 13.7255 MeanPredS: 453.9216 nMSES: 0.054118
% Window: 1 Doppler: 550 AccuracyS: 46.6019 MeanPredS: 476.2136 nMSES: 0.051352
% Window: 2 Doppler: 0 AccuracyS: 24.4898 MeanPredS: 103.0612 nMSES: 2.0895
% Window: 2 Doppler: 50 AccuracyS: 15.5556 MeanPredS: 108.3333 nMSES: 4.5667
% Window: 2 Doppler: 100 AccuracyS: 15.4545 MeanPredS: 150.9091 nMSES: 0.84545
% Window: 2 Doppler: 150 AccuracyS: 35 MeanPredS: 179 nMSES: 0.38444
% Window: 2 Doppler: 200 AccuracyS: 23.2143 MeanPredS: 204.0179 nMSES: 0.16908
% Window: 2 Doppler: 250 AccuracyS: 14.1304 MeanPredS: 234.7826 nMSES: 0.14261
% Window: 2 Doppler: 300 AccuracyS: 33.6842 MeanPredS: 318.4211 nMSES: 0.12251
% Window: 2 Doppler: 350 AccuracyS: 9.6774 MeanPredS: 327.957 nMSES: 0.13496
% Window: 2 Doppler: 400 AccuracyS: 25.2525 MeanPredS: 386.8687 nMSES: 0.059028
% Window: 2 Doppler: 450 AccuracyS: 27.3585 MeanPredS: 434.9057 nMSES: 0.052644
% Window: 2 Doppler: 500 AccuracyS: 13.7255 MeanPredS: 444.6078 nMSES: 0.043235
% Window: 2 Doppler: 550 AccuracyS: 35.9223 MeanPredS: 477.6699 nMSES: 0.038594
% Window: 3 Doppler: 0 AccuracyS: 28.5714 MeanPredS: 98.4694 nMSES: 2.2231
% Window: 3 Doppler: 50 AccuracyS: 20 MeanPredS: 112.2222 nMSES: 5.1111
% Window: 3 Doppler: 100 AccuracyS: 24.5455 MeanPredS: 112.7273 nMSES: 0.95455
% Window: 3 Doppler: 150 AccuracyS: 14 MeanPredS: 153.5 nMSES: 0.48333
% Window: 3 Doppler: 200 AccuracyS: 33.9286 MeanPredS: 195.9821 nMSES: 0.17467
% Window: 3 Doppler: 250 AccuracyS: 15.2174 MeanPredS: 241.8478 nMSES: 0.14391
% Window: 3 Doppler: 300 AccuracyS: 25.2632 MeanPredS: 318.4211 nMSES: 0.088012
% Window: 3 Doppler: 350 AccuracyS: 13.9785 MeanPredS: 329.0323 nMSES: 0.092385
% Window: 3 Doppler: 400 AccuracyS: 38.3838 MeanPredS: 384.3434 nMSES: 0.041824
% Window: 3 Doppler: 450 AccuracyS: 17.9245 MeanPredS: 436.3208 nMSES: 0.031563
% Window: 3 Doppler: 500 AccuracyS: 15.6863 MeanPredS: 448.0392 nMSES: 0.043529
% Window: 3 Doppler: 550 AccuracyS: 37.8641 MeanPredS: 472.3301 nMSES: 0.039798
% Window: 4 Doppler: 0 AccuracyS: 24.4898 MeanPredS: 103.0612 nMSES: 1.662
% Window: 4 Doppler: 50 AccuracyS: 11.1111 MeanPredS: 126.6667 nMSES: 6.0889
% Window: 4 Doppler: 100 AccuracyS: 6.3636 MeanPredS: 148.1818 nMSES: 1.4273
% Window: 4 Doppler: 150 AccuracyS: 23 MeanPredS: 215.5 nMSES: 0.67667
% Window: 4 Doppler: 200 AccuracyS: 30.3571 MeanPredS: 210.7143 nMSES: 0.11719
% Window: 4 Doppler: 250 AccuracyS: 28.2609 MeanPredS: 270.1087 nMSES: 0.13957
% Window: 4 Doppler: 300 AccuracyS: 26.3158 MeanPredS: 314.7368 nMSES: 0.10058
% Window: 4 Doppler: 350 AccuracyS: 26.8817 MeanPredS: 348.9247 nMSES: 0.081194
% Window: 4 Doppler: 400 AccuracyS: 39.3939 MeanPredS: 404.5455 nMSES: 0.026357
% Window: 4 Doppler: 450 AccuracyS: 10.3774 MeanPredS: 456.6038 nMSES: 0.034009
% Window: 4 Doppler: 500 AccuracyS: 29.4118 MeanPredS: 480.8824 nMSES: 0.024216
% Window: 4 Doppler: 550 AccuracyS: 47.5728 MeanPredS: 492.7184 nMSES: 0.027602
% Window: 5 Doppler: 0 AccuracyS: 23.4694 MeanPredS: 83.1633 nMSES: 2.1578
% Window: 5 Doppler: 50 AccuracyS: 30 MeanPredS: 102.2222 nMSES: 4.6444
% Window: 5 Doppler: 100 AccuracyS: 13.6364 MeanPredS: 138.1818 nMSES: 1.1364
% Window: 5 Doppler: 150 AccuracyS: 13 MeanPredS: 168.5 nMSES: 0.30556
% Window: 5 Doppler: 200 AccuracyS: 31.25 MeanPredS: 207.1429 nMSES: 0.19978
% Window: 5 Doppler: 250 AccuracyS: 31.5217 MeanPredS: 240.2174 nMSES: 0.081739
% Window: 5 Doppler: 300 AccuracyS: 33.6842 MeanPredS: 276.8421 nMSES: 0.069006
% Window: 5 Doppler: 350 AccuracyS: 29.0323 MeanPredS: 324.7312 nMSES: 0.064297
% Window: 5 Doppler: 400 AccuracyS: 32.3232 MeanPredS: 372.2222 nMSES: 0.031723
% Window: 5 Doppler: 450 AccuracyS: 15.0943 MeanPredS: 450.9434 nMSES: 0.044025
% Window: 5 Doppler: 500 AccuracyS: 22.549 MeanPredS: 459.8039 nMSES: 0.047843
% Window: 5 Doppler: 550 AccuracyS: 49.5146 MeanPredS: 487.8641 nMSES: 0.037712
% Window: 10 Doppler: 0 AccuracyS: 23.4694 MeanPredS: 67.3469 nMSES: 1.4286
% Window: 10 Doppler: 50 AccuracyS: 12.2222 MeanPredS: 73.8889 nMSES: 1.2111
% Window: 10 Doppler: 100 AccuracyS: 36.3636 MeanPredS: 99.0909 nMSES: 0.42727
% Window: 10 Doppler: 150 AccuracyS: 16 MeanPredS: 136.5 nMSES: 0.18556
% Window: 10 Doppler: 200 AccuracyS: 34.8214 MeanPredS: 219.6429 nMSES: 0.069196
% Window: 10 Doppler: 250 AccuracyS: 67.3913 MeanPredS: 253.8043 nMSES: 0.14913
% Window: 10 Doppler: 300 AccuracyS: 29.4737 MeanPredS: 272.1053 nMSES: 0.040643
% Window: 10 Doppler: 350 AccuracyS: 10.7527 MeanPredS: 322.5806 nMSES: 0.032697
% Window: 10 Doppler: 400 AccuracyS: 52.5253 MeanPredS: 400.5051 nMSES: 0.019413
% Window: 10 Doppler: 450 AccuracyS: 42.4528 MeanPredS: 429.717 nMSES: 0.022013
% Window: 10 Doppler: 500 AccuracyS: 20.5882 MeanPredS: 460.7843 nMSES: 0.029608
% Window: 10 Doppler: 550 AccuracyS: 35.9223 MeanPredS: 482.0388 nMSES: 0.040761
% Window: 15 Doppler: 0 AccuracyS: 21.4286 MeanPredS: 79.0816 nMSES: 1.7091
% Window: 15 Doppler: 50 AccuracyS: 28.8889 MeanPredS: 58.3333 nMSES: 1.3444
% Window: 15 Doppler: 100 AccuracyS: 3.6364 MeanPredS: 122.2727 nMSES: 1.1886
% Window: 15 Doppler: 150 AccuracyS: 18 MeanPredS: 148 nMSES: 0.26
% Window: 15 Doppler: 200 AccuracyS: 33.9286 MeanPredS: 188.8393 nMSES: 0.14453
% Window: 15 Doppler: 250 AccuracyS: 30.4348 MeanPredS: 259.7826 nMSES: 0.1
% Window: 15 Doppler: 300 AccuracyS: 14.7368 MeanPredS: 326.3158 nMSES: 0.094152
% Window: 15 Doppler: 350 AccuracyS: 31.1828 MeanPredS: 328.4946 nMSES: 0.025455
% Window: 15 Doppler: 400 AccuracyS: 30.303 MeanPredS: 380.303 nMSES: 0.035196
% Window: 15 Doppler: 450 AccuracyS: 1.8868 MeanPredS: 415.566 nMSES: 0.028302
% Window: 15 Doppler: 500 AccuracyS: 15.6863 MeanPredS: 481.8627 nMSES: 0.020882
% Window: 15 Doppler: 550 AccuracyS: 54.3689 MeanPredS: 497.5728 nMSES: 0.025355
% merged prediction
% Unrecognized function or variable 'sumAcc'.
% 
% Error in NR5G_Doppler (line 421)
%                 sumAccS(index) = sumAcc(index) + double((label == pred));
%  

% function [Data,Labels] = hGenerateTrainingData(dataSize, Dopplers, boundary, kk)
function [Data,Labels] = hGenerateTrainingData(dataSize, Dopplers, domain)
% Generate training data examples for channel estimation
% Run dataSize number of iterations to create random channel configurations
% and pass an OFDM-modulated fixed PDSCH grid with only the DM-RS symbols
% inserted. Perform perfect timing synchronization and OFDM demodulation,
% extracting the pilot symbols and performing linear interpolation at each
% iteration. Use perfect channel information to create the
% label data. The function returns 2 arrays - the training data and labels.

    fprintf('Starting data generation...\n')
    
    % Preallocate memory for the training data and labels
    spf_t = 15376;
    spf_f = 8568;
    spf_tf = 15376+8568;
    if domain == 0
        spf = spf_f;
    elseif domain == 1
        spf = spf_t;
    else
        spf = spf_tf;
    end
    numDopplerTypes = length(Dopplers);
    numExamples = dataSize * numDopplerTypes;
    Data = zeros([1 spf 2 numExamples]);
    Labels = categorical(zeros([numExamples, 1]));

    % List of possible channel profiles
    delayProfiles = {'TDL-A', 'TDL-B', 'TDL-C', 'TDL-D', 'TDL-E'};
    modTypes = {'QPSK', '16QAM', '64QAM'};       % '
    [gnb, pdsch] = hDeepLearningChanEstSimParameters();

    % Create the channel model object
    nTxAnts = gnb.NTxAnts;
    nRxAnts = gnb.NRxAnts;

    channel = nrTDLChannel; % TDL channel object
    channel.NumTransmitAntennas = nTxAnts;
    channel.NumReceiveAntennas = nRxAnts;

    % Use the value returned from <matlab:edit('hOFDMInfo') hOFDMInfo> to
    % set the the channel model sampling rate
    waveformInfo = hOFDMInfo(gnb);
    channel.SampleRate = waveformInfo.SamplingRate;

    % Get the maximum number of delayed samples by a channel multipath
    % component. This number is calculated from the channel path with the largest
    % delay and the implementation delay of the channel filter, and is required
    % to flush the channel filter to obtain the received signal.
    chInfo = info(channel);
    maxChDelay = ceil(max(chInfo.PathDelays*channel.SampleRate)) + chInfo.ChannelFilterDelay;

    index = 0;
    % Main loop for data generation, iterating over the number of examples
    % specified in the function call. Each iteration of the loop produces a
    % new channel realization with a random delay spread, doppler shift,
    % and delay profile. Every perturbed version of the transmitted
    % waveform with the DM-RS symbols is stored in trainData, and the
    % perfect channel realization in trainLabels.
    for Doppler_idx = 1:numDopplerTypes
      Doppler = DopplerTypes2num(Dopplers(Doppler_idx));             % Desired SNR in dB
      numSample = dataSize;
      for i = 1:numSample

        modulation = string(modTypes(randi([1 numel(modTypes)])));
        gnb.PDSCH.Modulation = modulation;
        pdsch.Modulation = modulation;

        numberOfBits = 612 * 14 * ModTypes2M(modulation);
        inputBits = randi([0 1], numberOfBits, 1);
        inputSym = lteSymbolModulate(inputBits, modulation);
        inputSym = reshape(inputSym, [612 14 1]);
        
        % Return DM-RS indices and symbols
        [~,dmrsIndices,dmrsSymbols,~] = hPDSCHResources(gnb,pdsch);

        % PDSCH mapping in grid associated with PDSCH transmission period
        pdschGrid = zeros(waveformInfo.NSubcarriers,waveformInfo.SymbolsPerSlot,nTxAnts);

        % PDSCH DM-RS precoding and mapping
        [~,dmrsAntIndices] = nrExtractResources(dmrsIndices,pdschGrid);
        pdschGrid = pdschGrid + inputSym;
%         pdschGrid(dmrsAntIndices) = pdschGrid(dmrsAntIndices) + dmrsSymbols;
        pdschGrid(dmrsAntIndices) = dmrsSymbols;

        % OFDM modulation of associated resource elements
        txWaveform_original = hOFDMModulate(gnb,pdschGrid);

        % Acquire linear interpolator coordinates for neural net preprocessing
    %     [rows,cols] = find(pdschGrid ~= 0);
    %     dmrsSubs = [rows, cols, ones(size(cols))];
    %     hest = zeros(size(pdschGrid));
    %     [l_hest,k_hest] = meshgrid(1:size(hest,2),1:size(hest,1));


    %     index = 0;
    %     % Main loop for data generation, iterating over the number of examples
    %     % specified in the function call. Each iteration of the loop produces a
    %     % new channel realization with a random delay spread, doppler shift,
    %     % and delay profile. Every perturbed version of the transmitted
    %     % waveform with the DM-RS symbols is stored in trainData, and the
    %     % perfect channel realization in trainLabels.
    %     for Doppler_idx = 1:numDopplerTypes
    %       Doppler = DopplerTypes2num(Dopplers(Doppler_idx));             % Desired SNR in dB
    %       numSample = dataSize;
    %       for i = 1:numSample
    % 
            % Release the channel to change nontunable properties
            channel.release

            % Pick a random seed to create different channel realizations
            channel.Seed = randi([1001 2000]);

            % Pick a random delay profile, delay spread, and maximum doppler shift
            channel.DelayProfile = string(delayProfiles(randi([1 numel(delayProfiles)])));
            channel.DelaySpread = randi([1 300])*1e-9;
            channel.MaximumDopplerShift = Doppler; %randi([5 400]);

            % Send data through the channel model. Append zeros at the end of
            % the transmitted waveform to flush channel content. These zeros
            % take into account any delay introduced in the channel, such as
            % multipath delay and implementation delay. This value depends on
            % the sampling rate, delay profile, and delay spread
            txWaveform = [txWaveform_original; zeros(maxChDelay, size(txWaveform_original,2))];
            [rxWaveform,pathGains,sampleTimes] = channel(txWaveform);

            % Add additive white Gaussian noise (AWGN) to the received time-domain
            % waveform. To take into account sampling rate, normalize the noise power.
           % The SNR is defined per RE for each receive antenna (3GPP TS 38.101-4).
            SNRdB = randi([-10 20]);  % Random SNR values between 0 and 10 dB
            SNR = 10^(SNRdB/20);    % Calculate linear noise gain
            N0 = 1/(sqrt(2.0*nRxAnts*double(waveformInfo.Nfft))*SNR);
            noise = N0*complex(randn(size(rxWaveform)),randn(size(rxWaveform)));
            rxWaveform = rxWaveform + noise;

            % Perform OFDM demodulation on the received data to recreate the
            % resource grid, including padding in case practical
            % synchronization results in an incomplete slot being demodulated
            rxGrid = hOFDMDemodulate(gnb,rxWaveform);
            [K,L,R] = size(rxGrid);
            if (L < waveformInfo.SymbolsPerSlot)
                rxGrid = cat(2,rxGrid,zeros(K,waveformInfo.SymbolsPerSlot-L,R));
            end

            % Linear interpolation
    %         dmrsRx = rxGrid(dmrsIndices);
    %         dmrsEsts = dmrsRx .* conj(dmrsSymbols);
    %         f = scatteredInterpolant(dmrsSubs(:,2),dmrsSubs(:,1),dmrsEsts);
    %         hest = f(l_hest,k_hest);

            % Split interpolated grid into real and imaginary components and
            % concatenate them along the third dimension, as well as for the
            % true channel response
    %         rx_grid = cat(3, real(hest), imag(hest));
            rx_grid = cat(3, real(rxGrid), imag(rxGrid));

            % Add generated training example and label to the respective arrays
            index = index + 1;
            subframe_t = reshape(cat(2, real(rxWaveform), imag(rxWaveform)), [1 spf_t 2]);
            subframe_f = reshape(rx_grid, [1 spf_f 2]);
            if domain == 0
                Data(:,:,:, index) = subframe_f;
            elseif domain == 1
                Data(:,:,:, index) = subframe_t;
            else
                Data(:,:,:, index) = cat(2, subframe_t, subframe_f);
            end

            Labels(index) = Dopplers(Doppler_idx);

      end
    end
        numFrames = size(Data, 4);
        shuffleIdx = randperm(numFrames);
        Data = Data(:, :, :, shuffleIdx);
        Labels = Labels(shuffleIdx);

        fprintf('Data generation complete.  numFrames=%d!\n ', numFrames);

end
function [gnb, pdsch] = hDeepLearningChanEstSimParameters()
    % Set waveform type and PDSCH numerology (SCS and CP type)
    simParameters.NRB = 51;                  % Bandwidth in number of resource blocks (51RBs at 30kHz SCS for 20MHz BW)
    simParameters.SubcarrierSpacing = 30;    % 15, 30, 60, 120, 240 (kHz)
    simParameters.CyclicPrefix = 'Normal';   % 'Normal' or 'Extended'
    simParameters.NCellID = 2;               % Cell identity

    % DL-SCH/PDSCH parameters
    simParameters.PDSCH.PRBSet = 0:simParameters.NRB-1; % PDSCH PRB allocation
    simParameters.PDSCH.SymbolSet = 0:13;           % PDSCH symbol allocation in each slot
    simParameters.PDSCH.EnableHARQ = true;          % Enable/disable HARQ, if disabled, single transmission with RV=0, i.e. no retransmissions

    simParameters.PDSCH.NLayers = 1;                % Number of PDSCH layers
    simParameters.NTxAnts = 1;                      % Number of PDSCH transmission antennas

    simParameters.NumCW = 1;                        % Number of codewords
    simParameters.PDSCH.TargetCodeRate = 490/1024;  % Code rate used to calculate transport block sizes
    simParameters.PDSCH.Modulation = '16QAM';       % 'QPSK', '16QAM', '64QAM', '256QAM'
    simParameters.NRxAnts = 1;                      % Number of UE receive antennas

    % DM-RS and antenna port configuration (TS 38.211 Section 7.4.1.1)
    simParameters.PDSCH.PortSet = 0:simParameters.PDSCH.NLayers-1; % DM-RS ports to use for the layers
    simParameters.PDSCH.PDSCHMappingType = 'A';     % PDSCH mapping type ('A'(slot-wise),'B'(non slot-wise))
    simParameters.PDSCH.DMRSTypeAPosition = 2;      % Mapping type A only. First DM-RS symbol position (2,3)
    simParameters.PDSCH.DMRSLength = 1;             % Number of front-loaded DM-RS symbols (1(single symbol),2(double symbol))
    simParameters.PDSCH.DMRSAdditionalPosition = 1; % Additional DM-RS symbol positions (max range 0...3)
    simParameters.PDSCH.DMRSConfigurationType = 2;  % DM-RS configuration type (1,2)
    simParameters.PDSCH.NumCDMGroupsWithoutData = 0;% CDM groups without data
    simParameters.PDSCH.NIDNSCID = 1;               % Scrambling identity (0...65535)
    simParameters.PDSCH.NSCID = 0;                  % Scrambling initialization (0,1)
    % Reserved PRB patterns (for CORESETs, forward compatibility etc)
    simParameters.PDSCH.Reserved.Symbols = [];      % Reserved PDSCH symbols
    simParameters.PDSCH.Reserved.PRB = [];          % Reserved PDSCH PRBs
    simParameters.PDSCH.Reserved.Period = [];       % Periodicity of reserved resources
    % PDSCH resource block mapping (TS 38.211 Section 7.3.1.6)
    simParameters.PDSCH.VRBToPRBInterleaving = 0;   % Disable interleaved resource mapping
    
    % Specify additional required fields for PDSCH
    simParameters.PDSCH.RNTI = 1;
    
    % The Xoh-PDSCH overhead value is taken to be 0 here
    simParameters.PDSCH.Xoh_PDSCH = 0;
    
    gnb = simParameters;
    pdsch = simParameters.PDSCH;

end
function Labelskk = Doppler2BoundLabel(Labels, Doppler, type)
   Labelskk = categorical(ones(length(Labels), 1));
   for j=1:length(Labels)
       if type == 0  % greater than or less
           if (DopplerTypes2num(Labels(j)) < Doppler) %true label < predictor label
               Labelskk(j) = categorical(-1);
           else
               Labelskk(j) = categorical(1);
           end
       else % equal or not
           if (DopplerTypes2num(Labels(j)) == Doppler)
               Labelskk(j) = categorical(-1);
           else
               Labelskk(j) = categorical(1);

           end
       end
   end
end
function Labelskk = fixabug(Labels)
   Labelskk = categorical(zeros(length(Labels), 1));
%    Labelskk = Labels;
   for j=1:length(Labels)
       label = Labels(j);
       Labelskk(j) = label;
   end
end
function bestFitLabel = bestFit(predictor, predLabel, trueLabel)
    if predLabel == "-1" %prediction: true label < predictor label
        if trueLabel < predictor % check if true label < predictor label
            bestFitLabel = trueLabel;
        else
            bestFitLabel = predictor;
        end
    else %prediction: true label >= predictor label
        if trueLabel >= predictor % check if true label >= predictor label
            bestFitLabel = trueLabel;
        else
            bestFitLabel = predictor;
        end
    end
end
function M = ModTypes2M(x)
if x == "QPSK" 
    M = 2;
elseif x == "16QAM" 
    M = 4;
elseif x == "64QAM" 
    M = 6;
end
end

function Doppler = DopplerTypes2num(x)
if x == "0" 
    Doppler = 0;
elseif x == "10" 
    Doppler = 10;
elseif x == "20" 
    Doppler = 20;
elseif x == "30" 
    Doppler = 30;
elseif x == "40" 
    Doppler = 40;
elseif x == "50" 
    Doppler = 50;
elseif x == "60" 
    Doppler = 60;
elseif x == "70" 
    Doppler = 70;
elseif x == "80" 
    Doppler = 80;
elseif x == "90" 
    Doppler = 90;
elseif x == "100" 
    Doppler = 100;
elseif x == "110" 
    Doppler = 110;
elseif x == "120" 
    Doppler = 120;
elseif x == "130" 
    Doppler = 130;
elseif x == "140" 
    Doppler = 140;
elseif x == "150" 
    Doppler = 150;
elseif x == "160" 
    Doppler = 160;
elseif x == "170" 
    Doppler = 170;
elseif x == "180" 
    Doppler = 180;
elseif x == "190" 
    Doppler = 190;
elseif x == "200" 
    Doppler = 200;
elseif x == "240" 
    Doppler = 240;
elseif x == "250" 
    Doppler = 250;
elseif x == "280" 
    Doppler = 280;
elseif x == "300" 
    Doppler = 300;
elseif x == "320" 
    Doppler = 320;
elseif x == "350" 
    Doppler = 350;
elseif x == "360" 
    Doppler = 360;
elseif x == "400" 
    Doppler = 400;
elseif x == "450" 
    Doppler = 450;
elseif x == "500" 
    Doppler = 500;
elseif x == "550" 
    Doppler = 550;
end
end



% function modulator = getModulator(modType, sps, fs)
% %getModulator Modulation function selector
% %   MOD = getModulator(TYPE,SPS,FS) returns the modulator function handle
% %   MOD based on TYPE. SPS is the number of samples per symbol and FS is 
% %   the sample rate.
% 
% switch modType
%   case "BPSK"
%     modulator = @(x)bpskModulator(x,sps);
%   case "QPSK"
%     modulator = @(x)qpskModulator(x,sps);
%   case "8PSK"
%     modulator = @(x)psk8Modulator(x,sps);
%   case "16QAM"
%     modulator = @(x)qam16Modulator(x,sps);
%   case "32QAM"
%     modulator = @(x)qam32Modulator(x,sps);
%   case "64QAM"
%     modulator = @(x)qam64Modulator(x,sps);
%   case "128QAM"
%     modulator = @(x)qam128Modulator(x,sps);
%   case "256QAM"
%     modulator = @(x)qam256Modulator(x,sps);
%   case "GFSK"
%     modulator = @(x)gfskModulator(x,sps);
%   case "CPFSK"
%     modulator = @(x)cpfskModulator(x,sps);
%   case "PAM4"
%     modulator = @(x)pam4Modulator(x,sps);
%   case "B-FM"
%     modulator = @(x)bfmModulator(x, fs);
%   case "DSB-AM"
%     modulator = @(x)dsbamModulator(x, fs);
%   case "SSB-AM"
%     modulator = @(x)ssbamModulator(x, fs);
% end
% end

% function src = getSource(modType, sps, spf, fs)
% %getSource Source selector for modulation types
% %    SRC = getSource(TYPE,SPS,SPF,FS) returns the data source
% %    for the modulation type TYPE, with the number of samples 
% %    per symbol SPS, the number of samples per frame SPF, and 
% %    the sampling frequency FS.
% 
% switch modType
%   case {"BPSK","GFSK","CPFSK"}
% %     M = 2;
%     M = 1;
%     src = @()randi([0 M-1],spf/sps,1);
%   case {"QPSK","PAM4"}
%     M = 4;
%     src = @()randi([0 M-1],spf/sps,1);
%   case "8PSK"
%     M = 8;
%     src = @()randi([0 M-1],spf/sps,1);
%   case "16QAM"
%     M = 16;
%     src = @()randi([0 M-1],spf/sps,1);
%   case "32QAM"
%     M = 32;
%     src = @()randi([0 M-1],spf/sps,1);
%   case "64QAM"
%     M = 64;
%     src = @()randi([0 M-1],spf/sps,1);
%   case "128QAM"
%     M = 128;
%     src = @()randi([0 M-1],spf/sps,1);
%   case "256QAM"
%     M = 256;
%     src = @()randi([0 M-1],spf/sps,1);
%   case {"B-FM","DSB-AM","SSB-AM"}
%     src = @()getAudio(spf,fs);
% end
% end

% function x = getAudio(spf,fs)
% %getAudio Audio source for analog modulation types
% %    A = getAudio(SPF,FS) returns the audio source A, with the 
% %    number of samples per frame SPF, and the sample rate FS.
% 
% persistent audioSrc audioRC
% 
% if isempty(audioSrc)
%   audioSrc = dsp.AudioFileReader('audio_mix_441.wav',...
%     'SamplesPerFrame',spf,'PlayCount',inf);
%   audioRC = dsp.SampleRateConverter('Bandwidth',30e3,...
%     'InputSampleRate',audioSrc.SampleRate,...
%     'OutputSampleRate',fs);
%   [~,decimFactor] = getRateChangeFactors(audioRC);
%   audioSrc.SamplesPerFrame = ceil(spf / fs * audioSrc.SampleRate / decimFactor) * decimFactor;
% end
% 
% x = audioRC(audioSrc());
% x = x(1:spf,1);
% end

% function frames = getNNFrames(rx,modType)
% %getNNOFDM Symbols Generate formatted OFDM Symbols for neural networks
% %   F = getNNOFDM Symbols(X,MODTYPE) formats the input X, into OFDM Symbols 
% %   that can be used with the neural network designed in this 
% %   example, and returns the OFDM Symbols in the output F.
% 
% frames = helperModClassFrameGenerator(rx,spf,spf,32,8);
% frameStore = helperModClassFrameStore(10,spf,categorical({modType}));
% add(frameStore,frames,modType);
% frames = get(frameStore);
% end

% function plotScores(score,labels)
% %plotScores Plot classification scores of OFDM Symbols
% %   plotScores(SCR,LABELS) plots the classification scores SCR as a stacked 
% %   bar for each frame. SCR is a matrix in which each row is the score for a 
% %   frame.
% 
% co = [0.08 0.9 0.49;
%   0.52 0.95 0.70;
%   0.36 0.53 0.96;
%   0.09 0.54 0.67;
%   0.48 0.99 0.26;
%   0.95 0.31 0.17;
%   0.52 0.85 0.95;
%   0.08 0.72 0.88;
%   0.12 0.45 0.69;
%   0.22 0.11 0.49;
%   0.65 0.54 0.71];
% figure; ax = axes('ColorOrder',co,'NextPlot','replacechildren');
% bar(ax,[score; nan(2,11)],'stacked'); legend(categories(labels),'Location','best');
% xlabel('Frame Number'); ylabel('Score'); title('Classification Scores')
% end

% function plotTimeDomain(rxTest,rxTestLabel,modulationTypes,fs)
% %plotTimeDomain Time domain plots of OFDM Symbols
% 
% numRows = ceil(length(modulationTypes) / 4);
% spf = size(rxTest,2);
% t = 1000*(0:spf-1)/fs;
% if size(rxTest,1) == 2
%   IQAsRows = true;
% else
%   IQAsRows = false;
% end
% for modType=1:length(modulationTypes)
%   subplot(numRows, 4, modType);
%   idxOut = find(rxTestLabel == modulationTypes(modType), 1);
%   if IQAsRows
%     rxI = rxTest(1,:,1,idxOut);
%     rxQ = rxTest(2,:,1,idxOut);
%   else
%     rxI = rxTest(1,:,1,idxOut);
%     rxQ = rxTest(1,:,2,idxOut);
%   end
%   plot(t,squeeze(rxI), '-'); grid on; axis equal; axis square
%   hold on
%   plot(t,squeeze(rxQ), '-'); grid on; axis equal; axis square
%   hold off
%   title(string(modulationTypes(modType)));
%   xlabel('Time (ms)'); ylabel('Amplitude')
% end
% end
% 
% function plotSpectrogram(rxTest,rxTestLabel,modulationTypes,fs,sps)
% %plotSpectrogram Spectrogram of OFDM Symbols
% 
% if size(rxTest,1) == 2
%   IQAsRows = true;
% else
%   IQAsRows = false;
% end
% numRows = ceil(length(modulationTypes) / 4);
% for modType=1:length(modulationTypes)
%   subplot(numRows, 4, modType);
%   idxOut = find(rxTestLabel == modulationTypes(modType), 1);
%   if IQAsRows
%     rxI = rxTest(1,:,1,idxOut);
%     rxQ = rxTest(2,:,1,idxOut);
%   else
%     rxI = rxTest(1,:,1,idxOut);
%     rxQ = rxTest(1,:,2,idxOut);
%   end
%   rx = squeeze(rxI) + 1i*squeeze(rxQ);
%   spectrogram(rx,kaiser(sps),0,spf,fs,'centered');
%   title(string(modulationTypes(modType)));
% end
% h = gcf; delete(findall(h.Children, 'Type', 'ColorBar'))
% end
% 
% function flag = isPlutoSDRInstalled
% %isPlutoSDRInstalled Check if ADALM-PLUTO is installed
% 
% spkg = matlabshared.supportpkg.getInstalled;
% flag = ~isempty(spkg) && any(contains({spkg.Name},'ADALM-PLUTO','IgnoreCase',true));
% end

% function y = bpskModulator(x,sps)
% %bpskModulator BPSK modulator with pulse shaping
% %   Y = bpskModulator(X,SPS) BPSK modulates the input X, and returns the 
% %   root-raised cosine pulse shaped signal Y. X must be a column vector 
% %   of values in the set [0 1]. The root-raised cosine filter has a 
% %   roll-off factor of 0.35 and spans four symbols. The output signal 
% %   Y has unit power.
% 
% persistent filterCoeffs
% if isempty(filterCoeffs)
%   filterCoeffs = rcosdesign(0.35, 4, sps);
% end
% % Modulate
% syms = pskmod(x,2);
% % Pulse shape
% y = filter(filterCoeffs, 1, upsample(syms,sps));
% end
% 
% function y = qpskModulator(x,sps)
% %qpskModulator QPSK modulator with pulse shaping
% %   Y = qpskModulator(X,SPS) QPSK modulates the input X, and returns the 
% %   root-raised cosine pulse shaped signal Y. X must be a column vector 
% %   of values in the set [0 3]. The root-raised cosine filter has a 
% %   roll-off factor of 0.35 and spans four symbols. The output signal 
% %   Y has unit power.
% 
% persistent filterCoeffs
% if isempty(filterCoeffs)
%   filterCoeffs = rcosdesign(0.35, 4, sps);
% end
% % Modulate
% syms = pskmod(x,4,pi/4);
% % Pulse shape
% y = filter(filterCoeffs, 1, upsample(syms,sps));
% end
% 
% function y = psk8Modulator(x,sps)
% %psk8Modulator 8-PSK modulator with pulse shaping
% %   Y = psk8Modulator(X,SPS) 8-PSK modulates the input X, and returns the 
% %   root-raised cosine pulse shaped signal Y. X must be a column vector 
% %   of values in the set [0 7]. The root-raised cosine filter has a 
% %   roll-off factor of 0.35 and spans four symbols. The output signal 
% %   Y has unit power.
% 
% persistent filterCoeffs
% if isempty(filterCoeffs)
%   filterCoeffs = rcosdesign(0.35, 4, sps);
% end
% % Modulate
% syms = pskmod(x,8);
% % Pulse shape
% y = filter(filterCoeffs, 1, upsample(syms,sps));
% end
% 
% function y = qam16Modulator(x,sps)
% %qam16Modulator 16-QAM modulator with pulse shaping
% %   Y = qam16Modulator(X,SPS) 16-QAM modulates the input X, and returns the 
% %   root-raised cosine pulse shaped signal Y. X must be a column vector 
% %   of values in the set [0 15]. The root-raised cosine filter has a 
% %   roll-off factor of 0.35 and spans four symbols. The output signal 
% %   Y has unit power.
% 
% persistent filterCoeffs
% if isempty(filterCoeffs)
%   filterCoeffs = rcosdesign(0.35, 4, sps);
% end
% % Modulate and pulse shape
% syms = qammod(x,16,'UnitAveragePower',true);
% % Pulse shape
% y = filter(filterCoeffs, 1, upsample(syms,sps));
% end
% 
% function y = qam32Modulator(x,sps)
% %qam16Modulator 16-QAM modulator with pulse shaping
% %   Y = qam16Modulator(X,SPS) 16-QAM modulates the input X, and returns the 
% %   root-raised cosine pulse shaped signal Y. X must be a column vector 
% %   of values in the set [0 15]. The root-raised cosine filter has a 
% %   roll-off factor of 0.35 and spans four symbols. The output signal 
% %   Y has unit power.
% 
% persistent filterCoeffs
% if isempty(filterCoeffs)
%   filterCoeffs = rcosdesign(0.35, 4, sps);
% end
% % Modulate and pulse shape
% syms = qammod(x,32,'UnitAveragePower',true);
% % Pulse shape
% y = filter(filterCoeffs, 1, upsample(syms,sps));
% end
% 
% function y = qam64Modulator(x,sps)
% %qam64Modulator 64-QAM modulator with pulse shaping
% %   Y = qam64Modulator(X,SPS) 64-QAM modulates the input X, and returns the 
% %   root-raised cosine pulse shaped signal Y. X must be a column vector 
% %   of values in the set [0 63]. The root-raised cosine filter has a 
% %   roll-off factor of 0.35 and spans four symbols. The output signal 
% %   Y has unit power.
% 
% persistent filterCoeffs
% if isempty(filterCoeffs)
%   filterCoeffs = rcosdesign(0.35, 4, sps);
% end
% % Modulate
% syms = qammod(x,64,'UnitAveragePower',true);
% % Pulse shape
% y = filter(filterCoeffs, 1, upsample(syms,sps));
% end
% 
% function y = qam128Modulator(x,sps)
% %qam64Modulator 64-QAM modulator with pulse shaping
% %   Y = qam64Modulator(X,SPS) 64-QAM modulates the input X, and returns the 
% %   root-raised cosine pulse shaped signal Y. X must be a column vector 
% %   of values in the set [0 63]. The root-raised cosine filter has a 
% %   roll-off factor of 0.35 and spans four symbols. The output signal 
% %   Y has unit power.
% 
% persistent filterCoeffs
% if isempty(filterCoeffs)
%   filterCoeffs = rcosdesign(0.35, 4, sps);
% end
% % Modulate
% syms = qammod(x,128,'UnitAveragePower',true);
% % Pulse shape
% y = filter(filterCoeffs, 1, upsample(syms,sps));
% end
% 
% function y = qam256Modulator(x,sps)
% %qam64Modulator 64-QAM modulator with pulse shaping
% %   Y = qam64Modulator(X,SPS) 64-QAM modulates the input X, and returns the 
% %   root-raised cosine pulse shaped signal Y. X must be a column vector 
% %   of values in the set [0 63]. The root-raised cosine filter has a 
% %   roll-off factor of 0.35 and spans four symbols. The output signal 
% %   Y has unit power.
% 
% persistent filterCoeffs
% if isempty(filterCoeffs)
%   filterCoeffs = rcosdesign(0.35, 4, sps);
% end
% % Modulate
% syms = qammod(x,256,'UnitAveragePower',true);
% % Pulse shape
% y = filter(filterCoeffs, 1, upsample(syms,sps));
% end
% 
% function y = pam4Modulator(x,sps)
% %pam4Modulator PAM4 modulator with pulse shaping
% %   Y = pam4Modulator(X,SPS) PAM4 modulates the input X, and returns the 
% %   root-raised cosine pulse shaped signal Y. X must be a column vector 
% %   of values in the set [0 3]. The root-raised cosine filter has a 
% %   roll-off factor of 0.35 and spans four symbols. The output signal 
% %   Y has unit power.
% 
% persistent filterCoeffs amp
% if isempty(filterCoeffs)
%   filterCoeffs = rcosdesign(0.35, 4, sps);
%   amp = 1 / sqrt(mean(abs(pammod(0:3, 4)).^2));
% end
% % Modulate
% syms = amp * pammod(x,4);
% % Pulse shape
% y = filter(filterCoeffs, 1, upsample(syms,sps));
% end
% 
% function y = gfskModulator(x,sps)
% %gfskModulator GFSK modulator
% %   Y = gfskModulator(X,SPS) GFSK modulates the input X and returns the 
% %   signal Y. X must be a column vector of values in the set [0 1]. The 
% %   BT product is 0.35 and the modulation index is 1. The output signal 
% %   Y has unit power.
% 
% persistent mod meanM
% if isempty(mod)
%   M = 2;
%   mod = comm.CPMModulator(...
%     'ModulationOrder', M, ...
%     'FrequencyPulse', 'Gaussian', ...
%     'BandwidthTimeProduct', 0.35, ...
%     'ModulationIndex', 1, ...
%     'SamplesPerSymbol', sps);
%   meanM = mean(0:M-1);
% end
% % Modulate
% y = mod(2*(x-meanM));
% end
% 
% function y = cpfskModulator(x,sps)
% %cpfskModulator CPFSK modulator
% %   Y = cpfskModulator(X,SPS) CPFSK modulates the input X and returns 
% %   the signal Y. X must be a column vector of values in the set [0 1]. 
% %   the modulation index is 0.5. The output signal Y has unit power.
% 
% persistent mod meanM
% if isempty(mod)
%   M = 2;
%   mod = comm.CPFSKModulator(...
%     'ModulationOrder', M, ...
%     'ModulationIndex', 0.5, ...
%     'SamplesPerSymbol', sps);
%   meanM = mean(0:M-1);
% end
% % Modulate
% y = mod(2*(x-meanM));
% end
% 
% function y = bfmModulator(x,fs)
% %bfmModulator Broadcast FM modulator
% %   Y = bfmModulator(X,FS) broadcast FM modulates the input X and returns
% %   the signal Y at the sample rate FS. X must be a column vector of
% %   audio samples at the sample rate FS. The frequency deviation is 75 kHz
% %   and the pre-emphasis filter time constant is 75 microseconds.
% 
% persistent mod
% if isempty(mod)
%   mod = comm.FMBroadcastModulator(...
%     'AudioSampleRate', fs, ...
%     'SampleRate', fs);
% end
% y = mod(x);
% end
% 
% function y = dsbamModulator(x,fs)
% %dsbamModulator Double sideband AM modulator
% %   Y = dsbamModulator(X,FS) double sideband AM modulates the input X and
% %   returns the signal Y at the sample rate FS. X must be a column vector of
% %   audio samples at the sample rate FS. The IF frequency is 50 kHz.
% 
% y = ammod(x,50e3,fs);
% end
% 
% function y = ssbamModulator(x,fs)
% %ssbamModulator Single sideband AM modulator
% %   Y = ssbamModulator(X,FS) single sideband AM modulates the input X and
% %   returns the signal Y at the sample rate FS. X must be a column vector of
% %   audio samples at the sample rate FS. The IF frequency is 50 kHz.
% 
% y = ssbmod(x,50e3,fs);
% end
