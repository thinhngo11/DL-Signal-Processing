%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 5/5/20 5G NR time-domain subframe-based input size = 614x14x1: 
% save('NR5G_SNR_sequence_S_type1', 'testAccuracyS') w/ 500
%  window: 1Mean accuracyS: 83.6027
%  window: 2Mean accuracyS: 92.7319
%  window: 3Mean accuracyS: 83.0279
%  window: 4Mean accuracyS: 96.9565
%  window: 5Mean accuracyS: 88.275

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

modulationTypes = ["QPSK", "16QAM", "64QAM"];

numModulationTypes = length(modulationTypes);
Dopplers = [0 60 120];
Delays = ["EPA" "EVA" "ETU"]; % 0 25us
numDopplerTypes = length(Dopplers);
numDelayTypes = length(Delays);

sps = 8;                % Samples per symbol

rng(1235)

spf = 8568;  %612*14; %subframe frequency domain based prediction 72 x 14

% dropoutRate = 0.5;
netWidth = 1;
filterSize = [1 sps];
poolSize = [1 2];
numHiddenUnits = 200;
% WINDOWS = [1 2 3 4 5 10 15];
maxEpochs = 12;  %is significant
miniBatchSize = 256;

WINDOWS = [1 2 3 4 5];
winlen = length(WINDOWS);


% SNRdB = 22;             % Desired SNR in dB
% SNR = 10^(SNRdB/20);    % Linear SNR  
rng('default');         % Configure random number generators


disp('Single input predictions');
SNRs = [-4, 0, 4, 8, 12, 16, 20, 24, 28, 32];
% SNRs = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32];
% SNRs = [-20, -16, -12, -8, -4, 0, 4, 8, 12, 16, 20, 24, 28, 32];
% SNRs = [-4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32];
snr_inc = SNRs(2) - SNRs(1);
boundary = SNRs(2 : length(SNRs)-1);
SNRTypes = categorical(SNRs);
numSNRTypes = length(SNRs);

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
  fullyConnectedLayer(length(SNRs), 'Name', 'FC1')
  softmaxLayer('Name', 'SoftMax')
  classificationLayer('Name', 'Output') ];

cnnLayers = layerGraph(modClassNet);
lgraph = connectLayers(cnnLayers,"fold/miniBatchSize","unfold/miniBatchSize");

testAccuracyS = zeros(length(WINDOWS), length(SNRs));
numFramesPerSNRType = 1000;
percentTrainSamples = 0.80;
percentValidationSamples = 0.10;
percentTestSamples = 0.10;

% Generate the training data
[Data, Labels] = hGenerateTrainingData(numFramesPerSNRType, SNRTypes);
numData = numFramesPerSNRType * length(SNRs);
%     InputV = 612  14;
Data = reshape(Data, [1 spf 2 numData]);

% Split into validation sets
valStart = 1;
valEnd = numData * percentValidationSamples;
rxValidation = Data(:,:,:,valStart:valEnd);
rxValidationLabel = Labels(valStart:valEnd);

% Split into test sets
testStart = 1 + valEnd;
testEnd = numData * percentTestSamples + valEnd;
rxTest = Data(:,:,:,testStart:testEnd);
rxTestLabel = Labels(testStart:testEnd);

%     lenTest = length(rxTestLabel);
%     sumAcc = zeros(length(SNRs), 1);
%     for i = 1:lenTest
%         index = find(SNRs == SNRTypes2num(rxTestLabel(i)));
%         sumAcc(index) = sumAcc(index) + 1;
%     end
% %     testAccuracyS(window, :) = 100 * length(SNRs) * sumAcc / lenTest;
%     for i = 1:numSNRTypes
%         disp("Swindow: " + window + " SNR: " + i + "   cnt: " + sumAcc(i));
%     end

% Split into training sets
trainStart = 1 + testEnd;
trainEnd = numData * percentTrainSamples + testEnd;
rxTraining = Data(:,:,:,trainStart:trainEnd);
rxTrainingLabel = Labels(trainStart:trainEnd);

validationFrequency = floor(numel(rxTrainingLabel)/miniBatchSize);

for window = 1:length(WINDOWS)

seqTraining = cell(length(rxTrainingLabel), 1);
seqTrainingLabel = zeros(length(rxTrainingLabel), 1);
rxTrainingArr = zeros(1, spf, 2, WINDOWS(window));
for i = 1:length(rxTrainingLabel)
    cnt = 1;
    rxTrainingArr(:, :, :, cnt) = rxTraining(:, :, :, i);
    for j = i+1:length(rxTrainingLabel)
        if cnt == WINDOWS(window)
            break
        end
        if rxTrainingLabel(j) == rxTrainingLabel(i)
            cnt = cnt + 1;
            rxTrainingArr(:, :, :, cnt) = rxTraining(:, :, :, j);
        end
    end
    if cnt == WINDOWS(window)
        seqTraining{i, 1} = rxTrainingArr;
        seqTrainingLabel(i, 1) = rxTrainingLabel(i);
    else
        for j = 1:length(rxTrainingLabel)
            if cnt == WINDOWS(window)
                break
            end
            if rxTrainingLabel(j) == rxTrainingLabel(i)
                cnt = cnt + 1;
                rxTrainingArr(:, :, :, cnt) = rxTraining(:, :, :, j);
            end
        end
        seqTraining{i, 1} = rxTrainingArr;
        seqTrainingLabel(i, 1) = rxTrainingLabel(i);
    end    
end

seqValidation = cell(length(rxValidationLabel), 1);
seqValidationLabel = zeros(length(rxValidationLabel), 1);
rxValidationArr = zeros(1, spf, 2, WINDOWS(window));
for i = 1:length(rxValidationLabel)
    cnt = 1;
    rxValidationArr(:, :, :, cnt) = rxValidation(:, :, :, i);
    for j = i+1:length(rxValidationLabel)
        if cnt == WINDOWS(window)
            break
        end
        if rxValidationLabel(j) == rxValidationLabel(i)
            cnt = cnt + 1;
            rxValidationArr(:, :, :, cnt) = rxValidation(:, :, :, j);
        end
    end
    if cnt == WINDOWS(window)
        seqValidation{i, 1} = rxValidationArr;
        seqValidationLabel(i, 1) = rxValidationLabel(i);
    else
        for j = 1:length(rxValidationLabel)
            if cnt == WINDOWS(window)
                break
            end
            if rxValidationLabel(j) == rxValidationLabel(i)
                cnt = cnt + 1;
                rxValidationArr(:, :, :, cnt) = rxValidation(:, :, :, j);
            end
        end
        seqValidation{i, 1} = rxValidationArr;
        seqValidationLabel(i, 1) = rxValidationLabel(i);
    end    
end

seqTest = cell(length(rxTestLabel), 1);
seqTestLabel = zeros(length(rxTestLabel), 1);
rxTestArr = zeros(1, spf, 2, WINDOWS(window));
for i = 1:length(rxTestLabel)
    cnt = 1;
    rxTestArr(:, :, :, cnt) = rxTest(:, :, :, i);
    for j = i+1:length(rxTestLabel)
        if cnt == WINDOWS(window)
            break
        end
        if rxTestLabel(j) == rxTestLabel(i)
            cnt = cnt + 1;
            rxTestArr(:, :, :, cnt) = rxTest(:, :, :, j);
        end
    end
    if cnt == WINDOWS(window)
        seqTest{i, 1} = rxTestArr;
        seqTestLabel(i, 1) = rxTestLabel(i);
    else
        for j = 1:length(rxTestLabel)
            if cnt == WINDOWS(window)
                break
            end
            if rxTestLabel(j) == rxTestLabel(i)
                cnt = cnt + 1;
                rxTestArr(:, :, :, cnt) = rxTest(:, :, :, j);
            end
        end
        seqTest{i, 1} = rxTestArr;
        seqTestLabel(i, 1) = rxTestLabel(i);
    end    
end

   options = trainingOptions('sgdm', ...
  'InitialLearnRate',2e-2, ...
  'MaxEpochs',maxEpochs, ...
  'MiniBatchSize',miniBatchSize, ...
  'Shuffle','every-epoch', ... 
  'Verbose',false, ...
  'ValidationPatience', 5, ...
  'ValidationData',{seqValidation,rxValidationLabel}, ...
  'ValidationFrequency',validationFrequency, ...
  'LearnRateSchedule', 'piecewise', ...
  'LearnRateDropPeriod', 9, ...
  'LearnRateDropFactor', 0.1, ...
  'ExecutionEnvironment', 'cpu');
% analyzeNetwork(modClassNet{kk});

trainNow = true;
if trainNow == true
    trainedNetS = trainNetwork(seqTraining,rxTrainingLabel,lgraph,options);
else
%   load trainedModulationClassificationNetwork
end

rxTestPred = classify(trainedNetS, seqTest);
TestPrednum = zeros((length(rxTestPred)/numSNRTypes), numSNRTypes);
TestLabelnum = zeros((length(rxTestPred)/numSNRTypes), numSNRTypes);
for SNR = 1:numSNRTypes
    j = 1;
    for i = 1:length(rxTestPred)
        if rxTestLabel(i) == SNRTypes(SNR) 
            TestPrednum(j, SNR) = SNRTypes2num(rxTestPred(i));
            TestLabelnum(j, SNR) = SNRTypes2num(SNRTypes(SNR));
            j = j + 1;
        end
    end
end
numTestDataperSNR = length(rxTestPred)/numSNRTypes;
meanTestPred = zeros(numSNRTypes, 1);
% stdTestPred = zeros(numSNRTypes, 1);
% movingTestPred = zeros(numTestDataperSNR, numSNRTypes);
% movingTestPredRound = zeros(numTestDataperSNR, numSNRTypes);
% movingtestAccuracy = zeros(numSNRTypes, 1);
testAccuracy = zeros(winlen, numSNRTypes, 1);
nMSE = zeros(winlen, numSNRTypes, 1);
MSE = zeros(winlen, numSNRTypes, 1);
% TestPrednum3 = zeros((length(rxTestPred)/numSNRTypes), 3);

for SNR = 1:numSNRTypes
    meanTestPred(SNR) = mean(TestPrednum(:,SNR));
%     stdTestPred(SNR) = std(TestPrednum(:,SNR));
%     movingTestPred(:, SNR) = movmean(TestPrednum(:,SNR), 25);
    sum_square_error = 0;
    for i = 1:numTestDataperSNR
        sum_square_error = sum_square_error + (movingTestPred(i,SNR) - SNRTypes2num(SNRTypes(SNR))) ^ 2;
    end
    MSE(window, SNR) = sum_square_error / numTestDataperSNR;
    if SNRTypes2num(SNRTypes(SNR)) == 0
        nMSE(window, SNR) = (sum_square_error / numTestDataperSNR) / (meanTestPred(SNR) ^ 2);
    else 
        nMSE(window, SNR) = (sum_square_error / numTestDataperSNR) / (SNRTypes2num(SNRTypes(SNR)) ^ 2);
    end
%     movingTestPredRound(:, SNR) = round(movmean(TestPrednum(:,SNR), 25)/snr_inc)*snr_inc;
%     movingtestAccuracy(SNR) = 100 * mean(movingTestPredRound(:, SNR) == SNRTypes2num(SNRTypes(SNR)));
    testAccuracy(window, SNR) = 100 * mean(TestPrednum(:, SNR) == SNRTypes2num(SNRTypes(SNR)));
    disp("Swindow: " + window + " SNR: " + SNRTypes2num(SNRTypes(SNR)) + "   Accuracy: " + testAccuracy(window, SNR) + "%   nMSE: " + nMSE(SNR));
end
% Single-CNN classifier accuracy
%     lenTest = length(rxTestLabel);
%     sumAcc = zeros(length(SNRs), 1);
%     total = zeros(length(SNRs), 1);
%     for i = 1:lenTest
%         index = find(SNRs == SNRTypes2num(rxTestLabel(i)));
%         sumAcc(index) = sumAcc(index) + double((SNRTypes2num(rxTestLabel(i)) == SNRTypes2num(rxTestPred(i))));
%         total(index) = total(index) + 1;
%     end
%     for i = 1:numSNRTypes
%         testAccuracyS(window, i) = 100 * sumAcc(i, 1) / total(i, 1);
%         disp("Swindow: " + window + " SNR: " + i + "   Accuracy: " + testAccuracyS(window, i));
%     end
    disp(" window: " + window + "Mean accuracyS: " + mean(testAccuracyS(window, :)));
end

%type1 = [-4, 0, 4, 8, 12, 16, 20, 24, 28, 32]
%type2 = [-4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
% type3 = [-20, -16, -12, -8, -4, 0, 4, 8, 12, 16, 20, 24, 28, 32]
% type4 = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32];
% 
save('NR5Gfs_SNR_sequence_S_type1', 'testAccuracyS')

% Swindow: 1 SNR: 1   Accuracy: 100 500
% Swindow: 1 SNR: 2   Accuracy: 91.6667
% Swindow: 1 SNR: 3   Accuracy: 95.4545
% Swindow: 1 SNR: 4   Accuracy: 92.8571
% Swindow: 1 SNR: 5   Accuracy: 86.2069
% Swindow: 1 SNR: 6   Accuracy: 90.9091
% Swindow: 1 SNR: 7   Accuracy: 94.5455
% Swindow: 1 SNR: 8   Accuracy: 90.566
% Swindow: 1 SNR: 9   Accuracy: 75.6098
% Swindow: 1 SNR: 10   Accuracy: 97.9592
%  window: 1Mean accuracyS: 91.5775
% Swindow: 2 SNR: 1   Accuracy: 100
% Swindow: 2 SNR: 2   Accuracy: 100
% Swindow: 2 SNR: 3   Accuracy: 88.6364
% Swindow: 2 SNR: 4   Accuracy: 100
% Swindow: 2 SNR: 5   Accuracy: 96.5517
% Swindow: 2 SNR: 6   Accuracy: 94.5455
% Swindow: 2 SNR: 7   Accuracy: 96.3636
% Swindow: 2 SNR: 8   Accuracy: 96.2264
% Swindow: 2 SNR: 9   Accuracy: 87.8049
% Swindow: 2 SNR: 10   Accuracy: 100
%  window: 2Mean accuracyS: 96.0128
% Swindow: 3 SNR: 1   Accuracy: 100
% Swindow: 3 SNR: 2   Accuracy: 100
% Swindow: 3 SNR: 3   Accuracy: 100
% Swindow: 3 SNR: 4   Accuracy: 100
% Swindow: 3 SNR: 5   Accuracy: 100
% Swindow: 3 SNR: 6   Accuracy: 100
% Swindow: 3 SNR: 7   Accuracy: 100
% Swindow: 3 SNR: 8   Accuracy: 100
% Swindow: 3 SNR: 9   Accuracy: 97.561
% Swindow: 3 SNR: 10   Accuracy: 100
%  window: 3Mean accuracyS: 99.7561
% Swindow: 4 SNR: 1   Accuracy: 100
% Swindow: 4 SNR: 2   Accuracy: 97.9167
% Swindow: 4 SNR: 3   Accuracy: 100
% Swindow: 4 SNR: 4   Accuracy: 100
% Swindow: 4 SNR: 5   Accuracy: 100
% Swindow: 4 SNR: 6   Accuracy: 100
% Swindow: 4 SNR: 7   Accuracy: 100
% Swindow: 4 SNR: 8   Accuracy: 100
% Swindow: 4 SNR: 9   Accuracy: 95.122
% Swindow: 4 SNR: 10   Accuracy: 100
%  window: 4Mean accuracyS: 99.3039
% Swindow: 5 SNR: 1   Accuracy: 100
% Swindow: 5 SNR: 2   Accuracy: 68.75
% Swindow: 5 SNR: 3   Accuracy: 95.4545
% Swindow: 5 SNR: 4   Accuracy: 100
% Swindow: 5 SNR: 5   Accuracy: 100
% Swindow: 5 SNR: 6   Accuracy: 100
% Swindow: 5 SNR: 7   Accuracy: 100
% Swindow: 5 SNR: 8   Accuracy: 100
% Swindow: 5 SNR: 9   Accuracy: 97.561
% Swindow: 5 SNR: 10   Accuracy: 95.9184
%  window: 5Mean accuracyS: 95.7684
% numframesperNSR=200
% Swindow: 1 SNR: 1   Accuracy: 93.75
% Swindow: 1 SNR: 2   Accuracy: 76.1905
% Swindow: 1 SNR: 3   Accuracy: 91.6667
% Swindow: 1 SNR: 4   Accuracy: 90.9091
% Swindow: 1 SNR: 5   Accuracy: 72.2222
% Swindow: 1 SNR: 6   Accuracy: 96
% Swindow: 1 SNR: 7   Accuracy: 69.5652
% Swindow: 1 SNR: 8   Accuracy: 87.5
% Swindow: 1 SNR: 9   Accuracy: 68.75
% Swindow: 1 SNR: 10   Accuracy: 89.4737
%  window: 1Mean accuracyS: 83.6027
% Swindow: 2 SNR: 1   Accuracy: 100
% Swindow: 2 SNR: 2   Accuracy: 95.2381
% Swindow: 2 SNR: 3   Accuracy: 79.1667
% Swindow: 2 SNR: 4   Accuracy: 95.4545
% Swindow: 2 SNR: 5   Accuracy: 94.4444
% Swindow: 2 SNR: 6   Accuracy: 100
% Swindow: 2 SNR: 7   Accuracy: 91.3043
% Swindow: 2 SNR: 8   Accuracy: 87.5
% Swindow: 2 SNR: 9   Accuracy: 100
% Swindow: 2 SNR: 10   Accuracy: 84.2105
%  window: 2Mean accuracyS: 92.7319
% Swindow: 3 SNR: 1   Accuracy: 93.75
% Swindow: 3 SNR: 2   Accuracy: 90.4762
% Swindow: 3 SNR: 3   Accuracy: 91.6667
% Swindow: 3 SNR: 4   Accuracy: 100
% Swindow: 3 SNR: 5   Accuracy: 88.8889
% Swindow: 3 SNR: 6   Accuracy: 80
% Swindow: 3 SNR: 7   Accuracy: 78.2609
% Swindow: 3 SNR: 8   Accuracy: 50
% Swindow: 3 SNR: 9   Accuracy: 62.5
% Swindow: 3 SNR: 10   Accuracy: 94.7368
%  window: 3Mean accuracyS: 83.0279
% Swindow: 4 SNR: 1   Accuracy: 100
% Swindow: 4 SNR: 2   Accuracy: 100
% Swindow: 4 SNR: 3   Accuracy: 100
% Swindow: 4 SNR: 4   Accuracy: 100
% Swindow: 4 SNR: 5   Accuracy: 100
% Swindow: 4 SNR: 6   Accuracy: 100
% Swindow: 4 SNR: 7   Accuracy: 69.5652
% Swindow: 4 SNR: 8   Accuracy: 100
% Swindow: 4 SNR: 9   Accuracy: 100
% Swindow: 4 SNR: 10   Accuracy: 100
%  window: 4Mean accuracyS: 96.9565
% Swindow: 5 SNR: 1   Accuracy: 100
% Swindow: 5 SNR: 2   Accuracy: 61.9048
% Swindow: 5 SNR: 3   Accuracy: 95.8333
% Swindow: 5 SNR: 4   Accuracy: 86.3636
% Swindow: 5 SNR: 5   Accuracy: 83.3333
% Swindow: 5 SNR: 6   Accuracy: 92
% Swindow: 5 SNR: 7   Accuracy: 69.5652
% Swindow: 5 SNR: 8   Accuracy: 100
% Swindow: 5 SNR: 9   Accuracy: 93.75
% Swindow: 5 SNR: 10   Accuracy: 100
%  window: 5Mean accuracyS: 88.275
 
function [Data,Labels] = hGenerateTrainingData(dataSize, SNRdBs)
% Generate training data examples for channel estimation
% Run dataSize number of iterations to create random channel configurations
% and pass an OFDM-modulated fixed PDSCH grid with only the DM-RS symbols
% inserted. Perform perfect timing synchronization and OFDM demodulation,
% extracting the pilot symbols and performing linear interpolation at each
% iteration. Use perfect channel information to create the
% label data. The function returns 2 arrays - the training data and labels.

    fprintf('Starting data generation...\n')

    % List of possible channel profiles
    delayProfiles = {'TDL-A', 'TDL-B', 'TDL-C', 'TDL-D', 'TDL-E'};

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

    % Return DM-RS indices and symbols
    [~,dmrsIndices,dmrsSymbols,~] = hPDSCHResources(gnb,pdsch);

    % PDSCH mapping in grid associated with PDSCH transmission period
    pdschGrid = zeros(waveformInfo.NSubcarriers,waveformInfo.SymbolsPerSlot,nTxAnts);

    % PDSCH DM-RS precoding and mapping
    [~,dmrsAntIndices] = nrExtractResources(dmrsIndices,pdschGrid);
    pdschGrid(dmrsAntIndices) = pdschGrid(dmrsAntIndices) + dmrsSymbols;

    % OFDM modulation of associated resource elements
    txWaveform_original = hOFDMModulate(gnb,pdschGrid);

    % Acquire linear interpolator coordinates for neural net preprocessing
    [rows,cols] = find(pdschGrid ~= 0);
    dmrsSubs = [rows, cols, ones(size(cols))];
    hest = zeros(size(pdschGrid));
    [l_hest,k_hest] = meshgrid(1:size(hest,2),1:size(hest,1));

    % Preallocate memory for the training data and labels
    numExamples = dataSize * length(SNRdBs);
    Data = zeros([612 14 2 numExamples]);
    Labels = categorical(zeros([numExamples, 1]));
    
    % Main loop for data generation, iterating over the number of examples
    % specified in the function call. Each iteration of the loop produces a
    % new channel realization with a random delay spread, doppler shift,
    % and delay profile. Every perturbed version of the transmitted
    % waveform with the DM-RS symbols is stored in trainData, and the
    % perfect channel realization in trainLabels.
    for SNR_idx = 1:length(SNRdBs)
      SNRdB = double(SNRdBs(SNR_idx));             % Desired SNR in dB

      for i = 1:dataSize
        % Release the channel to change nontunable properties
        channel.release

        % Pick a random seed to create different channel realizations
        channel.Seed = randi([1001 2000]);

        % Pick a random delay profile, delay spread, and maximum doppler shift
        channel.DelayProfile = string(delayProfiles(randi([1 numel(delayProfiles)])));
        channel.DelaySpread = randi([1 300])*1e-9;
        channel.MaximumDopplerShift = randi([5 400]);

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
%         SNRdB = randi([0 10]);  % Random SNR values between 0 and 10 dB
        SNR = 10^(SNRdB/20);    % Calculate linear noise gain
        N0 = 1/(sqrt(2.0*nRxAnts*double(waveformInfo.Nfft))*SNR);
        noise = N0*complex(randn(size(rxWaveform)),randn(size(rxWaveform)));
        rxWaveform = rxWaveform + noise;

        % Perfect synchronization. Use information provided by the channel
        % to find the strongest multipath component
        pathFilters = getPathFilters(channel); % Get path filters for perfect channel estimation
        [offset,~] = nrPerfectTimingEstimate(pathGains,pathFilters);

        rxWaveform = rxWaveform(1+offset:end, :);

        % Perform OFDM demodulation on the received data to recreate the
        % resource grid, including padding in case practical
        % synchronization results in an incomplete slot being demodulated
        rxGrid = hOFDMDemodulate(gnb,rxWaveform);
        [K,L,R] = size(rxGrid);
        if (L < waveformInfo.SymbolsPerSlot)
            rxGrid = cat(2,rxGrid,zeros(K,waveformInfo.SymbolsPerSlot-L,R));
        end

        % Perfect channel estimation, using the value of the path gains
        % provided by the channel. This channel estimate does not
        % include the effect of transmitter precoding
%         estChannelGridPerfect = nrPerfectChannelEstimate(pathGains, ...
%             pathFilters,gnb.NRB,gnb.SubcarrierSpacing,0,offset, ...
%            sampleTimes,gnb.CyclicPrefix);

        % Linear interpolation
        dmrsRx = rxGrid(dmrsIndices);
        dmrsEsts = dmrsRx .* conj(dmrsSymbols);
        f = scatteredInterpolant(dmrsSubs(:,2),dmrsSubs(:,1),dmrsEsts);
        hest = f(l_hest,k_hest);

        % Split interpolated grid into real and imaginary components and
        % concatenate them along the third dimension, as well as for the
        % true channel response
        rx_grid = cat(3, real(hest), imag(hest));
%         est_grid = cat(3, real(estChannelGridPerfect), ...
%             imag(estChannelGridPerfect));

        % Add generated training example and label to the respective arrays
        index = i + (SNR_idx - 1) * dataSize;
        Data(:,:,:,index) = rx_grid;
%         trainLabels(:,:,:,i) = est_grid;
        Labels(index) = SNRdBs(SNR_idx);

      end
    % Data generation tracker
    fprintf('%3.2f%% complete\n',SNR_idx/length(SNRdBs) *100);

    end
    % randomize frames
    numFrames = size(Data, 4);
    shuffleIdx = randperm(numFrames);
    Data = Data(:, :, :, shuffleIdx);
    Labels = Labels(shuffleIdx);

    fprintf('Data generation complete numframes=%d!\n', numFrames);
end

function BoundLabel = SNR2BoundLabel(SNR, boundary)
   if (SNR < boundary)
       BoundLabel = categorical(-1);
   else
       BoundLabel = categorical(1);
   end
end

%number of bits per symbol
function M = ModTypes2M(x)
if x == "QPSK" 
    M = 2;
elseif x == "16QAM" 
    M = 4;
elseif x == "64QAM" 
    M = 6;
end
end

function SNR = SNRTypes2num(x)
if x == "-24" 
    SNR = -24;
elseif x == "-25" 
    SNR = -25;
elseif x == "-26" 
    SNR = -26;
elseif x == "-28" 
    SNR = -28;
elseif x == "-30" 
    SNR = -30;
elseif x == "-32" 
    SNR = -32;
elseif x == "-34" 
    SNR = -34;
elseif x == "-35" 
    SNR = -35;
elseif x == "-36" 
    SNR = -36;
elseif x == "-22" 
    SNR = -22;
elseif x == "-20" 
    SNR = -20;
elseif x == "-18" 
    SNR = -18;
elseif x == "-16" 
    SNR = -16;
elseif x == "-15" 
    SNR = -15;
elseif x == "-14" 
    SNR = -14;
elseif x == "-12" 
    SNR = -12;
elseif x == "-10" 
    SNR = -10;
elseif x == "-8" 
    SNR = -8;
elseif x == "-6" 
    SNR = -6;
elseif x == "-5" 
    SNR = -5;
elseif x == "-4" 
    SNR = -4;
elseif x == "-2" 
    SNR = -2;
elseif x == "0" 
    SNR = 0;
elseif x == "2" 
    SNR = 2;
elseif x == "4" 
    SNR = 4;
elseif x == "5" 
    SNR = 5;
elseif x == "6" 
    SNR = 6;
elseif x == "8" 
    SNR = 8;
elseif x == "10" 
    SNR = 10;
elseif x == "12" 
    SNR = 12;
elseif x == "14" 
    SNR = 14;
elseif x == "15" 
    SNR = 15;
elseif x == "16" 
    SNR = 16;
elseif x == "18" 
    SNR = 18;
elseif x == "20" 
    SNR = 20;
elseif x == "22" 
    SNR = 22;
elseif x == "24" 
    SNR = 24;
elseif x == "25" 
    SNR = 25;
elseif x == "26" 
    SNR = 26;
elseif x == "28" 
    SNR = 28;
elseif x == "30" 
    SNR = 30;
elseif x == "32" 
    SNR = 32;
elseif x == "34" 
    SNR = 34;
elseif x == "35" 
    SNR = 35;
elseif x == "36" 
    SNR = 36;
elseif x == "40" 
    SNR = 40;
end
end



function modulator = getModulator(modType, sps, fs)
%getModulator Modulation function selector
%   MOD = getModulator(TYPE,SPS,FS) returns the modulator function handle
%   MOD based on TYPE. SPS is the number of samples per symbol and FS is 
%   the sample rate.

switch modType
  case "BPSK"
    modulator = @(x)bpskModulator(x,sps);
  case "QPSK"
    modulator = @(x)qpskModulator(x,sps);
  case "8PSK"
    modulator = @(x)psk8Modulator(x,sps);
  case "16QAM"
    modulator = @(x)qam16Modulator(x,sps);
  case "32QAM"
    modulator = @(x)qam32Modulator(x,sps);
  case "64QAM"
    modulator = @(x)qam64Modulator(x,sps);
  case "128QAM"
    modulator = @(x)qam128Modulator(x,sps);
  case "256QAM"
    modulator = @(x)qam256Modulator(x,sps);
  case "GFSK"
    modulator = @(x)gfskModulator(x,sps);
  case "CPFSK"
    modulator = @(x)cpfskModulator(x,sps);
  case "PAM4"
    modulator = @(x)pam4Modulator(x,sps);
  case "B-FM"
    modulator = @(x)bfmModulator(x, fs);
  case "DSB-AM"
    modulator = @(x)dsbamModulator(x, fs);
  case "SSB-AM"
    modulator = @(x)ssbamModulator(x, fs);
end
end

function src = getSource(modType, sps, spf, fs)
%getSource Source selector for modulation types
%    SRC = getSource(TYPE,SPS,SPF,FS) returns the data source
%    for the modulation type TYPE, with the number of samples 
%    per symbol SPS, the number of samples per frame SPF, and 
%    the sampling frequency FS.

switch modType
  case {"BPSK","GFSK","CPFSK"}
%     M = 2;
    M = 1;
    src = @()randi([0 M-1],spf/sps,1);
  case {"QPSK","PAM4"}
    M = 4;
    src = @()randi([0 M-1],spf/sps,1);
  case "8PSK"
    M = 8;
    src = @()randi([0 M-1],spf/sps,1);
  case "16QAM"
    M = 16;
    src = @()randi([0 M-1],spf/sps,1);
  case "32QAM"
    M = 32;
    src = @()randi([0 M-1],spf/sps,1);
  case "64QAM"
    M = 64;
    src = @()randi([0 M-1],spf/sps,1);
  case "128QAM"
    M = 128;
    src = @()randi([0 M-1],spf/sps,1);
  case "256QAM"
    M = 256;
    src = @()randi([0 M-1],spf/sps,1);
  case {"B-FM","DSB-AM","SSB-AM"}
    src = @()getAudio(spf,fs);
end
end

function x = getAudio(spf,fs)
%getAudio Audio source for analog modulation types
%    A = getAudio(SPF,FS) returns the audio source A, with the 
%    number of samples per frame SPF, and the sample rate FS.

persistent audioSrc audioRC

if isempty(audioSrc)
  audioSrc = dsp.AudioFileReader('audio_mix_441.wav',...
    'SamplesPerFrame',spf,'PlayCount',inf);
  audioRC = dsp.SampleRateConverter('Bandwidth',30e3,...
    'InputSampleRate',audioSrc.SampleRate,...
    'OutputSampleRate',fs);
  [~,decimFactor] = getRateChangeFactors(audioRC);
  audioSrc.SamplesPerFrame = ceil(spf / fs * audioSrc.SampleRate / decimFactor) * decimFactor;
end

x = audioRC(audioSrc());
x = x(1:spf,1);
end

function frames = getNNFrames(rx,modType)
%getNNOFDM Symbols Generate formatted OFDM Symbols for neural networks
%   F = getNNOFDM Symbols(X,MODTYPE) formats the input X, into OFDM Symbols 
%   that can be used with the neural network designed in this 
%   example, and returns the OFDM Symbols in the output F.

frames = helperModClassFrameGenerator(rx,spf,spf,32,8);
frameStore = helperModClassFrameStore(10,spf,categorical({modType}));
add(frameStore,frames,modType);
frames = get(frameStore);
end

function plotScores(score,labels)
%plotScores Plot classification scores of OFDM Symbols
%   plotScores(SCR,LABELS) plots the classification scores SCR as a stacked 
%   bar for each frame. SCR is a matrix in which each row is the score for a 
%   frame.

co = [0.08 0.9 0.49;
  0.52 0.95 0.70;
  0.36 0.53 0.96;
  0.09 0.54 0.67;
  0.48 0.99 0.26;
  0.95 0.31 0.17;
  0.52 0.85 0.95;
  0.08 0.72 0.88;
  0.12 0.45 0.69;
  0.22 0.11 0.49;
  0.65 0.54 0.71];
figure; ax = axes('ColorOrder',co,'NextPlot','replacechildren');
bar(ax,[score; nan(2,11)],'stacked'); legend(categories(labels),'Location','best');
xlabel('Frame Number'); ylabel('Score'); title('Classification Scores')
end

function plotTimeDomain(rxTest,rxTestLabel,modulationTypes,fs)
%plotTimeDomain Time domain plots of OFDM Symbols

numRows = ceil(length(modulationTypes) / 4);
spf = size(rxTest,2);
t = 1000*(0:spf-1)/fs;
if size(rxTest,1) == 2
  IQAsRows = true;
else
  IQAsRows = false;
end
for modType=1:length(modulationTypes)
  subplot(numRows, 4, modType);
  idxOut = find(rxTestLabel == modulationTypes(modType), 1);
  if IQAsRows
    rxI = rxTest(1,:,1,idxOut);
    rxQ = rxTest(2,:,1,idxOut);
  else
    rxI = rxTest(1,:,1,idxOut);
    rxQ = rxTest(1,:,2,idxOut);
  end
  plot(t,squeeze(rxI), '-'); grid on; axis equal; axis square
  hold on
  plot(t,squeeze(rxQ), '-'); grid on; axis equal; axis square
  hold off
  title(string(modulationTypes(modType)));
  xlabel('Time (ms)'); ylabel('Amplitude')
end
end

function plotSpectrogram(rxTest,rxTestLabel,modulationTypes,fs,sps)
%plotSpectrogram Spectrogram of OFDM Symbols

if size(rxTest,1) == 2
  IQAsRows = true;
else
  IQAsRows = false;
end
numRows = ceil(length(modulationTypes) / 4);
for modType=1:length(modulationTypes)
  subplot(numRows, 4, modType);
  idxOut = find(rxTestLabel == modulationTypes(modType), 1);
  if IQAsRows
    rxI = rxTest(1,:,1,idxOut);
    rxQ = rxTest(2,:,1,idxOut);
  else
    rxI = rxTest(1,:,1,idxOut);
    rxQ = rxTest(1,:,2,idxOut);
  end
  rx = squeeze(rxI) + 1i*squeeze(rxQ);
  spectrogram(rx,kaiser(sps),0,spf,fs,'centered');
  title(string(modulationTypes(modType)));
end
h = gcf; delete(findall(h.Children, 'Type', 'ColorBar'))
end

function flag = isPlutoSDRInstalled
%isPlutoSDRInstalled Check if ADALM-PLUTO is installed

spkg = matlabshared.supportpkg.getInstalled;
flag = ~isempty(spkg) && any(contains({spkg.Name},'ADALM-PLUTO','IgnoreCase',true));
end

function y = bpskModulator(x,sps)
%bpskModulator BPSK modulator with pulse shaping
%   Y = bpskModulator(X,SPS) BPSK modulates the input X, and returns the 
%   root-raised cosine pulse shaped signal Y. X must be a column vector 
%   of values in the set [0 1]. The root-raised cosine filter has a 
%   roll-off factor of 0.35 and spans four symbols. The output signal 
%   Y has unit power.

persistent filterCoeffs
if isempty(filterCoeffs)
  filterCoeffs = rcosdesign(0.35, 4, sps);
end
% Modulate
syms = pskmod(x,2);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end

function y = qpskModulator(x,sps)
%qpskModulator QPSK modulator with pulse shaping
%   Y = qpskModulator(X,SPS) QPSK modulates the input X, and returns the 
%   root-raised cosine pulse shaped signal Y. X must be a column vector 
%   of values in the set [0 3]. The root-raised cosine filter has a 
%   roll-off factor of 0.35 and spans four symbols. The output signal 
%   Y has unit power.

persistent filterCoeffs
if isempty(filterCoeffs)
  filterCoeffs = rcosdesign(0.35, 4, sps);
end
% Modulate
syms = pskmod(x,4,pi/4);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end

function y = psk8Modulator(x,sps)
%psk8Modulator 8-PSK modulator with pulse shaping
%   Y = psk8Modulator(X,SPS) 8-PSK modulates the input X, and returns the 
%   root-raised cosine pulse shaped signal Y. X must be a column vector 
%   of values in the set [0 7]. The root-raised cosine filter has a 
%   roll-off factor of 0.35 and spans four symbols. The output signal 
%   Y has unit power.

persistent filterCoeffs
if isempty(filterCoeffs)
  filterCoeffs = rcosdesign(0.35, 4, sps);
end
% Modulate
syms = pskmod(x,8);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end

function y = qam16Modulator(x,sps)
%qam16Modulator 16-QAM modulator with pulse shaping
%   Y = qam16Modulator(X,SPS) 16-QAM modulates the input X, and returns the 
%   root-raised cosine pulse shaped signal Y. X must be a column vector 
%   of values in the set [0 15]. The root-raised cosine filter has a 
%   roll-off factor of 0.35 and spans four symbols. The output signal 
%   Y has unit power.

persistent filterCoeffs
if isempty(filterCoeffs)
  filterCoeffs = rcosdesign(0.35, 4, sps);
end
% Modulate and pulse shape
syms = qammod(x,16,'UnitAveragePower',true);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end

function y = qam32Modulator(x,sps)
%qam16Modulator 16-QAM modulator with pulse shaping
%   Y = qam16Modulator(X,SPS) 16-QAM modulates the input X, and returns the 
%   root-raised cosine pulse shaped signal Y. X must be a column vector 
%   of values in the set [0 15]. The root-raised cosine filter has a 
%   roll-off factor of 0.35 and spans four symbols. The output signal 
%   Y has unit power.

persistent filterCoeffs
if isempty(filterCoeffs)
  filterCoeffs = rcosdesign(0.35, 4, sps);
end
% Modulate and pulse shape
syms = qammod(x,32,'UnitAveragePower',true);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end

function y = qam64Modulator(x,sps)
%qam64Modulator 64-QAM modulator with pulse shaping
%   Y = qam64Modulator(X,SPS) 64-QAM modulates the input X, and returns the 
%   root-raised cosine pulse shaped signal Y. X must be a column vector 
%   of values in the set [0 63]. The root-raised cosine filter has a 
%   roll-off factor of 0.35 and spans four symbols. The output signal 
%   Y has unit power.

persistent filterCoeffs
if isempty(filterCoeffs)
  filterCoeffs = rcosdesign(0.35, 4, sps);
end
% Modulate
syms = qammod(x,64,'UnitAveragePower',true);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end

function y = qam128Modulator(x,sps)
%qam64Modulator 64-QAM modulator with pulse shaping
%   Y = qam64Modulator(X,SPS) 64-QAM modulates the input X, and returns the 
%   root-raised cosine pulse shaped signal Y. X must be a column vector 
%   of values in the set [0 63]. The root-raised cosine filter has a 
%   roll-off factor of 0.35 and spans four symbols. The output signal 
%   Y has unit power.

persistent filterCoeffs
if isempty(filterCoeffs)
  filterCoeffs = rcosdesign(0.35, 4, sps);
end
% Modulate
syms = qammod(x,128,'UnitAveragePower',true);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end

function y = qam256Modulator(x,sps)
%qam64Modulator 64-QAM modulator with pulse shaping
%   Y = qam64Modulator(X,SPS) 64-QAM modulates the input X, and returns the 
%   root-raised cosine pulse shaped signal Y. X must be a column vector 
%   of values in the set [0 63]. The root-raised cosine filter has a 
%   roll-off factor of 0.35 and spans four symbols. The output signal 
%   Y has unit power.

persistent filterCoeffs
if isempty(filterCoeffs)
  filterCoeffs = rcosdesign(0.35, 4, sps);
end
% Modulate
syms = qammod(x,256,'UnitAveragePower',true);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end

function y = pam4Modulator(x,sps)
%pam4Modulator PAM4 modulator with pulse shaping
%   Y = pam4Modulator(X,SPS) PAM4 modulates the input X, and returns the 
%   root-raised cosine pulse shaped signal Y. X must be a column vector 
%   of values in the set [0 3]. The root-raised cosine filter has a 
%   roll-off factor of 0.35 and spans four symbols. The output signal 
%   Y has unit power.

persistent filterCoeffs amp
if isempty(filterCoeffs)
  filterCoeffs = rcosdesign(0.35, 4, sps);
  amp = 1 / sqrt(mean(abs(pammod(0:3, 4)).^2));
end
% Modulate
syms = amp * pammod(x,4);
% Pulse shape
y = filter(filterCoeffs, 1, upsample(syms,sps));
end

function y = gfskModulator(x,sps)
%gfskModulator GFSK modulator
%   Y = gfskModulator(X,SPS) GFSK modulates the input X and returns the 
%   signal Y. X must be a column vector of values in the set [0 1]. The 
%   BT product is 0.35 and the modulation index is 1. The output signal 
%   Y has unit power.

persistent mod meanM
if isempty(mod)
  M = 2;
  mod = comm.CPMModulator(...
    'ModulationOrder', M, ...
    'FrequencyPulse', 'Gaussian', ...
    'BandwidthTimeProduct', 0.35, ...
    'ModulationIndex', 1, ...
    'SamplesPerSymbol', sps);
  meanM = mean(0:M-1);
end
% Modulate
y = mod(2*(x-meanM));
end

function y = cpfskModulator(x,sps)
%cpfskModulator CPFSK modulator
%   Y = cpfskModulator(X,SPS) CPFSK modulates the input X and returns 
%   the signal Y. X must be a column vector of values in the set [0 1]. 
%   the modulation index is 0.5. The output signal Y has unit power.

persistent mod meanM
if isempty(mod)
  M = 2;
  mod = comm.CPFSKModulator(...
    'ModulationOrder', M, ...
    'ModulationIndex', 0.5, ...
    'SamplesPerSymbol', sps);
  meanM = mean(0:M-1);
end
% Modulate
y = mod(2*(x-meanM));
end

function y = bfmModulator(x,fs)
%bfmModulator Broadcast FM modulator
%   Y = bfmModulator(X,FS) broadcast FM modulates the input X and returns
%   the signal Y at the sample rate FS. X must be a column vector of
%   audio samples at the sample rate FS. The frequency deviation is 75 kHz
%   and the pre-emphasis filter time constant is 75 microseconds.

persistent mod
if isempty(mod)
  mod = comm.FMBroadcastModulator(...
    'AudioSampleRate', fs, ...
    'SampleRate', fs);
end
y = mod(x);
end

function y = dsbamModulator(x,fs)
%dsbamModulator Double sideband AM modulator
%   Y = dsbamModulator(X,FS) double sideband AM modulates the input X and
%   returns the signal Y at the sample rate FS. X must be a column vector of
%   audio samples at the sample rate FS. The IF frequency is 50 kHz.

y = ammod(x,50e3,fs);
end

function y = ssbamModulator(x,fs)
%ssbamModulator Single sideband AM modulator
%   Y = ssbamModulator(X,FS) single sideband AM modulates the input X and
%   returns the signal Y at the sample rate FS. X must be a column vector of
%   audio samples at the sample rate FS. The IF frequency is 50 kHz.

y = ssbmod(x,50e3,fs);
end