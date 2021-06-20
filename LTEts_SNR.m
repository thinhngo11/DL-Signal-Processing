%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 5/13/20 time-domain subframe-based input size = 1920x1 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% modulationTypes = categorical(["BPSK", "QPSK", "8PSK", ...
%   "16QAM", "64QAM", "PAM4", "GFSK", "CPFSK", ...
%   "B-FM", "DSB-AM", "SSB-AM"]);
% modulationTypes = categorical(["BPSK", "QPSK", "8PSK", ...
%   "16QAM", "64QAM", "PAM4", "GFSK", "CPFSK"]);
% modulationTypes = categorical(["BPSK", "QPSK", "8PSK", ...
%   "16QAM", "64QAM", "128QAM", "256QAM"]);
% modulationTypes = categorical(["BPSK", "QPSK", "8PSK", ...
%   "16QAM", "64QAM", "128QAM"]);
% modulationTypes = categorical(["BPSK", "QPSK", "8PSK", ...
%   "16QAM", "32QAM", "64QAM", "128QAM", "256QAM"]);
% modulationTypes = categorical(["BPSK", "QPSK", "8PSK", ...
%   "16QAM",% 
% modulationTypes = categorical(["QPSK", "16QAM", "64QAM"]);
% modulationTypes = categorical(["QPSK"]);
% modulationTypes = categorical(["BPSK", "8PSK", "64QAM"]);
modulationTypes = ["QPSK", "16QAM", "64QAM"];

numModulationTypes = length(modulationTypes);
% SNRs = [-20, -16, -12, -8, -4, 0, 4, 8, 12, 16, 20, 24, 28, 32];
% snr_inc = SNRs(2) - SNRs(1);
% boundary = SNRs(2 : length(SNRs)-1);
% SNRTypes = categorical(SNRs);
% numSNRTypes = length(SNRs);

% Dopplers = [0 20 40 60 80 100 120];
% Dopplers = [0 40 80 120 160 200];
Dopplers = [0 60 120];
% doppler_inc = Dopplers(2) - Dopplers(1);
Delays = ["EPA" "EVA" "ETU"]; % 0 25us
% Delays = [0 2 5]; % 0 25us
% boundary = Dopplers(2 : length(Dopplers)-1);
% boundaryLabel = [-1, 1];
% BoundLabelTypes = categorical(boundaryLabel);
% DopplerTypes = categorical(Dopplers);
numDopplerTypes = length(Dopplers);
numDelayTypes = length(Delays);

% numFramesPerModType = 100;
% numFramesPerModType = 1000;
percentTrainingSamples = 80;
percentValidationSamples = 10;
percentTestSamples = 10;

% sps = 8;                % Samples per symbol
sps = 8;                % Samples per symbol
% spf = 3840;             % Samples per frame
% symbolsPerFrame = spf / sps;
% fs = 200e3;             % Sample rate
% fc = [900e6 100e6];     % Center frequencies

rng(1235)
% tic

enb.NDLRB = 6;                 % Number of resource blocks
enb.CellRefP = 1;               % One transmit antenna port
enb.NCellID = 10;               % Cell ID
enb.CyclicPrefix = 'Normal';    % Normal cyclic prefix
enb.DuplexMode = 'FDD';         % FDD

% 128 REs / OFDM symbol
spf = 30720 / 2048 * 128; %1920 time-domain subframe-based
% spf = 72; %OFDM symbol based prediction
% spf = 1008; %subframe frequency domain based prediction 72 x 14

cfg.Seed = 1;                  % Channel seed
cfg.NRxAnts = 1;               % 1 receive antenna
cfg.DelayProfile = 'EVA';      % EVA delay spread
cfg.DopplerFreq = 120;         % 120Hz Doppler frequency
cfg.MIMOCorrelation = 'Low';   % Low (no) MIMO correlation
cfg.InitTime = 0;              % Initialize at time zero
cfg.NTerms = 16;               % Oscillators used in fading model
cfg.ModelType = 'GMEDS';       % Rayleigh fading model type
cfg.InitPhase = 'Random';      % Random initial phases
cfg.NormalizePathGains = 'On'; % Normalize delay profile power 
cfg.NormalizeTxAnts = 'On';    % Normalize for transmit antennas

cec.PilotAverage = 'UserDefined'; % Pilot averaging method
cec.FreqWindow = 9;               % Frequency averaging window in REs
cec.TimeWindow = 9;               % Time averaging window in REs

gridsize = lteDLResourceGridSize(enb);
K = gridsize(1);    % Number of subcarriers (72)
L = gridsize(2);    % Number of OFDM symbols in one subframe (14)
P = gridsize(3);    % Number of transmit antenna ports

txGrid = [];

% Number of bits needed is size of resource grid (K*L*P) * number of bits
% per symbol (2 for QPSK)
% numberOfBits = K*L*P*2; 

% dropoutRate = 0.5;
netWidth = 1;
filterSize = [1 sps];
poolSize = [1 2];
numHiddenUnits = 200;
% WINDOWS = [1 2 3 4 5 10 15];
% WINDOWS = [1 2 3 4 5];
WINDOWS = [1];
maxEpochs = 12;  %is significant
miniBatchSize = 256;

% rxTestPredAll = cell(length(WINDOWS), length(boundary));
% testAccuracyI = zeros(length(WINDOWS), length(boundary));
% testAccuracyM = zeros(length(WINDOWS), length(SNRs));
winlen = length(WINDOWS);


% SNRdB = 22;             % Desired SNR in dB
% SNR = 10^(SNRdB/20);    % Linear SNR  
rng('default');         % Configure random number generators


disp('Single input predictions');
% SNRs = [-4, 0, 4, 8, 12, 16, 20, 24, 28, 32];
% SNRs = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32];
SNRs = [-20, -16, -12, -8, -4, 0, 4, 8, 12, 16, 20, 24, 28, 32];
% SNRs = [-4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32];
% snr_inc = SNRs(2) - SNRs(1);
% boundary = SNRs(2 : length(SNRs)-1);
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
% kk = 1:length(boundary) + 1;
% frameStore = helperModClassFrameStore(...
%   numFramesPerModType*numModulationTypes*numSNRTypes*numDelayTypes*numDopplerTypes,spf,SNRTypes);
% spf = 3840;
numFramesPerSNRType = 100;
frameStore = helperModClassFrameStore(...
  numFramesPerSNRType*numSNRTypes*numModulationTypes*numDopplerTypes*numDelayTypes,spf,SNRTypes);
%   numFramesPerSNRType*numSNRTypes*numModulationTypes*numDopplerTypes*numDelayTypes*14,spf,SNRTypes);

for Modulation_idx = 1:numModulationTypes
    % Number of bits needed is size of resource grid (K*L*P) * number of bits
    % per symbol (2 for QPSK)
%     numberOfBits = K*L*P*2; 
    numberOfBits = K*L*P*ModTypes2M(modulationTypes(Modulation_idx));
    
for Doppler_idx = 1:numDopplerTypes
    cfg.DopplerFreq = Dopplers(Doppler_idx);         % 120Hz Doppler frequency

for Delay_idx = 1:numDelayTypes
    cfg.DelayProfile = Delays(Delay_idx);      % EVA delay spread

for SNR_idx = 1:numSNRTypes
    SNRdB = SNRs(SNR_idx);             % Desired SNR in dB
    SNR = 10^(SNRdB/20);    % Linear SNR  

    for fn = 1:numFramesPerSNRType/10
        txGrid = [];
        %11 subframes for sync purposes
        for sf = 0:10
            % Create random bit stream
            inputBits = randi([0 1], numberOfBits, 1); 

            % Modulate input bits (1008 = 72 x 14)
%             inputSym = lteSymbolModulate(inputBits,'QPSK');
            inputSym = lteSymbolModulate(inputBits,modulationTypes(Modulation_idx));

            % Set subframe number
            enb.NSubframe = mod(sf,10);

            % Generate empty subframe
            subframe = lteDLResourceGrid(enb);

            % Map input symbols to grid (72x14)
            subframe(:) = inputSym;

            % Generate synchronizing signals
            pssSym = ltePSS(enb);
            sssSym = lteSSS(enb);
            pssInd = ltePSSIndices(enb);
            sssInd = lteSSSIndices(enb);

            % Map synchronizing signals to the grid
            subframe(pssInd) = pssSym;
            subframe(sssInd) = sssSym;

            % Generate cell specific reference signal symbols and indices
            cellRsSym = lteCellRS(enb);
            cellRsInd = lteCellRSIndices(enb);

            % Map cell specific reference signal to grid
            subframe(cellRsInd) = cellRsSym;

            % Append subframe to grid to be transmitted 
            txGrid = [txGrid subframe]; %#ok (72 x 14)
        end
            [txWaveform,info] = lteOFDMModulate(enb,txGrid); 
            txGrid = txGrid(:,1:140);

            cfg.SamplingRate = info.SamplingRate;

            % Pass data through the fading channel model (1920 x1)
            rxWaveform = lteFadingChannel(cfg,txWaveform);

            % Calculate noise gain
            N0 = 1/(sqrt(2.0*enb.CellRefP*double(info.Nfft))*SNR);

            % Create additive white Gaussian noise
            noise = N0*complex(randn(size(rxWaveform)),randn(size(rxWaveform)));   

            % Add noise to the received time domain waveform
            rxWaveform = rxWaveform + noise;

            %Sync - removing first subframe as sync
            offset = lteDLFrameOffset(enb,rxWaveform);
%             rxWaveform = rxWaveform(1+offset:end,:);
            rxWaveform = rxWaveform(1+spf:end,:);
%             
%             % demodulation
%             rxGrid = lteOFDMDemodulate(enb,rxWaveform);
            
%             for s = 1:140
%                 OFDMSym = rxGrid(:, s);
% 
%                 % Remove transients from the beginning, trim to size, and normalize
% %                 frame = zeros([size(rxWaveform),1],class(rxWaveform));
% %                   frame(:,1) = rxWaveform;
%                 frame = zeros([size(OFDMSym),1],class(OFDMSym));
%                 frame(:,1) = OFDMSym;
%                 % Add to frame store
%     %             add(frameStore, frame, SNRTypes(SNR));
%                 add(frameStore, frame, SNRTypes(SNR_idx));
%             end
            for s = 1:10
%                 OFDMSym = rxGrid(:, 1 + (s - 1) * 14 : s * 14);
%                 subframe = reshape(OFDMSym, [1008 1]);
                % Remove transients from the beginning, trim to size, and normalize
%                 frame = zeros([size(rxWaveform),1],class(rxWaveform));
%                   frame(:,1) = rxWaveform;
                subframe = rxWaveform(1 + spf * (s - 1) : spf * s, :);
                frame = zeros([size(subframe),1],class(subframe));
                frame(:,1) = subframe;
                % Add to frame store
    %             add(frameStore, frame, SNRTypes(SNR));
                add(frameStore, frame, SNRTypes(SNR_idx));
            end
%         end
     end
end
end
end
end
[mcfsTraining,mcfsValidation,mcfsTest] = splitData(frameStore,...
  [percentTrainingSamples,percentValidationSamples,percentTestSamples]);
mcfsTraining.OutputFormat = "IQAsPages";
[rxTraining,rxTrainingLabel] = get(mcfsTraining);
mcfsValidation.OutputFormat = "IQAsPages";
[rxValidation,rxValidationLabel] = get(mcfsValidation);
mcfsTest.OutputFormat = "IQAsPages";
[rxTest,rxTestLabel] = get(mcfsTest);

% rxTraining = cat(3, rxTrainingI(1, :, :, :), rxTrainingI(1, :, :, :));
% rxValidation = cat(3, rxValidationI(1, :, :, :), rxValidationI(1, :, :, :));
% rxTest = cat(3, rxTestI(1, :, :, :), rxTestI(1, :, :, :));

validationFrequency = floor(numel(rxTrainingLabel)/miniBatchSize);

% parpool(7);
% parfor window = 1:length(WINDOWS)
% parpool(7);
testAccuracy = zeros(winlen, numSNRTypes, 1);
nMSE = zeros(winlen, numSNRTypes, 1);
MSE = zeros(winlen, numSNRTypes, 1);

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
%   fprintf('%s - Training the network\n', datestr(toc/86400,'HH:MM:SS'))
    trainedNetS = trainNetwork(seqTraining,rxTrainingLabel,lgraph,options);
else
%   load trainedModulationClassificationNetwork
end

% fprintf('%s - Classifying test OFDM Symbols\n', datestr(toc/86400,'HH:MM:SS'));
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
% testAccuracy = zeros(winlen, numSNRTypes, 1);
% nMSE = zeros(winlen, numSNRTypes, 1);
% MSE = zeros(winlen, numSNRTypes, 1);
TestPrednum3 = zeros((length(rxTestPred)/numSNRTypes), 3);

for SNR = 1:numSNRTypes
    meanTestPred(SNR) = mean(TestPrednum(:,SNR));
%     stdTestPred(SNR) = std(TestPrednum(:,SNR));
%     movingTestPred(:, SNR) = movmean(TestPrednum(:,SNR), 25);
    sum_square_error = 0;
    for i = 1:numTestDataperSNR
        sum_square_error = sum_square_error + (movingTestPred(i,SNR) - SNRTypes2num(SNRTypes(SNR))) ^ 2;
    end
    MSE(SNR) = sum_square_error / numTestDataperSNR;
    if SNRTypes2num(SNRTypes(SNR)) == 0
        nMSE(SNR) = (sum_square_error / numTestDataperSNR) / (meanTestPred(SNR) ^ 2);
    else 
        nMSE(SNR) = (sum_square_error / numTestDataperSNR) / (SNRTypes2num(SNRTypes(SNR)) ^ 2);
    end
%     movingTestPredRound(:, SNR) = round(movmean(TestPrednum(:,SNR), 25)/snr_inc)*snr_inc;
%     movingtestAccuracy(SNR) = 100 * mean(movingTestPredRound(:, SNR) == SNRTypes2num(SNRTypes(SNR)));
    testAccuracy(SNR) = 100 * mean(TestPrednum(:, SNR) == SNRTypes2num(SNRTypes(SNR)));
    disp("Swindow: " + window + " SNR: " + SNRTypes2num(SNRTypes(SNR)) + "   Accuracy: " + movingtestAccuracy(SNR) + "%   nMSE: " + nMSE(SNR));
end
% removing the 1st and the last SNR data due to one-sided average
% SNRsR = SNRs(2:numSNRTypes-1);
% TestPrednum = TestPrednum(:, 2:numSNRTypes-1);
% movingtestAccuracy = movingtestAccuracy(2:numSNRTypes-1);
% testAccuracy = testAccuracy(2:numSNRTypes-1);
% meanTestPred = meanTestPred(2:numSNRTypes-1);
% movingTestPred = movingTestPred(:, 2:numSNRTypes-1);
% movingTestPredRound = movingTestPredRound(:, 2:numSNRTypes-1);
% nMSE = nMSE(2:numSNRTypes-1);
% MSE = MSE(2:numSNRTypes-1);

% Single-CNN classifier accuracy
%     lenTest = length(rxTestLabel);
%     sumAcc = zeros(length(SNRs), 1);
%     for i = 1:lenTest
%         index = find(SNRs == SNRTypes2num(rxTestLabel(i)));
%         sumAcc(index) = sumAcc(index) + double((SNRTypes2num(rxTestLabel(i)) == SNRTypes2num(rxTestPred(i))));
%     end
%     testAccuracyS(window, :) = 100 * length(SNRs) * sumAcc / lenTest;
%     for i = 1:numSNRTypes
%     disp("Swindow: " + window + " SNR: " + i + "   Accuracy: " + testAccuracyS(window, i));
%     end
%     disp(" window: " + window + "Mean accuracyS: " + mean(testAccuracyS(window, :)));
end

TestPrednum3(:,1) = TestPrednum(:,3);
TestPrednum3(:,2) = TestPrednum(:,6);
TestPrednum3(:,3) = TestPrednum(:,9);


% for w = 1:winlen
%     disp(" window: " + w + "Mean accuracyS: " + mean(testAccuracyS(w, :)));
%     for i = 1:numSNRTypes
%         disp("Swindow: " + w + " SNR: " + i + "   Accuracy: " + testAccuracyS(w, i));
%     end
% end
%type1 = [-4, 0, 4, 8, 12, 16, 20, 24, 28, 32]
%type2 = [-4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
% type3 = [-20, -16, -12, -8, -4, 0, 4, 8, 12, 16, 20, 24, 28, 32]
% type4 = [-20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32];

figure
plot(SNRs, meanTestPred);
% title("Predicted vs True SNRs");
ylabel("Predicted SNR(dB)");
xlabel("True SNR(dB)");
xticks(SNRs);
yticks(SNRs);
grid on
xlim([min(SNRs), max(SNRs)]);
ylim([min(SNRs), max(SNRs)]);

figure
plot(TestPrednum3);
ylabel("Predicted SNR(dB)");
xlabel("OFDM Symbol Index");
yticks(SNRs);
xlim([0, numTestDataperSNR]);
ylim([min(SNRs), max(SNRs)]);


figure
yyaxis left
% plot(SNRs, nMSE);
semilogy(SNRs, nMSE);
% title("Normalized MSE vs True SNRs");
ylabel("Normalized MSE");
xlabel("True SNR(dB)");
xticks(SNRs);
grid on
xlim([min(SNRs), max(SNRs)]);
yyaxis right
semilogy(SNRs, MSE);
ylabel("MSE");


figure
plot(SNRs, MSE);
% title("MSE vs True SNRs");
ylabel("MSE");
xlabel("True SNR");
xticks(SNRs);
grid on
xlim([min(SNRs), max(SNRs)]);

xlim([-10 0]);


figure
plot(SNRs, testAccuracy);
% title("Accuracy vs True SNRs");
ylabel("Individual Accuracy (%)");
xlabel("True SNR");
xticks(SNRs);
yticks([0 10 20 30 40 50 60 70 80 90 100]);
grid on
ylim([0 100]);
xlim([min(SNRs), max(SNRs)]);


% save('LTEts_SNR_sequence_S_type1', 'testAccuracyS')
% function BoundLabel = SNR2BoundLabel(SNR, boundary)
%    if (SNR < boundary)
%        BoundLabel = categorical(-1);
%    else
%        BoundLabel = categorical(1);
%    end
% end

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