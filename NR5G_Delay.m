%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 5/5/20 5G NR freq-domain subframe-based input size = 614x14x1: 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% mode = 0; % generating data
% domain = 0; %0: freq, 1: time, 2: freq+time
% mode = 5; % predicting data (5:1-5, 10:10, 15;15); 20: for statistics
% type = 1; % 0: greater than or less; 1:equal or not
% scheme = 1; % 0: multi-class, 1: binary + merged binary
% size = 2; %2: for 2000l 3: 3000

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

WINDOWS = [1 2 3 4 5 10 15];
numWin = length(WINDOWS);


rng('default');         % Configure random number generators
sps = 8;                % Samples per symbol

delayProfiles = {'TDL-A', 'TDL-B', 'TDL-C', 'TDL-D', 'TDL-E'};
DelaysN = categorical([1, 2, 3, 4, 5]);
numDelays = length(DelaysN);
boundaryLabel = [-1, 1];
BoundLabelTypes = categorical(boundaryLabel);
numFramesPerDelayType = size * 1000;
percentTrainingSamples = 80;
percentValidationSamples = 10;
percentTestSamples = 10;
numData = numFramesPerDelayType*numDelays; 
numValidation = numData * percentValidationSamples/100;
numTest = numData * percentTestSamples/100;
numTraining = numData * percentTrainingSamples/100;
numTestperDelay = numTest/numDelays;

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
if scheme == 0
    numkk = 1;
    numLabel = numDelays;
elseif type == 0
%     numkk = numBound;
%     testAccuracyI = zeros(numWin, numkk); 
%     numLabel = length(boundaryLabel);
else
    numkk = numDelays;
    testAccuracyI = zeros(numWin, numkk); 
    numLabel = length(boundaryLabel);
end

    disp("mode=" + string(mode) + " type=" + string(type) + " domain=" + string(domain) + " scheme=" + string(scheme) + " numkk=" + string(numkk) + " numData=" + numData); 

if mode == 0
    % Generate the training data
    disp("Generate the training data");
%     [Data, Labels] = hGenerateTrainingData(numFramesPerDopplerType, DopplerTypes, boundary, kk);
    [Data, Labels] = hGenerateTrainingData(numFramesPerDelayType, DelaysN, domain);
    Data = reshape(Data, [1 spf 2 numData]);
    save('NR5G' + string(size) + '000_' + string(domain) + '_Delay.mat', 'Data', 'Labels', '-v7.3');
elseif mode < 100 % pred if 5,10 or 15;  >100 compute pred statistics only
    disp("predicting");
    load('NR5G' + string(size) + '000_' + string(domain) + '_Delay.mat', 'Data', 'Labels');
    
%     Labels = fixabug(Labels);
        
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
    trainEnd = numData * percentTrainingSamples/100 + testEnd;
    rxTraining = Data(:,:,:,trainStart:trainEnd);
    rxTrainingLabel = Labels(trainStart:trainEnd);

    rxTestPredAll = cell(numWin, numkk); 

    % dropoutRate = 0.5;
    netWidth = 1;
    filterSize = [1 sps];
    poolSize = [1 2];
    numHiddenUnits = 200;
    maxEpochs = 12;  %is significant
    miniBatchSize = 64; %256;

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
    
    validationFrequency = floor(numValidation/miniBatchSize);

    parpool(numkk);
  parfor kk = 1:numkk
% for kk = 1:numkk
        if scheme == 0
            rxValidationLabelkk = rxValidationLabel;
            rxTestLabelkk = rxTestLabel;
            rxTrainingLabelkk = rxTrainingLabel;
        else
            if type == 0
%                 Delay = Delays(kk+1);  % matlab bug? error on index of 8
%     %             Delay = boundary(kk);
            else
                Delay = DelaysN(kk);
            end
            rxValidationLabelkk = Delay2BoundLabel(rxValidationLabel, Delay, type);
            rxTestLabelkk = Delay2BoundLabel(rxTestLabel, Delay, type);
            rxTrainingLabelkk = Delay2BoundLabel(rxTrainingLabel, Delay, type);
        end
        
        div = 4;
        %parpool(winend-winstart+1);
        for window = winstart:winend
           seqTrainingdiv = cell(numTraining/div, 1);
           seqValidationdiv = cell(numValidation/div, 1);

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
                        disp("validation accuracy: " + string(info.FinalValidationAccuracy));
                    else
                        [trainedNetS, info] = trainNetwork(seqTrainingdiv,rxTrainingLabelkkdiv,layerGraph(trainedNetS),options);
                        disp("validation accuracy: " + string(info.FinalValidationAccuracy));
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
                disp("Window: " + WINDOWS(window) +" Delay: "+delayProfiles(kk)+" AccuracyI: "+testAccuracyI);                
            else
                % Single-CNN classifier accuracy
                rxTestPredS = zeros(1.5 * numTestperDelay, numDelays); %the merged predict = bottom
%                 sum_square_errorS = zeros(numDelays, 1);
                sumAccS = zeros(numDelays, 1);
                cntAcc = zeros(numDelays, 1);
                for i = 1:numTest
                    pred = rxTestPred(i);
                    label = rxTestLabel(i);
                    index = find(DelaysN == label);
                    sumAccS(index) = sumAccS(index) + double((label == pred));
                    cntAcc(index) = cntAcc(index) + 1;
                    rxTestPredS(cntAcc(index), index) = pred;
                end
                for i = 1:numDelays
                    testAccuracyS = 100 * sumAccS(i) / cntAcc(i);
                    disp("Window: " + WINDOWS(window) + " Delay: "+delayProfiles(i)+" AccuracyS: "+testAccuracyS);
                end            
            end
        end
 end    
    save('../../../work/bal718/NR5G_Delay_predAll_' + string(mode) + '_' + string(type) + '_' + string(domain) + '_' + string(scheme) + '.mat',  'rxTestPredAll', 'rxTestLabel', '-v7.3');
%    save('NR5G_Delay_predAll_' + string(mode) + '_' + string(type) + '_' + string(domain) + string(scheme) + '.mat',  'rxTestPredAll', 'rxTestLabel', '-v7.3');
end

% if mode > 0  % 5,10,15: pred & compute merged statistics; 105: statistics for 5, 11-: for 10, 115: for 15
%     if mode > 100
%         mode = mode - 100;
%         load('../../../work/bal718/NR5G_Delay_predAll_' + string(mode) + '_' + string(type) + '_' + string(domain) + string(scheme) + '.mat', 'rxTestPredAll', 'rxTestLabel');
% %         load('NR5G_Delay_predAll_' + string(mode) + '_' + string(type) + '_' + string(domain) + '_' + string(scheme) + '.mat', 'rxTestPredAll', 'rxTestLabel');
%     end
%     disp('merged prediction');
%     numTestperDelay = numTest/numDelays;
%     for w = winstart:winend
%         cntAcc = zeros(numDelays, 1);        
%         if scheme == 0  % multi-class 
%             rxTestPredS = zeros(1.5 * numTestperDelay, numDelays); %the merged predict = bottom
% %             sum_square_errorS = zeros(numDelays, 1);
%             rxTestPred = rxTestPredAll{w, 1};
%             sumAccS = zeros(numDelays, 1);
%             for i = 1:numTest
%                 pred = rxTestPred(i);
%                 label = rxTestLabel(i);
%                 index = find(delayProfiles == label);
%                 sumAccS(index) = sumAccS(index) + double((label == pred));
%                 cntAcc(index) = cntAcc(index) + 1;
%                 rxTestPredS(cntAcc(index), index) = pred;
%             end
%         elseif type == 0 % less/greater  
%         end
%         if scheme > 0  % binary
%         else
%             for i = 1:numDelays
%                 testAccuracyS = 100 * sumAccS(i) / cntAcc(i);
%                 disp("Window: " + WINDOWS(w) +" Delay: "+Delays(i)+" AccuracyS: "+testAccuracyS);
%             end            
%         end
%     end
% 
% end
function [Data,Labels] = hGenerateTrainingData(dataSize, DelaysN, domain)
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
    numDelaysplerTypes = length(DelaysN);
    numExamples = dataSize * numDelaysplerTypes;
    Data = zeros([1 spf 2 numExamples]);
    Labels = categorical(ones([numExamples, 1]));

%     % List of possible channel profiles
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

    for Delay_idx = 1:length(DelaysN)
%     for Doppler_idx = 1:numDelaysplerTypes
%       Doppler = DopplerTypes2num(Delays(Doppler_idx));             % Desired SNR in dB
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
    pdschGrid(dmrsAntIndices) = dmrsSymbols;

    % OFDM modulation of associated resource elements
    txWaveform_original = hOFDMModulate(gnb,pdschGrid);

    % Acquire linear interpolator coordinates for neural net preprocessing
%     [rows,cols] = find(pdschGrid ~= 0);
%     dmrsSubs = [rows, cols, ones(size(cols))];
%     hest = zeros(size(pdschGrid));
%     [l_hest,k_hest] = meshgrid(1:size(hest,2),1:size(hest,1));

%     % Main loop for data generation, iterating over the number of examples
%     % specified in the function call. Each iteration of the loop produces a
%     % new channel realization with a random delay spread, doppler shift,
%     % and delay profile. Every perturbed version of the transmitted
%     % waveform with the DM-RS symbols is stored in trainData, and the
%     % perfect channel realization in trainLabels.
%     for Channel_idx = 1:length(DelaysN)
% %       Doppler = double(Delays(Doppler_idx));             % Desired SNR in dB
% 
%       for i = 1:dataSize
        % Release the channel to change nontunable properties
        channel.release

        % Pick a random seed to create different channel realizations
        channel.Seed = randi([1001 2000]);

        % Pick a random delay profile, delay spread, and maximum doppler shift
%         channel.DelayProfile = string(Delays(randi([1 numel(Delays)])));
        channel.DelayProfile = string(delayProfiles(Delay_idx));
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
        SNRdB = randi([0 20]);  % Random SNR values between 0 and 10 dB
        SNR = 10^(SNRdB/20);    % Calculate linear noise gain
        N0 = 1/(sqrt(2.0*nRxAnts*double(waveformInfo.Nfft))*SNR);
        noise = N0*complex(randn(size(rxWaveform)),randn(size(rxWaveform)));
        rxWaveform = rxWaveform + noise;

%         % Perfect synchronization. Use information provided by the channel
%         % to find the strongest multipath component
%         pathFilters = getPathFilters(channel); % Get path filters for perfect channel estimation
%         [offset,~] = nrPerfectTimingEstimate(pathGains,pathFilters);
% 
%         rxWaveform = rxWaveform(1+offset:end, :);

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

            Labels(index) = DelaysN(Delay_idx);

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

function Labelskk = Delay2BoundLabel(Labels, Delay, type)
   Labelskk = categorical(ones(length(Labels), 1));
   for j=1:length(Labels)
       if type == 0  % greater than or less
%            if (DelayTypes2num(Labels(j)) < Delay) %true label < predictor label
%                Labelskk(j) = categorical(-1);
%            else
%                Labelskk(j) = categorical(1);
%            end
       else % equal or not
           if (Labels(j) == Delay)
               Labelskk(j) = categorical(-1);
           else
               Labelskk(j) = categorical(1);
           end
       end
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
