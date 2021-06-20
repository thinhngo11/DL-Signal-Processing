
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 8/21/20 for all domains, types, and modes
% 5/26/20 all 100% at 10000 numframes Doppler 5G NR time-domain subframe-based input size = 15,736: 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% mode = 0; % generating data
mode = 10; % predicting data (5:1-5, 10:10, 15;15); 20: for statistics
type = 0; % greater than or less; 1:equal or not
domain = 1; %0: freq, 1: time, 2: freq+time
scheme = 1; % 0: multi-class, 1: binary + merged binary

% 128 REs / OFDM symbol
spf_ts = 30720 / 2048 * 128; %1920 time-domain subframe-based
% spf = 72; %OFDM symbol based prediction
spf_fs = 1008; %subframe frequency domain based prediction 72 x 14
% spf_tf = 1920 + 1008;
spf_tfs = spf_fs + spf_ts;
if domain == 0
   spf = spf_fs;
elseif domain == 1
   spf = spf_ts;
else
   spf = spf_tfs;
end

% WINDOWS = [1 2 3 4 5];
% WINDOWS = [20];
WINDOWS = [1 2 3 4 5 10 15];
numWin = length(WINDOWS);

rng('default');         % Configure random number generators

% SNRs = [0, 10, 20];
% Dopplers = [0 20 40 60 80 100 120];
% Dopplers = [0 40 80 120 160 200];
% Dopplers = [0 40 80 120 160 200 240 280 320 360];
Dopplers = [0 50 100 150 200 250 300 350 400 450 500 550];
doppler_inc = Dopplers(2) - Dopplers(1);
% Delays = [0 2 5]; % 0 25us
boundary = Dopplers(2 : length(Dopplers));
boundaryLabel = [-1, 1];
BoundLabelTypes = categorical(boundaryLabel);
numDop = length(Dopplers);
numBound = numDop - 1;
modulationTypes = ["QPSK", "16QAM", "64QAM"];
Delays = ["EPA" "EVA" "ETU"]; % 0 25us
numModulationTypes = length(modulationTypes);
numDelayTypes = length(Delays);
numDopplerTypes = length(Dopplers);
numDop = length(Dopplers);
DopplerTypes = categorical(Dopplers);

if scheme == 0
    numkk = 1;
    testAccuracyS = zeros(numWin, numDop);
    numLabel = numDop;
elseif type == 0
  numkk = numBound;
  testAccuracyI = zeros(numWin, numkk); 
  numLabel = length(boundaryLabel);
else
  numkk = numDop;
  testAccuracyI = zeros(numWin, numkk); 
  numLabel = length(boundaryLabel);
end
rxTestPredAll = cell(numWin, numkk); 

numFramesPerDopplerType = 500;
percentTrainingSamples = 80;
percentValidationSamples = 10;
percentTestSamples = 10;
numData = numFramesPerDopplerType*numModulationTypes*numDop*numDelayTypes; 
numValidation = numData * percentValidationSamples/100;
numTest = numData * percentTestSamples/100;
numTraining = numData * percentTrainingSamples/100;
numTestperDop = numTest / numDop;

disp("mode=" + string(mode) + " type=" + string(type) + " domain=" + string(domain) + " scheme=" + string(scheme) + " numkk=" + string(numkk) + " numData=" + numData); 

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

    enb.NDLRB = 6;                 % Number of resource blocks
    enb.CellRefP = 1;               % One transmit antenna port
    enb.NCellID = 10;               % Cell ID
    enb.CyclicPrefix = 'Normal';    % Normal cyclic prefix
    enb.DuplexMode = 'FDD';         % FDD

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

%     numFramesPerDopplerType = 100;
    frameStore = helperModClassFrameStore(...
      numFramesPerDopplerType*numModulationTypes*numDopplerTypes*numDelayTypes,spf,DopplerTypes);
    %   numFramesPerDopplerType*numDopplerTypes*numModulationTypes*numDopplerTypes*numDelayTypes*14,spf,DopplerTypes);

    for Modulation_idx = 1:numModulationTypes
        % Number of bits needed is size of resource grid (K*L*P) * number of bits
        % per symbol (2 for QPSK)
    %     numberOfBits = K*L*P*2; 
        numberOfBits = K*L*P*ModTypes2M(modulationTypes(Modulation_idx));

      for Delay_idx = 1:numDelayTypes
        cfg.DelayProfile = Delays(Delay_idx);      % EVA delay spread

        for Doppler_idx = 1:numDopplerTypes
          cfg.DopplerFreq = Dopplers(Doppler_idx);         % 120Hz Doppler frequency

          for fn = 1:numFramesPerDopplerType/10
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

                for nn = 1:10

                % Pass data through the fading channel model (1920 x1)
                rxWaveform = lteFadingChannel(cfg,txWaveform);

                % Calculate noise gain
                SNRdB = randi([-10 20]);  % Random SNR values between 0 and 10 dB
                SNR = 10^(SNRdB/20);    % Calculate linear noise gain

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
                rxGrid = lteOFDMDemodulate(enb,rxWaveform);
                    if size(rxGrid, 2) == 140
                        break
                    end
                end

                for s = 1:10
                    if domain == 0 % freq-domain
                        OFDMSym = rxGrid(:, 1 + (s - 1) * 14 : s * 14);
                        subframe_f = reshape(OFDMSym, [1008 1]);
                        % Remove transients from the beginning, trim to size, and normalize
        %                 frame = zeros([size(rxWaveform),1],class(rxWaveform));
        %                 frame(:,1) = rxWaveform;
                        add(frameStore, subframe_f, DopplerTypes(Doppler_idx));
                    elseif domain == 1 % time-domain
                        subframe_t = rxWaveform(1 + spf * (s - 1) : spf * s, :);
                        add(frameStore, subframe_t, DopplerTypes(Doppler_idx));
                    else % freq+time
                        subframe = vertcat(subframe_t, subframe_f);
                        frame = zeros([size(subframe),1],class(subframe));
                        frame(:,1) = subframe;
                        % Add to frame store
            %             add(frameStore, frame, DopplerTypes(Doppler));
                        add(frameStore, frame, DopplerTypes(Doppler_idx));
                    end
                end
    %         end
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

    %     [rxTraining, rxTrainingLabel, rxValidation, rxValidationLabel, rxTest, rxTestLabel] = hGenerateTrainingData(spf, DopplerTypes, domain);
    %     Data = reshape(Data, [1 spf 2 numData]);
    save('LTE' + string(domain) + '_Doppler_500.mat', 'rxTraining', 'rxTrainingLabel', 'rxValidation', 'rxValidationLabel', 'rxTest', 'rxTestLabel', '-v7.3');
elseif mode < 100 % pred if 5,10 or 15;  >100 compute pred statistics only
    disp("predicting");
    load('LTE' + string(domain) + '_Doppler_500.mat', 'rxTraining', 'rxTrainingLabel', 'rxValidation', 'rxValidationLabel', 'rxTest', 'rxTestLabel');
        
    rxTestPredAll = cell(numWin, numkk); 
    disp("mode=" + string(mode) + " type=" + string(type) + " domain=" + string(domain) + " scheme=" + string(scheme) + " numkk=" + string(numkk) + " numData=" + numData); 

    % dropoutRate = 0.5;
    netWidth = 1;
    sps = 8;
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
                        if pred > 0
                            nSEB = nSEB + (pred - label) ^ 2 / pred ^ 2;
                        end
                    else 
                        nSEB = nSEB + (pred - label) ^ 2 / label ^ 2;
                    end                    
                end
                if nSEB == 0
                    nMSEB = 10^-5;
                else
                    nMSEB = nSEB / numTest;
                end
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
                    if label == 0
                        if pred > 0
                           sum_square_errorS(index) = sum_square_errorS(index) + (pred - label) ^ 2 / pred ^ 2;
                        end
                    else 
                        sum_square_errorS(index) = sum_square_errorS(index) + (pred - label) ^ 2 / label ^ 2;
                    end
                end
                for i = 1:numDop
                    MeanTestPredS = mean(rxTestPredS(1:cntAcc(i), i));
                    if sum_square_errorS(i) == 0
                        nMSES = 10^-5;
                    else
                        nMSES = sum_square_errorS(i) / cntAcc(i);
                    end
                    testAccuracyS = 100 * sumAccS(i) / cntAcc(i);
                    disp("Window: " + WINDOWS(window) + " Doppler: "+Dopplers(i)+" AccuracyS: "+testAccuracyS+" MeanPredS: "+MeanTestPredS+" nMSES: "+nMSES);
                end            
            end
        end
 end    
%     save('../../../work/bal718/LTE_Doppler_predAll_' + string(mode) + '_' + string(type) + '_' + string(domain) + '_' + string(scheme) + '.mat',  'rxTestPredAll', 'rxTestLabel', '-v7.3');
   save('LTE_Doppler_predAll_' + string(mode) + '_' + string(type) + '_' + string(domain) + string(scheme) + '.mat',  'rxTestPredAll', 'rxTestLabel', '-v7.3');
end

if mode > 0  % 5,10,15: pred & compute merged statistics; 105: statistics for 5, 11-: for 10, 115: for 15
    if mode > 100
        mode = mode - 100;
%         load('../../../work/bal718/NR5G_Doppler_predAll_' + string(mode) + '_' + string(type) + '_' + string(domain) + string(scheme) + '.mat', 'rxTestPredAll', 'rxTestLabel');
        load('LTE_Doppler_predAll_' + string(mode) + '_' + string(type) + '_' + string(domain) + '_' + string(scheme) + '.mat', 'rxTestPredAll', 'rxTestLabel');
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
                if label == 0
                    if pred > 0
                       sum_square_errorS(index) = sum_square_errorS(index) + (pred - label) ^ 2 / pred ^ 2;
                    end
                else 
                    sum_square_errorS(index) = sum_square_errorS(index) + (pred - label) ^ 2 / label ^ 2;
                end
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
                        if pred > 0
                            nSEB = nSEB + (pred - label) ^ 2 / pred ^ 2;
                        end
                    else 
                        nSEB = nSEB + (pred - label) ^ 2 / label ^ 2;
                    end                    
                end
                testAccuracyB = 100 * mean(rxTestPredB == rxTestLabel);
                if nSEB == 0
                    nMSEB = 10^-5; % to plot zero
                else
                    nMSEB = nSEB / numTest;
                end
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
                pred = bottom; 
                rxTestPredM(cntAcc(index), index) = pred;
                if label == 0
                   if pred > 0
                       sum_square_errorM(index) = sum_square_errorM(index) + (pred - label) ^ 2 / pred ^ 2;
                   end
                else 
                   sum_square_errorM(index) = sum_square_errorM(index) + (pred - label) ^ 2 / label ^ 2;
                end

                sumAccM(index) = sumAccM(index) + double(label == bottom);
                sumAccMR(index) = sumAccMR(index) + double((label == bottom) || (label == top));
                if abs(label - top) > abs(label - bottom)
                   pred = bottom;
                else
                    pred = top;
                end
                rxTestPredMR(cntAcc(index), index) = pred;
                if label == 0
                   if pred > 0
                       sum_square_errorMR(index) = sum_square_errorMR(index) + (pred - label) ^ 2 / pred ^ 2;
                   end
                else 
                   sum_square_errorMR(index) = sum_square_errorMR(index) + (pred - label) ^ 2 / label ^ 2;
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
                if sum_square_errorM(i) == 0
                        nMSEM = 10^-5;
                else
                        nMSEM = sum_square_errorM(i) / cntAcc(i);
                end
                if sum_square_errorMR(i) == 0
                        nMSEMR = 10^-5;
                else
                        nMSEMR = sum_square_errorMR(i) / cntAcc(i);
                end
                testAccuracyM = 100 * sumAccM(i) / cntAcc(i);
                testAccuracyMR = 100 * sumAccMR(i) / cntAcc(i);
                disp("Window: " + WINDOWS(w) +" Doppler: "+Dopplers(i)+" AccuracyM: "+testAccuracyM+" MeanPredM: "+MeanTestPredM+" nMSEM: "+nMSEM+" AccuracyMR: "+testAccuracyMR+" MeanPredMR: "+MeanTestPredMR+" nMSEMR: "+nMSEMR);
            end
        else
            for i = 1:numDop
                MeanTestPredS = mean(rxTestPredS(1:cntAcc(i), i));
                if sum_square_errorS(i) == 0
                        nMSES = 10^-5;
                else
                        nMSES = sum_square_errorS(i) / cntAcc(i);
                end
                testAccuracyS = 100 * sumAccS(i) / cntAcc(i);
                disp("Window: " + WINDOWS(w) +" Doppler: "+Dopplers(i)+" AccuracyS: "+testAccuracyS+" MeanPredS: "+MeanTestPredS+" nMSES: "+nMSES);
            end            
        end
    end

end
% LTE_Doppler_sequence 9000 samples
% mode=5 type=0 domain=0 scheme=0 numkk=1
% predicting
% validation accuracy: 30.6667
% Swindow: 1 Doppler: 1   Accuracy: 76.6667
% Swindow: 1 Doppler: 2   Accuracy: 36.6667
% Swindow: 1 Doppler: 3   Accuracy: 25.5556
% Swindow: 1 Doppler: 4   Accuracy: 5.5556
% Swindow: 1 Doppler: 5   Accuracy: 37.7778
% Swindow: 1 Doppler: 6   Accuracy: 13.3333
% Swindow: 1 Doppler: 7   Accuracy: 23.3333
% Swindow: 1 Doppler: 8   Accuracy: 3.3333
% Swindow: 1 Doppler: 9   Accuracy: 38.8889
% Swindow: 1 Doppler: 10   Accuracy: 61.1111
%  window: 1Mean accuracyS: 32.2222
% validation accuracy: 36.6667
% Swindow: 2 Doppler: 1   Accuracy: 78.8889
% Swindow: 2 Doppler: 2   Accuracy: 65.5556
% Swindow: 2 Doppler: 3   Accuracy: 37.7778
% Swindow: 2 Doppler: 4   Accuracy: 30
% Swindow: 2 Doppler: 5   Accuracy: 31.1111
% Swindow: 2 Doppler: 6   Accuracy: 15.5556
% Swindow: 2 Doppler: 7   Accuracy: 32.2222
% Swindow: 2 Doppler: 8   Accuracy: 28.8889
% Swindow: 2 Doppler: 9   Accuracy: 68.8889
% Swindow: 2 Doppler: 10   Accuracy: 25.5556
%  window: 2Mean accuracyS: 41.4444
% validation accuracy: 48
% Swindow: 3 Doppler: 1   Accuracy: 71.1111
% Swindow: 3 Doppler: 2   Accuracy: 70
% Swindow: 3 Doppler: 3   Accuracy: 32.2222
% Swindow: 3 Doppler: 4   Accuracy: 32.2222
% Swindow: 3 Doppler: 5   Accuracy: 44.4444
% Swindow: 3 Doppler: 6   Accuracy: 20
% Swindow: 3 Doppler: 7   Accuracy: 46.6667
% Swindow: 3 Doppler: 8   Accuracy: 34.4444
% Swindow: 3 Doppler: 9   Accuracy: 76.6667
% Swindow: 3 Doppler: 10   Accuracy: 36.6667
%  window: 3Mean accuracyS: 46.4444
% validation accuracy: 49.6667
% Swindow: 4 Doppler: 1   Accuracy: 91.1111
% Swindow: 4 Doppler: 2   Accuracy: 75.5556
% Swindow: 4 Doppler: 3   Accuracy: 52.2222
% Swindow: 4 Doppler: 4   Accuracy: 43.3333
% Swindow: 4 Doppler: 5   Accuracy: 46.6667
% Swindow: 4 Doppler: 6   Accuracy: 10
% Swindow: 4 Doppler: 7   Accuracy: 50
% Swindow: 4 Doppler: 8   Accuracy: 43.3333
% Swindow: 4 Doppler: 9   Accuracy: 81.1111
% Swindow: 4 Doppler: 10   Accuracy: 53.3333
%  window: 4Mean accuracyS: 54.6667
% validation accuracy: 44.8889
% Swindow: 5 Doppler: 1   Accuracy: 86.6667
% Swindow: 5 Doppler: 2   Accuracy: 55.5556
% Swindow: 5 Doppler: 3   Accuracy: 46.6667
% Swindow: 5 Doppler: 4   Accuracy: 22.2222
% Swindow: 5 Doppler: 5   Accuracy: 34.4444
% Swindow: 5 Doppler: 6   Accuracy: 43.3333
% Swindow: 5 Doppler: 7   Accuracy: 76.6667
% Swindow: 5 Doppler: 8   Accuracy: 26.6667
% Swindow: 5 Doppler: 9   Accuracy: 58.8889
% Swindow: 5 Doppler: 10   Accuracy: 64.4444
%  window: 5Mean accuracyS: 51.5556
% validation accuracy: 64.8889
% Swindow: 6 Doppler: 1   Accuracy: 98.8889
% Swindow: 6 Doppler: 2   Accuracy: 97.7778
% Swindow: 6 Doppler: 3   Accuracy: 52.2222
% Swindow: 6 Doppler: 4   Accuracy: 60
% Swindow: 6 Doppler: 5   Accuracy: 71.1111
% Swindow: 6 Doppler: 6   Accuracy: 30
% Swindow: 6 Doppler: 7   Accuracy: 78.8889
% Swindow: 6 Doppler: 8   Accuracy: 68.8889
% Swindow: 6 Doppler: 9   Accuracy: 80
% Swindow: 6 Doppler: 10   Accuracy: 81.1111
%  window: 6Mean accuracyS: 71.8889
% validation accuracy: 59.8889
% Swindow: 7 Doppler: 1   Accuracy: 100
% Swindow: 7 Doppler: 2   Accuracy: 96.6667
% Swindow: 7 Doppler: 3   Accuracy: 36.6667
% Swindow: 7 Doppler: 4   Accuracy: 54.4444
% Swindow: 7 Doppler: 5   Accuracy: 78.8889
% Swindow: 7 Doppler: 6   Accuracy: 38.8889
% Swindow: 7 Doppler: 7   Accuracy: 54.4444
% Swindow: 7 Doppler: 8   Accuracy: 78.8889
% Swindow: 7 Doppler: 9   Accuracy: 77.7778
% Swindow: 7 Doppler: 10   Accuracy: 64.4444
% window: 7Mean accuracyS: 68.1111
% LTE_Doppler_sequence 45000
% mode=5 type=0 domain=0 scheme=0 numkk=1
% predicting
% validation accuracy: 54.7778
% Swindow: 1 Doppler: 1   Accuracy: 62.4444
% Swindow: 1 Doppler: 2   Accuracy: 57.1111
% Swindow: 1 Doppler: 3   Accuracy: 46.2222
% Swindow: 1 Doppler: 4   Accuracy: 43.3333
% Swindow: 1 Doppler: 5   Accuracy: 55.3333
% Swindow: 1 Doppler: 6   Accuracy: 49.1111
% Swindow: 1 Doppler: 7   Accuracy: 44.2222
% Swindow: 1 Doppler: 8   Accuracy: 60
% Swindow: 1 Doppler: 9   Accuracy: 60.2222
% Swindow: 1 Doppler: 10   Accuracy: 67.1111
%  window: 1Mean accuracyS: 54.5111
% validation accuracy: 73.5778
% Swindow: 2 Doppler: 1   Accuracy: 87.3333
% Swindow: 2 Doppler: 2   Accuracy: 78.8889
% Swindow: 2 Doppler: 3   Accuracy: 64.4444
% Swindow: 2 Doppler: 4   Accuracy: 66.2222
% Swindow: 2 Doppler: 5   Accuracy: 70.4444
% Swindow: 2 Doppler: 6   Accuracy: 74.2222
% Swindow: 2 Doppler: 7   Accuracy: 64.8889
% Swindow: 2 Doppler: 8   Accuracy: 70.6667
% Swindow: 2 Doppler: 9   Accuracy: 78
% Swindow: 2 Doppler: 10   Accuracy: 76
%  window: 2Mean accuracyS: 73.1111
% validation accuracy: 78.6
% Swindow: 3 Doppler: 1   Accuracy: 96.6667
% Swindow: 3 Doppler: 2   Accuracy: 81.1111
% Swindow: 3 Doppler: 3   Accuracy: 75.5556
% Swindow: 3 Doppler: 4   Accuracy: 76.4444
% Swindow: 3 Doppler: 5   Accuracy: 84.6667
% Swindow: 3 Doppler: 6   Accuracy: 57.3333
% Swindow: 3 Doppler: 7   Accuracy: 73.7778
% Swindow: 3 Doppler: 8   Accuracy: 91.7778
% Swindow: 3 Doppler: 9   Accuracy: 72.8889
% Swindow: 3 Doppler: 10   Accuracy: 83.1111
%  window: 3Mean accuracyS: 79.3333
% validation accuracy: 88.7556
% Swindow: 4 Doppler: 1   Accuracy: 98.6667
% Swindow: 4 Doppler: 2   Accuracy: 94.8889
% Swindow: 4 Doppler: 3   Accuracy: 82.6667
% Swindow: 4 Doppler: 4   Accuracy: 77.7778
% Swindow: 4 Doppler: 5   Accuracy: 85.1111
% Swindow: 4 Doppler: 6   Accuracy: 90.8889
% Swindow: 4 Doppler: 7   Accuracy: 86
% Swindow: 4 Doppler: 8   Accuracy: 86.8889
% Swindow: 4 Doppler: 9   Accuracy: 83.5556
% Swindow: 4 Doppler: 10   Accuracy: 94.4444
%  window: 4Mean accuracyS: 88.0889
% validation accuracy: 91.1111
% Swindow: 5 Doppler: 1   Accuracy: 99.1111
% Swindow: 5 Doppler: 2   Accuracy: 94.6667
% Swindow: 5 Doppler: 3   Accuracy: 80.2222
% Swindow: 5 Doppler: 4   Accuracy: 90
% Swindow: 5 Doppler: 5   Accuracy: 86
% Swindow: 5 Doppler: 6   Accuracy: 89.5556
% Swindow: 5 Doppler: 7   Accuracy: 85.7778
% Swindow: 5 Doppler: 8   Accuracy: 91.5556
% Swindow: 5 Doppler: 9   Accuracy: 92.2222
% Swindow: 5 Doppler: 10   Accuracy: 97.7778
%  window: 5Mean accuracyS: 90.6889
% validation accuracy: 96.0222
% Swindow: 6 Doppler: 1   Accuracy: 100
% Swindow: 6 Doppler: 2   Accuracy: 96.4444
% Swindow: 6 Doppler: 3   Accuracy: 94.6667
% Swindow: 6 Doppler: 4   Accuracy: 91.3333
% Swindow: 6 Doppler: 5   Accuracy: 96.2222
% Swindow: 6 Doppler: 6   Accuracy: 92.4444
% Swindow: 6 Doppler: 7   Accuracy: 97.5556
% Swindow: 6 Doppler: 8   Accuracy: 96.8889
% Swindow: 6 Doppler: 9   Accuracy: 98.4444
% Swindow: 6 Doppler: 10   Accuracy: 97.1111
%  window: 6Mean accuracyS: 96.1111
% validation accuracy: 96.4
% Swindow: 7 Doppler: 1   Accuracy: 100
% Swindow: 7 Doppler: 2   Accuracy: 100
% Swindow: 7 Doppler: 3   Accuracy: 95.3333
% Swindow: 7 Doppler: 4   Accuracy: 95.5556
% Swindow: 7 Doppler: 5   Accuracy: 95.1111
% Swindow: 7 Doppler: 6   Accuracy: 96.2222
% Swindow: 7 Doppler: 7   Accuracy: 99.1111
% Swindow: 7 Doppler: 8   Accuracy: 98.4444
% Swindow: 7 Doppler: 9   Accuracy: 97.5556
% Swindow: 7 Doppler: 10   Accuracy: 100
%  window: 7Mean accuracyS: 97.7333
% LTE_Doppler_sequence
% mode=5 type=0 domain=0 scheme=0 numkk=1
% predicting
% validation accuracy: 46.4889
% validation accuracy: 51.8667
% Swindow: 1 Doppler: 1   Accuracy: 73.5556
% Swindow: 1 Doppler: 2   Accuracy: 64.2222
% Swindow: 1 Doppler: 3   Accuracy: 41.5556
% Swindow: 1 Doppler: 4   Accuracy: 40
% Swindow: 1 Doppler: 5   Accuracy: 34.8889
% Swindow: 1 Doppler: 6   Accuracy: 52.6667
% Swindow: 1 Doppler: 7   Accuracy: 46.2222
% Swindow: 1 Doppler: 8   Accuracy: 52.2222
% Swindow: 1 Doppler: 9   Accuracy: 57.7778
% Swindow: 1 Doppler: 10   Accuracy: 59.5556
%  window: 1Mean accuracyS: 52.2667
% validation accuracy: 63.4222
% validation accuracy: 70.5333
% Swindow: 2 Doppler: 1   Accuracy: 92.6667
% Swindow: 2 Doppler: 2   Accuracy: 76.6667
% Swindow: 2 Doppler: 3   Accuracy: 61.1111
% Swindow: 2 Doppler: 4   Accuracy: 66.6667
% Swindow: 2 Doppler: 5   Accuracy: 69.3333
% Swindow: 2 Doppler: 6   Accuracy: 68
% Swindow: 2 Doppler: 7   Accuracy: 66.8889
% Swindow: 2 Doppler: 8   Accuracy: 68.8889
% Swindow: 2 Doppler: 9   Accuracy: 70.4444
% Swindow: 2 Doppler: 10   Accuracy: 70.8889
%  window: 2Mean accuracyS: 71.1556
% validation accuracy: 74.3111
% validation accuracy: 78
% Swindow: 3 Doppler: 1   Accuracy: 98
% Swindow: 3 Doppler: 2   Accuracy: 87.3333
% Swindow: 3 Doppler: 3   Accuracy: 66.4444
% Swindow: 3 Doppler: 4   Accuracy: 63.3333
% Swindow: 3 Doppler: 5   Accuracy: 76.6667
% Swindow: 3 Doppler: 6   Accuracy: 79.3333
% Swindow: 3 Doppler: 7   Accuracy: 72
% Swindow: 3 Doppler: 8   Accuracy: 78.6667
% Swindow: 3 Doppler: 9   Accuracy: 79.7778
% Swindow: 3 Doppler: 10   Accuracy: 86.4444
%  window: 3Mean accuracyS: 78.8
% validation accuracy: 79.8222
% validation accuracy: 85.6
% Swindow: 4 Doppler: 1   Accuracy: 98.4444
% Swindow: 4 Doppler: 2   Accuracy: 86.6667
% Swindow: 4 Doppler: 3   Accuracy: 81.1111
% Swindow: 4 Doppler: 4   Accuracy: 79.7778
% Swindow: 4 Doppler: 5   Accuracy: 83.3333
% Swindow: 4 Doppler: 6   Accuracy: 88.4444
% Swindow: 4 Doppler: 7   Accuracy: 84.8889
% Swindow: 4 Doppler: 8   Accuracy: 84.4444
% Swindow: 4 Doppler: 9   Accuracy: 83.1111
% Swindow: 4 Doppler: 10   Accuracy: 91.1111
%  window: 4Mean accuracyS: 86.1333
% validation accuracy: 80.6222
% validation accuracy: 87.2444
% Swindow: 5 Doppler: 1   Accuracy: 99.7778
% Swindow: 5 Doppler: 2   Accuracy: 92.2222
% Swindow: 5 Doppler: 3   Accuracy: 84
% Swindow: 5 Doppler: 4   Accuracy: 77.7778
% Swindow: 5 Doppler: 5   Accuracy: 83.1111
% Swindow: 5 Doppler: 6   Accuracy: 89.1111
% Swindow: 5 Doppler: 7   Accuracy: 88
% Swindow: 5 Doppler: 8   Accuracy: 90.2222
% Swindow: 5 Doppler: 9   Accuracy: 89.1111
% Swindow: 5 Doppler: 10   Accuracy: 94.2222
%  window: 5Mean accuracyS: 88.7556
% validation accuracy: 88.5333
% validation accuracy: 95.6
% Swindow: 6 Doppler: 1   Accuracy: 100
% Swindow: 6 Doppler: 2   Accuracy: 96.6667
% Swindow: 6 Doppler: 3   Accuracy: 95.1111
% Swindow: 6 Doppler: 4   Accuracy: 91.1111
% Swindow: 6 Doppler: 5   Accuracy: 94.6667
% Swindow: 6 Doppler: 6   Accuracy: 92.8889
% Swindow: 6 Doppler: 7   Accuracy: 95.3333
% Swindow: 6 Doppler: 8   Accuracy: 90.6667
% Swindow: 6 Doppler: 9   Accuracy: 95.1111
% Swindow: 6 Doppler: 10   Accuracy: 99.1111
%  window: 6Mean accuracyS: 95.0667
% validation accuracy: 90.6667
% validation accuracy: 96.3111
% Swindow: 7 Doppler: 1   Accuracy: 100
% Swindow: 7 Doppler: 2   Accuracy: 99.7778
% Swindow: 7 Doppler: 3   Accuracy: 97.3333
% Swindow: 7 Doppler: 4   Accuracy: 94.6667
% Swindow: 7 Doppler: 5   Accuracy: 96
% Swindow: 7 Doppler: 6   Accuracy: 95.5556
% Swindow: 7 Doppler: 7   Accuracy: 95.3333
% Swindow: 7 Doppler: 8   Accuracy: 95.1111
% Swindow: 7 Doppler: 9   Accuracy: 98
% Swindow: 7 Doppler: 10   Accuracy: 98.8889
%  window: 7Mean accuracyS: 97.0667
% merged prediction
% Window: 1 Doppler: 0 Accuracy: 73.5556 MeanPred: 37.4222 MSE: 8920.8889 nMSE: 6.3701
% Window: 1 Doppler: 40 Accuracy: 64.2222 MeanPred: 70.5778 MSE: 6791.1111 nMSE: 4.2444
% Window: 1 Doppler: 80 Accuracy: 41.5556 MeanPred: 105.0667 MSE: 7388.4444 nMSE: 1.1544
% Window: 1 Doppler: 120 Accuracy: 40 MeanPred: 135.1111 MSE: 6947.5556 nMSE: 0.48247
% Window: 1 Doppler: 160 Accuracy: 34.8889 MeanPred: 175.2 MSE: 6794.6667 nMSE: 0.26542
% Window: 1 Doppler: 200 Accuracy: 52.6667 MeanPred: 189.5111 MSE: 5923.5556 nMSE: 0.14809
% Window: 1 Doppler: 240 Accuracy: 46.2222 MeanPred: 212.8889 MSE: 10236.4444 nMSE: 0.17772
% Window: 1 Doppler: 280 Accuracy: 52.2222 MeanPred: 232.3556 MSE: 11854.2222 nMSE: 0.1512
% Window: 1 Doppler: 320 Accuracy: 57.7778 MeanPred: 265.8667 MSE: 12640 nMSE: 0.12344
% Window: 1 Doppler: 360 Accuracy: 59.5556 MeanPred: 282.4 MSE: 20497.7778 nMSE: 0.15816
% Window: 2 Doppler: 0 Accuracy: 92.6667 MeanPred: 8 MSE: 1479.1111 nMSE: 23.1111
% Window: 2 Doppler: 40 Accuracy: 76.6667 MeanPred: 53.6889 MSE: 2574.2222 nMSE: 1.6089
% Window: 2 Doppler: 80 Accuracy: 61.1111 MeanPred: 96.3556 MSE: 3640.8889 nMSE: 0.56889
% Window: 2 Doppler: 120 Accuracy: 66.6667 MeanPred: 121.8667 MSE: 2584.8889 nMSE: 0.17951
% Window: 2 Doppler: 160 Accuracy: 69.3333 MeanPred: 156.5333 MSE: 2328.8889 nMSE: 0.090972
% Window: 2 Doppler: 200 Accuracy: 68 MeanPred: 191.7333 MSE: 3075.5556 nMSE: 0.076889
% Window: 2 Doppler: 240 Accuracy: 66.8889 MeanPred: 223.2 MSE: 4625.7778 nMSE: 0.080309
% Window: 2 Doppler: 280 Accuracy: 68.8889 MeanPred: 254.8444 MSE: 5123.5556 nMSE: 0.065351
% Window: 2 Doppler: 320 Accuracy: 70.4444 MeanPred: 286.4889 MSE: 7349.3333 nMSE: 0.071771
% Window: 2 Doppler: 360 Accuracy: 70.8889 MeanPred: 318.3111 MSE: 8672 nMSE: 0.066914
% Window: 3 Doppler: 0 Accuracy: 98 MeanPred: 3.2889 MSE: 849.7778 nMSE: 78.561
% Window: 3 Doppler: 40 Accuracy: 87.3333 MeanPred: 45.6889 MSE: 1578.6667 nMSE: 0.98667
% Window: 3 Doppler: 80 Accuracy: 66.4444 MeanPred: 95.8222 MSE: 3868.4444 nMSE: 0.60444
% Window: 3 Doppler: 120 Accuracy: 63.3333 MeanPred: 131.4667 MSE: 3687.1111 nMSE: 0.25605
% Window: 3 Doppler: 160 Accuracy: 76.6667 MeanPred: 165.6889 MSE: 2488.8889 nMSE: 0.097222
% Window: 3 Doppler: 200 Accuracy: 79.3333 MeanPred: 199.9111 MSE: 1966.2222 nMSE: 0.049156
% Window: 3 Doppler: 240 Accuracy: 72 MeanPred: 242.9333 MSE: 2264.8889 nMSE: 0.039321
% Window: 3 Doppler: 280 Accuracy: 78.6667 MeanPred: 273.2444 MSE: 1984 nMSE: 0.025306
% Window: 3 Doppler: 320 Accuracy: 79.7778 MeanPred: 300.8 MSE: 4408.8889 nMSE: 0.043056
% Window: 3 Doppler: 360 Accuracy: 86.4444 MeanPred: 342.8444 MSE: 2926.2222 nMSE: 0.022579
% Window: 4 Doppler: 0 Accuracy: 98.4444 MeanPred: 2.2222 MSE: 458.6667 nMSE: 92.88
% Window: 4 Doppler: 40 Accuracy: 86.6667 MeanPred: 44.2667 MSE: 718.2222 nMSE: 0.44889
% Window: 4 Doppler: 80 Accuracy: 81.1111 MeanPred: 88.8 MSE: 1269.3333 nMSE: 0.19833
% Window: 4 Doppler: 120 Accuracy: 79.7778 MeanPred: 123.5556 MSE: 1507.5556 nMSE: 0.10469
% Window: 4 Doppler: 160 Accuracy: 83.3333 MeanPred: 161.7778 MSE: 1102.2222 nMSE: 0.043056
% Window: 4 Doppler: 200 Accuracy: 88.4444 MeanPred: 199.4667 MSE: 725.3333 nMSE: 0.018133
% Window: 4 Doppler: 240 Accuracy: 84.8889 MeanPred: 233.9556 MSE: 1770.6667 nMSE: 0.030741
% Window: 4 Doppler: 280 Accuracy: 84.4444 MeanPred: 271.7333 MSE: 1980.4444 nMSE: 0.025261
% Window: 4 Doppler: 320 Accuracy: 83.1111 MeanPred: 306.1333 MSE: 2680.8889 nMSE: 0.026181
% Window: 4 Doppler: 360 Accuracy: 91.1111 MeanPred: 348.0889 MSE: 2154.6667 nMSE: 0.016626
% Window: 5 Doppler: 0 Accuracy: 99.7778 MeanPred: 0.62222 MSE: 174.2222 nMSE: 450
% Window: 5 Doppler: 40 Accuracy: 92.2222 MeanPred: 43.2889 MSE: 359.1111 nMSE: 0.22444
% Window: 5 Doppler: 80 Accuracy: 84 MeanPred: 83.8222 MSE: 657.7778 nMSE: 0.10278
% Window: 5 Doppler: 120 Accuracy: 77.7778 MeanPred: 122.4889 MSE: 1315.5556 nMSE: 0.091358
% Window: 5 Doppler: 160 Accuracy: 83.1111 MeanPred: 161.4222 MSE: 1578.6667 nMSE: 0.061667
% Window: 5 Doppler: 200 Accuracy: 89.1111 MeanPred: 197.4222 MSE: 657.7778 nMSE: 0.016444
% Window: 5 Doppler: 240 Accuracy: 88 MeanPred: 240.7111 MSE: 952.8889 nMSE: 0.016543
% Window: 5 Doppler: 280 Accuracy: 90.2222 MeanPred: 274.4889 MSE: 1230.2222 nMSE: 0.015692
% Window: 5 Doppler: 320 Accuracy: 89.1111 MeanPred: 311.2889 MSE: 1998.2222 nMSE: 0.019514
% Window: 5 Doppler: 360 Accuracy: 94.2222 MeanPred: 352.8889 MSE: 1109.3333 nMSE: 0.0085597
% Window: 6 Doppler: 0 Accuracy: 100 MeanPred: 0 MSE: 0 nMSE: 0
% Window: 6 Doppler: 40 Accuracy: 96.6667 MeanPred: 41.3333 MSE: 53.3333 nMSE: 0.033333
% Window: 6 Doppler: 80 Accuracy: 95.1111 MeanPred: 80.3556 MSE: 149.3333 nMSE: 0.023333
% Window: 6 Doppler: 120 Accuracy: 91.1111 MeanPred: 119.2 MSE: 373.3333 nMSE: 0.025926
% Window: 6 Doppler: 160 Accuracy: 94.6667 MeanPred: 159.1111 MSE: 227.5556 nMSE: 0.0088889
% Window: 6 Doppler: 200 Accuracy: 92.8889 MeanPred: 199.2 MSE: 451.5556 nMSE: 0.011289
% Window: 6 Doppler: 240 Accuracy: 95.3333 MeanPred: 238.8444 MSE: 231.1111 nMSE: 0.0040123
% Window: 6 Doppler: 280 Accuracy: 90.6667 MeanPred: 278.3111 MSE: 401.7778 nMSE: 0.0051247
% Window: 6 Doppler: 320 Accuracy: 95.1111 MeanPred: 316.8889 MSE: 245.3333 nMSE: 0.0023958
% Window: 6 Doppler: 360 Accuracy: 99.1111 MeanPred: 359.0222 MSE: 110.2222 nMSE: 0.00085048
% Window: 7 Doppler: 0 Accuracy: 100 MeanPred: 0 MSE: 0 nMSE: 0
% Window: 7 Doppler: 40 Accuracy: 99.7778 MeanPred: 40.0889 MSE: 3.5556 nMSE: 0.0022222
% Window: 7 Doppler: 80 Accuracy: 97.3333 MeanPred: 81.7778 MSE: 192 nMSE: 0.03
% Window: 7 Doppler: 120 Accuracy: 94.6667 MeanPred: 120.1778 MSE: 192 nMSE: 0.013333
% Window: 7 Doppler: 160 Accuracy: 96 MeanPred: 159.6444 MSE: 85.3333 nMSE: 0.0033333
% Window: 7 Doppler: 200 Accuracy: 95.5556 MeanPred: 197.9556 MSE: 295.1111 nMSE: 0.0073778
% Window: 7 Doppler: 240 Accuracy: 95.3333 MeanPred: 239.6444 MSE: 206.2222 nMSE: 0.0035802
% Window: 7 Doppler: 280 Accuracy: 95.1111 MeanPred: 280.7111 MSE: 99.5556 nMSE: 0.0012698
% Window: 7 Doppler: 320 Accuracy: 98 MeanPred: 319.3778 MSE: 74.6667 nMSE: 0.00072917
% Window: 7 Doppler: 360 Accuracy: 98.8889 MeanPred: 358.1333 MSE: 359.1111 nMSE: 0.0027709
% 
function [rxTraining,rxTrainingLabel, rxValidation, rxValidationLabel, rxTest, rxTestLabel] = hGenerateTrainingData(spf, Dopplers, domain)
modulationTypes = ["QPSK", "16QAM", "64QAM"];
Delays = ["EPA" "EVA" "ETU"]; % 0 25us
numModulationTypes = length(modulationTypes);
numDelayTypes = length(Delays);
numDopplerTypes = length(Dopplers);
DopplerTypes = categorical(Dopplers);

enb.NDLRB = 6;                 % Number of resource blocks
enb.CellRefP = 1;               % One transmit antenna port
enb.NCellID = 10;               % Cell ID
enb.CyclicPrefix = 'Normal';    % Normal cyclic prefix
enb.DuplexMode = 'FDD';         % FDD

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

numFramesPerDopplerType = 100;
frameStore = helperModClassFrameStore(...
  numFramesPerDopplerType*numModulationTypes*numDopplerTypes*numDelayTypes,spf,DopplerTypes);
%   numFramesPerDopplerType*numDopplerTypes*numModulationTypes*numDopplerTypes*numDelayTypes*14,spf,DopplerTypes);

for Modulation_idx = 1:numModulationTypes
    % Number of bits needed is size of resource grid (K*L*P) * number of bits
    % per symbol (2 for QPSK)
%     numberOfBits = K*L*P*2; 
    numberOfBits = K*L*P*ModTypes2M(modulationTypes(Modulation_idx));
    
  for Delay_idx = 1:numDelayTypes
    cfg.DelayProfile = Delays(Delay_idx);      % EVA delay spread

    for Doppler_idx = 1:numDopplerTypes
      cfg.DopplerFreq = Dopplers(Doppler_idx);         % 120Hz Doppler frequency

      for fn = 1:numFramesPerDopplerType/10
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
            
            for nn = 1:10

            % Pass data through the fading channel model (1920 x1)
            rxWaveform = lteFadingChannel(cfg,txWaveform);

            % Calculate noise gain
            SNRdB = randi([-10 20]);  % Random SNR values between 0 and 10 dB
            SNR = 10^(SNRdB/20);    % Calculate linear noise gain

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
            rxGrid = lteOFDMDemodulate(enb,rxWaveform);
                if size(rxGrid, 2) == 140
                    break
                end
            end

            for s = 1:10
                if domain == 0 % freq-domain
                    OFDMSym = rxGrid(:, 1 + (s - 1) * 14 : s * 14);
                    subframe_f = reshape(OFDMSym, [1008 1]);
                    % Remove transients from the beginning, trim to size, and normalize
    %                 frame = zeros([size(rxWaveform),1],class(rxWaveform));
    %                 frame(:,1) = rxWaveform;
                    add(frameStore, frame_f, DopplerTypes(Doppler_idx));
                elseif domain == 1 % time-domain
                    subframe_t = rxWaveform(1 + spf * (s - 1) : spf * s, :);
                    add(frameStore, frame_t, DopplerTypes(Doppler_idx));
                else % freq+time
                    subframe = vertcat(subframe_t, subframe_f);
                    frame = zeros([size(subframe),1],class(subframe));
                    frame(:,1) = subframe;
                    % Add to frame store
        %             add(frameStore, frame, DopplerTypes(Doppler));
                    add(frameStore, frame, DopplerTypes(Doppler_idx));
                end
            end
%         end
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

% function [Data,Labels] = hGenerateTrainingData(dataSize, Dopplers, boundary, kk)
% function [Data,Labels] = hGenerateTrainingData(dataSize, Dopplers, domain)
% % Generate training data examples for channel estimation
% % Run dataSize number of iterations to create random channel configurations
% % and pass an OFDM-modulated fixed PDSCH grid with only the DM-RS symbols
% % inserted. Perform perfect timing synchronization and OFDM demodulation,
% % extracting the pilot symbols and performing linear interpolation at each
% % iteration. Use perfect channel information to create the
% % label data. The function returns 2 arrays - the training data and labels.
% 
%     fprintf('Starting data generation...\n')
% 
%     % List of possible channel profiles
%     delayProfiles = {'TDL-A', 'TDL-B', 'TDL-C', 'TDL-D', 'TDL-E'};
% 
%     [gnb, pdsch] = hDeepLearningChanEstSimParameters();
% 
%     % Create the channel model object
%     nTxAnts = gnb.NTxAnts;
%     nRxAnts = gnb.NRxAnts;
% 
%     channel = nrTDLChannel; % TDL channel object
%     channel.NumTransmitAntennas = nTxAnts;
%     channel.NumReceiveAntennas = nRxAnts;
% 
%     % Use the value returned from <matlab:edit('hOFDMInfo') hOFDMInfo> to
%     % set the the channel model sampling rate
%     waveformInfo = hOFDMInfo(gnb);
%     channel.SampleRate = waveformInfo.SamplingRate;
% 
%     % Get the maximum number of delayed samples by a channel multipath
%     % component. This number is calculated from the channel path with the largest
%     % delay and the implementation delay of the channel filter, and is required
%     % to flush the channel filter to obtain the received signal.
%     chInfo = info(channel);
%     maxChDelay = ceil(max(chInfo.PathDelays*channel.SampleRate)) + chInfo.ChannelFilterDelay;
% 
%     % Return DM-RS indices and symbols
%     [~,dmrsIndices,dmrsSymbols,~] = hPDSCHResources(gnb,pdsch);
% 
%     % PDSCH mapping in grid associated with PDSCH transmission period
%     pdschGrid = zeros(waveformInfo.NSubcarriers,waveformInfo.SymbolsPerSlot,nTxAnts);
% 
%     % PDSCH DM-RS precoding and mapping
%     [~,dmrsAntIndices] = nrExtractResources(dmrsIndices,pdschGrid);
%     pdschGrid(dmrsAntIndices) = pdschGrid(dmrsAntIndices) + dmrsSymbols;
% 
%     % OFDM modulation of associated resource elements
%     txWaveform_original = hOFDMModulate(gnb,pdschGrid);
% 
%     % Acquire linear interpolator coordinates for neural net preprocessing
%     [rows,cols] = find(pdschGrid ~= 0);
%     dmrsSubs = [rows, cols, ones(size(cols))];
%     hest = zeros(size(pdschGrid));
%     [l_hest,k_hest] = meshgrid(1:size(hest,2),1:size(hest,1));
% 
%         % Preallocate memory for the training data and labels
%     spf_t = 15376;
%     spf_f = 8568;
%     spf_tf = 15376+8568;
%     if domain == 0
%         spf = spf_f;
%     elseif doamin == 1
%         spf = spf_t;
%     else
%         spf = spf_tf;
%     end
%     numDopplerTypes = length(Dopplers);
%     numExamples = dataSize * numDopplerTypes;
%     Data = zeros([1 spf 2 numExamples]);
%     Labels = categorical(ones([numExamples, 1]));
% 
%     index = 0;
%     % Main loop for data generation, iterating over the number of examples
%     % specified in the function call. Each iteration of the loop produces a
%     % new channel realization with a random delay spread, doppler shift,
%     % and delay profile. Every perturbed version of the transmitted
%     % waveform with the DM-RS symbols is stored in trainData, and the
%     % perfect channel realization in trainLabels.
%     for Doppler_idx = 1:numDopplerTypes
%       Doppler = DopplerTypes2num(Dopplers(Doppler_idx));             % Desired SNR in dB
% 
%       numSample = dataSize;
%       for i = 1:numSample
%         % Release the channel to change nontunable properties
%         channel.release
% 
%         % Pick a random seed to create different channel realizations
%         channel.Seed = randi([1001 2000]);
% 
%         % Pick a random delay profile, delay spread, and maximum doppler shift
%         channel.DelayProfile = string(delayProfiles(randi([1 numel(delayProfiles)])));
%         channel.DelaySpread = randi([1 300])*1e-9;
%         channel.MaximumDopplerShift = Doppler; %randi([5 400]);
% 
%         % Send data through the channel model. Append zeros at the end of
%         % the transmitted waveform to flush channel content. These zeros
%         % take into account any delay introduced in the channel, such as
%         % multipath delay and implementation delay. This value depends on
%         % the sampling rate, delay profile, and delay spread
%         txWaveform = [txWaveform_original; zeros(maxChDelay, size(txWaveform_original,2))];
%         [rxWaveform,pathGains,sampleTimes] = channel(txWaveform);
% 
%         % Add additive white Gaussian noise (AWGN) to the received time-domain
%         % waveform. To take into account sampling rate, normalize the noise power.
%        % The SNR is defined per RE for each receive antenna (3GPP TS 38.101-4).
%         SNRdB = randi([0 10]);  % Random SNR values between 0 and 10 dB
%         SNR = 10^(SNRdB/20);    % Calculate linear noise gain
%         N0 = 1/(sqrt(2.0*nRxAnts*double(waveformInfo.Nfft))*SNR);
%         noise = N0*complex(randn(size(rxWaveform)),randn(size(rxWaveform)));
%         rxWaveform = rxWaveform + noise;
% 
%         % Perform OFDM demodulation on the received data to recreate the
%         % resource grid, including padding in case practical
%         % synchronization results in an incomplete slot being demodulated
%         rxGrid = hOFDMDemodulate(gnb,rxWaveform);
%         [K,L,R] = size(rxGrid);
%         if (L < waveformInfo.SymbolsPerSlot)
%             rxGrid = cat(2,rxGrid,zeros(K,waveformInfo.SymbolsPerSlot-L,R));
%         end
% 
%         % Linear interpolation
%         dmrsRx = rxGrid(dmrsIndices);
%         dmrsEsts = dmrsRx .* conj(dmrsSymbols);
%         f = scatteredInterpolant(dmrsSubs(:,2),dmrsSubs(:,1),dmrsEsts);
%         hest = f(l_hest,k_hest);
% 
%         % Split interpolated grid into real and imaginary components and
%         % concatenate them along the third dimension, as well as for the
%         % true channel response
%         rx_grid = cat(3, real(hest), imag(hest));
% 
%         % Add generated training example and label to the respective arrays
%         index = index + 1;
%         subframe_t = reshape(cat(2, real(rxWaveform), imag(rxWaveform)), [1 spf_t 2]);
%         subframe_f = reshape(rx_grid, [1 spf_f 2]);
%         if domain == 0
%             Data(:,:,:, index) = subframe_f;
%         elseif doamin == 1
%             Data(:,:,:, index) = subframe_t;
%         else
%             Data(:,:,:, index) = cat(2, subframe_t, subframe_f);
%         end
% 
%         Labels(index) = Dopplers(Doppler_idx);
% 
%       end
%     end
%     numFrames = size(Data, 4);
%     shuffleIdx = randperm(numFrames);
%     Data = Data(:, :, :, shuffleIdx);
%     Labels = Labels(shuffleIdx);
% 
%     fprintf('Data generation complete.  numFrames=%d!\n ', numFrames);
% 
% end
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
           if (DopplerTypes2num(Labels(j)) < Doppler) %predictor label < data label
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

% function [seqTraining, seqValidation, seqTest] = create_sequence(window, WINDOWS, spf, rxTraining, rxTrainingLabelkk, rxValidation, rxValidationLabelkk, rxTest, rxTestLabelkk)
%     numTest = length(rxTestLabelkk);
%     numTraining = length(rxTrainingLabelkk);
%     numValidation = length(rxValidationLabelkk);
%     seqTraining = cell(numTraining, 1);
%     seqValidation = cell(numValidation, 1);
%     seqTest = cell(numTest, 1);
% 
%     rxTrainingArr = zeros(1, spf, 2, WINDOWS(window));
%     for i = 1:numTraining
%         cnt = 1;
%         rxTrainingArr(:, :, :, cnt) = rxTraining(:, :, :, i);
%         for j = i+1:numTraining
%             if rxTrainingLabelkk(j) == rxTrainingLabelkk(i)
%                 cnt = cnt + 1;
%                 rxTrainingArr(:, :, :, cnt) = rxTraining(:, :, :, j);
%                 if cnt == WINDOWS(window)
%                     break
%                 end
%             end
%         end
%         if cnt == WINDOWS(window)
%             seqTraining{i, 1} = rxTrainingArr;
%         else
%             for j = 1:numTraining
%                 if rxTrainingLabelkk(j) == rxTrainingLabelkk(i)
%                     cnt = cnt + 1;
%                     rxTrainingArr(:, :, :, cnt) = rxTraining(:, :, :, j);
%                     if cnt == WINDOWS(window)
%                         break
%                     end
%                 end
%             end
%             seqTraining{i, 1} = rxTrainingArr;
%         end    
%     end
% 
%     rxValidationArr = zeros(1, spf, 2, WINDOWS(window));
%     for i = 1:numValidation
%         cnt = 1;
%         rxValidationArr(:, :, :, cnt) = rxValidation(:, :, :, i);
%         for j = i+1:numValidation
%             if rxValidationLabelkk(j) == rxValidationLabelkk(i)
%                 cnt = cnt + 1;
%                 rxValidationArr(:, :, :, cnt) = rxValidation(:, :, :, j);
%                 if cnt == WINDOWS(window)
%                     break
%                 end
%             end
%         end
%         if cnt == WINDOWS(window)
%             seqValidation{i, 1} = rxValidationArr;
%         else
%             for j = 1:numValidation
%                 if rxValidationLabelkk(j) == rxValidationLabelkk(i)
%                     cnt = cnt + 1;
%                     rxValidationArr(:, :, :, cnt) = rxValidation(:, :, :, j);
%                     if cnt == WINDOWS(window)
%                         break
%                     end
%                 end
% 
%             end
%             seqValidation{i, 1} = rxValidationArr;
%         end    
%     end
% 
%     rxTestArr = zeros(1, spf, 2, WINDOWS(window));
%     for i = 1:numTest
%         cnt = 1;
%         rxTestArr(:, :, :, cnt) = rxTest(:, :, :, i);
%         for j = i+1:numTest
%             if rxTestLabelkk(j) == rxTestLabelkk(i)
%                 cnt = cnt + 1;
%                 rxTestArr(:, :, :, cnt) = rxTest(:, :, :, j);
%                 if cnt == WINDOWS(window)
%                     break
%                 end
%             end
%         end
%         if cnt == WINDOWS(window)
%             seqTest{i, 1} = rxTestArr;
%         else
%             for j = 1:numTest
%                 if rxTestLabelkk(j) == rxTestLabelkk(i)
%                     cnt = cnt + 1;
%                     rxTestArr(:, :, :, cnt) = rxTest(:, :, :, j);
%                     if cnt == WINDOWS(window)
%                         break
%                     end
%                 end
%             end
%             seqTest{i, 1} = rxTestArr;
%         end    
%     end
% end

%number of bits per symbol
% function M = ModTypes2M(x)
% if x == "QPSK" 
%     M = 2;
% elseif x == "16QAM" 
%     M = 4;
% elseif x == "64QAM" 
%     M = 6;
% end
% end
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
elseif x == "250" 
    Doppler = 250;
elseif x == "300" 
    Doppler = 300;
elseif x == "350" 
    Doppler = 350;
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
