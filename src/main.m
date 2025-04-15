% Implementation of Adaptive Equalization for Rayleigh FSF Channels
% Based on the research paper with BPSK, QPSK and 8PSK modulations
% Input: man.png image

clear all; close all; clc;

%% Parameters
modTypes = [2, 4, 8];  % BPSK, QPSK, 8PSK
snr = 15;              % Signal-to-noise ratio in dB
filterLength = 31;     % Increased filter length for better equalization
channelTaps = 7;       % Channel impulse response length
pilotPercentage = 0.08; % Pilot data percentage (8% as mentioned in paper)

%% Load and prepare input image
try
    img = imread('man.png');
    if size(img, 3) == 3
        img = rgb2gray(img);  % Convert to grayscale if RGB
    end
catch
    error('Error: Could not find man.png image file.');
end

% Display original image
figure;
imshow(img);
title('Original Image');

% Store original image dimensions
imgSize = size(img);
origImg = img; % Keep original for comparison

% Convert image to bit stream
imgBin = de2bi(double(img(:)), 8, 'left-msb');
imgBits = imgBin(:);

% Results storage
berBeforeLMS = zeros(1, length(modTypes));
berAfterLMS = zeros(1, length(modTypes));
berBeforeRLS = zeros(1, length(modTypes));
berAfterRLS = zeros(1, length(modTypes));

%% Process each modulation type
for modIdx = 1:length(modTypes)
    M = modTypes(modIdx);
    bitsPerSymbol = log2(M);
    
    fprintf('\nProcessing %d-PSK Modulation...\n', M);
    
    % DIRECT APPROACH: Convert image to a stream of numbers first
    imgVector = double(img(:));
    
    % Convert numbers to bits (8 bits per pixel)
    imgBits = [];
    for i = 1:length(imgVector)
        pixelBits = de2bi(imgVector(i), 8, 'left-msb');
        imgBits = [imgBits; pixelBits(:)];
    end
    
    % Ensure bit stream length is a multiple of bitsPerSymbol
    paddingBits = mod((-1)*length(imgBits), bitsPerSymbol);
    if paddingBits > 0
        dataBits = [imgBits; zeros(paddingBits, 1)];
    else
        dataBits = imgBits;
    end
    
    % Group bits into symbols
    dataSymbolBits = reshape(dataBits, bitsPerSymbol, [])';
    dataSymbols = bi2de(dataSymbolBits, 'left-msb');
    
    % Generate pilot symbols for training
    numPilotSymbols = round(pilotPercentage * length(dataSymbols));
    pilotSymbols = randi([0 M-1], numPilotSymbols, 1);
    
    % Combine pilot and data symbols
    txSymbols = [pilotSymbols; dataSymbols];
    
    % Modulate using PSK with Gray coding
    txSignal = pskmod(txSymbols, M, pi/M, 'gray');
    
    %% Rayleigh Frequency Selective Fading Channel
    % Generate Rayleigh fading channel coefficients
    h = (randn(channelTaps, 1) + 1i*randn(channelTaps, 1))/sqrt(2*channelTaps);
    
    % Apply channel distortion
    rxChannel = filter(h, 1, txSignal);
    
    % Add AWGN noise
    rxSignal = awgn(rxChannel, snr, 'measured');
    
    % Demodulate received signal (before equalization)
    rxSymbols = pskdemod(rxSignal, M, pi/M, 'gray');
    
    %% LMS Equalization
    fprintf('Applying LMS Equalization...\n');

    % Manual LMS implementation (replacing comm.LinearEqualizer)
    % Initialize equalizer weights
    lmsWeights = zeros(filterLength, 1);
    stepSize = 0.01;

    % Extract training data
    trainRx = rxSignal(1:numPilotSymbols);
    trainTx = txSignal(1:numPilotSymbols);

    % Add progress reporting
    fprintf('  LMS Training: 0%%');

    % LMS Training
    for n = filterLength:length(trainRx)
        if mod(n, length(trainRx)/10) < 1
            fprintf('...%d%%', round((n-filterLength)/(length(trainRx)-filterLength)*100));
        end
        
        x = trainRx(n:-1:n-filterLength+1);  % Input vector
        y = lmsWeights' * x;                 % Filter output
        e = trainTx(n-filterLength+1) - y;   % Error
        lmsWeights = lmsWeights + stepSize * conj(e) * x;  % Update weights
    end
    fprintf('...done!\n');

    % Apply the trained LMS filter to the entire signal
    fprintf('  LMS Processing signal: 0%%');
    lmsEqOutput = zeros(size(rxSignal));
    for n = filterLength:length(rxSignal)
        if mod(n, length(rxSignal)/10) < 1
            fprintf('...%d%%', round((n-filterLength)/(length(rxSignal)-filterLength)*100));
        end
        x = rxSignal(n:-1:n-filterLength+1);
        lmsEqOutput(n-filterLength+1) = lmsWeights' * x;
    end
    fprintf('...done!\n');
    
    % Demodulate equalized signal
    lmsEqSymbols = pskdemod(lmsEqOutput, M, pi/M, 'gray');
    
    % Calculate BER (excluding pilot symbols)
    rxDataSymbols = rxSymbols(numPilotSymbols+1:end);
    lmsEqDataSymbols = lmsEqSymbols(numPilotSymbols+1:end);
    
    % Ensure equal lengths for comparison
    len = min([length(dataSymbols), length(rxDataSymbols), length(lmsEqDataSymbols)]);
    
    berBeforeLMS(modIdx) = sum(rxDataSymbols(1:len) ~= dataSymbols(1:len)) / len;
    berAfterLMS(modIdx) = sum(lmsEqDataSymbols(1:len) ~= dataSymbols(1:len)) / len;
    
    % OPTIMIZED reconstruction - avoid memory reallocation in loop
    fprintf('  Reconstructing images directly...\n');

    % For received data (pre-equalization)
    rxDataBits = zeros(len * bitsPerSymbol, 1);
    for i = 1:len
        bits = de2bi(rxDataSymbols(i), bitsPerSymbol, 'left-msb');
        rxDataBits((i-1)*bitsPerSymbol+1:i*bitsPerSymbol) = bits(:);
    end
    
    % Ensure we have enough bits for all pixels (each pixel is 8 bits)
    totalPixels = imgSize(1) * imgSize(2);
    pixelBits = rxDataBits(1:min(length(rxDataBits), totalPixels*8));
    
    % Reshape to get 8 bits per pixel
    if length(pixelBits) < totalPixels*8
        pixelBits = [pixelBits; zeros(totalPixels*8 - length(pixelBits), 1)];
    end
    
    % Convert bytes to pixels
    pixelBytes = reshape(pixelBits, 8, [])';
    pixelValues = bi2de(pixelBytes, 'left-msb');
    
    % Reshape to original image dimensions
    rxImg = reshape(uint8(pixelValues), imgSize);
    
    % Same approach for LMS equalized image
    lmsEqDataBits = zeros(len * bitsPerSymbol, 1);
    for i = 1:len
        bits = de2bi(lmsEqDataSymbols(i), bitsPerSymbol, 'left-msb');
        lmsEqDataBits((i-1)*bitsPerSymbol+1:i*bitsPerSymbol) = bits(:);
    end
    
    pixelBits = lmsEqDataBits(1:min(length(lmsEqDataBits), totalPixels*8));
    if length(pixelBits) < totalPixels*8
        pixelBits = [pixelBits; zeros(totalPixels*8 - length(pixelBits), 1)];
    end
    pixelBytes = reshape(pixelBits, 8, [])';
    pixelValues = bi2de(pixelBytes, 'left-msb');
    lmsEqImg = reshape(uint8(pixelValues), imgSize);
    
    %% RLS Equalization
    fprintf('Applying RLS Equalization...\n');
    
    % Manual RLS implementation
    rlsWeights = zeros(filterLength, 1);
    P = eye(filterLength) / 0.1;  % Initial correlation matrix
    lambda = 0.99;  % Forgetting factor

    % Add progress reporting
    fprintf('  RLS Training: 0%%');

    % RLS Training
    for n = filterLength:length(trainRx)
        if mod(n, length(trainRx)/10) < 1
            fprintf('...%d%%', round((n-filterLength)/(length(trainRx)-filterLength)*100));
        end
        
        x = trainRx(n:-1:n-filterLength+1);  % Input vector
        
        % Compute Kalman gain
        k = (P * x) / (lambda + x' * P * x);
        
        % Compute a priori error
        y = rlsWeights' * x;                 % Filter output
        e = trainTx(n-filterLength+1) - y;   % Error
        
        % Update weights
        rlsWeights = rlsWeights + k * conj(e);
        
        % Update inverse correlation matrix
        P = (P - k * x' * P) / lambda;
    end
    fprintf('...done!\n');

    % Apply the trained RLS filter to the entire signal
    fprintf('  RLS Processing signal: 0%%');
    rlsEqOutput = zeros(size(rxSignal));
    for n = filterLength:length(rxSignal)
        if mod(n, length(rxSignal)/10) < 1
            fprintf('...%d%%', round((n-filterLength)/(length(rxSignal)-filterLength)*100));
        end
        x = rxSignal(n:-1:n-filterLength+1);
        rlsEqOutput(n-filterLength+1) = rlsWeights' * x;
    end
    fprintf('...done!\n');
    
    % Demodulate equalized signal
    rlsEqSymbols = pskdemod(rlsEqOutput, M, pi/M, 'gray');
    
    % Calculate BER (excluding pilot symbols)
    rlsEqDataSymbols = rlsEqSymbols(numPilotSymbols+1:end);
    
    berBeforeRLS(modIdx) = sum(rxDataSymbols(1:len) ~= dataSymbols(1:len)) / len;
    berAfterRLS(modIdx) = sum(rlsEqDataSymbols(1:len) ~= dataSymbols(1:len)) / len;
    
    % Reconstruct image after RLS equalization - FIXED to prevent grid pattern
    fprintf('  Reconstructing RLS image: ');

    % Pre-allocate memory for efficiency
    rlsEqDataBits = zeros(len * bitsPerSymbol, 1);
    for i = 1:length(rlsEqDataSymbols(1:len))
        bits = de2bi(rlsEqDataSymbols(i), bitsPerSymbol, 'left-msb');
        rlsEqDataBits((i-1)*bitsPerSymbol+1:i*bitsPerSymbol) = bits(:);
        
        % Add progress indicator every 10%
        if mod(i, round(len/10)) == 0
            fprintf('.');
        end
    end
    fprintf(' done!\n');
    rlsEqDataBits = rlsEqDataBits(1:min(length(rlsEqDataBits), length(imgBits)));
    rlsEqImg = reconstructImage(rlsEqDataBits, imgSize);
    
    % Replace the current display code (after RLS reconstruction) with this combined plotting approach:
    % Create combined figure for this modulation type with 3 subplots
    figure;
    subplot(1, 3, 1);
    imshow(uint8(rxImg));
    title(sprintf('%d-PSK: Received Image', M));

    subplot(1, 3, 2);
    imshow(uint8(lmsEqImg));
    title(sprintf('%d-PSK: LMS Equalized', M));

    subplot(1, 3, 3);
    imshow(uint8(rlsEqImg));
    title(sprintf('%d-PSK: RLS Equalized', M));

    % Add a main title for the figure
    sgtitle(sprintf('%d-PSK Modulation Results', M));
end

%% Display BER Results
fprintf('\n=== BER Results Summary ===\n\n');

% LMS Results
fprintf('LMS Equalizer Results:\n');
fprintf('Modulation | BER Before Equalization | BER After Equalization\n');
fprintf('----------|------------------------|----------------------\n');
for i = 1:length(modTypes)
    fprintf('%d-PSK     | %.6f                | %.6f\n', ...
        modTypes(i), berBeforeLMS(i), berAfterLMS(i));
end

% RLS Results
fprintf('\nRLS Equalizer Results:\n');
fprintf('Modulation | BER Before Equalization | BER After Equalization\n');
fprintf('----------|------------------------|----------------------\n');
for i = 1:length(modTypes)
    fprintf('%d-PSK     | %.6f                | %.6f\n', ...
        modTypes(i), berBeforeRLS(i), berAfterRLS(i));
end

%% Helper function to reconstruct image from bit stream - FIXED function
function img = reconstructImage(bits, imgSize)
    % Ensure we have exactly the right number of bits
    numPixels = imgSize(1) * imgSize(2);
    reqBits = numPixels * 8;
    
    if length(bits) < reqBits
        % Pad with zeros if needed
        bits = [bits; zeros(reqBits - length(bits), 1)];
    elseif length(bits) > reqBits
        % Truncate if too long
        bits = bits(1:reqBits);
    end
    
    % Follow the exact approach from implementation.m that works correctly
    % First reshape to group bits into bytes (8 bits per pixel)
    img_bits = reshape(bits, 8, [])';
    
    % Convert to pixel values
    img_bytes = bi2de(img_bits, 'left-msb');
    
    % Then reshape to image dimensions as uint8
    img = reshape(uint8(img_bytes), imgSize);
end