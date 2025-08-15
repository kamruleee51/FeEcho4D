%% Interactive radial-slice & time navigator with annotation status
% Keys:
%   ←/→ : previous/next time index (t)
%   ↑/↓ : previous/next slice index (sliceNum)
%     e : exit
%
% Display:
%   - Current slice number (1..maxSlice) and time (1..numTimes)
%   - Annotation status: counts PNGs in ./timeXXX/segmented/*.png

clc
close all
clear all

%% Configuration
maxSlice = 37;                          % total number of radial slices (e.g., 0:5:180 → 37)
casePath = pwd;                         % base directory (repo root with time folders)
videoDir = fullfile(casePath, 'video'); % where .avi files live

% Count available time points by enumerating AVI files in ./video
aviList  = dir(fullfile(videoDir, '*.avi'));
numTimes = numel(aviList);

WhichTime = 1;  % reference time index for annotation listing (kept as in your code)

%% State variables (current indices)
t = 1;          % current time index
sliceNum = 1;   % current slice index

%% Main loop
figure('Name','Radial Slice/Time Navigator — ↑/↓ change slice, ←/→ change time, e exit');
while true

    % --- Wrap time index t into [1, numTimes] ---
    if t > numTimes
        t = 1;
    end
    if t < 1
        t = numTimes;
    end

    % --- Wrap slice index sliceNum into [1, maxSlice] ---
    if sliceNum > maxSlice
        sliceNum = 1;
    end
    if sliceNum < 1
        sliceNum = maxSlice;
    end

    % --- Console readout (optional) ---
    disp('------------------------');
    disp(['Slice number = ' num2str(sliceNum)]);
    disp(['Time        = ' num2str(t)]);
    fprintf('\n');

    % --- Annotation folder contents (counts segmented PNGs) ---
    segmentedDir = fullfile(casePath, sprintf('time%03d', WhichTime), 'segmented');
    Contents = dir(fullfile(segmentedDir, '*.png'));

    % --- Load current image: ./timeTTT/sliceSSStimeTTT.png ---
    imgPath = fullfile(casePath, ...
                       sprintf('time%03d', t), ...
                       sprintf('slice%03dtime%03d.png', sliceNum, t));

    if ~exist(imgPath, 'file')
        % If image is missing, show a warning frame instead of erroring out
        warning('Missing image: %s', imgPath);
        imshow(zeros(256,256,'uint8'), 'InitialMagnification', 'fit');  % placeholder black image
        title('Image not found', 'Color', 'r');
    else
        img = imread(imgPath);
        imshow(img, 'InitialMagnification', 'fit');
    end

    % --- On-image HUD text: degree/slice, time, and annotation status ---
    hudSlice = sprintf('Radial slice (↓ ↑) = %d/%d', sliceNum, maxSlice);
    hudTime  = sprintf('Time slice (→ ←) = %d/%d', t, numTimes);

    if isempty(Contents)
        hudAnno = 'Annotated = None';
    else
        hudAnno = sprintf('Annotated = Slice%03d.png', numel(Contents));
    end

    hudText  = sprintf('%s  ||  %s  ||  %s', hudSlice, hudTime, hudAnno);
    text(5, 20, hudText, 'Color', 'red', 'FontSize', 14, 'FontWeight', 'bold');

    % --- Wait for key press & update indices ---
    waitforbuttonpress;
    keyVal = double(get(gcf, 'CurrentCharacter'));   % numeric key code

    % Arrow key codes in MATLAB figure window:
    %  28: left, 29: right, 30: up, 31: down
    if     keyVal == 28         % ← : previous time
        t = t - 1;
    elseif keyVal == 29         % → : next time
        t = t + 1;
    elseif keyVal == 31         % ↓ : previous slice
        sliceNum = sliceNum - 1;
    elseif keyVal == 30         % ↑ : next slice
        sliceNum = sliceNum + 1;
    % Robust exit: accept ASCII 101 and the character 'e'
    elseif ismember(keyVal, [101, double('e')])   % 'e' to exit
        break;
    end
end

close all;
