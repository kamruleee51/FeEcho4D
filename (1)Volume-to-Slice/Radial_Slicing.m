%% Radial Slice Construction from 4D Echocardiography (TMI-aligned)
% Given a 4D sequence V ∈ R^{T × H' × W' × D'}, this script:
% 1) Selects an axial reference plane index Z_m (user-chosen slice index in depth D').
% 2) Uses anisotropic voxel spacing [s_x, s_y, s_z] from `scale.txt`.
% 3) Builds a regular 3D meshgrid X over the rotated volume at each time t.
% 4) Recenters coordinates to the selected LV center [C_x, C_y] (in pixels) and Z_m.
% 5) Applies uniform angular sampling θ ∈ [0, π] at 5° (S=37 slices): θ = π·s/(S−1), s=0..S−1.
% 6) Rotates coordinates via R_y(θ) and maps back to the original frame.
% 7) Uses cubic 3D interpolation at transformed coordinates to obtain V^t_θ.
% 8) Extracts the axial slice at Z_m from V^t_θ ⇒ I^t_θ ∈ R^{H × W}.
% 9) Stacks all angles to form V^t_θ ∈ R^{S × H × W}; over t, V_θ ∈ R^{T × S × H × W}.
%
% Notes on code-to-math mapping:
% - `spacing_values = [s_x, s_y, s_z]`; read from scale.txt
% - `lv_center = [C_x, C_y]` (pixels in the rotated/cropped frame)
% - `lv_start_end_frame` acts as Z_m (the axial slice index)
% - The angle loop 0:5:180 produces S = 37 angles including both 0° and 180°
% - Rotation axis is y (consistent with R_y(θ) in the paper)

%% Housekeeping
close all
clear all
clc

%% Parameters
bar_dist_mm = 20;        % Known bar distance in mm (used for scale calibration)
tui_spacing_mm = 0.5;    % Depth Unit Interval (TUI) distance in mm

%% Locate input videos
video_dir = pwd;
video_files = dir(fullfile(video_dir, 'video', '*.avi'));
num_time_frames = length(video_files);

largest_time_index = 1;  % Select a specific time frame for setup/calibration

%% Read selected video time frame
vid_data = read(VideoReader(fullfile(video_dir, 'video', ...
    sprintf('time%d.avi', largest_time_index))));
vid_data = reshape(vid_data(:,:,1,:), ...
    [size(vid_data,1), size(vid_data,2), size(vid_data,4)]);
num_frames = get(VideoReader(fullfile(video_dir, 'video', ...
    sprintf('time%d.avi', largest_time_index))), 'NumFrames');

%% Step 1: Select LV chamber (larger one)
figure('Name', 'Select LV (larger) — press Enter || Use ←/→ to browse slices');
lv_start_end_frame = [];
frame_index = 1;

while true
    imshow(vid_data(:,:,frame_index), 'InitialMagnification', 'fit');
    title('Select LV (larger) — press Enter || Use ←/→ to browse slices', ...
        'FontSize', 12, 'Color', 'r');
    key_press = waitforbuttonpress;
    key_val = double(get(gcf, 'CurrentCharacter'));

    if key_val == 28  % Left arrow
        frame_index = max(frame_index - 1, 1);
    elseif key_val == 29  % Right arrow
        frame_index = min(frame_index + 1, num_frames);
    elseif key_val == 13  % Enter key
        lv_start_end_frame = [lv_start_end_frame, frame_index];
        if length(lv_start_end_frame) == 1
            break;
        end
    end
end
close all;

%% Extract selected LV frame
img_selected_frame = vid_data(:,:,lv_start_end_frame);

%% Step 2: Scale calibration using known bar distance
figure('Name', ['Select 2 points (' num2str(bar_dist_mm) ' mm)']);
imshow(img_selected_frame, 'InitialMagnification', 'fit');
title([num2str(bar_dist_mm) ' mm scale (bar dots = 5 mm)'], ...
    'FontSize', 12, 'Color', 'r');
scale_points = ginput(2);
close all;

% Compute pixel resolution (mm/pixel)
num_pixels = abs(scale_points(1,2) - scale_points(2,2));
pixel_resolution_mm = bar_dist_mm / num_pixels;

% Save scale info
fileID = fopen(fullfile(video_dir, 'scale.txt'), 'w');
fprintf(fileID, '%f ', [pixel_resolution_mm, pixel_resolution_mm, tui_spacing_mm]);
fclose(fileID);

%% Step 3: Crop image to LV region
figure('Name', 'Crop image — draw bounding box and adjust');
imshow(img_selected_frame, 'InitialMagnification', 'fit');
title('Crop image — draw bounding box and adjust', 'FontSize', 12, 'Color', 'r');
[~, crop_rect] = imcrop;
close(gcf);

% Apply crop to all slices in video
vol_cropped = zeros(round(crop_rect(4)), round(crop_rect(3)), size(vid_data, 3), 'uint8');
for slice_idx = 1:size(vid_data, 3)
    vol_cropped(:,:,slice_idx) = imcrop(vid_data(:,:,slice_idx), crop_rect);
end

%% Step 4: Align LV vertically
figure('Name', 'Draw apex–base line to align LV vertically (inverted U) and press ~');
imshow(vol_cropped(:,:,lv_start_end_frame), 'InitialMagnification', 'fit');
title('Draw apex–base line to align LV vertically (inverted U)', ...
    'FontSize', 12, 'Color', 'r');

% Get apex–base line from user
h_line = imline;
wait(h_line);
line_coords = getPosition(h_line);

% Compute rotation angle
line_slope = (line_coords(2,2) - line_coords(1,2)) / (line_coords(2,1) - line_coords(1,1));
angle_to_x_axis_deg = rad2deg(atan(line_slope));
angle_to_y_axis_deg = -(90 - abs(angle_to_x_axis_deg));

% Rotate cropped volume
vol_aligned = imrotate3(vol_cropped, angle_to_y_axis_deg, [0 0 1], 'cubic');

%% Step 5: Confirm crop & rotation
figure('Name', 'Is crop and aligning okay?');
imshowpair(vol_cropped(:,:,lv_start_end_frame), ...
           vol_aligned(:,:,lv_start_end_frame), 'montage');
title('Is crop and aligning okay? If Yes, press Enter!', 'FontSize', 12, 'Color', 'r');
pause;
close all;

%% Step 6: Click LV center
figure('Name', 'Click the center of LV');
imshow(vol_aligned(:,:,lv_start_end_frame), 'InitialMagnification', 'fit');
title('Click the center of LV', 'FontSize', 12, 'Color', 'r');
lv_center = round(ginput(1));
close all;

% Save LV center coordinates
fileID = fopen(fullfile(video_dir, 'center.txt'), 'w');
fprintf(fileID, '%f ', lv_center);
fclose(fileID);

%% Step 7: Apply calibration, crop, and rotation to all time frames
spacing_values = readmatrix(fullfile(video_dir, 'scale.txt'));  % [s_x, s_y, s_z]
clear img_selected_frame vid_data vol_aligned vol_cropped;

% Iterate over all time points: t = 1..T
for t_idx = 1:num_time_frames
    % Load volume V^t ∈ R^{H' × W' × D'}
    raw_t = read(VideoReader(fullfile(video_dir, 'video', sprintf('time%d.avi', t_idx))));
    raw_t = reshape(raw_t(:,:,1,:), [size(raw_t,1), size(raw_t,2), size(raw_t,4)]);

    % Output directory for this t
    time_folder = fullfile(video_dir, sprintf('time%03d', t_idx));
    mkdir(time_folder);

    % Apply cropping to all slices in depth D'
    vol_cropped = zeros(round(crop_rect(4)), round(crop_rect(3)), size(raw_t, 3), 'uint8');
    for z_idx = 1:size(raw_t, 3)
        vol_cropped(:,:,z_idx) = imcrop(raw_t(:,:,z_idx), crop_rect);
    end

    % Apply in-plane rotation for LV vertical alignment
    vol_rotated = imrotate3(vol_cropped, angle_to_y_axis_deg, [0 0 1], 'cubic');

    % ----- Build 3D meshgrid X over V^t with anisotropic spacing -----
    [X, Y, Z] = meshgrid( ...
        (1:size(vol_rotated,2)) * spacing_values(1), ... % x_i = i*s_x
        (1:size(vol_rotated,1)) * spacing_values(2), ... % y_j = j*s_y
        (1:size(vol_rotated,3)) * spacing_values(3));    % z_k = k*s_z

    % ----- Center to LV [C_x, C_y] and reference slice Z_m -----
    Xc = X - (lv_center(1) * spacing_values(1));
    Yc = Y - (lv_center(2) * spacing_values(2));
    Zc = Z - (lv_start_end_frame * spacing_values(3));   % lv_start_end_frame = Z_m

    coords_centered = [Xc(:), Yc(:), Zc(:)];
    clear Xc Yc Zc;

    % ----- Uniform angular sampling θ ∈ [0, π] at 5° increments (S = 37) -----
    save_index = 1;
    for rotation_angle_deg = 0:5:180
        % Rotation matrix R_y(θ) about y-axis
        R_y = [ cosd(rotation_angle_deg), 0, sind(rotation_angle_deg); ...
                0,                      1, 0; ...
               -sind(rotation_angle_deg), 0, cosd(rotation_angle_deg) ];

        % Rotate in centered coords
        rotated_centered = (R_y * coords_centered')';

        % Map back to original frame coords
        Xr = reshape(rotated_centered(:,1), size(X)) + (lv_center(1) * spacing_values(1));
        Yr = reshape(rotated_centered(:,2), size(Y)) + (lv_center(2) * spacing_values(2));
        Zr = reshape(rotated_centered(:,3), size(Z)) + (lv_start_end_frame * spacing_values(3));

        % ----- Cubic interpolation -----
        vol_theta = uint8(interp3(X, Y, Z, double(vol_rotated), Xr, Yr, Zr, 'cubic'));

        % Extract axial slice at Z_m (I^t_θ)
        I_theta = vol_theta(:,:,lv_start_end_frame);

        % Save 2D radial slice
        imwrite(I_theta, fullfile(time_folder, ...
            sprintf('slice%03dtime%03d.png', save_index, t_idx)), 'png');
        save_index = save_index + 1;

        clear vol_theta rotated_centered Xr Yr Zr I_theta;
    end

    clear coords_centered vol_rotated X Y Z raw_t;
end