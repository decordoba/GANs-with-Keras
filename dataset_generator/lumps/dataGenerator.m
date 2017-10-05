function dataGenerator(save)

FIRST_SAMPLE_NUM = 0;
LAST_SAMPLE_NUM = 59999;
num_samples = LAST_SAMPLE_NUM - FIRST_SAMPLE_NUM + 1;
disp(['Samples generated: ', num2str(num_samples)]);

IMAGE_SIZE = [64, 64];
MEAN_NUMBER_LUMPS = 200;
DC = 10;
LUMP_FUNCTION = 'GaussLmp';
PARS = [1, 2];  % [1, 10]

SAVE = true;  % true: save images, false: show images
if nargin > 0
    SAVE = save;
end

if SAVE
    % create folders
    folder_dataset = 'dataset/';
    folder_lumps = 'lumps/';
    mkdir(folder_dataset);
    mkdir(folder_lumps);
    extension = '.txt';
else
    close;
    disp('Warning, images are only shown, not saved!');
end

% save or show data
for i = FIRST_SAMPLE_NUM:LAST_SAMPLE_NUM
    [matrix, num_lumps, pos_lumps] = LumpyBgnd(IMAGE_SIZE, MEAN_NUMBER_LUMPS, DC, LUMP_FUNCTION, PARS);
    if SAVE
        % save data
        filename = [num2str(i, '%06d'), '_', num2str(num_lumps, '%03d'), extension];
        dlmwrite([folder_dataset, filename], matrix, 'precision', 6)
        dlmwrite([folder_lumps, filename], pos_lumps, 'precision', 6)
    else
        % show data
        imshow(matrix, [])
    end

    if mod(i, 100) == 0
        c = clock;
        disp([num2str(c(4), '%02d'), ':', num2str(c(5), '%02d'), '.', num2str(int8(c(6)), '%02d'), '  ', 'It. ', num2str(i)])
    end
end