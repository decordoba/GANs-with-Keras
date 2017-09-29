FIRST_SAMPLE_NUM = 0;
LAST_SAMPLE_NUM = 59999;
num_samples = LAST_SAMPLE_NUM - FIRST_SAMPLE_NUM + 1

IMAGE_SIZE = [64, 64];
MEAN_NUMBER_LUMPS = 200;
DC = 10;
LUMP_FUNCTION = 'GaussLmp';
PARS = [1, 10];

% create folders
folder_dataset = 'dataset/';
folder_lumps = 'lumps/';
mkdir(folder_dataset);
mkdir(folder_lumps);
extension = '.txt';

% save data
for i = FIRST_SAMPLE_NUM:LAST_SAMPLE_NUM
    [matrix, num_lumps, pos_lumps] = LumpyBgnd(IMAGE_SIZE, MEAN_NUMBER_LUMPS, DC, LUMP_FUNCTION, PARS);
    filename = [num2str(i, '%06d'), '_', num2str(num_lumps, '%03d'), extension];
    dlmwrite([folder_dataset, filename], matrix, 'precision', 6)
    dlmwrite([folder_lumps, filename], pos_lumps, 'precision', 6)
    %save([folder_dataset, filename], 'matrix', '-ascii');
    %save([folder_lumps, filename], 'pos_lumps', '-ascii');
    
    if mod(i, 100) == 0
        c = clock;
        disp([num2str(c(4), '%02d'), ':', num2str(c(5), '%02d'), '.', num2str(int8(c(6)), '%02d'), '  ', 'It. ', num2str(i)])
    end
end