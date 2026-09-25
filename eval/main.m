% TG-Net evaluation entry point.
%
% Computes the seven metrics reported in the paper (mean Dice, mean IoU,
% weighted F-measure, S-measure, mean/max E-measure, MAE, plus sensitivity and
% specificity) by sweeping a 256-step threshold grid, following the protocol of
% Fan et al. (Structure-measure, ICCV'17; Enhanced-alignment measure, IJCAI'18)
% and Margolin et al. (Weighted F-measure, CVPR'14).
%
% Usage
% -----
%   1. Produce prediction maps with:
%        python test.py --config configs/ham10000.yaml
%      They land in results/<dataset>/ as 8-bit PNGs named <case>.png.
%
%   2. Set DATASET below to the folder that holds both the ground truth
%      (`<DATASET>/masks/`) and the predictions (`<DATASET>/outputs/`), then
%      run this script from the `eval/` directory:
%        >> main
%
% Results are printed to the command window and written to `../res/`.

clear all;
close all;
clc;

% ---- 1. Path configuration ------------------------------------------------
% Root folder holding one subfolder per dataset, each with `masks/` and
% `outputs/`. Override this to point at your own results directory.
ResultMapPath = fullfile('..', 'results');

% Datasets to evaluate, each must exist as a subfolder of ResultMapPath and
% contain a `masks/` folder with the ground truth.
Datasets = {'HAM10000'};

% Subfolder of each dataset holding the prediction maps.
PredictionDir = 'outputs';

% ---- 2. Evaluation configuration ------------------------------------------
ResDir = fullfile('..', 'res');
ResName = '_result.txt';

% 256-step threshold sweep from 1 down to 0.
Thresholds = 1:-1/255:0;

datasetNum = numel(Datasets);

for d = 1:datasetNum
    tic;
    dataset = Datasets{d};
    fprintf('Processing %d/%d: %s Dataset\n', d, datasetNum, dataset);

    ResPath = fullfile(ResDir, [dataset '-mat']);
    if ~exist(ResPath, 'dir')
        mkdir(ResPath);
    end
    resTxt = fullfile(ResDir, [dataset ResName]);
    fileID = fopen(resTxt, 'w');

    gtPath = fullfile(ResultMapPath, dataset, 'masks');
    resMapPath = fullfile(ResultMapPath, dataset, PredictionDir);

    imgFiles = dir(fullfile(resMapPath, '*.png'));
    imgNUM = numel(imgFiles);

    if imgNUM == 0
        warning('No .png predictions found in %s -- skipping %s.', ...
                resMapPath, dataset);
        fclose(fileID);
        continue;
    end

    fprintf('  ground truth : %s\n', gtPath);
    fprintf('  predictions  : %s (%d files)\n', resMapPath, imgNUM);

    [threshold_Fmeasure, threshold_Emeasure, threshold_IoU] = ...
        deal(zeros(imgNUM, length(Thresholds)));
    [threshold_Precion, threshold_Recall] = ...
        deal(zeros(imgNUM, length(Thresholds)));
    [threshold_Sensitivity, threshold_Specificity, threshold_Dice] = ...
        deal(zeros(imgNUM, length(Thresholds)));

    [Smeasure, wFmeasure, MAE] = deal(zeros(1, imgNUM));

    for i = 1:imgNUM
        name = imgFiles(i).name;

        % ---- load ground truth ----
        gt = imread(fullfile(gtPath, name));
        if (ndims(gt) > 2)
            gt = rgb2gray(gt);
        end
        if ~islogical(gt)
            gt = gt(:, :, 1) > 128;
        end

        % ---- load prediction ----
        resmap = imread(fullfile(resMapPath, name));

        % Predictions are exported at the original image size, so a mismatch
        % means the ground truth is the odd one out. Resize it and report.
        if size(resmap, 1) ~= size(gt, 1) || size(resmap, 2) ~= size(gt, 2)
            gt = imresize(gt, size(resmap));
            fprintf(['Resizing ground truth: size mismatch for %s ' ...
                     '(prediction %dx%d)\n'], ...
                    name, size(resmap, 1), size(resmap, 2));
        end

        resmap = im2double(resmap(:, :, 1));

        % S-measure (ICCV'17)
        Smeasure(i) = StructureMeasure(resmap, logical(gt));

        % Weighted F-measure (CVPR'14)
        wFmeasure(i) = original_WFb(resmap, logical(gt));

        MAE(i) = mean2(abs(double(logical(gt)) - resmap));

        [threshold_E, threshold_F, threshold_Pr, threshold_Rec, threshold_Iou] = ...
            deal(zeros(1, length(Thresholds)));
        [threshold_Spe, threshold_Dic] = deal(zeros(1, length(Thresholds)));

        for t = 1:length(Thresholds)
            threshold = Thresholds(t);
            [threshold_Pr(t), threshold_Rec(t), threshold_Spe(t), ...
             threshold_Dic(t), threshold_F(t), threshold_Iou(t)] = ...
                Fmeasure_calu(resmap, double(gt), size(gt), threshold);

            Bi_resmap = zeros(size(resmap));
            Bi_resmap(resmap > threshold) = 1;
            threshold_E(t) = Enhancedmeasure(Bi_resmap, gt);
        end

        threshold_Emeasure(i, :) = threshold_E;
        threshold_Fmeasure(i, :) = threshold_F;
        threshold_Sensitivity(i, :) = threshold_Rec;
        threshold_Specificity(i, :) = threshold_Spe;
        threshold_Dice(i, :) = threshold_Dic;
        threshold_IoU(i, :) = threshold_Iou;

        if mod(i, 100) == 0 || i == imgNUM
            fprintf('  evaluating: %d/%d\n', i, imgNUM);
        end
    end

    % ---- aggregate ----
    mae = mean2(MAE);
    Sm = mean2(Smeasure);
    wFm = mean2(wFmeasure);

    column_E = mean(threshold_Emeasure, 1);
    meanEm = mean(column_E);
    maxEm = max(column_E);

    column_Sen = mean(threshold_Sensitivity, 1);
    meanSen = mean(column_Sen);
    maxSen = max(column_Sen);

    column_Spe = mean(threshold_Specificity, 1);
    meanSpe = mean(column_Spe);
    maxSpe = max(column_Spe);

    column_Dic = mean(threshold_Dice, 1);
    meanDic = mean(column_Dic);
    maxDic = max(column_Dic);

    column_IoU = mean(threshold_IoU, 1);
    meanIoU = mean(column_IoU);
    maxIoU = max(column_IoU);

    save(fullfile(ResPath, 'res_test.mat'), ...
         'Sm', 'mae', 'column_Dic', 'column_Sen', 'column_Spe', 'column_E', ...
         'column_IoU', 'maxDic', 'maxEm', 'maxSen', 'maxSpe', 'maxIoU', ...
         'meanIoU', 'meanDic', 'meanEm', 'meanSen', 'meanSpe');

    summaryFormat = ['(Dataset:%s) meanDic:%.3f;meanIoU:%.3f;wFm:%.3f;Sm:%.3f;' ...
                     'meanEm:%.3f;MAE:%.3f;maxEm:%.3f;maxDice:%.3f;maxIoU:%.3f;' ...
                     'meanSen:%.3f;maxSen:%.3f;meanSpe:%.3f;maxSpe:%.3f.\n'];
    fprintf(fileID, summaryFormat, dataset, meanDic, meanIoU, wFm, Sm, ...
            meanEm, mae, maxEm, maxDic, maxIoU, meanSen, maxSen, meanSpe, maxSpe);
    fprintf(summaryFormat, dataset, meanDic, meanIoU, wFm, Sm, ...
            meanEm, mae, maxEm, maxDic, maxIoU, meanSen, maxSen, meanSpe, maxSpe);

    fclose(fileID);
    toc;
end
