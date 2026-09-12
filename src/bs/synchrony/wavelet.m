% Behavioral wavelet synchrony for paired Head and motion signals.
% Run through: bs wavelet
% External dependency: ASToolbox2018 AWCOG and MeanGAIN (see docs/guide.md).
clear;
clc;
resolved = getenv('BS_RESOLVED_CONFIG');
if isempty(resolved)
    error('Run this analysis with bs wavelet so all paths are resolved consistently.');
end
cfg = jsondecode(resolved);
manifest = jsondecode(fileread(cfg.wavelet_manifest));
data_dir = cfg.prepared_dir;
output_dir = cfg.wavelet_results_dir;
maps_dir = fullfile(output_dir, 'maps');
addpath(genpath(cfg.wavelet_toolbox_dir));
if isempty(which('AWCOG')) || isempty(which('MeanGAIN'))
    error('Install ASToolbox2018 and configure wavelet_toolbox_dir.');
end
% MeanGAIN must propagate an invalid scale to the whole time point.
probe = MeanGAIN([1, NaN; 1, 1], [1; 2], 0, 0);
if abs(probe(1) - 1) > 1e-12 || ~isnan(probe(2))
    error('MeanGAIN semantics differ: expected abs(mean(complex WCO)) with NaN propagation.');
end
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

log_file = fullfile(output_dir, 'log_bs_mea_awcog_macro_022_049Hz.txt');
if isfile(log_file), delete(log_file); end
diary(log_file);

%  분석 파라미터
SAVE_MAPS = false;  % true이면 그룹 x Phase x 신호별 coherence map 저장
if SAVE_MAPS && ~exist(maps_dir, 'dir'), mkdir(maps_dir); end %#ok<UNRCH>

groups = cellstr(string(manifest.groups));
sessions = cellstr(string(manifest.sessions));

signal_names  = {'pose_Rx', 'pose_Ry', 'pose_Rz', 'mea_z'};
signal_labels = {'Pitch', 'Yaw', 'Roll', 'MEA'};

source_hz = 25;
target_hz = 10;
dt = 1 / target_hz;
dj = 1 / 30;
resample_ratio = target_hz / source_hz;

% Behavioral analysis frequency band
freq_low_hz  = 0.22;
freq_high_hz = 0.49;
low_period   = 1 / freq_high_hz;  % 약 2.0408초
up_period    = 1 / freq_low_hz;   % 약 4.5455초

% AWCOG가 위 period 범위만 생성하므로 MeanGAIN은 전체 period 사용
meanGain_low_period = 0;
meanGain_up_period  = 0;

% AWCOG 파라미터: Behavioral coherence parameters
pad     = 0;
mother  = 'Morlet';
beta    = 6.0;
gamma   = [];
wt_size = 5;
ws_size = 10;
n_sur   = 0;
p_ar    = 1;
q_ar    = 1;

% Fisher's Z: Applied after raw-score averaging
fisher_epsilon = 1e-6;

%  결과 저장용 변수
session_rows = {};
analysis_errors = {};

fprintf('============================================================\n');
fprintf('  WTC Head BS + MEA Macro Pipeline (AWCOG)\n');
fprintf('  입력: Pitch, Yaw, Roll, MEA\n');
fprintf('  리샘플링: %d Hz -> %d Hz\n', source_hz, target_hz);
fprintf('  주파수 대역: %.2f~%.2f Hz\n', freq_low_hz, freq_high_hz);
fprintf('  주기 범위: %.4f~%.4f초\n', low_period, up_period);
fprintf('  분석 단위: Phase 전체(Macro), 60초 분절 없음\n');
fprintf('  Fisher Z: 모든 평균 완료 후 적용 (epsilon %.0e)\n', ...
    fisher_epsilon);
fprintf('============================================================\n\n');

%  메인 루프
for gi = 1:length(groups)
    group_name = groups{gi};

    for si = 1:length(sessions)
        session_name = sessions{si};
        p1_file = fullfile(data_dir, group_name, ...
            sprintf('%s_P1_%s_slim.csv', group_name, session_name));
        p2_file = fullfile(data_dir, group_name, ...
            sprintf('%s_P2_%s_slim.csv', group_name, session_name));

        if ~isfile(p1_file) || ~isfile(p2_file)
            msg = sprintf('%s %s - P1/P2 파일 없음', ...
                group_name, session_name);
            fprintf('[ERROR] %s\n', msg);
            analysis_errors{end+1} = msg; %#ok<SAGROW>
            continue;
        end

        % ----------------------------------------------------
        % 통합 입력 검증
        % ----------------------------------------------------
        try
            opts1 = detectImportOptions(p1_file);
            opts2 = detectImportOptions(p2_file);
            df1 = readtable(p1_file, opts1);
            df2 = readtable(p2_file, opts2);

            required_names = [{'frame', 'timestamp'}, signal_names];
            if ~all(ismember(required_names, df1.Properties.VariableNames)) || ...
                    ~all(ismember(required_names, df2.Properties.VariableNames))
                error(['필수 열(frame, timestamp, pose_Rx, pose_Ry, ' ...
                    'pose_Rz, mea_z)이 없습니다.']);
            end

            expected_n = manifest.sample_counts.(group_name).(session_name);
            if height(df1) ~= expected_n || height(df2) ~= expected_n
                error('원본 길이 오류: P1=%d, P2=%d, expected=%d', ...
                    height(df1), height(df2), expected_n);
            end
            if ~isequal(double(df1.frame), double(df2.frame))
                error('P1/P2 frame이 일치하지 않습니다.');
            end
            if any(abs(double(df1.timestamp) - double(df2.timestamp)) > 1e-9)
                error('P1/P2 timestamp가 일치하지 않습니다.');
            end
            if any(diff(double(df1.frame)) <= 0) || ...
                    any(diff(double(df1.timestamp)) <= 0)
                error('frame 또는 timestamp가 단조 증가하지 않습니다.');
            end
        catch err
            msg = sprintf('%s %s 입력 검증 실패: %s', ...
                group_name, session_name, err.message);
            fprintf('[ERROR] %s\n', msg);
            analysis_errors{end+1} = msg; %#ok<SAGROW>
            continue;
        end

        n_resampled = floor(expected_n * resample_ratio);
        t_original = linspace(0, 1, expected_n);
        t_resampled = linspace(0, 1, n_resampled);
        raw_signal_scores = nan(1, length(signal_names));

        % ----------------------------------------------------
        % Pitch, Yaw, Roll, MEA에 동일한 AWCOG 계산
        % ----------------------------------------------------
        for signal_index = 1:length(signal_names)
            column_name = signal_names{signal_index};
            display_name = signal_labels{signal_index};

            try
                p1_raw = double(df1.(column_name))';
                p2_raw = double(df2.(column_name))';
                if any(~isfinite(p1_raw)) || any(~isfinite(p2_raw))
                    error('원본 신호에 NaN 또는 Inf가 있습니다.');
                end

                % STEP 1: 25 Hz -> 10 Hz 선형 리샘플링
                p1_signal = interp1( ...
                    t_original, p1_raw, t_resampled, 'linear');
                p2_signal = interp1( ...
                    t_original, p2_raw, t_resampled, 'linear');
                if any(~isfinite(p1_signal)) || any(~isfinite(p2_signal))
                    error('리샘플링된 신호에 NaN 또는 Inf가 있습니다.');
                end

                % STEP 2: Phase 전체에 AWCOG 수행
                % 네 개의 출력만 요청하여 surrogate p-value 계산 생략
                [WCO, ~, periods, coi] = AWCOG( ...
                    p1_signal, p2_signal, dt, dj, ...
                    low_period, up_period, pad, mother, beta, gamma, ...
                    wt_size, ws_size, n_sur, p_ar, q_ar);

                % STEP 3: 각 시간점의 COI 경계 밖 값 제외
                coi_mask = periods(:) > coi(:).';
                WCO(coi_mask) = NaN;

                if SAVE_MAPS
                    fig = figure('Visible', 'off'); %#ok<UNRCH>
                    imagesc((0:size(WCO, 2)-1) * dt, ...
                        log2(periods), abs(WCO));
                    set(gca, 'YDir', 'normal');
                    hold on;
                    plot((0:length(coi)-1) * dt, log2(coi), ...
                        'k--', 'LineWidth', 2);
                    hold off;
                    title(sprintf( ...
                        '%s %s %s Macro Coherence %.2f-%.2f Hz', ...
                        group_name, session_name, display_name, ...
                        freq_low_hz, freq_high_hz), ...
                        'Interpreter', 'none');
                    xlabel('Time (s)');
                    ylabel('Period (log2 s)');
                    colormap('jet');
                    colorbar;
                    saveas(fig, fullfile(maps_dir, sprintf( ...
                        '%s_%s_%s_macro_022_049Hz.png', ...
                        group_name, session_name, column_name)));
                    close(fig);
                end

                % STEP 4: 주파수 평균 후 Phase 전체 시간 평균
                % MeanGAIN은 abs(mean(complex WCO, frequency)) 반환
                mean_gain = MeanGAIN(WCO, periods, ...
                    meanGain_low_period, meanGain_up_period);
                raw_score = mean(mean_gain, 'omitnan');
                if ~isfinite(raw_score)
                    error('Macro raw coherence 결과가 NaN 또는 Inf입니다.');
                end
                raw_signal_scores(signal_index) = raw_score;

            catch err
                msg = sprintf('%s %s %s: %s', ...
                    group_name, session_name, display_name, err.message);
                fprintf('[ERROR] %s\n', msg);
                analysis_errors{end+1} = msg; %#ok<SAGROW>
                raw_signal_scores(signal_index) = NaN;
            end
        end

        if any(~isfinite(raw_signal_scores))
            msg = sprintf('%s %s - 네 신호의 Macro 점수가 완전하지 않음', ...
                group_name, session_name);
            fprintf('[SKIP] %s\n', msg);
            analysis_errors{end+1} = msg; %#ok<SAGROW>
            continue;
        end

        % STEP 5: 종합 점수는 raw coherence를 먼저 평균한다.
        raw_head_avg = mean(raw_signal_scores(1:3));
        raw_all4_avg = mean(raw_signal_scores);

        % STEP 6: 모든 평균 완료 후 각각 Fisher's Z를 적용한다.
        fisher_signal_scores = fisher_z_transform( ...
            raw_signal_scores, fisher_epsilon);
        fisher_head_avg = fisher_z_transform(raw_head_avg, fisher_epsilon);
        fisher_all4_avg = fisher_z_transform(raw_all4_avg, fisher_epsilon);

        session_rows(end+1, :) = [ ...
            {group_name, session_name, expected_n, n_resampled}, ...
            num2cell(raw_signal_scores), ...
            {raw_head_avg, raw_all4_avg}, ...
            num2cell(fisher_signal_scores), ...
            {fisher_head_avg, fisher_all4_avg}]; %#ok<SAGROW>

        fprintf(['[%s %s] raw Pitch=%.6f Yaw=%.6f Roll=%.6f ' ...
            'MEA=%.6f | Head=%.6f All4=%.6f\n'], ...
            group_name, session_name, raw_signal_scores(1), ...
            raw_signal_scores(2), raw_signal_scores(3), ...
            raw_signal_scores(4), raw_head_avg, raw_all4_avg);
    end
end

%  통합 결과 CSV 저장
expected_result_rows = length(groups) * length(sessions);
if size(session_rows, 1) ~= expected_result_rows
    fprintf('\n[FAILED] 완료 결과 %d/%d개\n', ...
        size(session_rows, 1), expected_result_rows);
    if ~isempty(analysis_errors)
        fprintf('  - %s\n', analysis_errors{:});
    end
    diary off;
    error('입력 또는 분석 오류가 있어 통합 결과 CSV를 갱신하지 않았습니다.');
end

column_names = { ...
    'group', 'session', 'source_samples', 'resampled_samples', ...
    'pose_Rx_raw', 'pose_Ry_raw', 'pose_Rz_raw', 'mea_score_raw', ...
    'bs_score_avg_raw', 'all4_score_avg_raw', ...
    'pose_Rx_FisherZ', 'pose_Ry_FisherZ', 'pose_Rz_FisherZ', ...
    'mea_score_FisherZ', 'bs_score_avg_FisherZ', ...
    'all4_score_avg_FisherZ'};

session_table = cell2table(session_rows, 'VariableNames', column_names);
keys = strcat(string(session_table.group), "|", string(session_table.session));
if length(unique(keys)) ~= expected_result_rows
    diary off;
    error('통합 결과에 중복된 group-session 행이 있습니다.');
end

numeric_result = session_table{:, 3:end};
if any(~isfinite(numeric_result), 'all')
    diary off;
    error('통합 결과에 NaN 또는 Inf가 있어 CSV를 저장하지 않았습니다.');
end

output_file = fullfile(output_dir, ...
    'bs_mea_macro_scores_awcog_022_049Hz.csv');
writetable(session_table, output_file);

fprintf('\n============================================================\n');
fprintf('  완료: %d/%d개 그룹-Phase\n', ...
    height(session_table), expected_result_rows);
fprintf('  결과: %s\n', output_file);
fprintf('  로그: %s\n', log_file);
fprintf('============================================================\n');

diary off;


%% Local function: Fisher's Z transformation
function z = fisher_z_transform(values, epsilon)
    if any(values < -1 | values > 1, 'all')
        error('Fisher Z 입력값이 [-1, 1] 범위를 벗어났습니다.');
    end

    clipped = min(max(values, -1 + epsilon), 1 - epsilon);
    z = atanh(clipped);
end
