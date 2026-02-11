% =========================================================================
% data_generator_needles_only.m
% 功能：追加生成 [随机 Needle/Geometric Drop] (5000组)
% 修复：加入 Taskkill 强制清理僵尸进程，解决无限重启死循环
% 输出：自动接续 (目前将从 4039 开始)
% =========================================================================
clear; clc;

% --- 1. 配置与初始化 ---
% ⚠️ 数据文件夹
data_folder = 'E:\hkust\Meta_AI_Project\data_cross_drop_9k\';
if ~exist(data_folder, 'dir'), mkdir(data_folder); end

% 定义任务量
num_cross_existed = 4000; 
num_needles = 5000;       
total_target = num_cross_existed + num_needles; 

% Lumerical 连接配置
lumerical_path = 'C:\Program Files\Lumerical\v202\'; % 请确认版本号
setenv('PATH', [getenv('PATH') ';' lumerical_path 'bin']);
path(path, [lumerical_path 'api\matlab']);

% 物理参数
n_pixels = 32;
pixel_size = 340e-9 / n_pixels;
meta_thickness = 50e-9;

% --- 智能断点续跑逻辑 ---
files = dir(fullfile(data_folder, 'sample_mixed_*.mat'));
current_count = length(files);

% 强制起始点：自动接在现有文件后面
% 如果你现在有 4038 个文件，start_index 就会自动变成 4039
start_index = max(current_count + 1, num_cross_existed + 1);

fprintf('----------------------------------------\n');
fprintf('🏭 Needle 追加任务启动 (抗干扰版)\n');
fprintf('📊 已有文件数: %d\n', current_count);
fprintf('🚀 本次任务范围: %d -> %d\n', start_index, total_target);
fprintf('----------------------------------------\n');

% --- 2. 主循环 ---
for i = start_index : total_target
    
    % 初始化画布
    mask = zeros(n_pixels, n_pixels);
    x = linspace(-1, 1, n_pixels);
    [X, Y] = meshgrid(x, x);
    
    if mod(i, 10) == 0
        fprintf('正在生成样本 %d / %d ...\n', i, total_target);
    end
    
    % ==========================================================
    % 🎨 生成 Geometric Drop (Needle)
    % ==========================================================
    shape_id = 99;
    params = zeros(1, 5); 
    
    enable_symmetry = (rand() > 0.5);
    num_shapes = randi([2, 5]);
    
    for k = 1:num_shapes
        type = randi([1, 4]); 
        cx = (rand()-0.5)*1.5; cy = (rand()-0.5)*1.5;
        rot = rand()*360; rad = deg2rad(rot);
        X_t = (X - cx)*cos(rad) - (Y - cy)*sin(rad);
        Y_t = (X - cx)*sin(rad) + (Y - cy)*cos(rad);
        
        temp_mask = zeros(n_pixels, n_pixels);
        switch type
            case 1 % Rect
                w = 0.15 + 0.5*rand(); h = 0.15 + 0.5*rand();
                temp_mask = (abs(X_t)<w) & (abs(Y_t)<h);
            case 2 % Circle
                r = 0.15 + 0.4*rand();
                temp_mask = (X_t.^2 + Y_t.^2) < r^2;
            case 3 % Cross
                L = 0.4+0.4*rand(); W = 0.1+0.15*rand();
                temp_mask = ((abs(X_t)<L)&(abs(Y_t)<W)) | ((abs(X_t)<W)&(abs(Y_t)<L));
            case 4 % Ring
                r_out = 0.3+0.4*rand(); r_in = r_out - (0.1+0.2*rand());
                R_dist = sqrt(X_t.^2 + Y_t.^2);
                temp_mask = (R_dist > r_in) & (R_dist < r_out);
        end
        
        is_add = (k==1) || (rand() > 0.3); 
        if is_add
            mask = mask | temp_mask;
        else
            mask = mask & (~temp_mask);
        end
    end
    
    if enable_symmetry
        mask = mask | rot90(mask, 1) | rot90(mask, 2) | rot90(mask, 3);
    end
    
    mask = double(mask > 0);
    
    % ==========================================================
    % 3. FDTD 仿真 (核弹级稳健连接版)
    % ==========================================================
    
    is_connected = false;
    retry_count = 0;
    
    while ~is_connected
        try
            % 1. 检查句柄是否存在且有效
            if exist('h', 'var') && ~isempty(h)
                % 尝试轻量测试
                appevalscript(h, ' '); 
                % 尝试传数据
                appputvar(h, 'mask_data', mask);
                % 成功
                is_connected = true;
            else
                error('Handle h is empty or undefined');
            end
            
        catch
            % 2. 报错处理
            retry_count = retry_count + 1;
            fprintf('⚠️ 连接故障 (第 %d 次尝试重连)...\n', retry_count);
            
            % A. 试图关闭旧句柄
            try appclose(h); catch; end
            clear h; 
            
            % B. 🔥🔥🔥 核心大招：强制杀掉 Windows 后台 FDTD 进程 🔥🔥🔥
            % 防止僵尸进程占用端口导致 misconnecting
            [~, ~] = system('taskkill /F /IM fdtd-solutions.exe /T');
            
            % C. 重新配置路径
            path(path, [lumerical_path 'api\matlab']);
            
            % D. 重启
            fprintf('🔄 正在重启 Lumerical... ');
            try
                h = appopen('fdtd');
                fprintf('✅ 指令已发送\n');
            catch
                fprintf('❌ 启动失败，等待 5 秒后再试\n');
                pause(5);
                continue; 
            end
            
            % E. 初始化等待 (加长到 8 秒，确保 API 就绪)
            fprintf('⏳ 初始化等待 (8秒)...\n');
            pause(8); 
        end
        
        % 防止死循环，重试超过 5 次跳过
        if retry_count > 5
            warning('❌ 无法建立连接，跳过样本 %d', i);
            break; 
        end
    end
    
    if ~is_connected
        continue;
    end

    % 3. 传输剩余变量
    appputvar(h, 'px_size', pixel_size);
    appputvar(h, 'N', n_pixels);
    appputvar(h, 'h_meta', meta_thickness);
    
    % --- Step 1: FDTD 设置 ---
    code_step1 = [ ...
        'switchtolayout; selectall; delete; ', ...
        'addfdtd; ', ...
        'set("dimension", "3D"); ', ...
        'set("simulation time", 2e-11); ', ...
        'set("x", 0); set("x span", N*px_size); ', ...
        'set("y", 0); set("y span", N*px_size); ', ...
        'set("z max", 1.5e-6); ', ...
        'set("z min", -1.5e-6); ', ...
        'set("mesh accuracy", 3); ', ...
        'set("x min bc", "Periodic"); set("x max bc", "Periodic"); ', ...
        'set("y min bc", "Periodic"); set("y max bc", "Periodic"); ', ...
        'set("z min bc", "PML"); set("z max bc", "PML"); ', ...
        'set("pml layers", 64); '];
    appevalscript(h, code_step1);
    
    % --- Step 2: 基底设置 ---
    appevalscript(h, 'if(getnamednumber("Substrate")>0) { select("Substrate"); delete; }'); 
    appevalscript(h, 'if(materialexists("SiO2 (Glass) - Palik")==0){addmaterial("Dielectric");setmaterial("Dielectric","name","SiO2 (Glass) - Palik");setmaterial("SiO2 (Glass) - Palik","Refractive Index",1.45);}');
    appevalscript(h, 'addrect; set("name", "Substrate");');
    appevalscript(h, 'set("override mesh order from material database", 1); set("mesh order", 3);');
    xy_span = n_pixels * pixel_size * 2;
    cmd_geo = ['set("x", 0); set("x span", ' num2str(xy_span) '); set("y", 0); set("y span", ' num2str(xy_span) '); set("z min", -5e-6); set("z max", 0);'];
    appevalscript(h, cmd_geo);
    appevalscript(h, 'set("material", "SiO2 (Glass) - Palik"); set("alpha", 0.3);');
   
    % --- Step 3: 结构绘制 ---
    max_retries = 3;
    is_success = false;
    code_step3 = [ ...
        'unselectall; ', ...  
        'if(materialexists("Au (Gold) - Palik")==0){addmaterial("Dielectric");setmaterial("Dielectric","name","Au (Gold) - Palik");setmaterial("Au (Gold) - Palik","Refractive Index",0.17,3.15);} ', ...
        'redrawoff; ', ... 
        'for(i=1:N){ for(j=1:N){ ', ...
        '  if(mask_data(i,j)==1){ ', ...
        '    addrect; ', ...
        '    set("x", (i-N/2-0.5)*px_size); set("x span", px_size); ', ...
        '    set("y", (j-N/2-0.5)*px_size); set("y span", px_size); ', ...
        '    set("z min", 0); set("z max", h_meta); ', ...
        '    set("material", "Au (Gold) - Palik"); ', ...
        '  } ', ...
        '}} ', ...
        'redrawon; ']; 
    
    for attempt = 1:max_retries
        try
            appevalscript(h, code_step3);
            is_success = true;
            break; 
        catch
            fprintf('⚠️ Step 3 重试 %d ...\n', attempt);
            appevalscript(h, 'select("Au (Gold) - Palik"); delete;'); 
        end
    end
    if ~is_success, error('❌ 结构绘制失败'); end
    
    % --- Step 4: 光源与监视器 ---
    code_step4 = [ ...
        'addplane; set("injection axis", "z-axis"); set("direction", "backward"); ', ...
        'set("z", 0.8e-6); set("x", 0); set("x span", N*px_size); set("y", 0); set("y span", N*px_size); ', ...
        'set("wavelength start", 400e-9); set("wavelength stop", 800e-9); ', ...
        'addpower; set("name", "T_monitor"); set("monitor type", "2D Z-normal"); ', ...
        'set("override global monitor settings", 1); set("frequency points", 1000); ', ...
        'set("z", -0.8e-6); set("x", 0); set("x span", N*px_size); set("y", 0); set("y span", N*px_size); '];
    appevalscript(h, code_step4);
    
    % --- Step 5: 运行与保存 ---
    temp_file = fullfile(pwd, '../data_raw/temp_sim.fsp');
    save_cmd = ['save("', replace(temp_file, '\', '/'), '");'];
    appevalscript(h, save_cmd);
    
    try
        appevalscript(h, 'run;');
    catch
        warning('❌ FDTD 运行错误，跳过该样本');
        continue;
    end
    
    appevalscript(h, 'T_res = getresult("T_monitor", "T");');
    appevalscript(h, 'T_val = T_res.T;');
    appevalscript(h, 'lambda = T_res.lambda;');
    
    T_val = appgetvar(h, 'T_val');
    lambda = appgetvar(h, 'lambda');
    
    % 保存结果
    filename = fullfile(data_folder, sprintf('sample_mixed_%d.mat', i));
    save(filename, 'mask', 'T_val', 'lambda', 'params', 'shape_id');
end
fprintf('✅ Needle save as: %s\n', data_folder);