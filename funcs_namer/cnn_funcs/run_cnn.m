function [x_cnn, py_runtime] = run_cnn(x_in, create_patches, patches_full_fn, ...
    patch_params, cnn_model_name, cnn_tmp_path, gpu_str)

% scale image so that it's magnitude ranges from 0 to 1
im_scale = max(abs(x_in(:)));
x_in = x_in / im_scale;

% create patches from the image if designated in the "create_patches"
% boolean
if create_patches
    x_test = createpatches(x_in, patch_params.patch_sz, patch_params.patch_stsz);
    x_test = permute(x_test,[3 1 2]);
    x_test = cat(4,real(x_test),imag(x_test));
    save(patches_full_fn, 'x_test')
end

output_filename = 'tmp_output_patches.mat';

% initialize python script paths
py_script_path = cnn_tmp_path;
py_input_filename = patches_full_fn;
py_input_var = 'x_test';
py_output_filename = [cnn_tmp_path, output_filename];
py_out_var = 'x_test_output';
py_sc_name = 'run_namer_cnn.py ';
model_filename = [cnn_tmp_path, cnn_model_name];

% boolean to decided if you want to see python output
print_output = true;

% create string to call python script with input and output variable names
system_command_str = ['python ', py_script_path, py_sc_name, ...
    py_input_filename, ' ', py_input_var, ' ', ...
    py_output_filename, ' ', py_out_var, ' ', ...
    model_filename, ' ', gpu_str];

% call python script
py_runtime_0 = tic;
if print_output
    sys_status = system(system_command_str);
else
    [sys_status, ~] = system(system_command_str);
end
py_runtime = toc(py_runtime_0);

% read in ouput of python script
load(py_output_filename,py_out_var);
patches_out = x_test_output;  %%% IMPORTANT! THIS MUST MATCH "py_out_var" string
patches_out = squeeze(patches_out);
patches_out = patches_out(:,:,:,1) + 1j*patches_out(:,:,:,2);

% sew patches together
output_sz = size(x_in);
artifact_image = sewpatches(patches_out,output_sz, patch_params.patch_sz, patch_params.patch_stsz);
x_cnn = x_in-artifact_image;

x_cnn = x_cnn * im_scale;
    
end