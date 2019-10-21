clear all; close all;

%% Find and import all audio training files

current_path = strcat(mfilename('fullpath'), '.m');

[filepath,name,ext] = fileparts(current_path);

files = dir(fullfile(strrep(filepath,'src','data/TRAIN')));
















%% Extract only speech region from data files













%% Do frequency analysis