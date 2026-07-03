%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%   LICHTENBERG ALGORITHM (LA) FOR NONLINEAR UNCONSTRAINED
%                 AND CONSTRAINED OPTIMIZATION 
%
% AUTHORS: João Luiz Junho Pereira, Guilherme Ferreira Gomes and
% Sebastião Simões;
%
% A fastest Lichtenberg Algorithm(LA) applied to optimization inspired on
% storms with intraclouding radial lightning
%
%Please cite this algorithm as:
%
%PEREIRA, J. L. J. ; FRANCISCO, M. B. ; Diniz, C. A. ; OLIVER, G. A. ; CUNHA JR., S. S. ; GOMES, G. F. . 
%Lichtenberg Algorithm: A Novel Hybrid PHYSICS-Based Meta-Heuristic For Global Optimization. 
%EXPERT SYSTEMS WITH APPLICATIONS, 2020. https://doi.org/10.1016/j.eswa.2020.114522

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
close all
clc
format long
tic

set(0,'DefaultAxesFontName', 'Times New Roman')
set(0,'DefaultAxesFontSize', 14)
set(0,'DefaultTextFontname', 'Times New Roman')
set(0,'DefaultTextFontSize', 26)

% Search space
% Welded Beam Design Problem
LB=[0.1 0.1  0.1  0.1]; %Lower Bounds (Welded Beam Design Problem)
UB=[2.0 10.0 10.0 2.0]; %Upper bounds

d = length(LB);
% Optimizator Parameters
pop = 10*d;        %Population
n_iter = 100;      %Max number os iterations/gerations
ref = 0.4;         %if more than zero, a second LF is created with refinement % the size of the other
Np = 100000;       %Number of Particles (If 3D, better more than 10000)
S_c = 1;           %Stick Probability: Percentage of particles that can don´t stuck in the
                   %cluster. Between 0 and 1. Near 0 there are more aggregate, the density of
                   %cluster is bigger and difusity is low. Near 1 is the opposite. 
Rc = 150;          %Creation Radius (if 3D, better be less than 80, untill 150)
M = 0;             %If M = 0, no lichtenberg figure is created (it is loaded a optimized figure); if 1, a single is created and used in all iterations; If 2, one is created for each iteration.(creating an LF figure takes about 2 min)
im = 0;            %If im = 0, no figure is shown on the Matlab screen (faster). If one, yes. 
                   %But only for 2D and 3D

[x,fval,iter,state,population]=LA_optimization(@objective,d,im,pop,LB,UB,ref,n_iter,Np,Rc,S_c,M,@constraint);

toc