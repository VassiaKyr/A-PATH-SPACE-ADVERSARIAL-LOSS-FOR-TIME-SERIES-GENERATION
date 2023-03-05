function X = generate_double_well_data(mu_0, sgm_0)
%
% Simulates up to a 3-dimensional sde system with two^k wells.
% The potential = -(X_k^3 - a_k*X_k)
%
% ---- INPUT PARAMETERS ----
%       mu_0: mean vector for the initial distributions (scalar)
%       sgm_0: standard deviation for the initial distributions (scalar)
%       
% ---- OUTPUT PARAMETERS ----
%
% (c) yannis.pantazis@gmail.com, CausalPath, CSD, UOC, 2016
%

N =  length(mu_0); % dimension of the stochastic process

T_end = 1000;
dt = 0.01;
NoT = floor(T_end/dt)+1;

sgm = 1;  %  sigma of the Brownian motion

% sub-sampling
R = 5;
dt1 = dt/R;  

X = zeros(NoT, N);

Xnew = mu_0 + sgm_0*randn(1,N); % starting instant
t = 0;  % starting time

j = 1;
while t <= T_end
    % save
    X(j,:) = Xnew;
    j = j+1;
    
    % noise term
    dw1 = sqrt(dt1) * randn(1, R);
    dw2 = sqrt(dt1) * randn(1, R);
%     dw3 = sqrt(dt1) * randn(1, R);
    
    % advance the process with EULER-MARUYAMA numerical scheme
    Xnew(1) = Xnew(1) - (Xnew(1).^3-2.25*Xnew(1))*dt + sgm * sum(dw1,2);
    Xnew(2) = Xnew(2) - (Xnew(2).^3-1*Xnew(2))*dt + sgm * sum(dw2,2); 
%     if N>=2
%         Xnew(:,2) = Xnew(:,2) - (Xnew(:,2).^3-0.5^2*Xnew(:,2))*dt + sgm *sum(dw2,2);
%     end
%     if N>=3
%         Xnew(:,3) = Xnew(:,3) - (Xnew(:,3).^3-0.75^2*Xnew(:,3))*dt + sgm *sum(dw3,2);
%     end
   
    t = t + dt; % advance time
end

