% Combining MPC and MHE, MATLAB code
clear 
close all
clc

%% Before runig code, download CASADI (https://web.casadi.org/)
import casadi.*
import casadi.*

T = 0.2; %[s]
N = 60; % prediction horizon
rob_diam = 0.3;

v_max = 0.6; v_min = -v_max;
omega_max = pi/2; omega_min = -omega_max;

x = SX.sym('x'); y = SX.sym('y'); theta = SX.sym('theta');
states = [x;y;theta]; n_states = length(states);

v = SX.sym('v'); omega = SX.sym('omega');
controls = [v;omega]; n_controls = length(controls);
rhs = [v*cos(theta);v*sin(theta);omega]; % system r.h.s

f = Function('f',{states,controls},{rhs}); % nonlinear mapping function f(x,u)
U = SX.sym('U',n_controls,N); % Decision variables (controls)
P = SX.sym('P',n_states + n_states);
% parameters (which include at the initial state of the robot and the reference state)

X = SX.sym('X',n_states,(N+1));
% A vector that represents the states over the optimization problem.

obj = 0; % Objective function
g = [];  % constraints vector

Q = zeros(3,3); Q(1,1) = 1;Q(2,2) = 5;Q(3,3) = 0.1; % weighing matrices (states)
R = zeros(2,2); R(1,1) = 0.5; R(2,2) = 0.05; % weighing matrices (controls)

st  = X(:,1); % initial state
g = [g;st-P(1:3)]; % initial condition constraints
for k = 1:N
    st = X(:,k);  con = U(:,k);
    obj = obj+(st-P(4:6))'*Q*(st-P(4:6)) + con'*R*con; % calculate obj
    st_next = X(:,k+1);
    f_value = f(st,con);
    st_next_euler = st+ (T*f_value);
    g = [g;st_next-st_next_euler]; % compute constraints
end
% make the decision variable one column  vector
OPT_variables = [reshape(X,3*(N+1),1);reshape(U,2*N,1)];

nlp_prob = struct('f', obj, 'x', OPT_variables, 'g', g, 'p', P);

opts = struct;
opts.ipopt.max_iter = 2000;
opts.ipopt.print_level =0;%0,3
opts.print_time = 0;
opts.ipopt.acceptable_tol =1e-8;
opts.ipopt.acceptable_obj_change_tol = 1e-6;

solver = nlpsol('solver', 'ipopt', nlp_prob,opts);


args = struct;

args.lbg(1:3*(N+1)) = 0;
args.ubg(1:3*(N+1)) = 0;

args.lbx(1:3:3*(N+1),1) = -2; %state x lower bound
args.ubx(1:3:3*(N+1),1) = 2; %state x upper bound
args.lbx(2:3:3*(N+1),1) = -2; %state y lower bound
args.ubx(2:3:3*(N+1),1) = 2; %state y upper bound
args.lbx(3:3:3*(N+1),1) = -inf; %state theta lower bound
args.ubx(3:3:3*(N+1),1) = inf; %state theta upper bound

args.lbx(3*(N+1)+1:2:3*(N+1)+2*N,1) = v_min; %v lower bound
args.ubx(3*(N+1)+1:2:3*(N+1)+2*N,1) = v_max; %v upper bound
args.lbx(3*(N+1)+2:2:3*(N+1)+2*N,1) = omega_min; %omega lower bound
args.ubx(3*(N+1)+2:2:3*(N+1)+2*N,1) = omega_max; %omega upper bound
%----------------------------------------------
% ALL OF THE ABOVE IS JUST A PROBLEM SET UP

%% MHE part
N_MHE = 10; % prediction horizon

% r = SX.sym('r'); alpha = SX.sym('alpha'); % range and bearing
measurement_rhs = [sqrt(x^2+y^2); atan(y/x)];
h = Function('h',{states},{measurement_rhs}); % MEASUREMENT MODEL
%y_tilde = [r;alpha];

% Decision variables
Um = SX.sym('U',n_controls,N_MHE);    %(controls)
Xm = SX.sym('X',n_states,(N_MHE+1));  %(states) [remember multiple shooting]

Pm = SX.sym('P', 2 , N_MHE + (N_MHE+1));
% parameters (include r and alpha measurements as well as controls measurements)
% Synthesize the measurments
con_cov = diag([0.005 deg2rad(2)]).^2;
meas_cov = diag([0.1 deg2rad(2)]).^2;
V = inv(sqrt(meas_cov)); % weighing matrices (output)  y_tilde - y
W = inv(sqrt(con_cov)); % weighing matrices (input)   u_tilde - u

objm = 0; % Objective function
gm = [];  % constraints vector

for k = 1:N_MHE+1
    stm = Xm(:,k);
    h_x = h(stm);
    y_tilde = Pm(:,k);
    objm = objm+ (y_tilde-h_x)' * V * (y_tilde-h_x); % calculate obj
end

for k = 1:N_MHE
    conm = Um(:,k);
    u_tilde = Pm(:,N_MHE+ k);
    objm = objm+ (u_tilde-conm)' * W * (u_tilde-conm); % calculate obj
end

% multiple shooting constraints
for k = 1:N_MHE
    stm = Xm(:,k);  conm = Um(:,k);
    st_nextm = Xm(:,k+1);
    f_valuem = f(stm,conm);
    st_next_eulerm = stm+ (T*f_valuem);
    gm = [gm;st_nextm-st_next_eulerm]; % compute constraints
end

% make the decision variable one column  vector
OPT_variablesm = [reshape(Xm,3*(N_MHE+1),1);reshape(Um,2*N_MHE,1)];

nlp_mhe = struct('f', objm, 'x', OPT_variablesm, 'g', gm, 'p', Pm);

optsm = struct;
optsm.ipopt.max_iter = 2000;
optsm.ipopt.print_level =0;%0,3
optsm.print_time = 0;
optsm.ipopt.acceptable_tol =1e-8;
optsm.ipopt.acceptable_obj_change_tol = 1e-6;

solverm = nlpsol('solver', 'ipopt', nlp_mhe,optsm);


argsm = struct;

argsm.lbg(1:3*(N_MHE)) = 0;
argsm.ubg(1:3*(N_MHE)) = 0;

argsm.lbx(1:3:3*(N_MHE+1),1) = -2; %state x lower bound
argsm.ubx(1:3:3*(N_MHE+1),1) = 2; %state x upper bound
argsm.lbx(2:3:3*(N_MHE+1),1) = -2; %state y lower bound
argsm.ubx(2:3:3*(N_MHE+1),1) = 2; %state y upper bound
argsm.lbx(3:3:3*(N_MHE+1),1) = -pi/2; %state theta lower bound
argsm.ubx(3:3:3*(N_MHE+1),1) = pi/2; %state theta upper bound

argsm.lbx(3*(N_MHE+1)+1:2:3*(N_MHE+1)+2*N_MHE,1) = v_min; %v lower bound
argsm.ubx(3*(N_MHE+1)+1:2:3*(N_MHE+1)+2*N_MHE,1) = v_max; %v upper bound
argsm.lbx(3*(N_MHE+1)+2:2:3*(N_MHE+1)+2*N_MHE,1) = omega_min; %omega lower bound
argsm.ubx(3*(N_MHE+1)+2:2:3*(N_MHE+1)+2*N_MHE,1) = omega_max; %omega upper bound
%----------------------------------------------
% ALL OF THE ABOVE IS JUST A PROBLEM SET UP

% THE SIMULATION LOOP SHOULD START FROM HERE
%-------------------------------------------
t0 = 0;
x0 = [0.1 ; 0.1 ; 0.0];    % initial condition.
xs = [2 ; 2 ; 0.0]; % Reference posture.

xx(:,1) = x0; % xx contains the history of states
t(1) = t0;

u0 = zeros(N,2);        % two control inputs for each robot
X0 = repmat(x0,1,N+1)'; % initialization of the states decision variables

sim_tim = 60; % total sampling times

% Start MPC
mpciter = 0;
xx1 = [];
u_cl=[];

% the main simulaton loop... it works as long as the error is greater
% than 10^-6 and the number of mpc steps is less than its maximum
% value.
tic

X_estimate = []; % X_estimate contains the MHE estimate of the states
U_estimate = []; % U_estimate contains the MHE estimate of the controls

U0m = zeros(N_MHE,2);   % two control inputs for each robot
X0m = zeros(N_MHE+1,3); % initialization of the states decision variables
y_measurements = (x0(1:2))';

k=1;

r = [];
alpha = [];

while(norm((x0-xs),2) > 0.05 && mpciter < sim_tim / T)
    args.p   = [x0;xs]; % set the values of the parameters vector
    % initial value of the optimization variables
    args.x0  = [reshape(X0',3*(N+1),1);reshape(u0',2*N,1)];
    sol = solver('x0', args.x0, 'lbx', args.lbx, 'ubx', args.ubx,...
        'lbg', args.lbg, 'ubg', args.ubg,'p',args.p);
    u = reshape(full(sol.x(3*(N+1)+1:end))',2,N)'; % get controls only from the solution
    xx1(:,1:3,mpciter+1)= reshape(full(sol.x(1:3*(N+1)))',3,N+1)'; % get solution TRAJECTORY
    u_cl= [u_cl ; u(1,:)];
    t(mpciter+1) = t0;
    % Apply the control and shift the solution
    [t0, x0, u0] = shift(T, t0, x0, u,f,con_cov);
    
    r = [r; sqrt(x0(1)^2+x0(2)^2)  + sqrt(meas_cov(1,1))*randn(1)];
    alpha = [alpha; atan2(x0(2),x0(1))      + sqrt(meas_cov(2,2))*randn(1)];

    y_measurements=[r , alpha];

     if mpciter < N_MHE
    
     else
 
       if mpciter ==N_MHE
          U0m= u_cl(1:N_MHE,:); 
          X0m(:,1:2) = [y_measurements(1:N_MHE+1,1).*cos(y_measurements(1:N_MHE+1,2)),...
          y_measurements(1:N_MHE+1,1).*sin(y_measurements(1:N_MHE+1,2))];
       end
 
       argsm.p   = [y_measurements(k:k+N_MHE,:)',u_cl(k:k+N_MHE-1,:)'];
       % initial value of the optimization variables
       argsm.x0  = [reshape(X0m',3*(N_MHE+1),1);reshape(U0m',2*N_MHE,1)];
       solm = solverm('x0', argsm.x0, 'lbx', argsm.lbx, 'ubx', argsm.ubx,...
                      'lbg', argsm.lbg, 'ubg', argsm.ubg,'p',argsm.p);
       U_sol = reshape(full(solm.x(3*(N_MHE+1)+1:end))',2,N_MHE)'; % get controls only from the solution
       X_sol = reshape(full(solm.x(1:3*(N_MHE+1)))',3,N_MHE+1)'; % get solution TRAJECTORY
       X_estimate = [X_estimate;X_sol(N_MHE+1,:)];
       U_estimate = [U_estimate;U_sol(N_MHE,:)];
    
       x0=X_sol(N_MHE+1,:)';
       % Shift trajectory to initialize the next step
       X0m = [X_sol(2:end,:);X_sol(end,:)];
       U0m = [U_sol(2:end,:);U_sol(end,:)];
        k = k +1;
     end
    
     xx(:,mpciter+2) = x0;
     X0 = reshape(full(sol.x(1:3*(N+1)))',3,N+1)'; % get solution TRAJECTORY
        % Shift trajectory to initialize the next step
     X0 = [X0(2:end,:);X0(end,:)];
     mpciter
     mpciter = mpciter + 1;
end
toc


%-----------------------------------------
%-----------------------------------------
%-----------------------------------------
%    Start Plotting 
%-----------------------------------------
%-----------------------------------------
%-----------------------------------------

figure(1) %% Reference
subplot(311)
plot(t,xs(1) * ones(1, length(t)),'k','linewidth',1.5); hold on
grid on
subplot(312)
plot(t,xs(2) * ones(1, length(t)),'k','linewidth',1.5); hold on
grid on


figure(1) %% Measurement 
subplot(311)
plot(t,r.*cos(alpha),'r','linewidth',1.5); hold on
grid on
subplot(312)
plot(t,r.*sin(alpha),'r','linewidth',1.5); hold on
grid on

figure(1)  %% Actual output
subplot(311)
plot(t,xx(1,1:end-1),'b','linewidth',1.5); axis([0 t(end) 0 4]);hold on
legend('Reference','Measurement','MPC','Location','northwest')
ylabel('x (m)')
grid on

subplot(312)
plot(t,xx(2,1:end-1),'b','linewidth',1.5); axis([0 t(end) 0 4]);hold on
ylabel('y (m)')
grid on

subplot(313)
plot(t,xx(3,1:end-1),'b','linewidth',1.5); axis([0 t(end) -pi pi]);hold on
xlabel('time (seconds)')
ylabel('\theta (rad)')
grid on

figure(2) %% control signals
subplot(211)
stairs(t,u_cl(:,1),'k','linewidth',1.5); axis([0 t(end) -0.8 0.8])
ylabel('v (rad/s)')
grid on

subplot(212)
stairs(t,u_cl(:,2),'m','linewidth',1.5); axis([0 t(end) -2 2])
xlabel('time (seconds)')
ylabel('\omega (rad/s)')
grid on