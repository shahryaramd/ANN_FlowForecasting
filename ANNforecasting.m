%% ANN Reservoir Inflow Forecasting  
% Obtain forecast reservoir inflow for lead tmes of 1-7 days using ANN
% based forecasting model, setup for Pensacola dam's drainage basin.
% A set of metrics are calculated at the end to assess the performance.


% INPUTS
% Time series of:
% 1. GFS forecast (7 days lead) - Precipitation, Min/Max Temperature
% 2. Insitu data from GSOD - - Precipitation, Min/Max Temp, Windspeed
% 3. Observed Antecedent Streamflow
% 4. Observed Antecedent Soil Moisture

% OUTPUTS
% Time series of forecast streamflow for the training period (variable 'y')
% and validation period (variable 'yV') for lead 1-7 days

% Developed by: Shahryar Khalique Ahmad (skahmad@uw.edu)

%%
clear all;
close all
% rng('default')
corr=zeros(7,1);
corrV=corr;
rmse=corr;
nrmse=corr;
rmseV=corr;
nse=corr;
nseV=corr;
range= [1461:4100];  % range for dates in the training data 
rangeV=[1:1460];   % range for dates in the validation data 

sz=4100;
for nn = 1:7
    
    % Read the GFS forecast data
    dat=dlmread(['Predictors/Forecast_GFS_L' num2str(nn) '.txt']);
    for i=1:size(dat,1)
        dayorg(i,1)=datenum(sprintf('%d', dat(i,1)),'yyyymmdd');
    end
    datg=dlmread('Predictors/Antecedent_GSOD.txt');     % Antecedent precip, min/max temp, windspeed
    dprecL(:,nn)=dat(1:sz,2);
    dtmaxL(:,nn)=dat(1:sz,3);
    dtminL(:,nn)=dat(1:sz,4);
    dwspdL(:,nn)=dat(1:sz,5);

end
dprecL=[datg(3:sz+2,2) datg(2:sz+1,2) datg(1:sz,2) dprecL];
dtmaxL=[datg(2:sz+1,3) datg(1:sz,3) dtmaxL];
dtminL=[datg(2:sz+1,4) datg(1:sz,4) dtminL];
dwspdL=[datg(2:sz+1,5) datg(1:sz,5) dwspdL];

datobs=xlsread('Predictors/ObservedFlow.xlsx');    % Observed streamflow (antecedent)
sm=dlmread('Predictors/Soilmoisture_GLDAS.txt');   % Observed GLDAS soil mositure (antecedent)
for nn=1:7
    QobsL(:,nn)=datobs(:,nn+7);         % Lagged observed flows, used as reference for L1-L7 forecasts
end
QobsL(isnan(QobsL))=0;

%% Baseflow Separation using Recursive Digital Filter
rL =zeros(size(dat,1),7);
b =rL;
bo=rL;
by =zeros(7,size(range,1));
byV=zeros(7,size(rangeV,1));
ry=by;
ryV=byV;

% Calibrated Parameters for baseflow separation filter 
a=0.05;
BFImax=0.73;
for nl=7:-1:1 
    % Define hindcast flow
    nn=8-nl;
    QhcL(:,nn)=datobs(:,nl+1);
    for k=2:size(dat,1)
        b(k,nn)=min(QhcL(k,nn),((1 - BFImax) * a * b(k-1,nn) + ( 1 - a ) * BFImax * QhcL(k,nn)) / ( 1 - a * BFImax) );
        r(k,nn)=QhcL(k,nn)-b(k,nn);
    end
end

netsFin = cell(1,7);
%% Model Training
rng(7)          % Control random number generation
hiddenLayerSize = 10;
for nn=1:7   % Model for each lead time
    days = dayorg(range);       %calibration period dates
    daysV = dayorg(rangeV);     %validation period dates
    obs(:,nn) = QobsL(range,nn);    
    obsV(:,nn) = QobsL(rangeV,nn);
    
    % IF-ELSE loop to use forecast flow/baseflow/runoff at previous lead
    % for subsequent day's forecasts starting L2
    if nn==1
            xQ = [QhcL(range,1) QhcL(range,2) ];
            xQV = [QhcL(rangeV,1) QhcL(rangeV,2)];
            bQ = [b(range,1) b(range,2) b(range,3) ];
            bQV = [b(rangeV,1) b(rangeV,2) b(rangeV,3)];
            rQ = [r(range,1) r(range,2) r(range,3) ];
            rQV = [r(rangeV,1) r(rangeV,2) r(rangeV,3)];
        elseif nn==2
            xQ =  [y(nn-1,:)'  QhcL(range,1)  ];
            xQV = [yV(nn-1,:)' QhcL(rangeV,1) ];
            bQ =  [by(nn-1,:)'  b(range,1) b(range,2) ];
            bQV = [byV(nn-1,:)' b(rangeV,1) b(rangeV,2)];
            rQ =  [ry(nn-1,:)'  r(range,1) r(range,2) ];
            rQV = [ryV(nn-1,:)' r(rangeV,1) r(rangeV,2)];
        elseif nn==3
            xQ = [y(nn-1,:)' y(nn-2,:)' ];
            xQV = [yV(nn-1,:)' yV(nn-2,:)'  ];
            bQ =  [by(nn-1,:)'  by(nn-2,:)' b(range,1)];
            bQV = [byV(nn-1,:)' byV(nn-2,:)' b(rangeV,1)];
            rQ =  [ry(nn-1,:)'  ry(nn-2,:)' r(range,1)];
            rQV = [ryV(nn-1,:)' ryV(nn-2,:)' r(rangeV,1)];
        else
            xQ = [y(nn-1,:)' y(nn-2,:)' ];
            xQV = [yV(nn-1,:)' yV(nn-2,:)' ];
            bQ =  [by(nn-1,:)'  by(nn-2,:)' by(nn-3,:)' ];
            bQV = [byV(nn-1,:)' byV(nn-2,:)' byV(nn-3,:)'];
            rQ =  [ry(nn-1,:)'  ry(nn-2,:)' ry(nn-3,:)' ];
            rQV = [ryV(nn-1,:)' ryV(nn-2,:)' ryV(nn-3,:)'];
    end
    %  Predcitor nodes
    x=[dprecL(range,nn+3) dprecL(range,nn+2) dprecL(range,nn+1)...  
        dtmaxL(range,nn+1)    ...  
        dtminL(range,nn+1)  ... 
        sm(range,6) sm(range,5) ...
        rQ bQ ];
    xV=[dprecL(rangeV,nn+3) dprecL(rangeV,nn+2) dprecL(rangeV,nn+1)...  
        dtmaxL(rangeV,nn+1) ...  
        dtminL(rangeV,nn+1)  ...
        sm(rangeV,6) sm(rangeV,5) ...
        rQV bQV];
        
    % Start NN 
    x = x';
    xV = xV';
    t = obs(:,nn)';
    tV = obsV(:,nn)';
    trainFcn = 'trainlm'; % Training function: Levenberg-Marquardt.
    % Create a Fitting Network
    net = feedforwardnet(hiddenLayerSize,trainFcn);
    net.performFcn='msereg';

    net.trainParam.epochs = 300;
    net.trainParam.goal = 1e-5;
    net.performParam.regularization = 0.5;

    % Setup Division of Data for Training, Validation, Testing
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 15/100;
    % Train the Network
    numNN=20;
    perfs = zeros(1, numNN);
    nets = cell(1, numNN);   
    for i = 1:numNN
        nets{i} = train(net, x, t);
        neti = nets{i};
        yt(i,:) = neti(x);
        tmp = yt(i,:);
        tmp(isnan(tmp)) = 0;
        yt(i,:) = tmp;
        perfs(i) = mse(neti, t, yt);
        ytV(i,:) = neti(xV);
        tmp = ytV(i,:);
        tmp(isnan(tmp)) = 0;
        ytV(i,:) = tmp;
        ct = corrcoef(tV,ytV(i,:));
        cx(i)=ct(1,2);
    end
    [idv,idx]=max(cx);
    netsFin{nn}  = nets{idx};   %evaluate for best trained ANN      
    y(nn,:) = yt(idx,:);
    yV(nn,:)= ytV(idx,:);


    %calculate baseflow of modeled streamflow y, yV
    for k=2:size(y,2)
        by(nn,k)=min(y(nn,k),((1 - BFImax) * a * by(nn,k-1) + ( 1 - a ) * BFImax * y(nn,k)) / ( 1 - a * BFImax) );
        ry(nn,k)=y(nn,k)-by(nn,k);
    end
    for k=2:size(yV,2)
        byV(nn,k)=min(yV(nn,k),((1 - BFImax) * a * byV(nn,k-1) + ( 1 - a ) * BFImax * yV(nn,k)) / ( 1 - a * BFImax) );
        ryV(nn,k)=yV(nn,k)-byV(nn,k);
    end
    
    % Training metrics
    e = gsubtract(t,y(nn,:));
    rmse(nn,1) = (perform(net,t,y(nn,:)))^0.5;
    nrmse(nn,1) = (perform(net,t,y(nn,:)))^0.5/std(t);
    c = corrcoef(t,y(nn,:));
    corr(nn,1)=c(1,2);
    E = t- y(nn,:);
    SSE = sum(E.^2);
    u = mean(t);
    SSU = sum((t - u).^2);
    nse(nn,1) = 1 - SSE/SSU;

    % Validation metrics
    e = gsubtract(tV,yV(nn,:));
    rmseV(nn,1) = (perform(net,tV,yV(nn,:)))^0.5;
    nrmseV(nn,1) = (perform(net,tV,yV(nn,:)))^0.5/std(tV);
    c = corrcoef(tV,yV(nn,:));
    corrV(nn,1)=c(1,2);
    E = tV- yV(nn,:);
    SSE = sum(E.^2);
    u = mean(tV);
    SSU = sum((tV - u).^2);
    nseV(nn,1) = 1 - SSE/SSU;
end
statsnnCal=[nse corr];      %stats
statsnnVal=[nseV rmseV]     

%% Training-Validation Streamflow Plots
obs_plV=obsV';
obs_pl=obs';
% Validation plot
figure
for nl=1:3
    subplot(3,1,nl)
    nn=3*nl-2;
    hold on
    if nn==6
        nn=7;
    end
    plot(daysV',yV(nn,:),'b')
    plot(daysV',obs_plV(nn,:),'r--')
    ylabel('Flow, cfs')
    ylim([0 3e5])
    datetick('x', 'mmm yy')
end
legend('NN Validated','Observed')

% Training plot
figure
for nl=1:3
    subplot(3,1,nl)
    nn=3*nl-2;
    hold on
    if nn==6
        nn=7;
    end 
    plot(days',y(nn,:),'b','LineWidth',0.8)
    plot(days',obs_pl(nn,:),'r--','LineWidth',0.8)
    ylim([0 20e4])
    ylabel('Flow, cfs')
    datetick('x', 'mmm yy')
end
legend('NN Trained','Observed')
%% plot baseflow
figure, hold on
plot(daysV',yV(1,:))
plot(daysV',byV(1,:))
ylabel('flow (cfs)')
legend('Streamflow,','Baseflow')
 datetick('x', 'mmm yy')
title('Separated Baseflow for Pensacola basin for lead 1')

%% training-testing metrics plot
figure
subplot(1,2,1)
hold on
plot([1:7], corr, 'b-.')
plot([1:7], corrV, 'b-o')
% ylim([0 1])
xlabel('Lead, days')
title('Correlation')
legend('NN Calibrated','NN Validated')
subplot(1,2,2)
hold on
plot([1:7], nse, 'k-.')
plot([1:7], nseV, 'b-o')
% ylim([-0.2 1])
xlabel('Lead, days')
legend('NN Calibrated','NN Validated')
title('NSE')

