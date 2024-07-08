function Data_flt = LowPassFilter(Data, SampFreq, LP_Band, Order)

% Data inputs:
%   Data        n x m matrix with m variables
%   SampFreq    Data sampling frequency
%   LP_Band     Low-edge frequency (in Hz)
%   Order       Butterworth filter order

% Data outputs
%   Data_flt    Band-pass detrended and filtered data


%% filtering data
Wnn_emg = LP_Band/(SampFreq/2);
[ae2,be2] = butter(Order,Wnn_emg);
Data_flt = filtfilt(ae2,be2,(Data));


%% plotting filtered result
%figure
%plot(Data,'b', 'linewidth', 1.5);
%hold on
%plot(Data_flt, 'Color', 'r', 'linewidth', 2);
%box off
%title(['Column ' num2str(i)])
%title(['Low-pass (' num2str(LP_Band) ' Hz) Ch ' num2str(i)]);
%ylabel('uV'); xlabel('Time (s)')
%legend('Raw', 'Filtered'); legend box off
%set(gca, 'fontsize', 12)


