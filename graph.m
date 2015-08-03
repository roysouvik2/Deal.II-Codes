X = importdata('Smoothing_Term.txt');
Y = importdata('L2_Error.txt');
plot(X,Y)
xlabel('Smoothing Parameter');
ylabel('L2 Error');
title('Graph of smoothing parameter vs L2 error of the solution');