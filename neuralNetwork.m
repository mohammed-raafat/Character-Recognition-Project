clc;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Neural Network %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55\

%hd5l three inputs a w b w c arkam 3shwa2ya 
a= rot90(classesTypesSVM/1000);
b= rot90(Target/1000);
c= rot90(predict/1000);  

n=rot90(predict)*0.05;  %n dh noise aw nsbt al error f kol input 3shan abyn en al inputs de real numbers 
y=a*5+b.*c+7*c+n; %equation bdrb al weight fe kol input 3shan atl3 output 
I=[a; b; c]; %7tet al inputs f variable 
O=y; % w al y al hwa al output  fe variable 
R=[0 1; 0 1 ; 0 1]; %bgeb al min w al max bta3 kol input w kda kda al output dymn bytl3 ya zero y one 3shan by3dy 3la threshold  
S=[5 1]; %size of input=5 w al size of output=1
net = newff([0 1;0 1 ;0 1],S,{'tansig','purelin'}); %function bulti in ghza 3shan a3ml create network
net=train(net,I,O); %b3ml training ll network 
O1=sim(net,I); %brsm al network 
plot(1:271,O,1:271,O1);
scatter(O,O1);
