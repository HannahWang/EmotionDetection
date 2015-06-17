clear all
clc

arduino=serial('/dev/cu.usbmodem1411', 'BaudRate', 9600);

fopen(arduino);

x=linspace(1 ,8000, 8000);

for i=1:length(x)
    %y(i)=float(fscanf(arduino, '%d'))/1024.0*5.0;
    y(i)=fscanf(arduino, '%d');
    y(i)=y(i)/1024*5;
end

fclose(arduino);
disp('making plot..')
plot(x, y);
for i=1:4000
    if y(i)<1.8
        tmp(i) = 0;
    else
        tmp(i)=y(i);
    end
end

[final, final2] = findpeaks(tmp);
for i =1:length(final2)-1
    over(i) = final2(i+1)-final2(i);
end
over=over*4*1.04/1000;
%time domain
SDNN=std(over);
RMSSD=sqrt(dot(over, over));
for i =1:length(over)-1
    ab(i) = over(i+1)-over(i);
end
%ab=abs(ab);
NN50=0;
for i=1:length(ab)
    if(ab(i)>=0.05 || ab(i)<=-0.05)
        NN50=NN50+1;
    end
end
pNN50=NN50/length(over);
SDSD=std(ab);
SDNN
RMSSD
NN50
pNN50
SDSD

%frequency domain
%overf=fft(over);

   
  
    